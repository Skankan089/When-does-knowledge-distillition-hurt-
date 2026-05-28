"""
Offline Teacher-Forced Scoring Script
=======================================
Scores gold summaries through Qwen2.5-32B and Qwen2.5-14B teachers to extract
token-level logits/logprobs for distillation. Uses teacher-forced scoring
(NOT free generation) — a single forward pass per batch.

Why teacher-forced?
  1.  Token alignment: teacher and student predict the SAME gold tokens
      → KL, EWAD, and CPDP are well-defined
  2.  ~10–50× faster than autoregressive generation
  3.  Deterministic (no sampling noise)
  4.  Standard approach in distillation literature

Usage:
    python generate_teacher_outputs.py --teacher 32b
    python generate_teacher_outputs.py --teacher 14b
    python generate_teacher_outputs.py --teacher 32b --split train --batch-size 4
"""

import os
import sys
import json
import math
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


def load_dataset():
    """Load and split BanSum dataset."""
    print(f"\n{'='*80}")
    print("LOADING BANSUM DATASET")
    print(f"{'='*80}")
    
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples loaded: {len(data)}")
    
    # Shuffle with fixed seed for reproducibility
    np.random.seed(SEED)
    indices = np.random.permutation(len(data))
    data = [data[i] for i in indices]
    
    # Split
    total = len(data)
    train_end = int(TRAIN_SPLIT * total)
    val_end = train_end + int(VAL_SPLIT * total)
    
    splits = {
        'train': data[:train_end],
        'validation': data[train_end:val_end],
        'test': data[val_end:]
    }
    
    for name, split_data in splits.items():
        print(f"  {name}: {len(split_data)} samples")
    
    # Limit samples if configured
    if MAX_SAMPLES is not None:
        for name in splits:
            splits[name] = splits[name][:MAX_SAMPLES]
        print(f"\n  [LIMITED TO {MAX_SAMPLES} SAMPLES PER SPLIT]")
    
    return splits


class HuggingFaceTeacherBackend:
    """
    HuggingFace Transformers backend for teacher-forced scoring.
    Loads model in 4-bit quantization for memory efficiency.
    
    Teacher-forced scoring: given (article, gold_summary), we concatenate them
    and run a SINGLE forward pass to extract the teacher's log-probability
    distribution over each gold summary token. No autoregressive generation.
    """
    
    def __init__(self, model_name, quantization="4bit"):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        print(f"\nLoading model: {model_name}")
        print(f"Quantization: {quantization}")
        self.model_name = model_name
        self.is_gemma = "gemma" in model_name.lower()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Quantization config
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = None
        
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }
        if quant_config:
            load_kwargs["quantization_config"] = quant_config

        # Gemma models need eager attention (sliding window + SDPA incompatible)
        if self.is_gemma:
            load_kwargs["attn_implementation"] = "eager"
            print("Gemma detected — using eager attention")
        else:
            try:
                import flash_attn  # noqa: F401
                load_kwargs["attn_implementation"] = "flash_attention_2"
                print("Using Flash Attention 2")
            except ImportError:
                print("Flash Attention 2 not installed — using default attention")
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        
        self.vocab_size = len(self.tokenizer)
        print(f"Model loaded. Vocab size: {self.vocab_size}")
        
        # GPU memory usage
        if torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"GPU memory used: {mem_gb:.2f} GB")

    def build_prompt(self, text):
        prompt = TEACHER_PROMPT_TEMPLATE.format(text=text)
        if not self.is_gemma:
            return prompt, True

        messages = [{"role": "user", "content": prompt.rstrip()}]
        try:
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return chat_prompt, False
        except Exception as exc:
            print(f"Gemma chat template failed ({exc}); falling back to raw prompt.")
            return prompt, True
    
    def score_gold_summaries(self, texts, gold_summaries, max_input_tokens, max_output_tokens):
        """
        Teacher-forced scoring: extract per-token logprobs for gold summaries
        via a SINGLE forward pass (no generation).
        
        For a causal LM with input [t0, t1, ..., tN]:
            logits[i] predicts token[i+1]
        
        So if input = [prompt_0, ..., prompt_{P-1}, summary_0, ..., summary_{S-1}]:
            - logits[P-1]     → P(summary_0 | prompt)
            - logits[P+j-1]   → P(summary_j | prompt + summary_0..j-1)
        
        Args:
            texts: list of article texts
            gold_summaries: list of gold summary texts
            max_input_tokens: max tokens for the article/prompt
            max_output_tokens: max tokens for the gold summary
            
        Returns:
            list of dicts, each containing:
                - 'summary': gold summary text
                - 'token_ids': list of gold summary token IDs
                - 'top_k_logprobs': list of [(token_id, logprob), ...] per position
        """
        results = []
        
        # Tokenize all items to determine boundaries and prepare batch
        all_input_ids = []
        all_attention_masks = []
        all_summary_starts = []    # Where summary begins (in unpadded sequence)
        all_summary_token_ids = [] # Gold summary token IDs
        
        # Pre-compute actual token length of each gold summary so we never
        # truncate it. max_output_tokens is used as a hard safety ceiling only.
        actual_summary_tokens = [
            self.tokenizer(
                gs,
                truncation=False,
                add_special_tokens=False,
                return_tensors=None,
            )['input_ids']
            for gs in gold_summaries
        ]
        # Per-sample budget: actual length + 10 buffer, capped at max_output_tokens
        per_sample_max = [
            min(len(ids) + 10, max_output_tokens)
            for ids in actual_summary_tokens
        ]

        for idx, (text, gold_summary) in enumerate(zip(texts, gold_summaries)):
            prompt, add_prompt_special_tokens = self.build_prompt(text)
            
            # Tokenize prompt and summary SEPARATELY to know the boundary
            prompt_enc = self.tokenizer(
                prompt, 
                truncation=True, 
                max_length=max_input_tokens, 
                add_special_tokens=add_prompt_special_tokens,
                return_tensors=None,
            )
            summary_enc = self.tokenizer(
                gold_summary, 
                truncation=True,
                # Use the actual gold-length budget so nothing is truncated
                max_length=per_sample_max[idx],
                add_special_tokens=False,  # No BOS/EOS — raw summary tokens
                return_tensors=None,
            )
            
            prompt_ids = prompt_enc['input_ids']
            summary_ids = summary_enc['input_ids']
            
            # Concatenate: [prompt_tokens, summary_tokens]
            full_ids = prompt_ids + summary_ids
            full_mask = [1] * len(full_ids)
            
            all_input_ids.append(full_ids)
            all_attention_masks.append(full_mask)
            all_summary_starts.append(len(prompt_ids))
            all_summary_token_ids.append(summary_ids)
        
        # Left-pad for batching (causal LM convention)
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id
        
        padded_input_ids = []
        padded_attention_masks = []
        
        for ids, mask in zip(all_input_ids, all_attention_masks):
            pad_len = max_len - len(ids)
            padded_input_ids.append([pad_id] * pad_len + ids)
            padded_attention_masks.append([0] * pad_len + mask)
        
        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long).to(self.model.device)
        attention_mask_tensor = torch.tensor(padded_attention_masks, dtype=torch.long).to(self.model.device)
        
        # ===== SINGLE FORWARD PASS (the whole point!) =====
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
            )
            logits = outputs.logits  # (batch, seq_len, vocab_size)
        
        # Extract per-token logprobs for summary positions
        for i in range(len(texts)):
            unpadded_len = len(all_input_ids[i])
            pad_len = max_len - unpadded_len
            
            # Summary starts at this position in the PADDED sequence
            summary_start_padded = pad_len + all_summary_starts[i]
            summary_ids = all_summary_token_ids[i]
            summary_len = len(summary_ids)
            
            token_logprobs_list = []
            
            for t in range(summary_len):
                # Causal LM: logits[pos] predicts token[pos+1]
                # To get P(summary_token_t | prompt, summary_0..t-1):
                #   logit position = summary_start_padded - 1 + t
                logit_pos = summary_start_padded - 1 + t
                
                logprobs = F.log_softmax(logits[i, logit_pos].float(), dim=-1)
                
                gold_token_id = summary_ids[t]
                gold_logprob = logprobs[gold_token_id].item()

                # Top-k extraction. We also append the gold token if it is
                # outside top-k, but keep gold_in_top_k so downstream code can
                # detect low teacher support instead of silently trusting it.
                top_k = min(LOGIT_TOP_K, logprobs.shape[0])
                top_k_vals, top_k_ids = torch.topk(logprobs, k=top_k)
                top_k_id_list = top_k_ids.cpu().tolist()
                
                top_k_entries = list(zip(
                    top_k_id_list,
                    top_k_vals.cpu().tolist()
                ))
                gold_in_top_k = gold_token_id in top_k_id_list
                if not gold_in_top_k:
                    top_k_entries.append((gold_token_id, gold_logprob))

                token_logprobs_list.append(top_k_entries)
            
            results.append({
                'summary': gold_summaries[i],
                'token_ids': summary_ids,
                'top_k_logprobs': token_logprobs_list,
                'gold_logprobs': [
                    entries[[tok for tok, _ in entries].index(summary_ids[t])][1]
                    for t, entries in enumerate(token_logprobs_list)
                ],
                'gold_in_top_k': [
                    summary_ids[t] in [tok for tok, _ in entries[:LOGIT_TOP_K]]
                    for t, entries in enumerate(token_logprobs_list)
                ],
            })
        
        # Free GPU memory
        del logits, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results


def save_teacher_outputs(results, output_dir, split_name, is_first_batch):
    """Save teacher outputs to disk as JSONL for streaming during training."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{split_name}.jsonl")
    mode = 'w' if is_first_batch else 'a'
    
    with open(output_file, mode, encoding='utf-8') as f:
        for result in results:
            record = {
                'summary': result['summary'],
                'token_ids': result['token_ids'],
                'top_k_logprobs': result['top_k_logprobs'],
            }
            if 'gold_logprobs' in result:
                record['gold_logprobs'] = result['gold_logprobs']
            if 'gold_in_top_k' in result:
                record['gold_in_top_k'] = result['gold_in_top_k']
            if 'original_id' in result:
                record['original_id'] = result['original_id']
            if 'gold_summary' in result:
                record['gold_summary'] = result['gold_summary']
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def generate_for_split(backend, data, output_dir, split_name, batch_size=4):
    """
    Score gold summaries for an entire data split using teacher-forced forward passes.
    """
    print(f"\n{'='*80}")
    print(f"TEACHER-FORCED SCORING — {split_name.upper()}")
    print(f"{'='*80}")
    print(f"Samples: {len(data)}, Batch size: {batch_size}")
    
    # Check for existing progress (resume support)
    output_file = os.path.join(output_dir, f"{split_name}.jsonl")
    existing_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_count = sum(1 for _ in f)
        print(f"Found {existing_count} existing outputs. Resuming from sample {existing_count}.")
    
    data_remaining = data[existing_count:]
    if len(data_remaining) == 0:
        print("All samples already processed. Skipping.")
        return
    
    num_batches = math.ceil(len(data_remaining) / batch_size)
    is_first_write = (existing_count == 0)
    
    for batch_idx in tqdm(range(num_batches), desc=f"  {split_name}"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(data_remaining))
        batch_data = data_remaining[start:end]
        
        # Extract text AND gold summaries from batch
        texts = [item[DATASET_TEXT_KEY] for item in batch_data]
        gold_summaries = [item[DATASET_SUMMARY_KEY] for item in batch_data]

        # Per-batch adaptive ceiling: max actual gold token count + 20 buffer.
        # This ensures no gold summary token is ever truncated during scoring.
        batch_gold_token_counts = [
            len(backend.tokenizer(
                gs, truncation=False, add_special_tokens=False, return_tensors=None
            )['input_ids'])
            for gs in gold_summaries
        ]
        adaptive_max_output = max(batch_gold_token_counts) + 20

        try:
            results = backend.score_gold_summaries(
                texts=texts,
                gold_summaries=gold_summaries,
                max_input_tokens=TEACHER_MAX_INPUT_TOKENS,
                max_output_tokens=adaptive_max_output,
            )
            
            # Add original data IDs for traceability
            for j, result in enumerate(results):
                result['original_id'] = batch_data[j].get('ID', existing_count + start + j)
                result['gold_summary'] = gold_summaries[j]
            
            # Save incrementally
            save_teacher_outputs(
                results, output_dir, split_name, 
                is_first_batch=(is_first_write and batch_idx == 0)
            )
            
        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at batch {batch_idx} (batch_size={batch_size}). "
                  f"Try reducing --batch-size.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Try processing one by one as fallback
            print("  Falling back to batch_size=1 for this batch...")
            for j in range(len(batch_data)):
                try:
                    single_max = len(backend.tokenizer(
                        gold_summaries[j], truncation=False,
                        add_special_tokens=False, return_tensors=None
                    )['input_ids']) + 20
                    single_result = backend.score_gold_summaries(
                        texts=[texts[j]],
                        gold_summaries=[gold_summaries[j]],
                        max_input_tokens=TEACHER_MAX_INPUT_TOKENS,
                        max_output_tokens=single_max,
                    )
                    single_result[0]['original_id'] = batch_data[j].get('ID', existing_count + start + j)
                    single_result[0]['gold_summary'] = gold_summaries[j]
                    save_teacher_outputs(single_result, output_dir, split_name, is_first_batch=False)
                except Exception as inner_e:
                    print(f"    ERROR on sample {existing_count + start + j}: {inner_e}")
                    continue
            
        except Exception as e:
            print(f"\n  ERROR at batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Skipping batch. Progress saved up to this point.")
            continue
    
    # Verify count
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            final_count = sum(1 for _ in f)
        print(f"\n  Completed: {final_count}/{len(data)} samples saved to {output_file}")


def build_metadata(model_name, output_dir, splits, splits_to_run, batch_size, backend):
    return {
        "teacher_model": model_name,
        "quantization": TEACHER_QUANTIZATION,
        "mode": "teacher_forced_scoring_adaptive_length",
        "description": (
            "Single forward pass per batch — scores full gold summaries. "
            "max_output_tokens is computed per-batch from actual gold token counts "
            "so no gold token is ever truncated. Each gold token's logprob is "
            "stored even when it falls outside top-k, and gold_in_top_k records "
            "whether the teacher genuinely ranked it in the saved top-k."
        ),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "dataset": DATASET_FILE,
        "tokenizer_name": model_name,
        "vocab_size": backend.vocab_size,
        "qwen_compatible": "qwen" in model_name.lower(),
        "cross_tokenizer_teacher": "qwen" not in model_name.lower(),
        "logit_top_k": LOGIT_TOP_K,
        "max_input_tokens": TEACHER_MAX_INPUT_TOKENS,
        "max_output_tokens": "adaptive (gold token count + 20 per batch)",
        "batch_size": batch_size,
        "output_dir": output_dir,
        "splits_scored": splits_to_run,
        "samples_per_split": {name: len(splits[name]) for name in splits_to_run},
    }


def save_metadata(output_dir, metadata):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    import config as cfg
    parser = argparse.ArgumentParser(description="Teacher-forced scoring for distillation")
    parser.add_argument(
        "--teacher", 
        type=str, 
        required=True, 
        choices=["32b", "14b", "qwen3_14b", "gemma3_12b"],
        help="Which teacher to run: 32b | 14b | qwen3_14b | gemma3_12b"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset path (default: from config.py)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "validation", "test", "all"],
        help="Which data split to score"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TEACHER_BATCH_SIZE,
        help=f"Batch size for scoring (default: {TEACHER_BATCH_SIZE})"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Quick test: limit to 1000 samples (overrides config)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap samples per split (e.g. 5000 for a smoke test). Outputs go to teacher_outputs_smoke/."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for this teacher run."
    )
    args = parser.parse_args()

    if args.test_mode and not cfg.TEST_MODE:
        cfg.MAX_SAMPLES = 1000
        cfg.TEACHER_32B_OUTPUTS = os.path.join(cfg.BASE_DIR, "teacher_outputs_test", "teacher_32b")
        cfg.TEACHER_14B_OUTPUTS = os.path.join(cfg.BASE_DIR, "teacher_outputs_test", "teacher_14b")
        globals()['MAX_SAMPLES'] = 1000
        globals()['TEACHER_32B_OUTPUTS'] = cfg.TEACHER_32B_OUTPUTS
        globals()['TEACHER_14B_OUTPUTS'] = cfg.TEACHER_14B_OUTPUTS
        print("\n*** --test-mode: limiting to 1000 samples, using test output dirs ***\n")

    if args.max_samples is not None:
        cfg.MAX_SAMPLES = args.max_samples
        cfg.TEACHER_32B_OUTPUTS = os.path.join(cfg.BASE_DIR, "teacher_outputs_smoke", "teacher_32b")
        cfg.TEACHER_14B_OUTPUTS = os.path.join(cfg.BASE_DIR, "teacher_outputs_smoke", "teacher_14b")
        globals()['MAX_SAMPLES'] = args.max_samples
        globals()['TEACHER_32B_OUTPUTS'] = cfg.TEACHER_32B_OUTPUTS
        globals()['TEACHER_14B_OUTPUTS'] = cfg.TEACHER_14B_OUTPUTS
        print(f"\n*** --max-samples {args.max_samples}: smoke-test mode, outputs → teacher_outputs_smoke/ ***\n")
    
    # Override dataset if provided
    if args.dataset is not None:
        cfg.DATASET_FILE = args.dataset
        globals()['DATASET_FILE'] = args.dataset
        print(f"\n*** Dataset overridden → {args.dataset} ***\n")

    # Select model and output directory
    if args.teacher == "32b":
        model_name = TEACHER_32B_MODEL
        output_dir = TEACHER_32B_OUTPUTS
    elif args.teacher == "qwen3_14b":
        model_name = "Qwen/Qwen3-14B"
        parent_dir = "teacher_outputs_smoke" if args.max_samples is not None else "teacher_outputs"
        output_dir = os.path.join(cfg.BASE_DIR, parent_dir, "teacher_qwen3_14b")
    elif args.teacher == "gemma3_12b":
        model_name = "google/gemma-3-12b-it"
        parent_dir = "teacher_outputs_smoke" if args.max_samples is not None else "teacher_outputs"
        output_dir = os.path.join(cfg.BASE_DIR, parent_dir, "teacher_gemma3_12b")
    else:
        model_name = TEACHER_14B_MODEL
        output_dir = TEACHER_14B_OUTPUTS

    if args.output_dir is not None:
        output_dir = args.output_dir
    
    print(f"\n{'='*80}")
    print(f"DUAL-TEACHER KNOWLEDGE DISTILLATION — TEACHER-FORCED SCORING")
    print(f"{'='*80}")
    print(f"Teacher: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Dataset: {DATASET_FILE}")
    print(f"Mode: Teacher-forced (single forward pass per batch, no generation)")
    print(f"Batch size: {args.batch_size}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load dataset
    splits = load_dataset()
    
    # Load teacher model
    backend = HuggingFaceTeacherBackend(model_name, quantization=TEACHER_QUANTIZATION)
    
    # Score for requested splits
    splits_to_run = list(splits.keys()) if args.split == "all" else [args.split]
    metadata = build_metadata(model_name, output_dir, splits, splits_to_run, args.batch_size, backend)
    save_metadata(output_dir, metadata)
    
    for split_name in splits_to_run:
        generate_for_split(
            backend=backend,
            data=splits[split_name],
            output_dir=output_dir,
            split_name=split_name,
            batch_size=args.batch_size,
        )
    
    print(f"\n{'='*80}")
    print("TEACHER-FORCED SCORING COMPLETE!")
    print(f"{'='*80}")
    print(f"Outputs saved to: {output_dir}")
    
    metadata["completed_timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_metadata(output_dir, metadata)


if __name__ == "__main__":
    main()
