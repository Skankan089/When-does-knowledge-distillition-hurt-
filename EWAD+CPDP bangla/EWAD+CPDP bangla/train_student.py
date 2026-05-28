"""
Student Distillation Training Script
======================================
Trains Qwen2.5-3B student with LoRA using pre-computed teacher logits.
Supports all 8 experiment configurations (3 baselines + 5 ablations).

Usage:
    python train_student.py --experiment ewad_full
    python train_student.py --experiment baseline_no_distill
    python train_student.py --experiment ewad_cpdp
    
For all experiments:
    python run_all_experiments.py
"""

import os
import sys
import json
import math
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from ewad_cpdp_loss import (
    DualTeacherDistillationLoss,
)


# ============================================================================
# Dataset
# ============================================================================
class IndexedJsonl:
    """
    Random-access JSONL reader that keeps only byte offsets in RAM.

    Teacher files are multi-GB; parsing them into Python lists multiplies
    memory use badly. This reader scans once to build offsets, then decodes
    only the requested line inside __getitem__.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.offsets = []
        self._fh = None

        with open(filepath, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __bool__(self):
        return len(self.offsets) > 0

    def trim(self, length):
        if length < len(self.offsets):
            self.offsets = self.offsets[:length]

    def _file(self):
        if self._fh is None or self._fh.closed:
            self._fh = open(self.filepath, "rb")
        return self._fh

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError(idx)
        f = self._file()
        f.seek(self.offsets[idx])
        line = f.readline()
        return json.loads(line.decode("utf-8"))


class DistillationDataset(Dataset):
    """
    Dataset that combines gold labels with pre-computed teacher outputs.
    Loads teacher outputs from JSONL files lazily for memory efficiency.
    """
    
    def __init__(
        self, 
        data_samples,          # list of dicts from BanSum
        teacher_32b_file,      # path to teacher_32b JSONL
        teacher_14b_file,      # path to teacher_14b JSONL
        cpdp_teacher_32b_file, # optional Qwen-compatible CPDP teacher JSONL
        cpdp_teacher_14b_file, # optional Qwen-compatible CPDP teacher JSONL
        tokenizer,
        max_input_tokens,
        max_output_tokens,
        use_distillation=True,
    ):
        self.data_samples = data_samples
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.use_distillation = use_distillation
        
        # Index teacher outputs lazily. The train teacher files can be >10 GB;
        # only offsets are kept in RAM.
        self.teacher_32b_outputs = None
        self.teacher_14b_outputs = None
        self.cpdp_teacher_32b_outputs = None
        self.cpdp_teacher_14b_outputs = None
        self.teacher_32b_qwen_compatible = True
        self.teacher_14b_qwen_compatible = True
        self.cpdp_teacher_32b_qwen_compatible = True
        self.cpdp_teacher_14b_qwen_compatible = True
        
        if use_distillation and teacher_32b_file and os.path.exists(teacher_32b_file):
            print(f"  Indexing 32B teacher outputs from {teacher_32b_file}...")
            self.teacher_32b_outputs = IndexedJsonl(teacher_32b_file)
            self.teacher_32b_qwen_compatible = self._is_qwen_compatible(teacher_32b_file)
            if not self.teacher_32b_qwen_compatible:
                print("  32B teacher is cross-tokenizer; using it for EWAD support/gating only.")
            print(f"  Indexed {len(self.teacher_32b_outputs)} teacher 32B outputs")
        
        if use_distillation and teacher_14b_file and os.path.exists(teacher_14b_file):
            print(f"  Indexing 14B teacher outputs from {teacher_14b_file}...")
            self.teacher_14b_outputs = IndexedJsonl(teacher_14b_file)
            self.teacher_14b_qwen_compatible = self._is_qwen_compatible(teacher_14b_file)
            if not self.teacher_14b_qwen_compatible:
                print("  14B teacher is cross-tokenizer; using it for EWAD support/gating only.")
            print(f"  Indexed {len(self.teacher_14b_outputs)} teacher 14B outputs")

        if use_distillation and cpdp_teacher_32b_file and os.path.exists(cpdp_teacher_32b_file):
            print(f"  Indexing CPDP teacher A outputs from {cpdp_teacher_32b_file}...")
            self.cpdp_teacher_32b_outputs = IndexedJsonl(cpdp_teacher_32b_file)
            self.cpdp_teacher_32b_qwen_compatible = self._is_qwen_compatible(cpdp_teacher_32b_file)
            if not self.cpdp_teacher_32b_qwen_compatible:
                print("  WARNING: CPDP teacher A is cross-tokenizer and will be ignored.")
            print(f"  Indexed {len(self.cpdp_teacher_32b_outputs)} CPDP teacher A outputs")

        if use_distillation and cpdp_teacher_14b_file and os.path.exists(cpdp_teacher_14b_file):
            print(f"  Indexing CPDP teacher B outputs from {cpdp_teacher_14b_file}...")
            self.cpdp_teacher_14b_outputs = IndexedJsonl(cpdp_teacher_14b_file)
            self.cpdp_teacher_14b_qwen_compatible = self._is_qwen_compatible(cpdp_teacher_14b_file)
            if not self.cpdp_teacher_14b_qwen_compatible:
                print("  WARNING: CPDP teacher B is cross-tokenizer and will be ignored.")
            print(f"  Indexed {len(self.cpdp_teacher_14b_outputs)} CPDP teacher B outputs")
        
        # Validate alignment across every loaded teacher stream. All JSONL
        # files must follow the same shuffled split order; if one is short,
        # cap all streams to the shared prefix instead of silently shifting.
        loaded_streams = [
            ("EWAD teacher A", self.teacher_32b_outputs),
            ("EWAD teacher B", self.teacher_14b_outputs),
            ("CPDP teacher A", self.cpdp_teacher_32b_outputs),
            ("CPDP teacher B", self.cpdp_teacher_14b_outputs),
        ]
        lengths = [len(self.data_samples)]
        for name, stream in loaded_streams:
            if stream is not None:
                lengths.append(len(stream))
                if len(stream) != len(self.data_samples):
                    print(f"  WARNING: {name} outputs ({len(stream)}) != data samples ({len(self.data_samples)})")

        min_len = min(lengths)
        if min_len != len(self.data_samples):
            print(f"  Using shared aligned prefix: {min_len} samples")
            self.data_samples = self.data_samples[:min_len]
            if self.teacher_32b_outputs:
                self.teacher_32b_outputs.trim(min_len)
            if self.teacher_14b_outputs:
                self.teacher_14b_outputs.trim(min_len)
            if self.cpdp_teacher_32b_outputs:
                self.cpdp_teacher_32b_outputs.trim(min_len)
            if self.cpdp_teacher_14b_outputs:
                self.cpdp_teacher_14b_outputs.trim(min_len)

    def _is_qwen_compatible(self, teacher_file):
        metadata_path = os.path.join(os.path.dirname(teacher_file), "metadata.json")
        if not os.path.exists(metadata_path):
            print(f"  WARNING: Missing metadata for {teacher_file}; assuming Qwen-compatible.")
            return True
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as exc:
            print(f"  WARNING: Could not read {metadata_path}: {exc}; assuming Qwen-compatible.")
            return True
        model_name = str(metadata.get("teacher_model", "")).lower()
        tokenizer_name = str(metadata.get("tokenizer_name", model_name)).lower()
        return "qwen" in model_name or "qwen" in tokenizer_name

    @staticmethod
    def _gold_support_mask(teacher_record, gold_token_ids, summary_len, qwen_compatible=True):
        """
        Mark positions where the teacher genuinely placed the gold token in
        its saved top-k. New teacher files store gold_in_top_k explicitly;
        older files fall back to membership in top_k_logprobs.

        For cross-tokenizer teachers such as Gemma, positions do not align
        with Qwen student tokens. In that case we repeat the sample-level
        support rate across Qwen positions and use it only as a gate.
        """
        if not teacher_record:
            return [0] * summary_len

        topk = teacher_record.get('top_k_logprobs', [])
        explicit = teacher_record.get('gold_in_top_k', None)

        if not qwen_compatible:
            if explicit:
                support = sum(1 for x in explicit if x) / max(len(explicit), 1)
            else:
                support = 0.0
            return [float(support)] * summary_len

        compare_len = min(len(topk), len(gold_token_ids), summary_len)

        mask = []
        for t in range(compare_len):
            if explicit is not None and t < len(explicit):
                mask.append(1 if explicit[t] else 0)
            else:
                topk_ids = {int(entry[0]) for entry in topk[t]} if topk[t] else set()
                mask.append(1 if int(gold_token_ids[t]) in topk_ids else 0)

        if len(mask) < summary_len:
            mask.extend([0] * (summary_len - len(mask)))
        return mask[:summary_len]
    
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        text = sample[DATASET_TEXT_KEY]
        gold_summary = sample[DATASET_SUMMARY_KEY]
        
        # Match the teacher-forced scoring prefix exactly. Otherwise the
        # student logits are conditioned on raw article text while the teacher
        # logits are conditioned on an instruction prompt.
        prompt = TEACHER_PROMPT_TEMPLATE.format(text=text)

        # Tokenize prompt (with special tokens / BOS)
        input_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_input_tokens,
            add_special_tokens=True,
            return_tensors=None,
        )
        
        # Tokenize gold summary (NO special tokens — raw continuation)
        target_enc = self.tokenizer(
            gold_summary,
            truncation=True,
            max_length=self.max_output_tokens,
            add_special_tokens=False,
            return_tensors=None,
        )
        
        prompt_ids = input_enc['input_ids']
        summary_ids = target_enc['input_ids']
        teacher_gold_ids = list(summary_ids)
        eos_id = self.tokenizer.eos_token_id
        eos_added = False
        if eos_id is not None:
            if len(summary_ids) >= self.max_output_tokens:
                summary_ids = summary_ids[:self.max_output_tokens - 1] + [eos_id]
            else:
                summary_ids = summary_ids + [eos_id]
            eos_added = True
        prompt_len = len(prompt_ids)
        summary_len = len(summary_ids)
        
        # Concatenate: [prompt_tokens, summary_tokens]
        # This is the correct format for decoder-only causal LM summarization
        full_input_ids = prompt_ids + summary_ids
        full_attention_mask = [1] * len(full_input_ids)
        
        # Labels: -100 for prompt positions (don't compute loss there),
        #         actual token IDs for summary positions
        full_labels = [-100] * prompt_len + summary_ids
        
        result = {
            'input_ids': full_input_ids,
            'attention_mask': full_attention_mask,
            'labels': full_labels,
            'summary_start': prompt_len,    # where summary begins
            'summary_len': summary_len,     # how many summary tokens
        }
        
        # Add teacher outputs if available
        if self.use_distillation:
            teacher_len = None
            if self.teacher_32b_outputs and idx < len(self.teacher_32b_outputs):
                teacher_record = self.teacher_32b_outputs[idx]
                if self.teacher_32b_qwen_compatible:
                    result['teacher_32b_logprobs'] = teacher_record.get('top_k_logprobs', [])
                    teacher_len = len(result['teacher_32b_logprobs'])
                result['teacher_32b_token_ids'] = teacher_record.get('token_ids', [])
                result['teacher_32b_gold_mask'] = self._gold_support_mask(
                    teacher_record, teacher_gold_ids, summary_len,
                    qwen_compatible=self.teacher_32b_qwen_compatible,
                )
            
            if self.teacher_14b_outputs and idx < len(self.teacher_14b_outputs):
                teacher_record = self.teacher_14b_outputs[idx]
                if self.teacher_14b_qwen_compatible:
                    result['teacher_14b_logprobs'] = teacher_record.get('top_k_logprobs', [])
                    if teacher_len is None:
                        teacher_len = len(result['teacher_14b_logprobs'])
                    else:
                        teacher_len = min(teacher_len, len(result['teacher_14b_logprobs']))
                result['teacher_14b_token_ids'] = teacher_record.get('token_ids', [])
                result['teacher_14b_gold_mask'] = self._gold_support_mask(
                    teacher_record, teacher_gold_ids, summary_len,
                    qwen_compatible=self.teacher_14b_qwen_compatible,
                )

            if teacher_len is None:
                teacher_len = summary_len
            if eos_added and summary_len > 0:
                teacher_len = min(teacher_len, summary_len - 1)
            else:
                teacher_len = min(teacher_len, summary_len)

            result['teacher_mask'] = [1] * teacher_len + [0] * (summary_len - teacher_len)

            if (
                self.cpdp_teacher_32b_outputs and idx < len(self.cpdp_teacher_32b_outputs)
                and self.cpdp_teacher_32b_qwen_compatible
            ):
                result['cpdp_teacher_32b_logprobs'] = self.cpdp_teacher_32b_outputs[idx].get('top_k_logprobs', [])

            if (
                self.cpdp_teacher_14b_outputs and idx < len(self.cpdp_teacher_14b_outputs)
                and self.cpdp_teacher_14b_qwen_compatible
            ):
                result['cpdp_teacher_14b_logprobs'] = self.cpdp_teacher_14b_outputs[idx].get('top_k_logprobs', [])
        
        return result


def sparse_batch_to_tensors(batch_top_k_logprobs, max_summary_len, vocab_size):
    """
    Pack sparse teacher top-k JSON into small tensors.

    Returns a dict instead of a dense (B, T, V) tensor. With Qwen vocab this
    changes teacher memory from hundreds of MB per batch to a few MB at most.
    """
    batch_size = len(batch_top_k_logprobs)
    max_top_k = 1
    for seq in batch_top_k_logprobs:
        for top_k in seq[:max_summary_len]:
            max_top_k = max(max_top_k, len(top_k or []))

    indices = torch.full((batch_size, max_summary_len, max_top_k), -1, dtype=torch.long)
    logprobs = torch.zeros((batch_size, max_summary_len, max_top_k), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_summary_len, max_top_k), dtype=torch.bool)

    for b, seq in enumerate(batch_top_k_logprobs):
        for t, top_k in enumerate(seq[:max_summary_len]):
            if not top_k:
                continue
            k = 0
            for entry in top_k:
                if not entry or len(entry) < 2:
                    continue
                token_id = int(entry[0])
                if 0 <= token_id < vocab_size and k < max_top_k:
                    indices[b, t, k] = token_id
                    logprobs[b, t, k] = float(entry[1])
                    mask[b, t, k] = True
                    k += 1

    return {
        "indices": indices,
        "logprobs": logprobs,
        "mask": mask,
        "vocab_size": vocab_size,
    }


def collate_fn(batch, tokenizer, vocab_size, use_distillation=True):
    """
    Custom collate function for concatenated [article + summary] sequences.
    
    - Left-pads input_ids and attention_mask (causal LM convention)
    - Left-pads labels with -100 (matching the left-padding)
    - Tracks summary_start per sample (adjusted for left-padding)
    - Pads teacher logprobs to max summary length
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    max_seq_len = max(len(item['input_ids']) for item in batch)
    max_summary_len = max(item['summary_len'] for item in batch)
    
    input_ids = []
    attention_masks = []
    labels = []
    summary_starts = []  # adjusted for left-padding
    teacher_masks = []
    teacher_gold_masks = {
        'teacher_32b_gold_mask': [],
        'teacher_14b_gold_mask': [],
    }
    
    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_seq_len - seq_len
        
        # Left-pad input_ids and attention_mask
        input_ids.append([pad_id] * pad_len + item['input_ids'])
        attention_masks.append([0] * pad_len + item['attention_mask'])
        
        # Left-pad labels with -100 (padding positions have no loss)
        labels.append([-100] * pad_len + item['labels'])
        
        # Adjust summary_start for left-padding
        summary_starts.append(item['summary_start'] + pad_len)

        if use_distillation and 'teacher_mask' in item:
            tm = item['teacher_mask']
            if len(tm) < max_summary_len:
                tm = tm + [0] * (max_summary_len - len(tm))
            else:
                tm = tm[:max_summary_len]
            teacher_masks.append(tm)

        if use_distillation:
            for mask_key in teacher_gold_masks:
                if mask_key in item:
                    gm = item[mask_key]
                    if len(gm) < max_summary_len:
                        gm = gm + [0] * (max_summary_len - len(gm))
                    else:
                        gm = gm[:max_summary_len]
                    teacher_gold_masks[mask_key].append(gm)
    
    result = {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'summary_starts': summary_starts,      # list of ints (per sample)
        'max_summary_len': max_summary_len,     # for teacher logprob alignment
    }

    if use_distillation and teacher_masks:
        result['teacher_mask'] = torch.tensor(teacher_masks, dtype=torch.float32)

    if use_distillation:
        for mask_key, masks in teacher_gold_masks.items():
            if masks:
                result[mask_key] = torch.tensor(masks, dtype=torch.float32)
    
    # Handle teacher logprobs as sparse top-k tensors. Do not expand to
    # full vocab; that is the RAM/VRAM killer for 896-token summaries.
    if use_distillation:
        for teacher_key in [
            'teacher_32b_logprobs', 'teacher_14b_logprobs',
            'cpdp_teacher_32b_logprobs', 'cpdp_teacher_14b_logprobs',
        ]:
            if any(item.get(teacher_key) for item in batch):
                batch_logprobs = [item.get(teacher_key, []) for item in batch]
                result[teacher_key] = sparse_batch_to_tensors(
                    batch_logprobs,
                    max_summary_len=max_summary_len,
                    vocab_size=vocab_size,
                )
    
    return result


def move_teacher_batch_to_device(teacher_batch, device):
    if teacher_batch is None:
        return None
    if isinstance(teacher_batch, dict):
        moved = {}
        for key, value in teacher_batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved
    return teacher_batch.to(device).float()


# ============================================================================
# Mid-Training Evaluation
# ============================================================================
def run_epoch_eval(model, tokenizer, val_samples, device, n_samples=50, max_new_tokens=256):
    """
    Generate predictions for n_samples and compute ROUGE-L + char-ngram
    semantic similarity. Callable mid-epoch and end-of-epoch.
    """
    from rouge_score import rouge_scorer as _rs

    class _SpaceTok:
        def tokenize(self, t): return t.split()

    scorer = _rs.RougeScorer(['rougeL'], tokenizer=_SpaceTok())
    preds, refs = [], []

    model.eval()
    for d in val_samples[:n_samples]:
        prompt = TEACHER_PROMPT_TEMPLATE.format(text=d[DATASET_TEXT_KEY])
        enc = tokenizer(
            prompt, return_tensors='pt', truncation=True,
            max_length=STUDENT_MAX_INPUT_TOKENS,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        pred = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        preds.append(pred)
        refs.append(d[DATASET_SUMMARY_KEY])

    rouge_l = float(np.mean([
        scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(preds, refs)
    ]))

    sem_sim = 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as _cos
        texts = preds + refs
        vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), min_df=1)
        mat = vec.fit_transform(texts)
        n = len(preds)
        sem_sim = float(np.mean([_cos(mat[i], mat[n + i])[0][0] for i in range(n)]))
    except Exception:
        pass

    return rouge_l, sem_sim


# ============================================================================
# Training Loop
# ============================================================================
def train_student(experiment_name: str, resume_from: str = None, start_epoch: int = 0):
    """
    Train the student model for a specific experiment configuration.
    """
    assert experiment_name in EXPERIMENTS, f"Unknown experiment: {experiment_name}. Choose from: {list(EXPERIMENTS.keys())}"
    
    exp_config = EXPERIMENTS[experiment_name]
    
    print(f"\n{'='*80}")
    print(f"DUAL-TEACHER DISTILLATION — STUDENT TRAINING")
    print(f"{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"Description: {exp_config['description']}")
    print(f"Distillation: {exp_config['use_distillation']}")
    print(f"EWAD: {exp_config['use_ewad']}")
    print(f"CPDP: {exp_config['use_cpdp']}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ===== Load tokenizer =====
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    
    print(f"\nLoading tokenizer: {STUDENT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Use len(tokenizer) instead of tokenizer.vocab_size to include all special/added tokens
    # Qwen2.5 has token IDs beyond vocab_size (e.g., 151643 when vocab_size=151643)
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size} (base: {tokenizer.vocab_size})")
    
    # ===== Load dataset =====
    print("\nLoading dataset...")
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    np.random.seed(SEED)
    indices = np.random.permutation(len(all_data))
    all_data = [all_data[i] for i in indices]
    
    total = len(all_data)
    train_end = int(TRAIN_SPLIT * total)
    val_end = train_end + int(VAL_SPLIT * total)
    
    splits = {
        'train': all_data[:train_end],
        'validation': all_data[train_end:val_end],
        'test': all_data[val_end:]
    }
    
    if MAX_SAMPLES is not None:
        for name in splits:
            splits[name] = splits[name][:MAX_SAMPLES]
    
    for name, s in splits.items():
        print(f"  {name}: {len(s)} samples")
    
    # ===== Create datasets =====
    use_distill = exp_config['use_distillation']
    
    # Determine teacher files based on experiment
    teacher_32b_train = os.path.join(TEACHER_32B_OUTPUTS, "train.jsonl") if use_distill else None
    teacher_14b_train = os.path.join(TEACHER_14B_OUTPUTS, "train.jsonl") if use_distill else None
    teacher_32b_val = os.path.join(TEACHER_32B_OUTPUTS, "validation.jsonl") if use_distill else None
    teacher_14b_val = os.path.join(TEACHER_14B_OUTPUTS, "validation.jsonl") if use_distill else None
    cpdp_teacher_32b_train = None
    cpdp_teacher_14b_train = None
    cpdp_teacher_32b_val = None
    cpdp_teacher_14b_val = None
    if use_distill and exp_config.get("use_cpdp", False):
        cpdp_dir_32b = globals().get("CPDP_TEACHER_32B_OUTPUTS", None)
        cpdp_dir_14b = globals().get("CPDP_TEACHER_14B_OUTPUTS", None)
        if cpdp_dir_32b:
            cpdp_teacher_32b_train = os.path.join(cpdp_dir_32b, "train.jsonl")
            cpdp_teacher_32b_val = os.path.join(cpdp_dir_32b, "validation.jsonl")
        if cpdp_dir_14b:
            cpdp_teacher_14b_train = os.path.join(cpdp_dir_14b, "train.jsonl")
            cpdp_teacher_14b_val = os.path.join(cpdp_dir_14b, "validation.jsonl")
    
    # For single-teacher experiments, nullify the unused teacher
    teacher_weights = exp_config.get("teacher_weights", None)
    if teacher_weights:
        if teacher_weights.get("32b", 0) == 0:
            teacher_32b_train = None
            teacher_32b_val = None
        if teacher_weights.get("14b", 0) == 0:
            teacher_14b_train = None
            teacher_14b_val = None
    
    print("\nCreating training dataset...")
    train_dataset = DistillationDataset(
        data_samples=splits['train'],
        teacher_32b_file=teacher_32b_train,
        teacher_14b_file=teacher_14b_train,
        cpdp_teacher_32b_file=cpdp_teacher_32b_train,
        cpdp_teacher_14b_file=cpdp_teacher_14b_train,
        tokenizer=tokenizer,
        max_input_tokens=STUDENT_MAX_INPUT_TOKENS,
        max_output_tokens=STUDENT_MAX_OUTPUT_TOKENS,
        use_distillation=use_distill,
    )
    
    print("\nCreating validation dataset...")
    val_dataset = DistillationDataset(
        data_samples=splits['validation'],
        teacher_32b_file=teacher_32b_val,
        teacher_14b_file=teacher_14b_val,
        cpdp_teacher_32b_file=cpdp_teacher_32b_val,
        cpdp_teacher_14b_file=cpdp_teacher_14b_val,
        tokenizer=tokenizer,
        max_input_tokens=STUDENT_MAX_INPUT_TOKENS,
        max_output_tokens=STUDENT_MAX_OUTPUT_TOKENS,
        use_distillation=use_distill,
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=STUDENT_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, vocab_size, use_distill),
        num_workers=0,
        pin_memory=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=STUDENT_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, vocab_size, use_distill),
        num_workers=0,
        pin_memory=False,
    )
    
    # ===== Load student model with LoRA =====
    print(f"\nLoading student model: {STUDENT_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if STUDENT_BF16 else torch.float32,
        device_map="auto",
    )
    
    if STUDENT_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    
    # Apply or load LoRA
    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        print(f"\nResuming LoRA adapter from: {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        print("\nApplying LoRA...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # ===== Loss function =====
    print("\nInitializing loss function...")
    loss_fn = DualTeacherDistillationLoss(
        vocab_size=vocab_size,
        experiment_config=exp_config,
    )
    print(f"  Mode: {'distillation' if use_distill else 'baseline CE'}")
    if use_distill and exp_config.get('use_ewad'):
        print(f"  EWAD mode: {exp_config['use_ewad']}")
    if use_distill and exp_config.get('use_cpdp'):
        print(f"  CPDP μ: {CPDP_MU}")
    
    # ===== Optimizer & Scheduler =====
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STUDENT_LEARNING_RATE,
        weight_decay=STUDENT_WEIGHT_DECAY,
    )
    
    total_steps = len(train_loader) * STUDENT_NUM_EPOCHS // STUDENT_GRADIENT_ACCUMULATION
    warmup_steps = int(STUDENT_WARMUP_RATIO * total_steps)
    
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"\nTraining config:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Effective batch: {STUDENT_BATCH_SIZE * STUDENT_GRADIENT_ACCUMULATION}")
    print(f"  Save every: {STUDENT_SAVE_STEPS} optimizer steps")
    print(f"  Max sequence tokens: {STUDENT_MAX_SEQUENCE_TOKENS}")
    
    # ===== Output directory =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{experiment_name}_{timestamp}"
    if resume_from:
        run_tag += "_resume"
    output_dir = os.path.join(STUDENT_OUTPUT_DIR, run_tag)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # Save experiment config
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump({
            "experiment_name": experiment_name,
            "config": exp_config,
            "student_model": STUDENT_MODEL,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "batch_size": STUDENT_BATCH_SIZE,
            "grad_accum": STUDENT_GRADIENT_ACCUMULATION,
            "save_steps": STUDENT_SAVE_STEPS,
            "max_sequence_tokens": STUDENT_MAX_SEQUENCE_TOKENS,
            "lr": STUDENT_LEARNING_RATE,
            "epochs": STUDENT_NUM_EPOCHS,
            "resume_from": resume_from,
            "ewad_teacher_32b_outputs": TEACHER_32B_OUTPUTS,
            "ewad_teacher_14b_outputs": TEACHER_14B_OUTPUTS,
            "cpdp_teacher_32b_outputs": globals().get("CPDP_TEACHER_32B_OUTPUTS", ""),
            "cpdp_teacher_14b_outputs": globals().get("CPDP_TEACHER_14B_OUTPUTS", ""),
            "ewad_tau_w": EWAD_TAU_W,
            "ewad_k": EWAD_K,
            "ewad_delta": EWAD_DELTA,
            "cpdp_mu": CPDP_MU,
            "timestamp": timestamp,
        }, f, indent=2)
    
    # ===== Training Loop =====
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")

    steps_per_epoch = len(train_loader) // STUDENT_GRADIENT_ACCUMULATION
    global_step = start_epoch * steps_per_epoch
    best_val_loss = float('inf')
    best_rouge_l = 0.0
    last_rouge_l = 0.0
    last_sem_sim = 0.0
    training_log = []

    # Fast-forward scheduler to match start_epoch
    if start_epoch > 0:
        print(f"  Fast-forwarding scheduler by {global_step} steps (start_epoch={start_epoch})...")
        for _ in range(global_step):
            scheduler.step()
        print(f"  LR after fast-forward: {scheduler.get_last_lr()[0]:.2e}")

    for epoch in range(start_epoch, STUDENT_NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_diagnostics = {}
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{STUDENT_NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            summary_starts = batch['summary_starts']  # list of ints
            max_summary_len = batch['max_summary_len']
            teacher_mask = batch.get('teacher_mask', None)
            if teacher_mask is not None:
                teacher_mask = teacher_mask.to(model.device)
            teacher_32b_gold_mask = batch.get('teacher_32b_gold_mask', None)
            teacher_14b_gold_mask = batch.get('teacher_14b_gold_mask', None)
            if teacher_32b_gold_mask is not None:
                teacher_32b_gold_mask = teacher_32b_gold_mask.to(model.device)
            if teacher_14b_gold_mask is not None:
                teacher_14b_gold_mask = teacher_14b_gold_mask.to(model.device)

            # Proactive skip: very long sequences cause VRAM spikes
            if input_ids.shape[1] > STUDENT_MAX_SEQUENCE_TOKENS:
                print(f"\n  [SKIP] batch {batch_idx} seq_len={input_ids.shape[1]} > {STUDENT_MAX_SEQUENCE_TOKENS}. Skipping.")
                del input_ids, attention_mask, labels
                torch.cuda.empty_cache()
                continue

            try:
                # Forward pass on full [article + summary] sequence
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                all_logits = outputs.logits  # (B, full_seq_len, vocab)

                # ===== Causal shift + summary extraction =====
                # In causal LM: logits[t] predicts token[t+1]
                # For summary token at position s, the predicting logit is at position s-1
                # Teacher logprobs[j] = P(summary_token_j | prompt, summary_0..j-1)
                #   which aligns with student logits[summary_start - 1 + j]
                #
                # Extract summary-aligned logits: positions [summary_start-1, summary_start-1+summary_len)
                # We collect per-sample into a padded (B, max_summary_len, vocab) tensor

                B = all_logits.size(0)
                V = all_logits.size(2)
                summary_logits = torch.zeros(B, max_summary_len, V, device=model.device, dtype=all_logits.dtype)
                summary_labels = torch.full((B, max_summary_len), -100, device=model.device, dtype=torch.long)

                for i in range(B):
                    s_start = summary_starts[i]
                    sample_labels = labels[i, s_start:]
                    s_len = (sample_labels != -100).sum().item()
                    if s_len == 0:
                        continue
                    logit_start = s_start - 1
                    logit_end = logit_start + s_len
                    summary_logits[i, :s_len, :] = all_logits[i, logit_start:logit_end, :]
                    summary_labels[i, :s_len] = sample_labels[:s_len]

                student_logits = summary_logits
                aligned_labels = summary_labels

                # Prepare teacher logprobs (already padded to max_summary_len by collate)
                teacher_32b_lp = batch.get('teacher_32b_logprobs', None)
                teacher_14b_lp = batch.get('teacher_14b_logprobs', None)
                cpdp_teacher_32b_lp = batch.get('cpdp_teacher_32b_logprobs', None)
                cpdp_teacher_14b_lp = batch.get('cpdp_teacher_14b_logprobs', None)

                teacher_32b_lp = move_teacher_batch_to_device(teacher_32b_lp, model.device)
                teacher_14b_lp = move_teacher_batch_to_device(teacher_14b_lp, model.device)
                cpdp_teacher_32b_lp = move_teacher_batch_to_device(cpdp_teacher_32b_lp, model.device)
                cpdp_teacher_14b_lp = move_teacher_batch_to_device(cpdp_teacher_14b_lp, model.device)

                # Compute loss
                loss, diagnostics = loss_fn(
                    student_logits=student_logits,
                    gold_labels=aligned_labels,
                    teacher_32b_logprobs=teacher_32b_lp,
                    teacher_14b_logprobs=teacher_14b_lp,
                    teacher_32b_gold_mask=teacher_32b_gold_mask,
                    teacher_14b_gold_mask=teacher_14b_gold_mask,
                    cpdp_teacher_32b_logprobs=cpdp_teacher_32b_lp,
                    cpdp_teacher_14b_logprobs=cpdp_teacher_14b_lp,
                    attention_mask=teacher_mask,
                )

                # Scale loss for gradient accumulation
                loss = loss / STUDENT_GRADIENT_ACCUMULATION
                loss.backward()

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if 'out of memory' in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    print(f"\n  [OOM] Skipped batch {batch_idx} (seq_len={input_ids.shape[1]}): {type(e).__name__}. Continuing.")
                    continue
                else:
                    raise  # re-raise non-OOM RuntimeErrors

            epoch_loss += loss.item() * STUDENT_GRADIENT_ACCUMULATION
            num_batches += 1
            
            # Accumulate diagnostics
            for k, v in diagnostics.items():
                epoch_diagnostics[k] = epoch_diagnostics.get(k, 0) + v
            
            # Gradient step
            if (batch_idx + 1) % STUDENT_GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), STUDENT_MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging — update tqdm postfix every optimizer step
                avg_loss = epoch_loss / max(num_batches, 1)
                lr = scheduler.get_last_lr()[0]
                nb = max(num_batches, 1)
                _d = epoch_diagnostics
                _pf = {
                    "loss":    f"{avg_loss:.4f}",
                    "ce":      f"{_d.get('ce_loss_mean', 0)/nb:.4f}",
                    "kd":      f"{_d.get('kd_loss_mean', 0)/nb:.4f}",
                    "gate":    f"{_d.get('gate_mean', 0)/nb:.3f}",
                    "supp":    f"{_d.get('teacher_support_mean', 0)/nb:.3f}",
                    "kdw":     f"{_d.get('kd_weight_mean', 0)/nb:.3f}",
                    "agr":     f"{_d.get('agreement_mean', 0)/nb:.3f}",
                    "c32":     f"{_d.get('conf_32b_mean', 0)/nb:.3f}",
                    "c14":     f"{_d.get('conf_14b_mean', 0)/nb:.3f}",
                    "lr":      f"{lr:.2e}",
                }
                # RL/sem only shown right after mid-eval, not at every step
                progress_bar.set_postfix(**_pf)

                if global_step % STUDENT_LOGGING_STEPS == 0:
                    log_entry = {
                        'step': global_step,
                        'epoch': epoch + 1,
                        'loss': avg_loss,
                        'lr': lr,
                    }
                    for k, v in epoch_diagnostics.items():
                        log_entry[k] = v / num_batches
                    training_log.append(log_entry)
                
                # Save checkpoint
                if global_step % STUDENT_SAVE_STEPS == 0:
                    ckpt_dir = os.path.join(output_dir, "checkpoints", f"step_{global_step}")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"\n  Checkpoint saved: {ckpt_dir}")

            # ── Mid-training eval every 2000 raw batch steps ─────────────
            if (batch_idx + 1) % 2000 == 0:
                print(f"\n  [Mid-eval @ batch {batch_idx+1}] Generating 50 val samples...")
                _rl, _sem = run_epoch_eval(
                    model, tokenizer, val_dataset.data_samples, model.device
                )
                model.train()
                torch.cuda.empty_cache()
                last_rouge_l = _rl
                last_sem_sim = _sem
                print(f"  ROUGE-L: {_rl:.4f}  |  SemanticSim: {_sem:.4f}")
                if _rl > best_rouge_l:
                    best_rouge_l = _rl
                    _best_dir = os.path.join(output_dir, "best_model")
                    model.save_pretrained(_best_dir)
                    tokenizer.save_pretrained(_best_dir)
                    print(f"  [NEW BEST ROUGE-L={_rl:.4f}] Saved to {_best_dir}")
                nb = max(num_batches, 1)
                _d = epoch_diagnostics
                progress_bar.set_postfix(
                    loss=f"{epoch_loss/nb:.4f}",
                    ce=f"{_d.get('ce_loss_mean', 0)/nb:.4f}",
                    kd=f"{_d.get('kd_loss_mean', 0)/nb:.4f}",
                    gate=f"{_d.get('gate_mean', 0)/nb:.3f}",
                    supp=f"{_d.get('teacher_support_mean', 0)/nb:.3f}",
                    kdw=f"{_d.get('kd_weight_mean', 0)/nb:.3f}",
                    agr=f"{_d.get('agreement_mean', 0)/nb:.3f}",
                    c32=f"{_d.get('conf_32b_mean', 0)/nb:.3f}",
                    c14=f"{_d.get('conf_14b_mean', 0)/nb:.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    RL=f"{_rl:.4f}",
                    sem=f"{_sem:.4f}",
                )

        # ===== End of epoch evaluation =====
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"\nEpoch {epoch+1} — Train Loss: {avg_epoch_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Validation"):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['labels'].to(model.device)
                val_summary_starts = batch['summary_starts']
                val_max_summary_len = batch['max_summary_len']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits = outputs.logits
                
                # Extract summary-aligned logits (same causal shift as training)
                B = all_logits.size(0)
                V = all_logits.size(2)
                summary_logits = torch.zeros(B, val_max_summary_len, V, device=model.device, dtype=all_logits.dtype)
                summary_labels = torch.full((B, val_max_summary_len), -100, device=model.device, dtype=torch.long)
                
                for i in range(B):
                    s_start = val_summary_starts[i]
                    sample_labels = labels[i, s_start:]
                    s_len = (sample_labels != -100).sum().item()
                    if s_len == 0:
                        continue
                    logit_start = s_start - 1
                    logit_end = logit_start + s_len
                    summary_logits[i, :s_len, :] = all_logits[i, logit_start:logit_end, :]
                    summary_labels[i, :s_len] = sample_labels[:s_len]
                
                # For validation, use CE loss only (fair comparison across experiments)
                ce_loss = F.cross_entropy(
                    summary_logits.reshape(-1, summary_logits.size(-1)),
                    summary_labels.reshape(-1),
                    ignore_index=-100
                )
                val_loss += ce_loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)
        print(f"  Val CE Loss: {avg_val_loss:.4f}")

        # ── ROUGE-L + semantic similarity (50 val samples) ─────────────────
        val_rouge_l, val_sem_sim = 0.0, 0.0
        try:
            val_rouge_l, val_sem_sim = run_epoch_eval(
                model, tokenizer, val_dataset.data_samples, model.device
            )
            model.train()
            torch.cuda.empty_cache()
            last_rouge_l = val_rouge_l
            last_sem_sim = val_sem_sim
            print(f"  Val ROUGE-L: {val_rouge_l:.4f}  |  SemanticSim: {val_sem_sim:.4f}")
        except Exception as _e:
            print(f"  Eval skipped: {_e}")
        # ─────────────────────────────────────────────────────────────────────

        # Save best model based on ROUGE-L (fallback to val loss if no ROUGE yet)
        _improved = (val_rouge_l > best_rouge_l) if val_rouge_l > 0 else (avg_val_loss < best_val_loss)
        if _improved:
            if val_rouge_l > 0:
                best_rouge_l = val_rouge_l
            best_val_loss = avg_val_loss
            best_dir = os.path.join(output_dir, "best_model")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  New best model! ROUGE-L={val_rouge_l:.4f}  ValLoss={avg_val_loss:.4f}")

        training_log.append({
            'epoch_end': epoch + 1,
            'train_loss': avg_epoch_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'val_rouge_l': val_rouge_l,
            'val_sem_sim': val_sem_sim,
            'best_rouge_l': best_rouge_l,
        })
    
    # ===== Save final model =====
    final_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Save training log
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE — {experiment_name}")
    print(f"{'='*80}")
    print(f"Final model: {final_dir}")
    print(f"Best model: {os.path.join(output_dir, 'best_model')}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best ROUGE-L:  {best_rouge_l:.4f}")
    print(f"Training log: {os.path.join(output_dir, 'training_log.json')}")
    
    return output_dir


def main():
    import config as cfg
    parser = argparse.ArgumentParser(description="Train student model with distillation")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=list(EXPERIMENTS.keys()),
        help=f"Experiment config to run. Options: {list(EXPERIMENTS.keys())}"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Epoch to start from (0-indexed). Use with --resume to skip completed epochs."
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override STUDENT_NUM_EPOCHS from config (e.g. 5 to extend training beyond the original 3)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Quick test: 1000 samples, 1 epoch (overrides config)"
    )
    parser.add_argument(
        "--teacher-32b-dir",
        type=str,
        default=None,
        help="Override EWAD/KD teacher A output directory, e.g. teacher_outputs/teacher_gemma3_12b"
    )
    parser.add_argument(
        "--teacher-14b-dir",
        type=str,
        default=None,
        help="Override EWAD/KD teacher B output directory, e.g. teacher_outputs/teacher_14b"
    )
    parser.add_argument(
        "--cpdp-teacher-32b-dir",
        type=str,
        default=None,
        help="Optional Qwen-compatible CPDP teacher A output directory"
    )
    parser.add_argument(
        "--cpdp-teacher-14b-dir",
        type=str,
        default=None,
        help="Optional Qwen-compatible CPDP teacher B output directory"
    )
    args = parser.parse_args()

    if args.test_mode and not cfg.TEST_MODE:
        cfg.MAX_SAMPLES = 1000
        cfg.STUDENT_NUM_EPOCHS = 1
        cfg.STUDENT_SAVE_STEPS = 50
        cfg.STUDENT_LOGGING_STEPS = 10
        cfg.STUDENT_GRADIENT_ACCUMULATION = 2
        cfg.TEACHER_32B_OUTPUTS = os.path.join(cfg.BASE_DIR, "teacher_outputs_test", "teacher_32b")
        cfg.TEACHER_14B_OUTPUTS = os.path.join(cfg.BASE_DIR, "teacher_outputs_test", "teacher_14b")
        cfg.STUDENT_OUTPUT_DIR = os.path.join(cfg.BASE_DIR, "student_outputs_test")
        # Update module-level names used by train_student()
        for k in ['MAX_SAMPLES', 'STUDENT_NUM_EPOCHS', 'STUDENT_SAVE_STEPS',
                  'STUDENT_LOGGING_STEPS', 'STUDENT_GRADIENT_ACCUMULATION',
                  'TEACHER_32B_OUTPUTS', 'TEACHER_14B_OUTPUTS', 'STUDENT_OUTPUT_DIR']:
            globals()[k] = getattr(cfg, k)
        print("\n*** --test-mode: 1000 samples, 1 epoch, test output dirs ***\n")

    if args.teacher_32b_dir:
        globals()['TEACHER_32B_OUTPUTS'] = args.teacher_32b_dir
        print(f"\n*** EWAD/KD teacher A override: {args.teacher_32b_dir} ***")
    if args.teacher_14b_dir:
        globals()['TEACHER_14B_OUTPUTS'] = args.teacher_14b_dir
        print(f"\n*** EWAD/KD teacher B override: {args.teacher_14b_dir} ***")
    if args.cpdp_teacher_32b_dir:
        globals()['CPDP_TEACHER_32B_OUTPUTS'] = args.cpdp_teacher_32b_dir
        print(f"\n*** CPDP teacher A override: {args.cpdp_teacher_32b_dir} ***")
    if args.cpdp_teacher_14b_dir:
        globals()['CPDP_TEACHER_14B_OUTPUTS'] = args.cpdp_teacher_14b_dir
        print(f"\n*** CPDP teacher B override: {args.cpdp_teacher_14b_dir} ***")

    if args.num_epochs is not None:
        import config as _cfg
        _cfg.STUDENT_NUM_EPOCHS = args.num_epochs
        globals()['STUDENT_NUM_EPOCHS'] = args.num_epochs
        print(f"\n*** --num-epochs: overriding STUDENT_NUM_EPOCHS -> {args.num_epochs} ***")

    output_dir = train_student(args.experiment, resume_from=args.resume, start_epoch=args.start_epoch)
    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
