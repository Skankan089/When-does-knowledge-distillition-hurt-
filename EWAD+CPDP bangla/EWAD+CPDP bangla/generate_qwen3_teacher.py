"""
generate_qwen3_teacher.py
=========================
Free-generation teacher outputs using Qwen3-14B (NF4).

Key difference from generate_teacher_outputs.py (teacher-forced scoring):
  - This script GENERATES text autoregressively (not a single forward pass).
  - max_new_tokens is set PER SAMPLE based on the gold summary's own token
    count (+buffer), so generated summaries approximately match gold length
    → higher ROUGE-L at evaluation time.
  - Output format is identical to the existing teacher outputs:
    JSONL with summary / token_ids / top_k_logprobs.

Checkpoint resume:
  Counts existing lines in the output JSONL and skips those samples.
  Safe to Ctrl+C and restart — no data is lost.

Usage:
    python generate_qwen3_teacher.py --split train
    python generate_qwen3_teacher.py --split all
    python generate_qwen3_teacher.py --split validation
"""

import os
import sys
import re
import json
import math
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_FILE, DATASET_TEXT_KEY, DATASET_SUMMARY_KEY,
    TRAIN_SPLIT, VAL_SPLIT, SEED, MAX_SAMPLES,
    LOGIT_TOP_K, TEACHER_MAX_INPUT_TOKENS,
)

# ─── Qwen3-14B settings ───────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-14B"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "teacher_outputs", "teacher_qwen3_14b")

SYSTEM_MSG = "আপনি একজন বাংলা সারাংশ বিশেষজ্ঞ।"

# Per-sample token budget: max_new_tokens = ceil(gold_token_count * SCALE) + BUFFER
TOKEN_BUFFER_SCALE = 1.1   # 10 % headroom over gold length
TOKEN_BUFFER       = 30    # flat extra tokens on top of scaled count
MIN_NEW_TOKENS     = 50    # never generate fewer than this


# ─── Dataset ──────────────────────────────────────────────────────────────────

def load_dataset():
    print(f"\n{'='*70}")
    print("LOADING DATASET")
    print(f"{'='*70}")

    with open(DATASET_FILE, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    np.random.seed(SEED)
    data = [data[i] for i in np.random.permutation(len(data))]

    total     = len(data)
    train_end = int(TRAIN_SPLIT * total)
    val_end   = train_end + int(VAL_SPLIT * total)

    splits = {
        "train":      data[:train_end],
        "validation": data[train_end:val_end],
        "test":       data[val_end:],
    }

    if MAX_SAMPLES is not None:
        splits = {k: v[:MAX_SAMPLES] for k, v in splits.items()}
        print(f"  [LIMITED TO {MAX_SAMPLES} SAMPLES PER SPLIT]")

    for name, sp in splits.items():
        print(f"  {name}: {len(sp)} samples")

    return splits


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"\nLoading tokenizer: {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading model: {MODEL_NAME} (NF4 4-bit) …")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_cfg,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  GPU memory allocated: {mem:.1f} GB")

    return tokenizer, model


# ─── Prompt building ──────────────────────────────────────────────────────────

def build_prompt(tokenizer, article: str) -> str:
    """Build Qwen3 chat-template prompt with thinking disabled."""
    # Truncate article so the prompt fits within TEACHER_MAX_INPUT_TOKENS
    ids = tokenizer(article, add_special_tokens=False)["input_ids"]
    if len(ids) > TEACHER_MAX_INPUT_TOKENS:
        article = tokenizer.decode(ids[:TEACHER_MAX_INPUT_TOKENS], skip_special_tokens=True)

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": (
            "নিচের বাংলা নিবন্ধটির একটি সংক্ষিপ্ত সারাংশ লিখুন।\n\n"
            f"নিবন্ধ:\n{article}"
        )},
    ]

    # Qwen3: disable chain-of-thought to avoid <think> output
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Fallback for non-Qwen3 tokenizers that don't support enable_thinking
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def gold_max_new_tokens(tokenizer, gold_summary: str) -> int:
    """
    Per-sample token budget derived from the gold summary's own token count.
    max_new_tokens = ceil(gold_tokens * SCALE) + BUFFER, at least MIN_NEW_TOKENS.
    """
    ids = tokenizer(gold_summary, add_special_tokens=False)["input_ids"]
    budget = int(math.ceil(len(ids) * TOKEN_BUFFER_SCALE)) + TOKEN_BUFFER
    return max(budget, MIN_NEW_TOKENS)


# ─── Generation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_one(tokenizer, model, article: str, max_new_tokens: int) -> dict:
    """
    Generate one summary (greedy) and capture per-token top-k logprobs.

    Returns dict with:
        summary           : decoded text (think-blocks stripped)
        token_ids         : list of generated token IDs
        top_k_logprobs    : list of [(token_id, logprob), ...] per position
        max_new_tokens_used: budget that was used
    """
    prompt = build_prompt(tokenizer, article)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                  # greedy — deterministic
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,               # capture per-step distributions
    )

    new_ids = out.sequences[0][input_len:].cpu()
    summary = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # Strip Qwen3 thinking blocks if they leaked through
    if "<think>" in summary:
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()

    # Extract top-k logprobs from out.scores
    # out.scores: tuple of (vocab_size,) tensors (one per generated step, batch=1)
    token_ids_list      = new_ids.tolist()
    top_k_logprobs_list = []

    for step_idx, score_tensor in enumerate(out.scores):
        if step_idx >= len(token_ids_list):
            break   # safety guard
        logprobs        = F.log_softmax(score_tensor[0].float(), dim=-1)
        k               = min(LOGIT_TOP_K, logprobs.shape[0])
        vals, idx_ids   = torch.topk(logprobs, k=k)
        top_k_logprobs_list.append(
            list(zip(idx_ids.cpu().tolist(), vals.cpu().tolist()))
        )

    torch.cuda.empty_cache()

    return {
        "summary":             summary,
        "token_ids":           token_ids_list,
        "top_k_logprobs":      top_k_logprobs_list,
        "max_new_tokens_used": max_new_tokens,
    }


# ─── Split runner ─────────────────────────────────────────────────────────────

def generate_for_split(tokenizer, model, data, output_dir, split_name):
    print(f"\n{'='*70}")
    print(f"Qwen3-14B FREE GENERATION — {split_name.upper()}")
    print(f"{'='*70}")
    print(f"Total samples: {len(data)}")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{split_name}.jsonl")

    # ── Checkpoint: count existing lines and resume from there ──────────────
    existing = 0
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        if existing > 0:
            print(f"  Checkpoint found: {existing} samples already done. Resuming …")

    remaining = data[existing:]
    if not remaining:
        print("  All samples already processed. Skipping.")
        return

    # Report expected per-sample budgets using first ≤100 remaining samples
    sample_for_stats = remaining[:min(100, len(remaining))]
    budgets = [gold_max_new_tokens(tokenizer, s[DATASET_SUMMARY_KEY])
               for s in sample_for_stats]
    print(f"  Gold-length token budget — avg: {sum(budgets)/len(budgets):.0f}, "
          f"min: {min(budgets)}, max: {max(budgets)}")

    # ── Generation loop ─────────────────────────────────────────────────────
    with open(output_file, "a", encoding="utf-8") as f_out:
        for idx, item in enumerate(tqdm(remaining, desc=f"  {split_name}")):
            article      = item[DATASET_TEXT_KEY]
            gold_summary = item[DATASET_SUMMARY_KEY]
            sample_idx   = existing + idx
            max_new      = gold_max_new_tokens(tokenizer, gold_summary)

            try:
                result = generate_one(tokenizer, model, article, max_new)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # Retry with a reduced budget
                reduced = max(max_new // 2, MIN_NEW_TOKENS)
                print(f"\n  OOM at sample {sample_idx}. Retrying with max_new={reduced} …")
                try:
                    result = generate_one(tokenizer, model, article, reduced)
                except Exception as e2:
                    print(f"  Retry also failed: {e2}. Skipping sample {sample_idx}.")
                    continue

            except Exception as e:
                print(f"\n  ERROR at sample {sample_idx}: {e}")
                continue

            record = {
                "original_id":         item.get("ID", sample_idx),
                "gold_summary":        gold_summary,
                "summary":             result["summary"],
                "token_ids":           result["token_ids"],
                "top_k_logprobs":      result["top_k_logprobs"],
                "max_new_tokens_used": result["max_new_tokens_used"],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()   # flush after every sample — safe to Ctrl+C anytime

    with open(output_file, encoding="utf-8") as f:
        final = sum(1 for _ in f)
    print(f"\n  Done: {final}/{len(data)} samples → {output_file}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-14B free generation with per-sample gold-length token budgets"
    )
    parser.add_argument(
        "--split", default="all",
        choices=["train", "validation", "test", "all"],
        help="Data split to process (default: all)",
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("Qwen3-14B TEACHER FREE GENERATION")
    print(f"{'='*70}")
    print(f"Model     : {MODEL_NAME}")
    print(f"Output    : {OUTPUT_DIR}")
    print(f"Dataset   : {DATASET_FILE}")
    print(f"Timestamp : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Budget    : gold_tokens × {TOKEN_BUFFER_SCALE} + {TOKEN_BUFFER}  (min {MIN_NEW_TOKENS})")

    splits = load_dataset()
    tokenizer, model = load_model()

    splits_to_run = list(splits.keys()) if args.split == "all" else [args.split]

    for split_name in splits_to_run:
        generate_for_split(tokenizer, model, splits[split_name],
                           OUTPUT_DIR, split_name)

    # ── Save metadata ────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata = {
        "teacher_model":      MODEL_NAME,
        "quantization":       "nf4_4bit",
        "mode":               "free_generation_length_adaptive",
        "description":        (
            "Autoregressive greedy generation. "
            f"max_new_tokens = ceil(gold_tokens × {TOKEN_BUFFER_SCALE}) + {TOKEN_BUFFER}, "
            f"min {MIN_NEW_TOKENS}."
        ),
        "token_buffer_scale": TOKEN_BUFFER_SCALE,
        "token_buffer":       TOKEN_BUFFER,
        "min_new_tokens":     MIN_NEW_TOKENS,
        "logit_top_k":        LOGIT_TOP_K,
        "max_input_tokens":   TEACHER_MAX_INPUT_TOKENS,
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":            DATASET_FILE,
        "splits":             splits_to_run,
        "samples_per_split":  {n: len(splits[n]) for n in splits_to_run},
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nMetadata saved → {OUTPUT_DIR}/metadata.json")
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
