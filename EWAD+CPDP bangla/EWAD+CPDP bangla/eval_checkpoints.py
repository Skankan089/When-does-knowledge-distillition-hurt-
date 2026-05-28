"""
Evaluate saved LoRA checkpoints on the BanSum test set.
Computes ROUGE-1/2/L only (fast, no BERTScore/SentSim).

Usage:
    python eval_checkpoints.py
    python eval_checkpoints.py --checkpoints path/to/ckpt1 path/to/ckpt2
    python eval_checkpoints.py --max_samples 500   # quick smoke-test
"""

import os, sys, json, argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"

# ── Import shared config / helpers from the training script ──────────────────
sys.path.insert(0, str(Path(__file__).parent))
from train_gemma4_e4b import (
    MODEL_NAME, BANSUM_FILE, BNB_CONFIG, GEMMA_CHAT_TEMPLATE,
    MAX_SEQ_LENGTH, MAX_NEW_TOKENS, EVAL_GEN_BATCH_SIZE, SEED,
    SpaceTokenizer, load_bansum, build_prompt_only,
    _unwrap_gemma4_clippable_linears,
)

from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from rouge_score import rouge_scorer as rouge_scorer_lib

# Override batch size to saturate GPU (was 4 — bumped to see OOM ceiling)
EVAL_GEN_BATCH_SIZE = 16


# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINTS = [
    r"gemma4_e2b_bansum_20260519_043115\checkpoint-14000",
    r"gemma4_e2b_bansum_20260519_043115\checkpoint-36000",
]


def load_processor():
    """Load + patch processor once (reused across checkpoints)."""
    print(f"\nLoading processor from {MODEL_NAME} ...")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    if processor.tokenizer.chat_template is None:
        processor.tokenizer.chat_template = GEMMA_CHAT_TEMPLATE
        print("  [patch] Injected Gemma chat template")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token    = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    return processor


def load_lora_model(checkpoint_dir: str):
    """Load base model (4-bit) + LoRA adapter from checkpoint."""
    print(f"\n  Loading base model ...")
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        quantization_config = BNB_CONFIG,
        device_map          = {"": "cuda:0"},
        trust_remote_code   = True,
        attn_implementation = "eager",
    )
    base = _unwrap_gemma4_clippable_linears(base)
    print(f"  Attaching LoRA adapter from {checkpoint_dir} ...")
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.eval()
    return model


def generate_and_score(model, processor, articles, references, device):
    """Generate summaries and compute ROUGE on-the-fly, showing live scores in tqdm."""
    scorer = rouge_scorer_lib.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        tokenizer=SpaceTokenizer(),
    )
    predictions = []
    r1_list, r2_list, rl_list = [], [], []

    with tqdm(total=len(articles), unit="sample", ncols=100, desc="  Generating") as bar:
        for i in range(0, len(articles), EVAL_GEN_BATCH_SIZE):
            batch_articles = articles[i : i + EVAL_GEN_BATCH_SIZE]
            batch_refs     = references[i : i + EVAL_GEN_BATCH_SIZE]
            prompts = [build_prompt_only(a) for a in batch_articles]
            enc = processor(
                text           = prompts,
                return_tensors = "pt",
                padding        = True,
                truncation     = True,
                max_length     = MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
            ).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens     = MAX_NEW_TOKENS,
                    do_sample          = False,
                    repetition_penalty = 1.1,
                    pad_token_id       = processor.tokenizer.pad_token_id,
                    eos_token_id       = processor.tokenizer.eos_token_id,
                )
            input_len = enc["input_ids"].shape[1]
            for out, ref in zip(out_ids, batch_refs):
                pred = processor.tokenizer.decode(
                    out[input_len:], skip_special_tokens=True
                ).replace("<end_of_turn>", "").strip()
                predictions.append(pred)
                s = scorer.score(ref, pred)
                r1_list.append(s["rouge1"].fmeasure)
                r2_list.append(s["rouge2"].fmeasure)
                rl_list.append(s["rougeL"].fmeasure)

            bar.update(len(batch_articles))
            bar.set_postfix({
                "R1": f"{np.mean(r1_list):.4f}",
                "R2": f"{np.mean(r2_list):.4f}",
                "RL": f"{np.mean(rl_list):.4f}",
            })

    return predictions, {
        "rouge1"   : float(np.mean(r1_list)),
        "rouge2"   : float(np.mean(r2_list)),
        "rougeL"   : float(np.mean(rl_list)),
        "n_samples": len(predictions),
    }


def eval_checkpoint(checkpoint_dir: str, processor, test_articles, test_references, out_dir: str):
    print("\n" + "=" * 70)
    print(f"EVALUATING: {checkpoint_dir}")
    print("=" * 70)

    model  = load_lora_model(checkpoint_dir)
    device = next(model.parameters()).device

    predictions, scores = generate_and_score(model, processor, test_articles, test_references, device)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    print(f"\n  ROUGE-1 : {scores['rouge1']:.4f}")
    print(f"  ROUGE-2 : {scores['rouge2']:.4f}")
    print(f"  ROUGE-L : {scores['rougeL']:.4f}")
    print(f"  Samples : {scores['n_samples']}")

    # Save per-checkpoint results
    os.makedirs(out_dir, exist_ok=True)
    ckpt_name  = Path(checkpoint_dir).name
    result_path = os.path.join(out_dir, f"rouge_{ckpt_name}.json")
    pred_path   = os.path.join(out_dir, f"predictions_{ckpt_name}.json")

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": checkpoint_dir, **scores}, f, indent=2, ensure_ascii=False)

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"reference": r, "prediction": p}
             for r, p in zip(test_references, predictions)],
            f, indent=2, ensure_ascii=False,
        )

    print(f"  Results  → {result_path}")
    print(f"  Preds    → {pred_path}")

    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS,
        help="List of checkpoint directories to evaluate",
    )
    p.add_argument(
        "--dataset", default=BANSUM_FILE,
        help="Path to BanSum JSON",
    )
    p.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit test set size (e.g. 500 for a quick run)",
    )
    p.add_argument(
        "--out_dir", default="eval_results",
        help="Directory to save evaluation results",
    )
    args = p.parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────────
    script_dir   = Path(__file__).parent
    dataset_path = args.dataset if os.path.isabs(args.dataset) else str(script_dir / args.dataset)

    dataset = load_bansum(dataset_path)
    test_ds = dataset["test"]

    if args.max_samples and args.max_samples < len(test_ds):
        import random; random.seed(SEED)
        indices = random.sample(range(len(test_ds)), args.max_samples)
        test_ds = test_ds.select(indices)
        print(f"  (Sampled {args.max_samples} from test set for quick eval)")

    test_articles   = list(test_ds["text"])
    test_references = list(test_ds["summary"])

    # ── Load processor once ───────────────────────────────────────────────────
    processor = load_processor()

    # ── Evaluate each checkpoint ──────────────────────────────────────────────
    all_results = {}
    for ckpt in args.checkpoints:
        ckpt = str(ckpt)
        if not os.path.isabs(ckpt):
            ckpt = str(script_dir / ckpt)
        scores = eval_checkpoint(ckpt, processor, test_articles, test_references, args.out_dir)
        all_results[Path(ckpt).name] = scores

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Checkpoint':<25}  {'ROUGE-1':>8}  {'ROUGE-2':>8}  {'ROUGE-L':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name, s in all_results.items():
        print(f"  {name:<25}  {s['rouge1']:>8.4f}  {s['rouge2']:>8.4f}  {s['rougeL']:>8.4f}")

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
