"""
evaluate_all_14k.py
-------------------
Evaluate A1, A2, A4, A5, A6 models on the 14,100-sample test set.
For each model: ROUGE-1/2/L, BLEU, BERTScore (P/R/F1), Semantic Similarity.

Skips generation if predictions.jsonl already exists.
Skips extended metrics if ext_metrics.json already exists.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# ── Config ─────────────────────────────────────────────────────────────────────
TEST_FILE   = "data_14k/splits/test.jsonl"
BERT_MODEL  = "bert-base-multilingual-cased"   # fully cached

MODELS = {
    "A1_CE":       "runs/a1_ce",
    "A2_KD":       "runs/a2_kd",
    "A4_Entropy":  "runs/a4_entropy_gate",
    "A5_Semantic": "runs/a5_semantic_gate",
    "A6_CHAD":     "runs_14k/a6_chad",          # already evaluated; reuse predictions
}

EVAL_BASE = Path("runs_14k/eval")
PYTHON    = sys.executable

GEN_ARGS = [
    "--max-source-length", "768",
    "--max-new-tokens",    "128",
    "--batch-size",        "4",
    "--num-beams",         "4",
]
# ────────────────────────────────────────────────────────────────────────────────


def run_generation(name: str, model_dir: str, out_dir: Path) -> None:
    """Run chad.evaluate_model to generate predictions if not already done."""
    pred_file = out_dir / "predictions.jsonl"
    if pred_file.exists():
        print(f"  [SKIP] predictions already exist for {name}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-m", "chad.evaluate_model",
        "--model-dir",  model_dir,
        "--test-file",  TEST_FILE,
        "--output-dir", str(out_dir),
    ] + GEN_ARGS
    print(f"  [GEN]  {name}  ({model_dir})")
    result = subprocess.run(cmd, check=True)
    print(f"  [OK]   generation done for {name}")


def load_predictions(out_dir: Path):
    from chad.data import load_jsonl  # type: ignore
    records = load_jsonl(str(out_dir / "predictions.jsonl"))
    preds = [r["prediction"] for r in records]
    refs  = [r["target"]     for r in records]
    return preds, refs


def compute_extended(name: str, out_dir: Path, preds: list, refs: list) -> dict:
    """Compute BERTScore + SemanticSim; cache to ext_metrics.json."""
    cache = out_dir / "ext_metrics.json"
    if cache.exists():
        print(f"  [SKIP] extended metrics already cached for {name}")
        with open(cache) as f:
            return json.load(f)

    # --- BERTScore ---
    print(f"  [BERT] BERTScore for {name}  (n={len(preds)}) ...")
    from bert_score import score as bscore  # type: ignore
    P, R, F1 = bscore(
        preds, refs,
        model_type=BERT_MODEL,
        lang="bn",
        verbose=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    bs_p  = float(P.mean())
    bs_r  = float(R.mean())
    bs_f1 = float(F1.mean())
    print(f"         BERTScore P={bs_p:.4f}  R={bs_r:.4f}  F1={bs_f1:.4f}")

    # --- Semantic Similarity ---
    print(f"  [SIM]  Semantic Similarity for {name} ...")
    from transformers import AutoTokenizer, AutoModel  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BERT_MODEL)
    mdl = AutoModel.from_pretrained(BERT_MODEL).to(device).eval()

    def embed(texts: list, bs: int = 32) -> np.ndarray:
        all_emb = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = mdl(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            all_emb.append(emb.cpu().numpy())
            if (i // bs) % 10 == 0:
                print(f"    {i + len(batch)}/{len(texts)}", end="\r")
        return np.vstack(all_emb)

    emb_p = embed(preds)
    emb_r = embed(refs)
    sims  = [float(cos_sim(emb_p[i:i+1], emb_r[i:i+1])[0][0]) for i in range(len(preds))]
    sem_sim = float(np.mean(sims))
    print(f"         Semantic Sim = {sem_sim:.4f}")

    # free GPU memory between models
    del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "bertscore_p":   bs_p,
        "bertscore_r":   bs_r,
        "bertscore_f1":  bs_f1,
        "semantic_sim":  sem_sim,
    }
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def load_rouge_bleu(out_dir: Path) -> dict:
    mfile = out_dir / "metrics.json"
    if not mfile.exists():
        return {}
    with open(mfile) as f:
        return json.load(f)


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    import sys
    sys.path.insert(0, ".")

    all_results: dict[str, dict] = {}

    for name, model_dir in MODELS.items():
        out_dir = EVAL_BASE / name.lower().replace(" ", "_")
        # A6 already has predictions in a dedicated dir
        if name == "A6_CHAD":
            out_dir = EVAL_BASE / "a6_chad"

        print(f"\n{'='*60}")
        print(f"  {name}  →  {out_dir}")
        print(f"{'='*60}")

        run_generation(name, model_dir, out_dir)
        preds, refs = load_predictions(out_dir)
        ext = compute_extended(name, out_dir, preds, refs)
        base = load_rouge_bleu(out_dir)
        all_results[name] = {**base, **ext}

    # ── Summary Table ──────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  BERTScore & Semantic Similarity — 14,100-sample test set")
    print("="*65)
    header = f"{'Model':<14} {'BS-P':>7} {'BS-R':>7} {'BS-F1':>7} {'SemSim':>8}"
    print(header)
    print("-"*65)
    for name, r in all_results.items():
        print(
            f"{name:<14} "
            f"{r.get('bertscore_p', 0):.4f} "
            f"{r.get('bertscore_r', 0):.4f} "
            f"{r.get('bertscore_f1', 0):.4f} "
            f"{r.get('semantic_sim', 0):.4f}"
        )
    print("="*65)

    # Save combined table to JSON
    out_path = EVAL_BASE / "comparison_14k.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
