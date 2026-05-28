"""Analyze counterfactual probe labels to produce reviewer-facing statistics.

Outputs:
  - overall KD helpful ratio
  - KD helpful ratio by source-length bin (short / medium / long)
  - comparison of teacher confidence features vs actual KD usefulness
  - JSON summary written to --output-file
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .data import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CHAD probe labels.")
    parser.add_argument("--probe-file", required=True, help="Output of build_probe_labels.py")
    parser.add_argument("--output-file", default="runs/analysis/probe_analysis.json")
    parser.add_argument(
        "--length-bins",
        nargs=2,
        type=int,
        default=[100, 300],
        metavar=("SHORT_MAX", "MEDIUM_MAX"),
        help="Word-count thresholds separating short/medium/long articles (default: 100 300)",
    )
    return parser.parse_args()


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _bin_label(source_words: float, short_max: int, medium_max: int) -> str:
    if source_words <= short_max:
        return "short"
    if source_words <= medium_max:
        return "medium"
    return "long"


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    xa = np.array(xs, dtype=np.float64)
    ya = np.array(ys, dtype=np.float64)
    if xa.std() == 0 or ya.std() == 0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.probe_file)
    if not rows:
        raise ValueError("Probe file is empty.")

    short_max, medium_max = args.length_bins[0], args.length_bins[1]

    # ── Overall stats ──────────────────────────────────────────────────────────
    labels = [int(r["kd_useful"]) for r in rows]
    deltas = [float(r["delta_val_loss"]) for r in rows]
    n_total = len(labels)
    n_helpful = sum(labels)
    helpful_ratio = n_helpful / max(n_total, 1)

    # ── By source-length bin ───────────────────────────────────────────────────
    bins: dict[str, list[int]] = {"short": [], "medium": [], "long": []}
    for row in rows:
        sw = float(row.get("source_words", 0))
        bin_key = _bin_label(sw, short_max, medium_max)
        bins[bin_key].append(int(row["kd_useful"]))

    length_bin_stats = {
        key: {
            "count": len(v),
            "helpful": sum(v),
            "helpful_ratio": sum(v) / max(len(v), 1),
        }
        for key, v in bins.items()
    }

    # ── Teacher confidence vs actual usefulness ────────────────────────────────
    # Key reviewer argument: "confidence ≠ usefulness"
    confidence_features = ["teacher_entropy", "teacher_max_prob", "teacher_margin"]
    agreement_features = ["gold_teacher_rouge_l", "semantic_agreement"]
    feature_correlations: dict[str, float] = {}

    for feat in confidence_features + agreement_features:
        pairs = [
            (float(r[feat]), float(r["kd_useful"]))
            for r in rows
            if feat in r and r[feat] is not None and not math.isnan(float(r[feat]))
        ]
        if pairs:
            xs, ys = zip(*pairs)
            feature_correlations[feat] = _pearson(list(xs), list(ys))
        else:
            feature_correlations[feat] = float("nan")

    # ── Delta distribution ─────────────────────────────────────────────────────
    delta_stats = {
        "mean": _safe_mean(deltas),
        "median": float(np.median(deltas)) if deltas else float("nan"),
        "std": float(np.std(deltas)) if deltas else float("nan"),
        "positive_ratio": sum(1 for d in deltas if d > 0) / max(len(deltas), 1),
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "probe_file": args.probe_file,
        "n_total": n_total,
        "n_helpful": n_helpful,
        "n_harmful_or_neutral": n_total - n_helpful,
        "helpful_ratio": helpful_ratio,
        "length_bins": {
            "thresholds": {"short_max_words": short_max, "medium_max_words": medium_max},
            "stats": length_bin_stats,
        },
        "delta_val_loss": delta_stats,
        "feature_correlations_with_kd_useful": feature_correlations,
        "interpretation": (
            "Positive correlation between teacher_entropy and kd_useful would mean "
            "KD helps more when the teacher is uncertain — counter-intuitive and "
            "supports the CHAD argument that confidence ≠ usefulness."
        ),
    }

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
