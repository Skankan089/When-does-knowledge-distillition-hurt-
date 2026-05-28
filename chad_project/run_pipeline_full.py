"""
run_pipeline_full.py — CHAD pipeline for the full BanSum dataset (141k records).

Stages:
  1. prepare_data  — split full BanSum JSON into train/val/test JSONL
  2. probe         — build counterfactual KD-helpfulness labels (uses A1 weights)
  3. train_gate    — fit GBM regression gate on probe labels
  4. score_gate    — score all ~139k training samples with the gate
  5. a6_chad       — train gated-KD student (main contribution)
  6. evaluate_a6   — evaluate A6 on the test set

NOTE: --no-generation-features is used in both probe and score_gate to keep
      feature sets consistent and to make gate scoring of 139k samples feasible
      (skips per-sample beam-search generation, removing gold_teacher_rouge_l
      from the feature set in both stages).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

# ─────────────────────────────── CONFIG ──────────────────────────────────────
TEACHER  = r"D:\summariser\not_more_than_limit\banglat5_bansum_20260218_213532\final_model"
STUDENT  = "csebuetnlp/banglat5_small"
A1_DIR   = r"D:\summariser\not_more_than_limit\ablation_results_bansum\A1_baseline_20260224_121256\best_model"

DATA_FILE  = r"D:\summariser\lmi\not more than limit\LMI\bansum_lte_1000_tokens.json"

TRAIN_FILE = "data_full/splits/train.jsonl"
VAL_FILE   = "data_full/splits/val.jsonl"
TEST_FILE  = "data_full/splits/test.jsonl"

# Training — 8 epochs over ~139k samples (~35k steps/epoch at batch=4)
EPOCHS          = 8
TRAIN_BATCH     = 4
EVAL_BATCH      = 4
GRAD_ACCUM      = 2
LR              = 5e-5
LOGGING_STEPS   = 500
EVAL_STEPS      = 2000
SAVE_STEPS      = 2000

PROBE_SIZE      = 5000
VAL_PROBE_SIZE  = 300
FEATURE_BATCH   = 4
SCORE_BATCH     = 4
MAX_SOURCE_LEN  = 768
TEMPERATURE     = 0.5

# Outputs
A6_DIR       = "runs_full/a6_chad"
PROBE_FILE   = "data_full/probes/kd_usefulness_probe.jsonl"
GATE_FILE    = "runs_full/gate/chad_gate.joblib"
GATE_METRICS = "runs_full/gate/metrics.json"
SCORES_FILE  = "data_full/gates/train_gate_scores.jsonl"
EVAL_BASE    = "runs_full/eval"
# ─────────────────────────────────────────────────────────────────────────────

COMMON_TRAIN = [
    "--train-file", TRAIN_FILE,
    "--val-file",   VAL_FILE,
    "--epochs",     str(EPOCHS),
    "--learning-rate", str(LR),
    "--train-batch-size", str(TRAIN_BATCH),
    "--eval-batch-size",  str(EVAL_BATCH),
    "--gradient-accumulation-steps", str(GRAD_ACCUM),
    "--logging-steps", str(LOGGING_STEPS),
    "--eval-steps",    str(EVAL_STEPS),
    "--save-steps",    str(SAVE_STEPS),
    "--max-source-length", str(MAX_SOURCE_LEN),
    "--bf16",
]


def run(step_name: str, args: list[str], skip_if: Path | None = None) -> None:
    """Run a module via subprocess; skip if `skip_if` already exists."""
    if skip_if and skip_if.exists():
        print(f"\n[SKIP] {step_name} — output already exists: {skip_if}")
        return
    cmd = [sys.executable, "-m"] + args
    print(f"\n{'='*70}")
    print(f"[RUN]  {step_name}")
    print(f"       {' '.join(cmd)}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAIL] {step_name} exited with code {result.returncode} after {elapsed:.0f}s")
        sys.exit(result.returncode)
    print(f"\n[OK]   {step_name} finished in {elapsed/60:.1f} min")


# ── Stage helpers ─────────────────────────────────────────────────────────────

def stage_prepare_data() -> None:
    """Split the full BanSum JSON into train/val/test JSONL files."""
    run(
        "Prepare data splits (full BanSum 141k)",
        [
            "chad.prepare_data",
            "--data-file",     DATA_FILE,
            "--output-dir",    "data_full/splits",
            "--text-field",    "main",
            "--summary-field", "sum1",
            "--id-field",      "ID",
            "--val-size",      "1000",
            "--test-size",     "1000",
        ],
        skip_if=Path("data_full/splits/train.jsonl"),
    )


def stage_probe() -> None:
    """Build counterfactual probe labels using A1 weights and grad_align.

    --no-generation-features: skips per-sample beam search so that
    gold_teacher_rouge_l is absent from BOTH probe and scoring features,
    keeping the feature set consistent.
    """
    run(
        "Probe labelling (grad_align, A1 student, no-gen-features)",
        [
            "chad.build_probe_labels",
            "--train-file",          TRAIN_FILE,
            "--val-file",            VAL_FILE,
            "--student-model-name",  A1_DIR,
            "--teacher-model-name",  TEACHER,
            "--output-file",         PROBE_FILE,
            "--method",              "grad_align",
            "--probe-size",          str(PROBE_SIZE),
            "--val-probe-size",      str(VAL_PROBE_SIZE),
            "--feature-batch-size",  str(FEATURE_BATCH),
            "--batch-size",          "1",
            "--max-source-length",   str(MAX_SOURCE_LEN),
            "--no-generation-features",
            "--temperature",         str(TEMPERATURE),
        ],
        skip_if=Path(PROBE_FILE),
    )


def stage_train_gate() -> None:
    """Train GBM regression gate from probe labels (grad_align_score target)."""
    run(
        "Gate training (GBM regressor)",
        [
            "chad.train_gate",
            "--probe-file",   PROBE_FILE,
            "--output-file",  GATE_FILE,
            "--metrics-file", GATE_METRICS,
            "--model-type",   "gbm",
        ],
        skip_if=Path(GATE_FILE),
    )


def stage_score_gate() -> None:
    """Score all ~139k training samples with the trained gate.

    --no-generation-features: must match what was used during probe building.
    Scoring without beam search makes the 139k-sample run feasible.
    """
    run(
        "Gate scoring (full ~139k train set, no-gen-features)",
        [
            "chad.score_gate",
            "--input-file",         TRAIN_FILE,
            "--gate-file",          GATE_FILE,
            "--teacher-model-name", TEACHER,
            "--student-model-name", A1_DIR,
            "--output-file",        SCORES_FILE,
            "--batch-size",         str(SCORE_BATCH),
            "--max-source-length",  str(MAX_SOURCE_LEN),
            "--no-generation-features",
        ],
        skip_if=Path(SCORES_FILE),
    )


def stage_a6_chad() -> None:
    """A6 — CHAD gated KD on full dataset (main contribution)."""
    run(
        "A6 CHAD gated KD (full dataset, 8 epochs, resume)",
        [
            "chad.train_student",
            "--mode", "chad",
            "--model-name",         STUDENT,
            "--teacher-model-name", TEACHER,
            "--gate-scores-file",   SCORES_FILE,
            "--output-dir",         A6_DIR,
        ] + COMMON_TRAIN + ["--early-stopping-patience", "5", "--temperature", str(TEMPERATURE)],
        skip_if=Path(A6_DIR) / "config.json",
    )


def stage_evaluate_a6() -> None:
    """Evaluate A6 on the held-out test set."""
    out_dir = f"{EVAL_BASE}/a6_chad"
    run(
        "Evaluate A6 CHAD (full dataset test set)",
        [
            "chad.evaluate_model",
            "--model-dir",         A6_DIR,
            "--test-file",         TEST_FILE,
            "--output-dir",        out_dir,
            "--batch-size",        str(EVAL_BATCH),
            "--num-beams",         "4",
            "--max-source-length", str(MAX_SOURCE_LEN),
        ],
        skip_if=Path(out_dir) / "metrics.json",
    )


# ─────────────────────────────── MAIN ────────────────────────────────────────

def main() -> None:
    print("CHAD full-dataset pipeline starting ...")
    print(f"Teacher : {TEACHER}")
    print(f"A1      : {A1_DIR}")
    print(f"Student : {STUDENT}")
    print(f"Data    : {DATA_FILE}")
    print(f"Outputs : data_full/  runs_full/")

    stage_prepare_data()
    stage_probe()
    stage_train_gate()
    stage_score_gate()
    stage_a6_chad()
    stage_evaluate_a6()

    print("\n" + "=" * 70)
    print("ALL STAGES COMPLETE")
    print("=" * 70)
    print(f"\nResults: {EVAL_BASE}/a6_chad/metrics.json")


if __name__ == "__main__":
    main()
