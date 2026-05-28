"""
run_pipeline_14k.py — CHAD pipeline re-run with a 14,100-sample test set.

Splits: val=1,000  |  test=14,100  |  train=~125,900
Outputs: data_14k/  runs_14k/   (does NOT touch existing data_full/ or runs_full/)

Stages:
  1. prepare_data  — re-split full BanSum with test=14100
  2. probe         — counterfactual KD-helpfulness labels (A1 weights)
  3. train_gate    — fit GBM regression gate
  4. score_gate    — score all ~125k training samples
  5. a6_chad       — train CHAD gated-KD student
  6. evaluate_a6   — evaluate on the 14,100-sample test set
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

TRAIN_FILE = "data_14k/splits/train.jsonl"
VAL_FILE   = "data_14k/splits/val.jsonl"
TEST_FILE  = "data_14k/splits/test.jsonl"

# Training — 8 epochs
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
A6_DIR       = "runs_14k/a6_chad"
PROBE_FILE   = "data_14k/probes/kd_usefulness_probe.jsonl"
GATE_FILE    = "runs_14k/gate/chad_gate.joblib"
GATE_METRICS = "runs_14k/gate/metrics.json"
SCORES_FILE  = "data_14k/gates/train_gate_scores.jsonl"
EVAL_BASE    = "runs_14k/eval"
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
    """Split the full BanSum JSON: val=1000, test=14100, train=rest."""
    run(
        "Prepare data splits (val=1000, test=14100, train=rest)",
        [
            "chad.prepare_data",
            "--data-file",     DATA_FILE,
            "--output-dir",    "data_14k/splits",
            "--text-field",    "main",
            "--summary-field", "sum1",
            "--id-field",      "ID",
            "--val-size",      "1000",
            "--test-size",     "14100",
        ],
        skip_if=Path("data_14k/splits/train.jsonl"),
    )


def stage_probe() -> None:
    """Build counterfactual probe labels (grad_align, A1 weights, no-gen-features)."""
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
    """Train GBM regression gate from probe labels."""
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
    """Score all ~125k training samples with the gate (no-gen-features)."""
    run(
        "Gate scoring (~125k train set, no-gen-features)",
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
    """A6 — CHAD gated KD (14k-split, 8 epochs)."""
    run(
        "A6 CHAD gated KD (14k-split, 8 epochs)",
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
    """Evaluate A6 on the 14,100-sample test set."""
    out_dir = f"{EVAL_BASE}/a6_chad"
    run(
        "Evaluate A6 CHAD (14,100-sample test set)",
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
    print("CHAD 14k-test pipeline starting ...")
    print(f"Teacher : {TEACHER}")
    print(f"A1      : {A1_DIR}")
    print(f"Student : {STUDENT}")
    print(f"Data    : {DATA_FILE}")
    print(f"Splits  : val=1,000 | test=14,100 | train=rest (~125,900)")
    print(f"Outputs : data_14k/  runs_14k/")

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
