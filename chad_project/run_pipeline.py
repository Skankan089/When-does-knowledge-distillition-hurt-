"""
run_pipeline.py — CHAD full overnight pipeline runner.

Usage:
    python run_pipeline.py

Each stage is skipped automatically if its output already exists.
Edit the CONFIG block below to tune paths and hyperparameters.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# ─────────────────────────────── CONFIG ──────────────────────────────────────
TEACHER = r"D:\summariser\not_more_than_limit\banglat5_filtered20k_20260511_141534\final_model"
STUDENT = "csebuetnlp/banglat5_small"

TRAIN_FILE = "data/splits/train.jsonl"
VAL_FILE   = "data/splits/val.jsonl"
TEST_FILE  = "data/splits/test.jsonl"

EPOCHS          = 13
TRAIN_BATCH     = 4          # per-device
EVAL_BATCH      = 4
GRAD_ACCUM      = 2          # effective batch = TRAIN_BATCH * GRAD_ACCUM * num_gpus
LR              = 5e-5
LOGGING_STEPS   = 100
EVAL_STEPS      = 500
SAVE_STEPS      = 500

PROBE_SIZE      = 1000       # training samples for the probe
VAL_PROBE_SIZE  = 200        # validation samples for gradient alignment
FEATURE_BATCH   = 4
SCORE_BATCH     = 4

# Outputs
A1_DIR          = "runs/a1_ce"
A2_DIR          = "runs/a2_kd"
A6_DIR          = "runs/a6_chad"
A4_DIR          = "runs/a4_entropy_gate"
A5_DIR          = "runs/a5_semantic_gate"

PROBE_FILE      = "data/probes/kd_usefulness_probe.jsonl"
GATE_FILE       = "runs/gate/chad_gate.joblib"
GATE_METRICS    = "runs/gate/metrics.json"
SCORES_FILE     = "data/gates/train_gate_scores.jsonl"

EVAL_BASE       = "runs/eval"
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

def stage_a1_ce() -> None:
    """A1 — Cross-entropy baseline."""
    run(
        "A1 CE baseline",
        [
            "chad.train_student",
            "--mode", "ce",
            "--model-name", STUDENT,
            "--output-dir", A1_DIR,
        ] + COMMON_TRAIN,
        skip_if=Path(A1_DIR) / "config.json",
    )


def stage_probe() -> None:
    """Build counterfactual probe labels using grad_align."""
    run(
        "Probe labelling (grad_align)",
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
        ],
        skip_if=Path(PROBE_FILE),
    )


def stage_train_gate() -> None:
    """Train the logistic gate from probe labels."""
    run(
        "Gate training",
        [
            "chad.train_gate",
            "--probe-file",   PROBE_FILE,
            "--output-file",  GATE_FILE,
            "--metrics-file", GATE_METRICS,
            "--model-type",   "logistic",
        ],
        skip_if=Path(GATE_FILE),
    )


def stage_score_gate() -> None:
    """Score full training set with gate."""
    run(
        "Gate scoring (full train set)",
        [
            "chad.score_gate",
            "--input-file",        TRAIN_FILE,
            "--gate-file",         GATE_FILE,
            "--teacher-model-name", TEACHER,
            "--output-file",       SCORES_FILE,
            "--batch-size",        str(SCORE_BATCH),
        ],
        skip_if=Path(SCORES_FILE),
    )


def stage_a2_kd() -> None:
    """A2 — Standard KD (no gate)."""
    run(
        "A2 Standard KD",
        [
            "chad.train_student",
            "--mode", "kd",
            "--model-name",        STUDENT,
            "--teacher-model-name", TEACHER,
            "--output-dir",        A2_DIR,
        ] + COMMON_TRAIN,
        skip_if=Path(A2_DIR) / "config.json",
    )


def stage_a6_chad() -> None:
    """A6 — CHAD gated KD (main contribution)."""
    run(
        "A6 CHAD gated KD",
        [
            "chad.train_student",
            "--mode", "chad",
            "--model-name",        STUDENT,
            "--teacher-model-name", TEACHER,
            "--gate-scores-file",  SCORES_FILE,
            "--output-dir",        A6_DIR,
        ] + COMMON_TRAIN,
        skip_if=Path(A6_DIR) / "config.json",
    )


def stage_a4_entropy_gate() -> None:
    """A4 — Entropy-only ablation gate (retrain gate on entropy features only, then train student)."""
    entropy_gate   = "runs/gate/entropy_gate.joblib"
    entropy_scores = "data/gates/train_entropy_scores.jsonl"

    run(
        "A4 entropy-only gate training",
        [
            "chad.train_gate",
            "--probe-file",   PROBE_FILE,
            "--output-file",  entropy_gate,
            "--metrics-file", "runs/gate/entropy_metrics.json",
            "--model-type",   "logistic",
            "--feature-columns",
                "teacher_entropy", "teacher_entropy_top5",
                "teacher_max_prob", "teacher_min_prob",
                "source_len", "target_len",
        ],
        skip_if=Path(entropy_gate),
    )

    run(
        "A4 entropy gate scoring",
        [
            "chad.score_gate",
            "--input-file",         TRAIN_FILE,
            "--gate-file",          entropy_gate,
            "--teacher-model-name", TEACHER,
            "--output-file",        entropy_scores,
            "--batch-size",         str(SCORE_BATCH),
            "--no-generation-features",
        ],
        skip_if=Path(entropy_scores),
    )

    run(
        "A4 CHAD-entropy student training",
        [
            "chad.train_student",
            "--mode", "chad",
            "--model-name",        STUDENT,
            "--teacher-model-name", TEACHER,
            "--gate-scores-file",  entropy_scores,
            "--output-dir",        A4_DIR,
        ] + COMMON_TRAIN,
        skip_if=Path(A4_DIR) / "config.json",
    )


def stage_a5_semantic_gate() -> None:
    """A5 — Semantic/ROUGE ablation gate."""
    sem_gate   = "runs/gate/semantic_gate.joblib"
    sem_scores = "data/gates/train_semantic_scores.jsonl"

    run(
        "A5 semantic-only gate training",
        [
            "chad.train_gate",
            "--probe-file",   PROBE_FILE,
            "--output-file",  sem_gate,
            "--metrics-file", "runs/gate/semantic_metrics.json",
            "--model-type",   "logistic",
            "--feature-columns",
                "rouge_l_teacher_vs_target", "source_len", "target_len",
        ],
        skip_if=Path(sem_gate),
    )

    run(
        "A5 semantic gate scoring",
        [
            "chad.score_gate",
            "--input-file",         TRAIN_FILE,
            "--gate-file",          sem_gate,
            "--teacher-model-name", TEACHER,
            "--output-file",        sem_scores,
            "--batch-size",         str(SCORE_BATCH),
        ],
        skip_if=Path(sem_scores),
    )

    run(
        "A5 CHAD-semantic student training",
        [
            "chad.train_student",
            "--mode", "chad",
            "--model-name",        STUDENT,
            "--teacher-model-name", TEACHER,
            "--gate-scores-file",  sem_scores,
            "--output-dir",        A5_DIR,
        ] + COMMON_TRAIN,
        skip_if=Path(A5_DIR) / "config.json",
    )


def stage_evaluate() -> None:
    """Evaluate all trained models on the test set."""
    models = {
        "a1_ce":         A1_DIR,
        "a2_kd":         A2_DIR,
        "a4_entropy":    A4_DIR,
        "a5_semantic":   A5_DIR,
        "a6_chad":       A6_DIR,
    }
    for tag, model_dir in models.items():
        out_dir = f"{EVAL_BASE}/{tag}"
        run(
            f"Evaluate {tag}",
            [
                "chad.evaluate_model",
                "--model-dir",   model_dir,
                "--test-file",   TEST_FILE,
                "--output-dir",  out_dir,
                "--batch-size",  str(EVAL_BATCH),
                "--num-beams",   "4",
            ],
            skip_if=Path(out_dir) / "metrics.json",
        )


# ─────────────────────────────── MAIN ────────────────────────────────────────

def main() -> None:
    print("CHAD overnight pipeline starting …")
    print(f"Teacher : {TEACHER}")
    print(f"Student : {STUDENT}")

    stage_a1_ce()
    stage_probe()
    stage_train_gate()
    stage_score_gate()
    stage_a2_kd()
    stage_a6_chad()
    stage_a4_entropy_gate()
    stage_a5_semantic_gate()
    stage_evaluate()

    print("\n" + "="*70)
    print("ALL STAGES COMPLETE")
    print("="*70)
    print(f"\nResults written under:  {EVAL_BASE}/")
    print("Per-model metrics.json files contain ROUGE-L and BLEU scores.")


if __name__ == "__main__":
    main()
