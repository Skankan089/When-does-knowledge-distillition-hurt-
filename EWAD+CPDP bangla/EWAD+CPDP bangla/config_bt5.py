"""
Configuration — BanglaT5-Small Distillation (EWAD + CPDP)
==========================================================
Student : csebuetnlp/banglat5_small  (60M, d_model=512, vocab=32128)
EWAD teacher : BanglaT5-fine-tuned   (247M, d_model=768, vocab=32128) — same vocab → full KL
CPDP teacher : MT5-XL-Sum-fine-tuned (500M, d_model=768, vocab=250112) — different vocab → encoder CPDP

Experiments:
  baseline   — gold CE only, no distillation
  ewad       — confidence-gated KL from BanglaT5 teacher
  ewad_cpdp  — ewad + encoder-space CPDP regularisation with MT5-XL-Sum
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_FILE = r"D:\bansum_filtered_20k\bansum_filtered_20k.json"

STUDENT_MODEL      = "csebuetnlp/banglat5_small"
EWAD_TEACHER_MODEL = r"D:\summariser\not_more_than_limit\banglat5_bansum_20260218_213532\checkpoint-28240"
CPDP_TEACHER_MODEL = r"D:\summariser\not_more_than_limit\mt5xlsum_bansum_20260219_062938\checkpoint-14000"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_outputs_bt5")

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_TEXT_KEY    = "main"
DATASET_SUMMARY_KEY = "sum2"
TRAIN_SPLIT  = 0.8
VAL_SPLIT    = 0.1
SEED         = 42
MAX_SAMPLES  = None   # set e.g. 2000 for a smoke test

# ── Tokenisation ──────────────────────────────────────────────────────────────
MAX_INPUT_TOKENS  = 512   # T5 family max position embeddings
MAX_TARGET_TOKENS = 200   # Bengali summaries are typically 80-150 BanglaT5 tokens

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE            = 8
GRADIENT_ACCUMULATION = 4      # effective batch = 32
LEARNING_RATE         = 3e-4   # seq2seq fine-tuning typical range 1e-4 – 5e-4
NUM_EPOCHS            = 6
WARMUP_RATIO          = 0.06
MAX_GRAD_NORM         = 1.0
EARLY_STOP_PATIENCE   = 2      # consecutive epochs without val ROUGE-L improvement

# ── EWAD ──────────────────────────────────────────────────────────────────────
# Confidence-gated KD: gate = sigmoid(K * (teacher_max_prob - DELTA))
# High max-prob → gate→1 → trust teacher KL.  Low max-prob → gate→0 → trust gold CE.
EWAD_CE_FLOOR = 0.3    # gold CE always gets at least 30% weight
EWAD_K        = 10.0   # sharpness of sigmoid gate
EWAD_DELTA    = 0.5    # max-prob threshold at gate midpoint

# ── CPDP ──────────────────────────────────────────────────────────────────────
# Encoder-space geometry regularisation.
# Learnable projections map each model's mean-pooled encoder output to a shared
# CPDP_PROJ_DIM-dimensional space, then cosine distances enforce:
#   d(student, BT5) - d(student, MT5) ≈ d(BT5, MT5)
CPDP_WEIGHT   = 0.05
CPDP_PROJ_DIM = 256

# ── Evaluation ────────────────────────────────────────────────────────────────
QUICK_EVAL_STEPS    = 1000   # evaluate every N train batches (mid-epoch)
QUICK_EVAL_SAMPLES  = 50     # val samples for quick eval
EPOCH_EVAL_SAMPLES  = 200    # val samples for end-of-epoch eval
EVAL_MAX_NEW_TOKENS = 200
EVAL_NUM_BEAMS      = 4
