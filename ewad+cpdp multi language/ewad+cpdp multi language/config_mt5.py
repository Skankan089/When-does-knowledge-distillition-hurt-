"""
Configuration for multilingual MT5 KD experiments
===================================================
Languages : persian, punjabi, vietnamese, marathi, thai
Student   : google/mt5-small
Teacher 1 : google/mt5-base            (fine-tuned → EWAD teacher)
Teacher 2 : csebuetnlp/mT5_XLSum       (fine-tuned → CPDP teacher)

Workflow:
  1.  python prepare_xlsum.py              # build data files
  2.  python train_teacher.py --model mt5base
  3.  python train_teacher.py --model mt5xl
  4.  python train_mt5_kd.py --experiment baseline
  4.  python train_mt5_kd.py --experiment ewad
  4.  python train_mt5_kd.py --experiment ewad_cpdp
"""

# ── Models ────────────────────────────────────────────────────────────────────
STUDENT_MODEL = "google/mt5-small"

# After running train_teacher.py the scripts below auto-save to these fixed paths.
# If you want to use the raw pretrained weights instead, swap in HF hub IDs.
EWAD_TEACHER_MODEL = "teacher_outputs_lang3/mt5base/best_model"   # google/mt5-base fine-tuned
CPDP_TEACHER_MODEL = "csebuetnlp/mT5_multilingual_XLSum"    # use pretrained directly (already trained on XL-Sum)

# HF hub names used by train_teacher.py
EWAD_TEACHER_HUB = "google/mt5-base"
CPDP_TEACHER_HUB = "csebuetnlp/mT5_XLSum"

# All MT5 variants share the same SentencePiece tokenizer → use one instance
TOKENIZER_NAME = "google/mt5-small"

# ── Dataset ───────────────────────────────────────────────────────────────────
LANGUAGES        = ['persian', 'punjabi', 'vietnamese', 'marathi', 'thai']
SAMPLES_PER_LANG = 1500
DATASET_CSV      = "xlsum_all_train.csv"
TRAIN_JSON       = "xlsum_5lang3_train.json"
VAL_JSON         = "xlsum_5lang3_val.json"
TEST_JSON        = "xlsum_5lang3_test.json"
DATASET_TEXT_KEY    = "text"
DATASET_SUMMARY_KEY = "summary"

TRAIN_RATIO = 0.80          # 1200 train / 150 val / 150 test per language
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
SEED        = 42

# Optional task prefix — MT5 family was trained with task prompts
TASK_PREFIX = "summarize: "

# ── Sequence lengths ──────────────────────────────────────────────────────────
MAX_INPUT_TOKENS  = 512
MAX_TARGET_TOKENS = 128

# ── Teacher fine-tuning ───────────────────────────────────────────────────────
TEACHER_EPOCHS        = 5
TEACHER_BATCH_SIZE    = 4
TEACHER_GRAD_ACCUM    = 4       # effective batch = 16
TEACHER_LR            = 5e-4
TEACHER_WARMUP_RATIO  = 0.1
TEACHER_MAX_GRAD_NORM = 1.0
TEACHER_EVAL_STEPS    = 100     # log ROUGE every N *batches*
TEACHER_QUICK_EVAL_N  = 60      # samples used for mid-epoch eval
TEACHER_EPOCH_EVAL_N  = 150     # samples used for end-of-epoch eval
TEACHER_EVAL_BEAMS    = 2
TEACHER_EVAL_MAX_NEW  = 64
TEACHER_OUTPUT_DIR    = "teacher_outputs_lang3"

# ── Student / KD training ─────────────────────────────────────────────────────
NUM_EPOCHS            = 5
BATCH_SIZE            = 4
GRADIENT_ACCUMULATION = 4       # effective batch = 16
LEARNING_RATE         = 3e-4
WARMUP_RATIO          = 0.1
MAX_GRAD_NORM         = 1.0
EARLY_STOP_PATIENCE   = 3

# ── Student eval ──────────────────────────────────────────────────────────────
QUICK_EVAL_STEPS   = 100        # evaluate every N batches mid-epoch
QUICK_EVAL_SAMPLES = 60
EPOCH_EVAL_SAMPLES = 150
EVAL_MAX_NEW_TOKENS = 64
EVAL_NUM_BEAMS      = 2

# ── EWAD hyperparameters ──────────────────────────────────────────────────────
# gate = sigmoid(K * (max_teacher_prob - DELTA))
# kd_weight = gate * (1 - CE_FLOOR)
EWAD_K        = 10.0
EWAD_DELTA    = 0.3
EWAD_CE_FLOOR = 0.3

# ── CPDP hyperparameters ──────────────────────────────────────────────────────
CPDP_PROJ_DIM = 256
CPDP_WEIGHT   = 0.1

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs_mt5_lang3"
