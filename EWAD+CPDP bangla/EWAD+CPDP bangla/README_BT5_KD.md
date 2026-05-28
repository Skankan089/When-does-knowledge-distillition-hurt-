# BanglaT5-Small Knowledge Distillation for Bangla Abstractive Summarisation

## Overview

This project implements a **seq2seq knowledge distillation (KD) pipeline** that compresses two large fine-tuned Bangla summarisation models into a 60-million-parameter student. The student (BanglaT5-small) is trained on 141,200 Bangla news articles using two complementary distillation objectives:

1. **EWAD** — Entropy-Weighted Adaptive Distillation: confidence-gated token-level KD from a same-vocabulary BanglaT5 teacher (247M).
2. **Encoder CPDP** — Cross-architecture Projected Distance Preservation: encoder-space geometry regularisation from an MT5-XL-Sum teacher (500M) whose vocabulary is incompatible with the student.

**Final test result (14,120 held-out samples, 4-beam decoding):**

| ROUGE-1 | ROUGE-2 | ROUGE-L |
|---------|---------|---------|
| 0.3163  | 0.1604  | **0.2587** |

---

## Repository Structure

```
├── config_bt5.py                       # All hyperparameters and paths
├── train_bt5_kd.py                     # Full training script (3 experiments)
├── eval_bt5.py                         # Test-set evaluation with live ROUGE-L bar
├── bansum_lte_1000_tokens.json         # Full 141,200-sample dataset (article + summaries)
└── student_outputs_bt5/
    ├── baseline_20260516_065411/       # Ablation: CE-only (ROUGE-L 0.2206 on 20k)
    ├── ewad_20260516_070916/           # Ablation: EWAD only (ROUGE-L 0.2231 on 20k)
    ├── ewad_cpdp_20260516_073001/      # Ablation: EWAD+CPDP (ROUGE-L 0.2328 on 20k) ← winner
    └── ewad_cpdp_20260516_121514/      # Final run: EWAD+CPDP on full 141k dataset
        ├── best_model/                 # Saved weights + tokeniser (HuggingFace format)
        │   ├── test_results.json       # Final ROUGE-1/2/L on 14,120 test samples
        │   └── test_predictions.json  # First 500 predictions vs references
        ├── experiment_config.json      # All training settings logged
        └── training_log.json           # Per-1000-batch ROUGE-L + loss checkpoints
```

---

## Models

### Student — BanglaT5-small

| Property | Value |
|---|---|
| HuggingFace ID | `csebuetnlp/banglat5_small` |
| Architecture | T5-small |
| Parameters | **60M** |
| Hidden size (d_model) | 512 |
| Vocabulary size | 32,128 |
| Tokeniser | SentencePiece (BanglaT5) |

This is the model being trained. It starts from the pre-trained `banglat5_small` checkpoint and is fine-tuned using knowledge distillation. The student handles both encoding (article) and decoding (summary generation).

---

### EWAD Teacher — BanglaT5 Fine-tuned

| Property | Value |
|---|---|
| Base | BanglaT5 (247M) |
| Fine-tuned checkpoint | `banglat5_bansum_20260218_213532/checkpoint-28240` |
| Parameters | **247M** |
| Hidden size (d_model) | 768 |
| Vocabulary size | 32,128 — **identical to student** |
| Role | Provides per-token probability distributions for KL distillation |
| Loaded dtype | `bfloat16` (frozen, eval mode) |

Because this teacher and the student share the exact same vocabulary (32,128 tokens), their output logit tensors have the same shape. This enables direct **full-vocabulary KL divergence** distillation — the student is trained to match the teacher's entire output distribution, not just the argmax token. This is the primary distillation signal.

---

### CPDP Teacher — MT5-XL-Sum Fine-tuned

| Property | Value |
|---|---|
| Base | MT5-XL-Sum (500M) |
| Fine-tuned checkpoint | `mt5xlsum_bansum_20260219_062938/checkpoint-14000` |
| Parameters | **500M** |
| Hidden size (d_model) | 768 |
| Vocabulary size | 250,112 — **incompatible with student** |
| Role | Provides encoder hidden states for geometric regularisation |
| Loaded dtype | `bfloat16` (frozen, eval mode) |

Because this teacher's vocabulary (250,112) does not match the student's (32,128), direct logit-level KD is impossible. Instead, only the **encoder's hidden states** are used. This teacher contributes a structural regularisation signal that teaches the student how to represent the input in a way consistent with a diverse, multilingual encoder. See the CPDP section below for the mathematical details.

---

## Dataset

### Full Training Dataset

| Property | Value |
|---|---|
| File | `bansum_lte_1000_tokens.json` |
| Total samples | **141,200** |
| Filter criterion | Articles ≤ 1,000 BanglaT5 tokens |
| Fields | `ID`, `main` (article), `sum1`, `sum2`, `sum3`, `token_count` |
| Summary used | `sum2` (mapped via `DATASET_SUMMARY_KEY`) |
| Article field | `main` (mapped via `DATASET_TEXT_KEY`) |

### Train / Validation / Test Split

The dataset is split **deterministically** using a fixed random seed (SEED=42) so that any re-run produces exactly the same partition:

```
Total: 141,200
  Train : 112,960  (80%)  ← gradient updates
  Val   :  14,120  (10%)  ← early stopping / best-model selection
  Test  :  14,120  (10%)  ← never seen during training, used only in eval_bt5.py
```

No data leakage: the test split (`data[val_end:]`) is silently skipped during training and is only accessed by `eval_bt5.py`. The validation set is used solely for checkpoint selection (no weight updates) — it does not contribute gradients.

### 20k Ablation Dataset

For the three ablation experiments (baseline / ewad / ewad_cpdp), a 20,000-sample filtered subset was used:

```
D:\bansum_filtered_20k\bansum_filtered_20k.json
  Train : 16,000
  Val   :  2,000
  Test  :  2,000 (unused in ablation)
```

---

## Distillation Methods

### 1. EWAD — Entropy-Weighted Adaptive Distillation

EWAD is a **confidence-gated** knowledge distillation loss. The core idea: the teacher's prediction is only trustworthy when the teacher is confident. When the teacher is uncertain (e.g. rare words, ambiguous phrasing), the student should rely on the gold summary instead of copying the teacher's confused distribution.

#### Gate Mechanism

For each decoder time step, the teacher's maximum output probability is used to compute a sigmoid gate:

```
gate = sigmoid(K × (max_teacher_prob − Δ))
     = sigmoid(10 × (max_teacher_prob − 0.5))
```

| Teacher max_prob | Gate value | KD weight | Behaviour |
|---|---|---|---|
| 0.95 | 0.982 | 0.687 | Copy teacher almost entirely |
| 0.77 | 0.783 | 0.548 | Balanced (observed mean during training) |
| 0.50 | 0.500 | 0.350 | Equally weighted |
| 0.30 | 0.119 | 0.083 | Almost entirely gold CE |
| 0.10 | 0.007 | 0.005 | Pure gold CE |

#### Loss Formula

```
kd_weight = gate × (1 − CE_FLOOR)     ∈ [0.0, 0.70]
ce_weight = 1 − kd_weight              ∈ [0.30, 1.00]

L_EWAD = ce_weight × CE(student, gold) + kd_weight × KL(teacher ‖ student)
```

- `CE_FLOOR = 0.3` — the gold cross-entropy loss always contributes at least 30% weight, preventing the student from becoming fully teacher-dependent.
- `K = 10.0` — sharpness of the sigmoid (makes the gate near-binary for confident/uncertain extremes).
- `Δ = 0.5` — the midpoint; gate=0.5 when teacher max_prob exactly equals 0.5.
- KL is summed over the full vocabulary (B, T, V) and averaged over valid (non-padding) token positions.
- Both logits are cast to `float32` before softmax to avoid numerical issues.

#### Training Diagnostics (live per-batch display)

| Key | Description |
|---|---|
| `ce` | Mean cross-entropy loss per token |
| `kl` | Mean KL divergence per token |
| `gate` | Mean gate value (not equal to sigmoid(conf) due to Jensen's inequality) |
| `conf` | Mean teacher max probability per token |
| `kdw` | Mean KD weight = gate × 0.7 |
| `loss` | Combined EWAD loss (+ CPDP if enabled) |

---

### 2. Encoder CPDP — Cross-architecture Projected Distance Preservation

CPDP is a **geometric regularisation** that uses the MT5 teacher even though its vocabulary is incompatible for logit-level distillation.

#### Core Idea

The mean-pooled encoder outputs of all three models (student, BanglaT5 teacher, MT5 teacher) are projected into a shared 256-dimensional space. The student is then penalised if its position in this space does not maintain the correct relative distances to both teachers.

Specifically, the constraint is:

```
d(Student, BT5) − d(Student, MT5) ≈ d(BT5, MT5)
```

where `d(a, b) = 1 − cosine_similarity(a, b)` is the cosine distance.

This prevents the student from collapsing its encoder representation to be identical to one teacher while ignoring the other. It preserves the geometry of the representation space as defined by the teachers.

#### Implementation

```python
class EncoderCPDP(nn.Module):
    proj_s  : Linear(512 → 256, no bias)   # student
    proj_t1 : Linear(768 → 256, no bias)   # BanglaT5 teacher
    proj_t2 : Linear(768 → 256, no bias)   # MT5 teacher
    # All projections initialised with orthogonal weights
```

1. Each model's encoder output (variable length) is **mask-weighted mean-pooled** → shape `(B, D)`.
2. Each vector is projected to 256-d and **L2-normalised** (unit sphere).
3. Cosine distances are computed between all pairs.
4. `d(BT5, MT5)` is **detached** — it is a fixed reference, not differentiated through the teachers (which are frozen).

```
L_CPDP = mean( (d_s_t1 − d_s_t2 − d_t1_t2)² )
```

#### Loss weight

```
L_total = L_EWAD + CPDP_WEIGHT × L_CPDP
        = L_EWAD + 0.05 × L_CPDP
```

The projections (`proj_s`, `proj_t1`, `proj_t2`) are trained jointly with the student. Only the student's encoder weights are updated by the CPDP gradient; the teachers are fully frozen.

---

## Training Configuration

All hyperparameters are centralised in `config_bt5.py`:

| Parameter | Value | Notes |
|---|---|---|
| `BATCH_SIZE` | 8 | Per-GPU batch |
| `GRADIENT_ACCUMULATION` | 4 | Effective batch = 32 |
| `LEARNING_RATE` | 3e-4 | AdamW |
| `NUM_EPOCHS` | 8 (final), 6 (ablation) | Hard cap |
| `WARMUP_RATIO` | 0.06 | 6% of total steps |
| `MAX_GRAD_NORM` | 1.0 | Gradient clipping |
| `EARLY_STOP_PATIENCE` | 2 | Consecutive epochs without val ROUGE-L improvement |
| `MAX_INPUT_TOKENS` | 512 | Truncated to T5 max positions |
| `MAX_TARGET_TOKENS` | 200 | Summary generation cap during training |
| `EVAL_MAX_NEW_TOKENS` | 200 | Beam search generation cap |
| `EVAL_NUM_BEAMS` | 4 | Beam search width |
| `QUICK_EVAL_STEPS` | 1000 | Batches between mid-epoch ROUGE evaluations |
| `QUICK_EVAL_SAMPLES` | 50 | Samples used for mid-epoch ROUGE |
| `EPOCH_EVAL_SAMPLES` | 200 | Samples used for epoch-end ROUGE (early stop decision) |
| `EWAD_CE_FLOOR` | 0.3 | Min CE weight |
| `EWAD_K` | 10.0 | Gate sharpness |
| `EWAD_DELTA` | 0.5 | Gate midpoint |
| `CPDP_WEIGHT` | 0.05 | Relative weight of CPDP loss |
| `CPDP_PROJ_DIM` | 256 | Shared projection dimension |
| `SEED` | 42 | Dataset shuffle + split |

### Mixed Precision

Training uses **bfloat16 autocast** (not float16):

```python
with autocast('cuda', dtype=torch.bfloat16):
    ...
```

This is critical. The teacher models are loaded in `bfloat16`. If `float16` autocast were used, the teacher logits (bf16) would overflow `float16`'s limited range (~65504 max), producing NaN values before the `.float()` cast. `bfloat16` shares the same 8-bit exponent range as `float32`, preventing overflow entirely.

The `GradScaler` is disabled when bfloat16 is used (scaling is only needed for fp16).

---

## Three Experiments — Ablation on 20k Samples

Before scaling to the full 141k dataset, three experiments were run on the 20k filtered subset to confirm the design:

| Experiment | Description | Best Val ROUGE-L |
|---|---|---|
| `baseline` | CE only — no distillation | 0.2206 |
| `ewad` | EWAD KD from BanglaT5 teacher | 0.2231 |
| `ewad_cpdp` | EWAD + encoder CPDP from MT5 | **0.2328** ← winner |

All three stopped early at epoch 3 (patience=2). `ewad_cpdp` showed the strongest generalisation at +1.22 points ROUGE-L over baseline and +0.97 over EWAD-alone.

---

## Full Dataset Training — ewad_cpdp on 141k

The winning configuration (`ewad_cpdp`) was then trained on the full 141,200-sample dataset.

### Scale

| | 20k run | 141k run |
|---|---|---|
| Train samples | 16,000 | 112,960 |
| Val samples | 2,000 | 14,120 |
| Batches per epoch | 2,000 | ~14,120 |
| Quick evals per epoch | 2 | ~14 |
| Approx. speed | ~6.3 it/s | ~5.9 it/s |

### Training progression (val ROUGE-L at key checkpoints)

| Epoch | Batch | ROUGE-L |
|---|---|---|
| 1 | 3,000 | 0.1914 |
| 1 | 7,000 | 0.2136 |
| 1 | 11,000 | 0.2352 |
| 1 | 14,000 | 0.2383 |
| 2 | 7,000 | 0.2500 |
| 2 | 9,000 | 0.2528 |
| 2 | 11,000 | 0.2540 |
| **2** | **12,000** | **0.2581** ← best saved |
| 2 | 14,000 | 0.2453 |

Training stopped via early stopping after the best checkpoint (epoch 2, batch 12,000) was not surpassed, saving the model at **ROUGE-L = 0.2581** on 50-sample validation.

### Output location

```
student_outputs_bt5/ewad_cpdp_20260516_121514/
  best_model/          ← use this for inference
  experiment_config.json
  training_log.json
```

---

## Final Test Evaluation

The best model was evaluated on the **held-out test split** (14,120 samples, never seen during training or checkpoint selection) using 4-beam decoding:

```
python eval_bt5.py \
  --model student_outputs_bt5/ewad_cpdp_20260516_121514/best_model \
  --dataset bansum_lte_1000_tokens.json
```

| Metric | Score |
|---|---|
| **ROUGE-1** | **0.3163** |
| **ROUGE-2** | **0.1604** |
| **ROUGE-L** | **0.2587** |
| Test samples | 14,120 |
| Decoding | 4-beam, max 200 new tokens |

The test ROUGE-L (0.2587) is within 0.0006 of the best validation ROUGE-L (0.2581), confirming the model generalises cleanly without overfitting to the validation set.

---

## Inference

Load and run the final model directly:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = r"student_outputs_bt5/ewad_cpdp_20260516_121514/best_model"
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda().eval()

article = "আপনার বাংলা নিবন্ধ এখানে লিখুন..."
inputs = tok(article, max_length=512, truncation=True, return_tensors="pt").to("cuda")

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=200, num_beams=4, early_stopping=True)

summary = tok.decode(output_ids[0], skip_special_tokens=True)
print(summary)
```

---

## Scripts Reference

### `train_bt5_kd.py`

```
python train_bt5_kd.py --experiment <baseline|ewad|ewad_cpdp>
                       [--epochs N]
                       [--max-samples N]         # smoke test cap
                       [--dataset /path/to.json] # override dataset
```

- Saves best model (by val ROUGE-L) to `student_outputs_bt5/<experiment>_<timestamp>/best_model/`
- Logs per-1000-batch and per-epoch ROUGE-L + loss to `training_log.json`
- Saves full config to `experiment_config.json`
- Mid-epoch checkpoint saved to `checkpoints/` every 1,000 batches

### `eval_bt5.py`

```
python eval_bt5.py --model <model_dir>
                   [--dataset /path/to.json]
                   [--samples N]        # 0 = all test samples
                   [--batch-size N]     # default 16
                   [--beams N]          # default 4
```

- Reconstructs the identical test split using SEED=42
- Displays live running ROUGE-L in the tqdm bar: `RL=0.XXXX`
- Saves `test_results.json` (ROUGE-1/2/L) and `test_predictions.json` (first 500) into the model directory
- Uses whitespace-split tokeniser for ROUGE scoring to correctly handle Bangla Unicode

---

## Technical Issues Resolved

| Problem | Root Cause | Fix |
|---|---|---|
| `RuntimeError: size 32128 must match 32100` | `resize_token_embeddings(tokenizer.vocab_size)` used `tokenizer.vocab_size=32100` but model config has `32128` | Removed `resize_token_embeddings` entirely — student already outputs 32128 logits natively |
| NaN loss from the first training step | Default `autocast('cuda')` uses `float16`; teacher bf16 logits overflow fp16 range before the `.float()` cast | Changed to `autocast('cuda', dtype=torch.bfloat16)` |
| `FutureWarning: torch_dtype is deprecated` | Old transformers `from_pretrained` API | Changed to `dtype=torch.bfloat16` |
| `FutureWarning: torch.cuda.amp` deprecated | Old PyTorch AMP API | Changed to `from torch.amp import autocast, GradScaler` with `GradScaler('cuda', ...)` |
| ROUGE scores ≈ 0.002 on test set | `rouge_score` default tokeniser applies ASCII normalisation before tokenising, stripping all Bangla Unicode characters | Replaced with custom `tokenizer=_Tok()` where `_Tok.tokenize = t.split()` (whitespace only, no normalisation) |

---

## Dependencies

```
torch >= 2.0
transformers >= 4.40
rouge_score
numpy
tqdm
```

Teachers require ~6 GB VRAM each in bfloat16. All three models (student + 2 teachers) fit on a single 24 GB GPU for ewad_cpdp training.
