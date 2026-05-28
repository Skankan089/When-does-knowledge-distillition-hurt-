# LMI — Dual-Teacher Knowledge Distillation for Bengali Abstractive Summarization

## Entropy-Weighted Agreement-Aware Distillation (EWAD) with Capacity-Proportional Divergence Preservation (CPDP)

---

## 1. Project Overview

This project implements a **novel dual-teacher knowledge distillation** framework for Bengali abstractive summarization. Two large Qwen2.5-Instruct teacher models (32B and 14B) distill their knowledge into a compact Qwen2.5-3B student model using our proposed **EWAD + CPDP** loss functions.

**Key Idea:** Rather than naively averaging teacher logits, we use *entropy-weighted confidence scores* and *agreement-gated fallback* to adaptively blend knowledge from teachers with complementary strengths:

| Teacher | Role | Strength |
|---------|------|----------|
| **Qwen2.5-32B-Instruct** | Primary task expert | Rich abstraction, nuanced phrasing, complex reasoning |
| **Qwen2.5-14B-Instruct** | Semantic regularizer | Faithfulness, compression restraint, distillation stability |
| **Qwen2.5-3B** (Student) | Deployment model | Lightweight, LoRA fine-tuned |

### Novelty Claims

1. **Hierarchical Capacity Distillation** — Distilling from both a high-capacity (32B) and mid-capacity (14B) teacher to improve robustness of a 3B Bengali summarizer
2. **EWAD Loss** — Token-level adaptive weighting via confidence scores + agreement gating with gold-label fallback when teachers disagree
3. **CPDP Loss** — Regularizer preserving the student's relative divergence from each teacher proportional to the teachers' mutual divergence
4. **Comprehensive 8-experiment ablation** demonstrating the contribution of each component

---

## 2. Architecture

```
┌─────────────────────┐      ┌─────────────────────┐
│  Qwen2.5-32B-Inst.  │      │  Qwen2.5-14B-Inst.  │
│  (4-bit NF4)        │      │  (4-bit NF4)         │
│  Teacher 1           │      │  Teacher 2           │
└────────┬────────────┘      └────────┬────────────┘
         │  top-50 logprobs            │  top-50 logprobs
         │  per token                  │  per token
         ▼                             ▼
   ┌─────────────────────────────────────────┐
   │     Offline Teacher-Forced Scoring      │
   │  (single forward pass, deterministic)   │
   └─────────────────┬───────────────────────┘
                     │  JSONL files
                     ▼
   ┌─────────────────────────────────────────┐
   │         Student Training Loop           │
   │  Qwen2.5-3B + LoRA (r=64, α=128)       │
   │                                         │
   │  L_total = L_EWAD + μ · L_CPDP          │
   └─────────────────────────────────────────┘
```

### Why Teacher-Forced Scoring?

1. **Token alignment** — Teacher and student predict the *same* gold tokens → KL, EWAD, and CPDP are well-defined
2. **10–50× faster** than autoregressive generation
3. **Deterministic** — no sampling noise
4. **Standard approach** in distillation literature

---

## 3. Mathematical Formulation

### 3.1 EWAD Loss (Entropy-Weighted Agreement-Aware Distillation)

**Per-token teacher confidence:**

$$C_i^t = 1 - \frac{H(p_i^t)}{\log|V|}$$

**Confidence-proportional weight:**

$$w_{32B}^t = \text{softmax}\left(\frac{C_{32B}^t}{\tau_w}\right)$$

**Teacher agreement (via JSD):**

$$A_t = 1 - \text{JSD}(p_{32B}^t \| p_{14B}^t)$$

**Agreement gate (sigmoid):**

$$\lambda_t = \sigma\left(k \cdot (A_t - \delta)\right)$$

**EWAD Loss:**

$$\mathcal{L}_{\text{EWAD}} = \frac{1}{T} \sum_t \left[ \lambda_t \left( w_{32B}^t \cdot \text{KL}(p_{32B}^t \| p_S^t) + w_{14B}^t \cdot \text{KL}(p_{14B}^t \| p_S^t) \right) + (1 - \lambda_t) \cdot \text{CE}(y_t^*, p_S^t) \right]$$

**Interpretation:** When teachers agree ($\lambda_t \to 1$), the student learns from the confidence-weighted blend of both teachers. When they disagree ($\lambda_t \to 0$), the student falls back to the gold label, avoiding noisy teacher signals.

### 3.2 CPDP Loss (Capacity-Proportional Divergence Preservation)

**Teacher mutual divergence (reference):**

$$\Delta^* = \text{KL}(p_{32B} \| p_{14B})$$

**CPDP Loss:**

$$\mathcal{L}_{\text{CPDP}} = \left| \frac{\text{KL}(p_{32B} \| p_S)}{H(p_S)} - \frac{\text{KL}(p_{14B} \| p_S)}{H(p_S)} - \Delta^* \right|^2$$

**Interpretation:** The student should maintain relative KL distances to each teacher that are proportional to the teachers' own mutual divergence — preventing the student from collapsing toward one teacher.

### 3.3 Combined Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{EWAD}} + \mu \cdot \mathcal{L}_{\text{CPDP}}$$

where $\mu = 0.05$ (default).

---

## 4. Dataset

**BanSum** — Bengali abstractive summarization dataset.

| Property | Value |
|----------|-------|
| Total samples | ~141,200 |
| Train / Val / Test | 80% / 10% / 10% |
| Filtered | Articles ≤ 1,000 tokens |
| Text field | `main` |
| Summary field | `sum2` |
| Random seed | 42 |

---

## 5. Experimental Setup — 8 Ablation Configurations

We run 8 experiments to isolate the contribution of each proposed component:

| # | Experiment | Distillation | EWAD | CPDP | Description |
|---|-----------|:-----------:|:----:|:----:|-------------|
| 1 | `baseline_no_distill` | ✗ | ✗ | ✗ | Student fine-tuned on gold labels only |
| 2 | `single_teacher_32b` | ✓ | ✗ | ✗ | Standard KD: 32B → 3B |
| 3 | `single_teacher_14b` | ✓ | ✗ | ✗ | Standard KD: 14B → 3B |
| 4 | `fixed_weights` | ✓ | ✗ | ✗ | Fixed α=0.7, β=0.3 dual-teacher KD |
| 5 | `confidence_only` | ✓ | Partial | ✗ | Confidence-weighted KD (no agreement gate) |
| 6 | `agreement_only` | ✓ | Partial | ✗ | Agreement-gated KD (no confidence weighting) |
| 7 | `ewad_full` | ✓ | ✓ | ✗ | Full EWAD (confidence + agreement) |
| 8 | `ewad_cpdp` | ✓ | ✓ | ✓ | **Full proposed system: EWAD + CPDP** |

Experiments 1–4 are **baselines**, 5–7 are **ablations**, and 8 is the **full proposed method**.

---

## 6. Evaluation Metrics

All models are evaluated on the held-out test set (~14,120 samples) using 6 metrics:

| Metric | Library | Notes |
|--------|---------|-------|
| **ROUGE-1** | `rouge_score` | Unigram overlap, space-tokenized for Bangla |
| **ROUGE-2** | `rouge_score` | Bigram overlap |
| **ROUGE-L** | `rouge_score` | Longest common subsequence |
| **BLEU-1/2/4** | `nltk` | With smoothing (method 1) |
| **BERTScore** | `bert_score` | F1, language=`bn` |
| **Semantic Similarity** | `sentence-transformers` | Cosine similarity using `paraphrase-multilingual-MiniLM-L12-v2` |

---

## 7. Training Hyperparameters

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 64 |
| Alpha (α) | 128 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

### Optimizer & Schedule

| Parameter | Value |
|-----------|-------|
| Batch size | 4 |
| Gradient accumulation | 8 (effective batch = 32) |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Warmup ratio | 5% |
| Weight decay | 0.01 |
| Max gradient norm | 1.0 |
| Precision | bfloat16 |
| Gradient checkpointing | Enabled |
| LR scheduler | Cosine |

### EWAD Hyperparameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Confidence temperature | $\tau_w$ | 1.0 |
| Agreement sharpness | $k$ | 5.0 |
| Agreement threshold | $\delta$ | 0.5 |
| Entropy epsilon | $\epsilon$ | 1e-8 |

### CPDP Hyperparameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| CPDP weight | $\mu$ | 0.05 |
| Epsilon | $\epsilon$ | 1e-8 |

### Teacher Scoring

| Parameter | Value |
|-----------|-------|
| Quantization | 4-bit NF4 (`bitsandbytes`) |
| Compute dtype | bfloat16 |
| Top-k logprobs saved | 50 per token |
| Max input tokens | 1,024 |
| Max output tokens | 256 |

---

## 8. Hardware Requirements

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 5090 (32 GB VRAM) |
| System RAM | 64 GB |
| Disk | ~500 GB free (teacher outputs are ~21 GB each for train split) |

The 32 GB VRAM enables loading either 32B or 14B teachers in 4-bit for scoring, plus training the 3B student with LoRA and gradient checkpointing — all on a single consumer GPU.

---

## 9. Project Structure

```
LMI/
├── config.py                          # Central configuration (all hyperparams, paths, experiments)
├── ewad_cpdp_loss.py                  # Novel loss functions (EWAD + CPDP)
├── generate_teacher_outputs.py        # Offline teacher-forced scoring
├── train_student.py                   # Student training loop (LoRA + distillation)
├── evaluate.py                        # Evaluation (ROUGE, BLEU, BERTScore, Semantic Sim.)
├── run_all_experiments.py             # Orchestrates full pipeline (all 8 experiments)
├── run_test.py                        # Quick sanity test with 1000 samples
├── README.md                          # This file
├── teacher_outputs/
│   ├── teacher_32b/
│   │   ├── train.jsonl                # 32B logprobs for train split
│   │   ├── validation.jsonl           # 32B logprobs for val split
│   │   └── test.jsonl                 # 32B logprobs for test split
│   └── teacher_14b/
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── test.jsonl
├── student_outputs/
│   └── <experiment_name>_<timestamp>/
│       ├── best_model/                # Best checkpoint (by val loss)
│       ├── checkpoint-<step>/         # Periodic checkpoints
│       └── training_log.json          # Loss curves
└── eval_results/
    └── <experiment_name>_<timestamp>/
        └── results.json               # All metrics
```

---

## 10. How to Run

### Step 1: Generate Teacher Outputs (one-time, ~2 days total)

```bash
# Score all splits with 32B teacher
python generate_teacher_outputs.py --teacher 32b --split train
python generate_teacher_outputs.py --teacher 32b --split validation
python generate_teacher_outputs.py --teacher 32b --split test

# Score all splits with 14B teacher
python generate_teacher_outputs.py --teacher 14b --split train
python generate_teacher_outputs.py --teacher 14b --split validation
python generate_teacher_outputs.py --teacher 14b --split test
```

> **Note:** The script has built-in resume support — if interrupted (e.g., OOM), simply re-run the same command and it will continue from where it stopped.

### Step 2: Train All 8 Experiments

```bash
# Run all experiments sequentially (skips teacher generation if already done)
python run_all_experiments.py --skip-teachers

# Or run a single experiment
python train_student.py --experiment ewad_cpdp
python train_student.py --experiment baseline_no_distill
```

### Step 3: Evaluate

```bash
# Evaluate a specific model
python evaluate.py --model-dir student_outputs/<experiment_dir>/best_model

# Full pipeline (teachers → train → evaluate → comparison table)
python run_all_experiments.py
```

### Quick Test Mode (1000 samples)

```bash
# Runs 3 key experiments on 1000 samples to verify the pipeline works
python run_test.py
```

Or set `TEST_MODE = True` in `config.py` to use test mode with any script.

---

## 11. Key Implementation Details

### Vocabulary Alignment

The teacher models (32B, 14B) have a vocabulary size of **151,936** while the student (3B) has **151,665** tokens (including added special tokens). Our `align_vocab_size()` function truncates the larger vocabulary to match the smaller and renormalizes log-probabilities:

```python
def align_vocab_size(student_logprobs, teacher_logprobs):
    min_v = min(student_logprobs.size(-1), teacher_logprobs.size(-1))
    # Truncate to common vocabulary
    student_logprobs = student_logprobs[..., :min_v]
    teacher_logprobs = teacher_logprobs[..., :min_v]
    # Renormalize
    student_logprobs -= torch.logsumexp(student_logprobs, dim=-1, keepdim=True)
    teacher_logprobs -= torch.logsumexp(teacher_logprobs, dim=-1, keepdim=True)
    return student_logprobs, teacher_logprobs
```

### Sparse-to-Dense Logprob Reconstruction

Only the top-50 token logprobs per position are stored on disk. During training, we reconstruct a full (sparse) probability distribution by assigning a uniform "tail" probability to unseen tokens:

```python
# For each position: top-50 (token_id, logprob) → full vocab tensor
# Remaining probability mass is distributed uniformly over unseen tokens
```

### Gradient Checkpointing Compatibility

All tensor reshaping uses `.reshape()` instead of `.view()` to support non-contiguous memory layouts from gradient checkpointing.

---

## 12. Dependencies

```
torch >= 2.0
transformers >= 4.40
peft (LoRA)
bitsandbytes (4-bit quantization)
rouge_score
nltk
bert_score
sentence-transformers
numpy
tqdm
```

---

## 13. Expected Results

The 8-experiment ablation is designed to demonstrate:

1. **Distillation helps**: Experiments 2–8 should outperform Experiment 1 (baseline)
2. **Dual-teacher beats single-teacher**: Experiment 4 should outperform Experiments 2 and 3
3. **Adaptive weighting helps**: Experiments 5–7 should outperform Experiment 4 (fixed weights)
4. **Full EWAD is best combination**: Experiment 7 should outperform Experiments 5 and 6
5. **CPDP regularization adds value**: Experiment 8 should outperform or match Experiment 7

The final comparison table (generated by `run_all_experiments.py`) presents all 6 metrics across all 8 experiments for clear ablation analysis.

---

## Author

Bengali NLP Research — Dual-Teacher Knowledge Distillation Project

## License

Research use only.
