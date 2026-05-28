# When Does Knowledge Distillation Hurt?

> A research repository exploring the **boundaries and failure modes of Knowledge Distillation (KD)** for low-resource and multilingual abstractive summarization using Bengali and multilingual NLP benchmarks.

---

## Overview

This repository contains three independent research projects that collectively investigate when, why, and how knowledge distillation can hurt — or help — a student model. Each project tackles the problem from a different angle: **diagnosing** KD failure cases, **mitigating** them via dual-teacher adaptive gating for a single language, and **scaling** the adaptive approach to multiple languages.

All three projects share a common thesis: *blindly applying KD is not always beneficial*. A teacher's soft labels can mislead the student on ambiguous or low-quality training examples, and naive KD weighting ignores this. The projects build progressively on one another to provide a complete answer.

---

## Repository Structure — Three Branches

| Branch | Folder | Language(s) | Task |
|--------|--------|-------------|------|
| [`chad-project`](#1-chad-project--counterfactual-helpfulness-aware-distillation) | `chad_project/` | Bengali | Counterfactual-gated KD via per-sample helpfulness probing |
| [`ewad-cpdp-bangla`](#2-ewad--cpdp-bangla--dual-teacher-adaptive-distillation) | `EWAD+CPDP bangla/` | Bengali | Dual-teacher EWAD + CPDP distillation (Qwen2.5 → BanglaT5) |
| [`ewad-cpdp-multilingual`](#3-ewad--cpdp-multilingual--cross-lingual-extension) | `ewad+cpdp multi language/` | Persian, Punjabi, Vietnamese, Marathi, Thai | EWAD + CPDP scaled to 5 languages via mT5 |

---

## 1. `chad-project` — Counterfactual Helpfulness-Aware Distillation

### What It Does

CHAD (**C**ounterfactual **H**elpfulness-**A**ware **D**istillation) answers the question: *For a given training sample, does applying KD actually help the student learn better than just using the gold label alone?*

It does so by building a **per-sample usefulness gate** that scores every training example on how beneficial KD would be for it. Only samples where KD is predicted to be useful receive a strong KD signal during training; the rest are trained with pure cross-entropy.

### Core Idea

The gate is trained using **counterfactual probe labels**. For each probe sample, CHAD measures:

- **Gradient Alignment** (`grad_align`, recommended): Cosine similarity between the mean validation CE gradient and the per-sample KD gradient. If the KD gradient points in the same direction as improving validation performance, the sample is labeled as KD-useful.
- **Simulation** (`simulation`, ablation): One-step lookahead — restore weights, take a CE step vs. a KD step, measure which improves validation loss more. Slower and noisier, but interpretable.

A binary label (`kd_useful = 1/0`) is derived from the cosine score via a threshold, and a sklearn pipeline (Imputer → Scaler → GBM Regressor) is trained on rich per-sample features extracted from both the teacher and student.

### Gated KD Loss

$$\mathcal{L}_{\text{CHAD}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \text{mean}\bigl(h(x) \cdot \mathcal{L}_{\text{KD}}^{(x)}\bigr)$$

where $h(x) \in [0, 1]$ is the gate score for sample $x$, and $\lambda = 0.5$ is the KD weight. Gate scores from a regression model are rescaled from $[-1, 1]$ to $[0, 1]$ via `clip((score + 1) / 2, 0, 1)`.

### Models

| Role | Model |
|------|-------|
| Teacher | `csebuetnlp/banglat5` (fine-tuned on BanSum) |
| Student | `csebuetnlp/banglat5_small` |
| Dataset | BanSum (~20k / ~141k Bengali news summarization) |

### Pipeline Stages

1. **`prepare_data`** — Splits raw BanSum JSON → `train/val/test.jsonl`
2. **`train_student --mode ce`** — Train A1 CE baseline (starting point for probe)
3. **`build_probe_labels`** — Compute counterfactual KD-usefulness labels on a probe subset
4. **`train_gate`** — Train sklearn GBM on probe features + labels
5. **`score_gate`** — Apply gate to all training samples, produce `gate_score` per sample
6. **`train_student --mode chad`** — Train A6 CHAD student with gated KD loss
7. **`evaluate_model`** — ROUGE-1/2/L + BLEU on test set

### Ablations

| ID | Model | Description |
|----|-------|-------------|
| A1 | `a1_ce` | Cross-entropy only baseline |
| A2 | `a2_kd` | Standard KD (gate weight = 1 for all) |
| A4 | `a4_entropy_gate` | Entropy-only gate (teacher confidence heuristic) |
| A5 | `a5_semantic_gate` | ROUGE/semantic gate (surface-text agreement) |
| A6 | `a6_chad` | **Full CHAD** — counterfactual-gated KD |

### Key Files

| File | Purpose |
|------|---------|
| `chad/losses.py` | `kd_loss_per_sample` — token-level KL with $\tau^2$ gradient compensation |
| `chad/build_probe_labels.py` | Gradient alignment and simulation labeling |
| `chad/train_gate.py` | sklearn Pipeline: Imputer → Scaler → GBM/LogisticReg/MLP |
| `chad/score_gate.py` | Score all training samples; rescale to $[0, 1]$ |
| `chad/train_student.py` | `GatedKDTrainer` overrides `compute_loss` for CHAD mode |
| `chad/features.py` | Teacher entropy/margin, student KL, ROUGE-L, semantic similarity |
| `chad/analyze_probe.py` | Statistics: helpful ratio by article length, feature–label correlations |
| `run_pipeline.py` | End-to-end 20k smoke-test pipeline (idempotent) |
| `run_pipeline_full.py` | Full 141k pipeline with GBM gate and larger probe |

### Dependencies

```
torch  transformers>=4.40  accelerate  numpy
scikit-learn  joblib  sacrebleu  sentencepiece  tqdm
```

---

## 2. `ewad-cpdp-bangla` — Dual-Teacher Adaptive Distillation

### What It Does

This project implements a **novel dual-teacher knowledge distillation** framework for Bengali abstractive summarization. Instead of one teacher, two large Qwen2.5-Instruct models (32B and 14B) distill knowledge into a compact Qwen2.5-3B student using two new loss functions — **EWAD** and **CPDP**.

The key question it addresses: *When two teachers disagree on a token, which should the student trust? And how can we prevent the student from collapsing toward one teacher?*

### Models

| Role | Model | Details |
|------|-------|---------|
| EWAD Teacher | `Qwen2.5-32B-Instruct` (4-bit NF4) | Primary — rich abstraction, nuanced phrasing |
| CPDP Teacher | `Qwen2.5-14B-Instruct` (4-bit NF4) | Regularizer — faithfulness, stability |
| Student | `Qwen2.5-3B` + LoRA (r=64, α=128) | Deployment model |
| Dataset | BanSum (~141k Bengali news articles) |

### EWAD Loss — Entropy-Weighted Agreement-Aware Distillation

**Per-token teacher confidence:**
$$C_i^t = 1 - \frac{H(p_i^t)}{\log|V|}$$

**Confidence-proportional weights** (softmax over confidences with temperature $\tau_w$):
$$w_{32B}^t = \text{softmax}\!\left(\frac{C_{32B}^t}{\tau_w}\right)$$

**Teacher agreement via Jensen-Shannon Divergence:**
$$A_t = 1 - \text{JSD}(p_{32B}^t \| p_{14B}^t) \quad \in [0, 1]$$

**Agreement gate** (sigmoid with sharpness $k=5$, threshold $\delta=0.5$):
$$\lambda_t = \sigma\!\bigl(k \cdot (A_t - \delta)\bigr)$$

**EWAD Loss with CE floor** (30% gold CE always preserved to prevent overfitting):

$$\mathcal{L}_{\text{EWAD}} = \frac{1}{T} \sum_t \Bigl[ \alpha_t \cdot \bigl(w_{32B}^t \cdot \text{KL}_{32B} + w_{14B}^t \cdot \text{KL}_{14B}\bigr) + \beta_t \cdot \text{CE}(y_t^*, p_S^t) \Bigr]$$

where the KD weight $\alpha_t = \lambda_t \cdot (1 - \text{CE\_floor}) \in [0,\ 0.7]$ and the CE weight $\beta_t = 1 - \alpha_t \in [0.3,\ 1.0]$, with $\text{CE\_floor} = 0.3$ ensuring gold labels always anchor the student.

**Interpretation:** When teachers agree ($\lambda_t \to 1$), the student blends knowledge from both teachers weighted by their confidence. When teachers disagree ($\lambda_t \to 0$), the student falls back to the gold label, avoiding corrupted teacher signals.

### CPDP Loss — Capacity-Proportional Divergence Preservation

Prevents the student from collapsing toward one teacher by enforcing that the student's relative KL distance to each teacher stays proportional to the teachers' own mutual divergence:

$$\mathcal{L}_{\text{CPDP}} = \left|\frac{\text{KL}(p_{32B} \| p_S)}{H(p_S)} - \frac{\text{KL}(p_{14B} \| p_S)}{H(p_S)} - \underbrace{\text{KL}(p_{32B} \| p_{14B})}_{\Delta^*}\right|^2$$

Student entropy $H(p_S)$ is **detached** to prevent CPDP from artificially pushing entropy up.

### Combined Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{EWAD}} + \mu \cdot \mathcal{L}_{\text{CPDP}}, \quad \mu = 0.05$$

### Teacher Logit Storage

Teachers are run **offline** (teacher-forced scoring — single forward pass over gold tokens). Only the **top-50 logprobs per token** are stored on disk. During training, a **sparse-to-dense reconstruction** distributes remaining probability mass uniformly over unseen tokens and renormalizes in log-space. This keeps disk usage feasible (~21 GB per split for 32B) while still enabling accurate KL computation.

### Vocabulary Alignment

The 32B/14B teachers have vocab size 151,936; the 3B student has 151,665. An `align_vocab_size()` function truncates to the smaller vocabulary and renormalizes log-probabilities.

### 8-Experiment Ablation Design

| # | Experiment | Distillation | EWAD | CPDP | Purpose |
|---|-----------|:-----------:|:----:|:----:|---------|
| 1 | `baseline_no_distill` | ✗ | ✗ | ✗ | CE-only baseline |
| 2 | `single_teacher_32b` | ✓ | ✗ | ✗ | Standard KD from 32B only |
| 3 | `single_teacher_14b` | ✓ | ✗ | ✗ | Standard KD from 14B only |
| 4 | `fixed_weights` | ✓ | ✗ | ✗ | Dual-teacher, fixed α=0.7, β=0.3 |
| 5 | `confidence_only` | ✓ | Partial | ✗ | Confidence weighting, no agreement gate |
| 6 | `agreement_only` | ✓ | Partial | ✗ | Agreement gate, no confidence weighting |
| 7 | `ewad_full` | ✓ | ✓ | ✗ | Full EWAD |
| 8 | `ewad_cpdp` | ✓ | ✓ | ✓ | **Full proposed system** |

### Key Files

| File | Purpose |
|------|---------|
| `ewad_cpdp_loss.py` | `EWADLoss`, `CPDPLoss`, `DualTeacherDistillationLoss` — full math implementation |
| `train_student.py` | LoRA training loop with lazy `IndexedJsonl` reader for multi-GB teacher files |
| `generate_teacher_outputs.py` | Offline teacher-forced scoring with top-50 logprob storage |
| `generate_qwen3_teacher.py` | Qwen3-series teacher generation variant |
| `train_bt5_kd.py` | BanglaT5 variant of the distillation training |
| `eval_bt5.py` / `eval_checkpoints.py` | ROUGE/BLEU/BERTScore evaluation |
| `config_bt5.py` | BanglaT5-specific config (EWAD single-teacher with confidence gate, encoder-space CPDP) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank | 64, α=128, dropout=0.05 |
| Effective batch | 32 (4 × 8 gradient accumulation) |
| Learning rate | 2e-4 (cosine LR with 5% warmup) |
| Epochs | 3 |
| Precision | bfloat16 |
| GPU | NVIDIA RTX 5090 (32 GB VRAM) |

### Dependencies

```
torch>=2.0  transformers>=4.40  peft  bitsandbytes
rouge_score  nltk  bert_score  sentence-transformers  numpy  tqdm
```

---

## 3. `ewad-cpdp-multilingual` — Cross-Lingual Extension

### What It Does

This project extends the **EWAD + CPDP framework to multilingual summarization** using the [XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum) dataset across five typologically diverse languages. The goal is to validate that adaptive dual-teacher distillation generalizes beyond Bengali and benefits low-resource multilingual settings.

### Languages & Dataset

| Property | Value |
|----------|-------|
| Languages | Persian · Punjabi · Vietnamese · Marathi · Thai |
| Dataset | XL-Sum (BBC multilingual abstractive summarization) |
| Samples per language | 1,500 |
| Split | 80% train / 10% val / 10% test (1200 / 150 / 150 per language) |
| Task prefix | `"summarize: "` |

### Models

| Role | Model | Details |
|------|-------|---------|
| EWAD Teacher | `google/mt5-base` (fine-tuned on XL-Sum) | Primary confidence-weighted teacher |
| CPDP Teacher | `csebuetnlp/mT5_multilingual_XLSum` | Pretrained XL-Sum checkpoint, divergence regularizer |
| Student | `google/mt5-small` | Lightweight multilingual deployment model |

All three models share the **same SentencePiece tokenizer** (`google/mt5-small`), so vocabulary alignment is exact and standard KL divergence applies without truncation.

### Experiments

| # | Experiment | Description |
|---|-----------|-------------|
| 1 | `baseline` | Cross-entropy fine-tuning only (no distillation) |
| 2 | `ewad` | Full EWAD dual-teacher distillation |
| 3 | `ewad_cpdp` | **Full system:** EWAD + CPDP regularization |

### Key Files

| File | Purpose |
|------|---------|
| `config_mt5.py` | All hyperparameters, model IDs, dataset paths, experiment configs |
| `prepare_xlsum.py` | Download and partition XL-Sum for the 5 target languages |
| `train_teacher.py` | Fine-tune `mt5-base` as the EWAD teacher |
| `train_mt5_kd.py` | Student training with EWAD/CPDP on mT5 architecture |
| `train_bt5_kd.py` | BanglaT5 KD variant (shared loss logic) |
| `kd_results.json` | Recorded experiment results |
| `first6 results.txt` | Summary of first 6 experiment outcomes |
| `paper.bib` / `compression_paper.bib` | Bibliography references |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 4, gradient accumulation 4 (effective batch = 16) |
| Learning rate | 5e-4, warmup 10% |
| Teacher fine-tuning epochs | 5 |
| Max input tokens | 512 |
| Max target tokens | 128 |

### How to Run

```bash
# 1. Build dataset splits
python prepare_xlsum.py

# 2. Fine-tune the EWAD teacher
python train_teacher.py --model mt5base

# 3. Train the student
python train_mt5_kd.py --experiment baseline
python train_mt5_kd.py --experiment ewad
python train_mt5_kd.py --experiment ewad_cpdp
```

### Dependencies

```
torch>=2.0  transformers>=4.40  datasets  sentencepiece
rouge_score  nltk  numpy  tqdm
```

---

## Research Connections Across Branches

The three branches form a coherent research arc:

```
chad-project
  ↓ asks: "Is this KD sample even helpful?"
  ↓ answer: counterfactual gate via gradient alignment

ewad-cpdp-bangla
  ↓ asks: "Given two teachers, how should we blend their signals?"
  ↓ answer: confidence-weighted + agreement-gated dual-teacher distillation (EWAD)
  ↓ asks: "How do we stop the student from collapsing toward one teacher?"
  ↓ answer: capacity-proportional divergence preservation (CPDP)

ewad-cpdp-multilingual
  ↓ asks: "Does EWAD + CPDP generalize beyond Bengali?"
  ↓ answer: validation across Persian, Punjabi, Vietnamese, Marathi, Thai
```

| Concept | CHAD | EWAD+CPDP Bangla | EWAD+CPDP Multilingual |
|---------|:----:|:----------------:|:----------------------:|
| Identifies harmful KD samples | ✓ | — | — |
| Adaptive per-token KD weighting | — | ✓ | ✓ |
| Teacher confidence signals | ✓ (features) | ✓ (gate) | ✓ (gate) |
| Teacher agreement / disagreement | — | ✓ (JSD gate) | ✓ (JSD gate) |
| Dual-teacher divergence preservation | — | ✓ (CPDP) | ✓ (CPDP) |
| Sparse logprob storage for large teachers | — | ✓ (top-50) | — |
| Multilingual generalization | — | — | ✓ |
| LoRA parameter-efficient training | — | ✓ | — |

---

## Evaluation Metrics

All three branches evaluate trained models using:

| Metric | Tool | Notes |
|--------|------|-------|
| **ROUGE-1 / ROUGE-2 / ROUGE-L** | `rouge_score` | Whitespace-tokenized (appropriate for Bengali/multilingual) |
| **BLEU** | `sacrebleu` / `nltk` | Corpus-level with smoothing |
| **BERTScore** | `bert_score` | Optional; requires GPU, language-specific model |
| **Semantic Similarity** | `sentence-transformers` | Cosine similarity via `paraphrase-multilingual-MiniLM-L12-v2` |

---

## Citation & License

Research use only. If you use or build upon any of these projects, please cite the corresponding work.

---

## Author

**Ankan Kumar Roy** — Bengali NLP Research, Dual-Teacher Knowledge Distillation Project  
BRAC University  
[ankan.kumar.roy1@g.bracu.ac.bd](mailto:ankan.kumar.roy1@g.bracu.ac.bd)
