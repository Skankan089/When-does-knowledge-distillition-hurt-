# Counterfactual Harm-Aware Distillation (CHAD) for BanSum — Full Reference

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Core Problem: When Does KD Hurt?](#2-the-core-problem-when-does-kd-hurt)
3. [The CHAD Solution](#3-the-chad-solution)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [Models and Dataset](#5-models-and-dataset)
6. [Repository Structure](#6-repository-structure)
7. [Module Reference (chad/)](#7-module-reference-chad)
8. [Pipeline Scripts](#8-pipeline-scripts)
9. [Step-by-Step Manual Pipeline](#9-step-by-step-manual-pipeline)
10. [Ablation Experiments](#10-ablation-experiments)
11. [Data Formats](#11-data-formats)
12. [Training Outputs and Artifacts](#12-training-outputs-and-artifacts)
13. [Key Hyperparameters Reference](#13-key-hyperparameters-reference)
14. [Design Notes and Caveats](#14-design-notes-and-caveats)

---

## 1. Project Overview

This repository implements **CHAD — Counterfactual Harm-Aware Distillation**, a selective knowledge distillation (KD) framework for abstractive text summarization in Bangla (Bengali). The system is applied to the **BanSum** dataset using a teacher–student pair of BanglaT5 models.

The central idea is that **not every training sample benefits from KD**. On some samples, following the teacher's soft output distribution actually degrades the student's validation performance. CHAD identifies these "harmful" samples using a counterfactual influence estimator and trains a lightweight gate that modulates the KD loss per sample. Only samples where KD is estimated to be helpful contribute non-trivially to the distillation objective.

The full pipeline produces five trained models for comparison:

| Tag | Description |
|-----|-------------|
| **A1** | Student trained with cross-entropy loss only (CE baseline) |
| **A2** | Student trained with standard KD — uniform gate weight of 1 on all samples |
| **A4** | Student trained with CHAD gate based on entropy/confidence features only |
| **A5** | Student trained with CHAD gate based on ROUGE/semantic agreement features only |
| **A6** | Student trained with the full CHAD counterfactual gate (main contribution) |

---

## 2. The Core Problem: When Does KD Hurt?

Standard knowledge distillation adds a KL-divergence term to the training loss that encourages the student to match the teacher's full output distribution (soft labels) rather than just predicting the ground-truth token:

```
L_total = L_CE(student, gold) + λ · KL(student || teacher)
```

This works well when the teacher's distribution is informative. But the teacher can be harmful on a given sample if:

- The teacher is **overconfident** on a wrong answer and strongly penalizes the correct token.
- The **article is ambiguous or noisy** and the teacher has learned a spurious pattern.
- The source is **very short or very long** and the teacher's compression strategy differs from what the student should learn.
- The teacher's **generated summary** does not match the gold reference (low ROUGE), meaning the soft labels are misleading.

Traditional heuristics for detecting these cases — such as filtering by teacher confidence (low entropy = good sample) — fail because **teacher confidence does not equal student-level helpfulness**. A confident teacher can still give wrong guidance; an uncertain teacher can still be useful. This is the central argument of CHAD.

---

## 3. The CHAD Solution

CHAD addresses this by **directly measuring** whether KD is helpful for each training sample via a counterfactual probe on a small subset of the training data:

1. **Sample a probe set** of ~1,000 training examples.
2. For each probe sample, compute a **counterfactual score** answering: "If I took one gradient step using KD on this sample, would the student's validation loss decrease more or less than if I used only cross-entropy?"
3. **Label** each sample as KD-helpful (`kd_useful = 1`) or KD-harmful/neutral (`kd_useful = 0`).
4. **Extract features** for each probe sample (teacher confidence statistics, surface text statistics, student-teacher divergence, ROUGE agreement, etc.).
5. **Train a gate model** (logistic regression or gradient boosting) that predicts KD usefulness from these features.
6. **Score every training sample** with the gate to produce a continuous weight `h(x) ∈ [0, 1]`.
7. **Train the final student** using a weighted KD loss where `h(x)` modulates how much each sample contributes to the distillation objective.

The gate generalizes the probe's counterfactual judgments to the entire training set at inference time using only cheap-to-compute features — no gradient computation needed during gate scoring.

---

## 4. Mathematical Formulation

### Gradient Alignment Scoring (default: `grad_align`)

For each probe sample `x`, the KD usefulness score is:

```
score(x) = cos( ∇θ L_val,  ∇θ L_KD(x) )
```

where:
- `∇θ L_val` is the **mean CE gradient** accumulated over a small validation probe set (computed once and shared across all probe samples — eliminates per-sample sampling variance).
- `∇θ L_KD(x)` is the gradient of the **per-sample KD loss** component only.

A **positive cosine** means the KD gradient points in the same direction as improving validation loss → KD is helpful on this sample.
A **negative cosine** means the KD gradient opposes validation improvement → KD is harmful on this sample.

This is a TracIn-style influence function approximation: stable, low-variance (validation noise is shared rather than per-sample), and directly citable in the literature.

### Simulation Scoring (ablation: `simulation`)

An alternative one-step lookahead:

1. Save current weights `θ₀`.
2. Take one CE step on sample `x`: `θ_CE = θ₀ - η · ∇CE`. Evaluate `L_val(θ_CE)`.
3. Restore `θ₀`. Take one KD step: `θ_KD = θ₀ - η · ∇(CE + λ·KD)`. Evaluate `L_val(θ_KD)`.
4. Label: `kd_useful = 1` if `L_val(θ_CE) - L_val(θ_KD) > threshold`.

More intuitive but noisier because each sample sees a different random val batch.

### KD Loss (per-sample, temperature-scaled)

The per-sample KD loss is the token-averaged, temperature-scaled KL divergence:

```
L_KD(x) = (τ² / |y|) · Σ_t KL( softmax(z_s^t / τ) || softmax(z_t^t / τ) )
```

where `z_s` are student logits, `z_t` are teacher logits, `τ` is the temperature, and the sum is over non-padding target tokens. The `τ²` factor compensates for the gradient scaling introduced by dividing by temperature.

### Final CHAD Training Loss

```
L_CHAD = L_CE(θ, x, y)  +  λ · h(x) · L_KD(x)
```

where `h(x) ∈ [0, 1]` is the gate score. For standard KD (A2), `h(x) = 1` for all samples.

---

## 5. Models and Dataset

### Student Model
**`csebuetnlp/banglat5_small`** — A small BanglaT5 sequence-to-sequence model trained on Bangla text. This is the model being distilled into.

### Teacher Model
**`csebuetnlp/banglat5`** — The full-size BanglaT5 model. **Must be fine-tuned on BanSum before use.** Using the raw pre-trained checkpoint as a teacher produces garbage KD signal because its logits reflect generic language modeling rather than summarization. The fine-tuned teacher checkpoint is referred to as `runs/teacher` throughout this document.

### Dataset: BanSum
- **`bansum_filtered_20k.json`** — A filtered 20k-record subset of the BanSum Bangla summarization dataset, included in this directory for development and smoke-testing.
- The full 141k-record dataset (`bansum_lte_1000_tokens.json`) is processed by `run_pipeline_full.py` and is not included due to size.
- Each raw record contains:
  - `main` — the article text (source, in Bangla)
  - `sum1` — the reference summary (target, in Bangla)
  - `ID` — a unique record identifier

---

## 6. Repository Structure

```
.
├── bansum_filtered_20k.json        # 20k-record BanSum subset (smoke-test data)
├── requirements.txt                # Python dependencies
├── run_pipeline.py                 # Automated end-to-end pipeline (20k dataset)
├── run_pipeline_full.py            # Automated end-to-end pipeline (full 141k dataset)
│
├── chad/                           # Core library package
│   ├── __init__.py                 # Package init, version string (0.1.0)
│   ├── data.py                     # Data loading, JSONL I/O, collation, Dataset classes
│   ├── features.py                 # Feature extraction (teacher/student stats, ROUGE, semantic sim)
│   ├── losses.py                   # KD loss functions (per-sample and mean)
│   ├── prepare_data.py             # CLI: split raw JSON into train/val/test JSONL
│   ├── build_probe_labels.py       # CLI: counterfactual probe labeling (grad_align/simulation)
│   ├── train_gate.py               # CLI: train usefulness gate from probe labels
│   ├── score_gate.py               # CLI: score full training set with trained gate
│   ├── train_student.py            # CLI: train student in CE / KD / CHAD modes
│   ├── evaluate_model.py           # CLI: generate summaries and compute ROUGE/BLEU
│   └── analyze_probe.py            # CLI: analyze probe labels for reviewer statistics
│
├── data/                           # Intermediate data for the 20k smoke-test run
│   ├── splits/                     # train/val/test JSONL splits
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   ├── test.jsonl
│   │   └── metadata.json
│   ├── smoke_splits/               # Alternative smaller smoke splits
│   │   ├── train.jsonl, val.jsonl, test.jsonl
│   │   ├── kd_probe.jsonl, kd_probe.summary.json
│   │   └── metadata.json
│   ├── probes/                     # Counterfactual probe label files
│   │   ├── kd_usefulness_probe.jsonl       # One row per probe sample with features + kd_useful
│   │   └── kd_usefulness_probe.summary.json
│   └── gates/                      # Per-sample gate scores for the training set
│       ├── train_gate_scores.jsonl         # CHAD gate scores
│       ├── train_gate_scores.summary.json
│       ├── train_entropy_scores.jsonl      # A4 entropy-gate scores
│       ├── train_entropy_scores.summary.json
│       ├── train_semantic_scores.jsonl     # A5 semantic-gate scores
│       └── train_semantic_scores.summary.json
│
├── data_full/                      # Intermediate data for the full 141k pipeline
│   ├── splits/                     # Full train/val/test JSONL splits
│   ├── probes/                     # Full probe labels
│   └── gates/                      # Full gate scores
│
├── runs/                           # Trained model checkpoints (20k run)
│   ├── a1_ce/                      # A1: CE-only student (final weights + 2 checkpoints)
│   ├── a2_kd/                      # A2: Standard KD student
│   ├── a4_entropy_gate/            # A4: Entropy-gated KD student
│   ├── a5_semantic_gate/           # A5: Semantic-gated KD student
│   ├── a6_chad_13ep/               # A6: CHAD counterfactual-gated student (13 epochs)
│   ├── eval/                       # Evaluation outputs per model
│   │   ├── a1_ce/                  # predictions.jsonl + metrics.json
│   │   ├── a2_kd/
│   │   ├── a4_entropy/
│   │   └── a5_semantic/
│   └── gate/                       # Trained gate artifacts
│       ├── chad_gate.joblib        # CHAD gate (sklearn Pipeline)
│       ├── entropy_gate.joblib     # A4 entropy-only gate
│       ├── semantic_gate.joblib    # A5 semantic-only gate
│       ├── metrics.json            # CHAD gate validation metrics
│       ├── entropy_metrics.json
│       └── semantic_metrics.json
│
└── runs_full/                      # Trained model checkpoints (full 141k run)
    ├── a6_chad/                    # A6 full-dataset model
    ├── eval/                       # Full-dataset evaluation outputs
    └── gate/                       # Full-dataset gate artifacts
```

---

## 7. Module Reference (chad/)

### chad/\_\_init\_\_.py

Declares the package and sets `__version__ = "0.1.0"`.

---

### chad/data.py

Provides all data I/O and PyTorch dataset/collator primitives used throughout the pipeline.

**Key functions:**

| Function | Description |
|----------|-------------|
| `read_bansum_records(path, text_field, summary_field, id_field, limit)` | Reads the raw BanSum JSON file (array or JSONL). Extracts `source`, `target`, and `id` fields. Skips records with empty source or target. |
| `load_jsonl(path, limit)` | Reads a JSONL file line-by-line. The optional `limit` caps records loaded. |
| `save_jsonl(records, path)` | Writes an iterable of dicts to a JSONL file. Creates parent directories automatically. |
| `split_records(records, val_size, test_size, seed)` | Reproducibly shuffles and splits into train/val/test. Sizes can be absolute integers or fractional floats. |
| `attach_gate_scores(records, gate_scores_file, default_score)` | Joins gate scores onto training records by `id`. Missing records get `default_score = 0.0`. Injects `gate_score` into training batches. |
| `move_to_device(batch, device)` | Moves every tensor in a batch dict to the given device. |

**Key classes:**

- **`Seq2SeqRecordDataset`** — A minimal `torch.utils.data.Dataset` wrapping a list of record dicts.
- **`Seq2SeqCollator`** — Tokenizes batches. Encodes `source` into `input_ids`/`attention_mask` and `target` into `labels` (padding masked with `-100`). If any record in the batch has a `gate_score` key, a `gate_weight` tensor is added to the batch — this is how CHAD injects per-sample weights into the trainer.

---

### chad/features.py

Computes all numeric features used to train and run the usefulness gate.

**Text metrics (no model required):**

| Feature | Description |
|---------|-------------|
| `source_chars` | Character count of the source article |
| `target_chars` | Character count of the reference summary |
| `source_words` | Whitespace token count of the source |
| `target_words` | Whitespace token count of the reference summary |
| `compression_ratio` | `target_words / source_words` |
| `novelty_ratio` | Fraction of summary words not present in the source |

**ROUGE metrics (no model required):**

| Function | Description |
|----------|-------------|
| `rouge_n_f1(candidate, reference, n)` | ROUGE-N F1 with whitespace tokenization |
| `rouge_l_f1(candidate, reference)` | ROUGE-L F1 via dynamic-programming LCS |

**Teacher-side distribution features** (`teacher_distribution_features`):
Per-sample statistics from teacher logits over the gold summary tokens, reflecting teacher confidence:

| Feature | Description |
|---------|-------------|
| `teacher_entropy` | Mean per-token entropy of the teacher's output distribution |
| `teacher_max_prob` | Mean per-token top-1 probability assigned by the teacher |
| `teacher_margin` | Mean per-token margin between top-1 and top-2 probabilities |

**Student-side distribution features** (`student_distribution_features`):
Per-sample statistics from student logits, plus student–teacher divergence:

| Feature | Description |
|---------|-------------|
| `student_ce_loss` | Mean per-token CE loss of the student on the gold summary |
| `student_entropy` | Mean per-token entropy of the student's output distribution |
| `student_teacher_kl` | Mean per-token KL(student ∥ teacher) |

**Generation-based features** (require teacher beam-search pass):

| Feature | Description |
|---------|-------------|
| `gold_teacher_rouge_l` | ROUGE-L between the teacher-generated summary and the gold reference |
| `semantic_agreement` | Cosine similarity between mean-pooled sentence embeddings of teacher-generated and gold summary (optional, requires `--semantic-model-name`) |

**`build_feature_rows`** assembles all of the above into a single list of feature dicts, one per record, ready for gate training or scoring.

**`semantic_similarity_scores`** uses mean-pooling over encoder hidden states from an arbitrary HuggingFace sentence encoder to compute cosine similarity between text pairs.

---

### chad/losses.py

Contains the KD loss implementations, separated from the trainer to allow clean reuse in probe label building.

**`kd_loss_per_sample(student_logits, teacher_logits, labels, temperature)`**

Returns a 1-D tensor of shape `(batch,)`. Computes token-level KL divergence over the gold sequence, applies `τ²` gradient compensation, masks padding tokens (`labels == -100`), and averages over the unmasked token count per sample.

Raises `ValueError` if student and teacher logit shapes differ (vocabulary mismatch).

**`kd_loss_mean(student_logits, teacher_logits, labels, temperature)`**

Returns the scalar mean of `kd_loss_per_sample`. Used in probe label building where a single scalar loss value is needed.

---

### chad/prepare_data.py

**CLI module.** Reads the raw BanSum JSON and produces three JSONL split files and a metadata summary.

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-file` | `bansum_filtered_20k.json` | Path to raw BanSum JSON |
| `--output-dir` | `data/splits` | Output directory |
| `--text-field` | `main` | JSON key for article text |
| `--summary-field` | `sum1` | JSON key for summary |
| `--id-field` | `ID` | JSON key for record ID |
| `--val-size` | `1000` | Validation set size (int or fraction) |
| `--test-size` | `1000` | Test set size (int or fraction) |
| `--seed` | `42` | Random seed for shuffle |
| `--limit` | `None` | Optional cap on total records read |

Writes `train.jsonl`, `val.jsonl`, `test.jsonl`, and `metadata.json` to the output directory. With the 20k file (1000 val + 1000 test), the training set will be approximately 18,000 records.

---

### chad/build_probe_labels.py

**CLI module.** The core counterfactual labeling step. Samples a probe subset of the training data, computes per-sample KD usefulness labels using either gradient alignment or simulation, and saves feature vectors alongside labels.

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-file` | required | JSONL training split |
| `--val-file` | required | JSONL validation split |
| `--student-model-name` | `csebuetnlp/banglat5_small` | Student checkpoint path or HuggingFace ID (use A1) |
| `--teacher-model-name` | required | Fine-tuned teacher checkpoint |
| `--output-file` | `data/probes/kd_usefulness_probe.jsonl` | Output probe JSONL |
| `--method` | `grad_align` | Labeling method: `grad_align` or `simulation` |
| `--probe-size` | `1000` | Number of training samples to label |
| `--val-probe-size` | `200` | Validation samples for gradient alignment |
| `--probe-lr` | `1e-5` | Learning rate for simulation mode only |
| `--lambda-kd` | `0.5` | KD loss weight (must match training) |
| `--temperature` | `2.0` | KD temperature (must match training) |
| `--threshold` | `0.0` | Cosine score threshold for binarizing labels |
| `--max-source-length` | `512` | Source tokenizer truncation |
| `--max-target-length` | `128` | Target tokenizer truncation |
| `--batch-size` | `1` | Batch size for the probe loop |
| `--val-batch-size` | `4` | Batch size for val loss evaluation |
| `--feature-batch-size` | `4` | Batch size for feature extraction passes |
| `--max-new-tokens` | `128` | Max tokens for teacher beam search |
| `--num-beams` | `4` | Beam width for teacher generation |
| `--semantic-model-name` | `None` | Optional sentence encoder for semantic agreement |
| `--no-generation-features` | off | Skip teacher beam search (omits `gold_teacher_rouge_l`) |
| `--seed` | `42` | Random seed |
| `--device` | `cuda` if available | Compute device |

**`grad_align` method (recommended):**
1. Compute the mean validation CE gradient `g_val` once over the val probe set (shared across all probe samples).
2. For each probe sample `x`, compute the per-sample KD gradient `g_kd`.
3. Score = `cos(g_val, g_kd)`. Label = `1` if score > threshold.

**`simulation` method (ablation):**
1. For each probe sample, restore weights, take a CE step, measure val loss.
2. Restore weights, take a KD step, measure val loss.
3. Label = `1` if CE val loss − KD val loss > threshold.

A companion `.summary.json` is written recording overall helpful/harmful counts and the helpful ratio.

---

### chad/train_gate.py

**CLI module.** Trains a scikit-learn classifier or regressor on the probe labels. The trained pipeline is serialized as a `.joblib` artifact.

The sklearn `Pipeline` wraps three steps: `SimpleImputer(strategy="median")` → `StandardScaler()` → estimator.

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--probe-file` | required | Probe JSONL from `build_probe_labels` |
| `--output-file` | `runs/gate/chad_gate.joblib` | Output `.joblib` path |
| `--metrics-file` | `runs/gate/metrics.json` | Output metrics JSON |
| `--model-type` | `gbm` | `logistic`, `mlp`, `ridge`, `mlp_regressor`, or `gbm` |
| `--feature-columns` | `None` (auto) | Explicit list of feature names to use |
| `--test-size` | `0.2` | Fraction of probe data for internal validation |
| `--seed` | `42` | Random seed |

**Model types:**
- `logistic` — `LogisticRegression(max_iter=2000, class_weight="balanced")`. Binary classification on `kd_useful`.
- `mlp` — `MLPClassifier(hidden_layer_sizes=(64, 32))`. Binary classification on `kd_useful`.
- `ridge` — `Ridge(alpha=1.0)`. Regression on continuous `grad_align_score`.
- `mlp_regressor` — `MLPRegressor(hidden_layer_sizes=(128, 64, 32))`. Regression on `grad_align_score`.
- `gbm` — `GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8)`. Default for the full pipeline. Regression on `grad_align_score`.

The `METADATA_COLUMNS` set (`id`, `source`, `target`, `kd_useful`, `ce_step_val_loss`, `kd_step_val_loss`, `delta_val_loss`, `grad_align_score`) is always excluded from features to prevent data leakage.

The saved `.joblib` stores `{"model": Pipeline, "feature_columns": [...], "gate_type": "classifier"|"regressor"}`.

---

### chad/score_gate.py

**CLI module.** Applies the trained gate to every record in the training set, producing a continuous gate score per sample. These scores are later joined onto training records to weight the KD loss.

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-file` | required | JSONL training set to score |
| `--gate-file` | required | `.joblib` gate artifact |
| `--teacher-model-name` | required | Fine-tuned teacher checkpoint (for feature extraction) |
| `--student-model-name` | `None` | Student checkpoint (required only if gate uses `student_*` features) |
| `--output-file` | `data/gates/train_gate_scores.jsonl` | Output JSONL with `gate_score` per record |
| `--max-source-length` | `512` | Tokenizer source length cap |
| `--max-target-length` | `128` | Tokenizer target length cap |
| `--batch-size` | `4` | Batch size for forward passes |
| `--no-generation-features` | off | Skip teacher beam search (must match probe step) |
| `--allow-missing-features` | off | Use median imputation for absent features (default: error) |
| `--limit` | `None` | Optional cap on records to score |
| `--device` | `cuda` if available | Compute device |

**Gate score calculation:**
- Regression gate: raw `predict()` output is rescaled from `[-1, 1]` to `[0, 1]` via `clip((score + 1) / 2, 0, 1)`.
- Classification gate: `predict_proba()[:, 1]` (probability of the helpful class).

The feature set used during scoring **must exactly match** what was used during probe building. A `ValueError` is raised if any gate feature is missing from the scored records (unless `--allow-missing-features` is passed).

Output JSONL: `{"id": "...", "gate_score": 0.72, ...all feature columns...}`. A companion `.summary.json` with count, mean, and median gate score is also written.

---

### chad/train_student.py

**CLI module.** Trains the student model in one of three modes. This is the central training script.

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | required | `ce`, `kd`, or `chad` |
| `--train-file` | required | JSONL training split |
| `--val-file` | required | JSONL validation split |
| `--gate-scores-file` | `None` | Gate scores JSONL (required for `chad` mode) |
| `--model-name` | `csebuetnlp/banglat5_small` | Student model path or HuggingFace ID |
| `--teacher-model-name` | `None` | Teacher checkpoint (required for `kd`/`chad`) |
| `--output-dir` | required | Final model save directory |
| `--max-source-length` | `512` | Source truncation |
| `--max-target-length` | `128` | Target truncation |
| `--epochs` | `3.0` | Training epochs |
| `--learning-rate` | `5e-5` | AdamW learning rate |
| `--train-batch-size` | `4` | Per-device training batch size |
| `--eval-batch-size` | `4` | Per-device eval batch size |
| `--gradient-accumulation-steps` | `1` | Gradient accumulation |
| `--lambda-kd` | `0.5` | KD loss weight |
| `--temperature` | `2.0` | KD temperature |
| `--logging-steps` | `50` | Log every N steps |
| `--eval-steps` | `500` | Evaluate every N steps |
| `--save-steps` | `500` | Save checkpoint every N steps |
| `--save-total-limit` | `2` | Maximum checkpoints retained |
| `--seed` | `42` | Random seed |
| `--fp16` | off | Enable FP16 mixed precision |
| `--bf16` | off | Enable BF16 mixed precision (recommended for A100/H100) |
| `--early-stopping-patience` | `0` | Stop if ROUGE-L does not improve for N evals (0 = disabled) |

**Modes:**
- **`ce`**: Pure cross-entropy training. No teacher. Gate scores ignored. Loss = `outputs.loss` from the HuggingFace model.
- **`kd`**: Standard KD. Requires a teacher. Gate weight is fixed at `1.0` for all samples: `L = CE + λ · mean(KD_per_sample)`.
- **`chad`**: Gated KD. Requires teacher and gate scores file. Loss = `CE + λ · mean(h(x) · KD_per_sample(x))`. On val batches (no `gate_weight` in batch), falls back to uniform weight of `1.0`.

**`GatedKDTrainer`** extends `Seq2SeqTrainer`, overriding `compute_loss` to implement the CHAD loss. The teacher model is frozen (`requires_grad_(False)`, `teacher.eval()`), loaded to the same device as the student, and called inside `torch.no_grad()`.

Uses `load_best_model_at_end=True` with `metric_for_best_model="rouge_l"`. The checkpoint with the best validation ROUGE-L is automatically saved to `--output-dir` at the end of training.

A custom `RougeProgressCallback` replaces the default `ProgressCallback` to display current and best ROUGE-L in the tqdm progress bar postfix.

**Resuming from checkpoint:** if `--output-dir` already contains checkpoints, training continues automatically from the latest one via `get_last_checkpoint`.

---

### chad/evaluate_model.py

**CLI module.** Generates summaries on the test set and computes ROUGE-1, ROUGE-2, ROUGE-L, and BLEU.

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-dir` | required | Trained model directory |
| `--test-file` | required | JSONL test split |
| `--output-dir` | required | Where to write `predictions.jsonl` and `metrics.json` |
| `--max-source-length` | `512` | Source truncation |
| `--max-new-tokens` | `128` | Max tokens to generate per summary |
| `--batch-size` | `4` | Inference batch size |
| `--num-beams` | `4` | Beam search width |
| `--limit` | `None` | Optional cap on test records |
| `--bertscore-model` | `None` | Optional BERTScore model name |
| `--device` | `cuda` if available | Compute device |

ROUGE is computed using the whitespace-tokenized implementation from `features.py`. BLEU is computed using `sacrebleu.corpus_bleu`. BERTScore is optional and requires the `bert_score` package.

**Outputs:**
- `predictions.jsonl` — one record per test example with `id`, `source`, `target`, `prediction`.
- `metrics.json` — aggregate ROUGE-1/2/L, BLEU, and optional BERTScore F1.

---

### chad/analyze_probe.py

**CLI module.** Produces statistics from the probe label file for analysis and reviewer evidence.

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--probe-file` | required | Probe JSONL from `build_probe_labels` |
| `--output-file` | `runs/analysis/probe_analysis.json` | Output JSON path |
| `--length-bins` | `100 300` | Word-count thresholds for short/medium/long article bins |

**Outputs (JSON):**
- Overall KD helpful ratio
- Per source-length bin (short ≤ 100 words / medium ≤ 300 words / long) KD helpful ratio and counts
- Pearson correlation between each feature (`teacher_entropy`, `teacher_max_prob`, `teacher_margin`, `gold_teacher_rouge_l`, `semantic_agreement`) and `kd_useful`
- Delta val loss distribution statistics (mean, median, std, positive ratio)
- Interpretation note

The central claim: if teacher confidence correlates weakly or inversely with actual KD usefulness, that supports the CHAD argument that heuristic confidence-based gating is insufficient.

---

## 8. Pipeline Scripts

### 8.1 run_pipeline.py (20k smoke-test)

Runs the entire CHAD pipeline end-to-end on the local `bansum_filtered_20k.json` data. Each stage is **skipped automatically** if its output file already exists, so the script is safe to re-run after interruption.

**Edit the CONFIG block at the top of the file before running:**

| Variable | Default | Description |
|----------|---------|-------------|
| `TEACHER` | (absolute path) | Path to fine-tuned teacher checkpoint |
| `STUDENT` | `csebuetnlp/banglat5_small` | Student model ID |
| `EPOCHS` | `13` | Training epochs for all student models |
| `TRAIN_BATCH` | `4` | Per-device training batch size |
| `GRAD_ACCUM` | `2` | Gradient accumulation (effective batch = 4 × 2 = 8) |
| `LR` | `5e-5` | Learning rate |
| `PROBE_SIZE` | `1000` | Number of probe samples |
| `VAL_PROBE_SIZE` | `200` | Validation probe samples for gradient alignment |
| `SCORE_BATCH` | `4` | Batch size for gate scoring |

**Execution order:**

1. `stage_a1_ce()` → `runs/a1_ce/config.json`
2. `stage_probe()` → `data/probes/kd_usefulness_probe.jsonl`
3. `stage_train_gate()` → `runs/gate/chad_gate.joblib`
4. `stage_score_gate()` → `data/gates/train_gate_scores.jsonl`
5. `stage_a2_kd()` → `runs/a2_kd/config.json`
6. `stage_a6_chad()` → `runs/a6_chad/config.json`
7. `stage_a4_entropy_gate()` → entropy gate + scores + `runs/a4_entropy_gate/config.json`
8. `stage_a5_semantic_gate()` → semantic gate + scores + `runs/a5_semantic_gate/config.json`
9. `stage_evaluate()` → `runs/eval/{a1_ce,a2_kd,a4_entropy,a5_semantic,a6_chad}/metrics.json`

```powershell
python run_pipeline.py
```

---

### 8.2 run_pipeline_full.py (full 141k dataset)

Runs a streamlined A6-only pipeline on the full ~141k BanSum dataset. Key differences from the smoke-test pipeline:

- Uses `data_full/` directories for all intermediate files.
- `--no-generation-features` in both probe and score_gate stages — skips per-sample beam search, making scoring 141k samples feasible, and keeps features consistent between probe training and gate inference.
- Gate type is **GBM regression** (`gbm`) on the continuous `grad_align_score`, providing finer-grained gate weights than a binary classifier.
- `PROBE_SIZE = 5000` (5× larger probe for better gate training on the larger dataset).
- `TEMPERATURE = 0.5` (sharper distributions on the larger dataset).
- `MAX_SOURCE_LEN = 768` (longer articles in the full dataset).
- `EPOCHS = 8`, `EVAL_STEPS = 2000`, `SAVE_STEPS = 2000` (adjusted for larger training set size).
- Uses an external pre-trained A1 checkpoint (`A1_DIR`) rather than training A1 in-pipeline.

**Stages:**

1. `stage_prepare_data()` → `data_full/splits/train.jsonl`
2. `stage_probe()` → `data_full/probes/kd_usefulness_probe.jsonl`
3. `stage_train_gate()` → `runs_full/gate/chad_gate.joblib`
4. `stage_score_gate()` → `data_full/gates/train_gate_scores.jsonl`
5. `stage_a6_chad()` → `runs_full/a6_chad/config.json`
6. `stage_evaluate_a6()` → `runs_full/eval/a6_chad/metrics.json`

```powershell
python run_pipeline_full.py
```

---

## 9. Step-by-Step Manual Pipeline

### Step 1 — Install dependencies

```powershell
python -m pip install -r requirements.txt
```

Required: `torch`, `transformers>=4.40`, `accelerate`, `numpy`, `scikit-learn`, `joblib`, `sacrebleu`, `sentencepiece`, `tqdm`.

A CUDA-capable GPU is strongly recommended. BF16 (`--bf16`) is supported on Ampere and newer GPUs (A100, RTX 3090+).

---

### Step 2 — Prepare data splits

```powershell
python -m chad.prepare_data `
  --data-file bansum_filtered_20k.json `
  --output-dir data/splits `
  --text-field main `
  --summary-field sum1 `
  --id-field ID `
  --val-size 1000 `
  --test-size 1000
```

---

### Step 3 — Fine-tune the teacher (REQUIRED)

> The teacher model must be fine-tuned on BanSum before it is used as a KD teacher. Skip this step only if you already have a BanSum-fine-tuned BanglaT5 checkpoint.

```powershell
python -m chad.train_student `
  --mode ce `
  --train-file data/splits/train.jsonl `
  --val-file data/splits/val.jsonl `
  --model-name csebuetnlp/banglat5 `
  --output-dir runs/teacher `
  --epochs 5 `
  --train-batch-size 4 `
  --learning-rate 5e-5
```

Use `runs/teacher` as `--teacher-model-name` in all subsequent steps.

---

### Step 4 — A1: Cross-entropy baseline

```powershell
python -m chad.train_student `
  --mode ce `
  --train-file data/splits/train.jsonl `
  --val-file data/splits/val.jsonl `
  --model-name csebuetnlp/banglat5_small `
  --output-dir runs/a1_ce `
  --epochs 13 `
  --train-batch-size 4 `
  --gradient-accumulation-steps 2 `
  --learning-rate 5e-5 `
  --eval-steps 500 `
  --save-steps 500 `
  --bf16
```

> The A1 checkpoint is the starting point for probe label computation. Using A1 (rather than the raw pre-trained student) ensures the probe observes a student that has already learned to summarize, making counterfactual measurements more meaningful.

---

### Step 5 — A2: Standard KD baseline

```powershell
python -m chad.train_student `
  --mode kd `
  --train-file data/splits/train.jsonl `
  --val-file data/splits/val.jsonl `
  --model-name csebuetnlp/banglat5_small `
  --teacher-model-name runs/teacher `
  --output-dir runs/a2_kd `
  --epochs 13 `
  --train-batch-size 4 `
  --gradient-accumulation-steps 2 `
  --learning-rate 5e-5 `
  --lambda-kd 0.5 `
  --temperature 2.0 `
  --eval-steps 500 `
  --save-steps 500 `
  --bf16
```

---

### Step 6 — Build counterfactual probe labels

```powershell
# Recommended (grad_align):
python -m chad.build_probe_labels `
  --train-file data/splits/train.jsonl `
  --val-file data/splits/val.jsonl `
  --student-model-name runs/a1_ce `
  --teacher-model-name runs/teacher `
  --output-file data/probes/kd_usefulness_probe.jsonl `
  --method grad_align `
  --probe-size 1000 `
  --val-probe-size 200 `
  --lambda-kd 0.5 `
  --feature-batch-size 4 `
  --batch-size 1
```

```powershell
# Ablation (simulation, noisier):
python -m chad.build_probe_labels `
  --train-file data/splits/train.jsonl `
  --val-file data/splits/val.jsonl `
  --student-model-name runs/a1_ce `
  --teacher-model-name runs/teacher `
  --output-file data/probes/kd_usefulness_probe_sim.jsonl `
  --method simulation `
  --probe-size 1000 `
  --val-probe-size 200 `
  --probe-lr 1e-5 `
  --lambda-kd 0.5
```

> Pass `--no-generation-features` to skip teacher beam search and omit `gold_teacher_rouge_l`. This is required for large-scale runs to keep gate scoring consistent and feasible.

---

### Step 7 — Train the helpfulness gate

```powershell
python -m chad.train_gate `
  --probe-file data/probes/kd_usefulness_probe.jsonl `
  --output-file runs/gate/chad_gate.joblib `
  --metrics-file runs/gate/metrics.json `
  --model-type logistic
```

For the full pipeline, use `--model-type gbm` (GBM regressor on the continuous `grad_align_score`).

---

### Step 8 — Analyze probe labels (optional)

```powershell
python -m chad.analyze_probe `
  --probe-file data/probes/kd_usefulness_probe.jsonl `
  --output-file runs/analysis/probe_analysis.json
```

Prints and saves statistics showing: what fraction of probe samples were KD-helpful, whether helpfulness correlates with teacher confidence, and how helpfulness varies by article length.

---

### Step 9 — Score the full training set

```powershell
python -m chad.score_gate `
  --input-file data/splits/train.jsonl `
  --teacher-model-name runs/teacher `
  --gate-file runs/gate/chad_gate.joblib `
  --output-file data/gates/train_gate_scores.jsonl `
  --batch-size 4
```

> Use `--no-generation-features` here if it was also used during probe building. The feature set must be identical between probe training and gate scoring.

---

### Step 10 — A6: CHAD gated-KD training

```powershell
python -m chad.train_student `
  --mode chad `
  --train-file data/splits/train.jsonl `
  --val-file data/splits/val.jsonl `
  --gate-scores-file data/gates/train_gate_scores.jsonl `
  --model-name csebuetnlp/banglat5_small `
  --teacher-model-name runs/teacher `
  --output-dir runs/a6_chad `
  --epochs 13 `
  --train-batch-size 4 `
  --gradient-accumulation-steps 2 `
  --learning-rate 5e-5 `
  --lambda-kd 0.5 `
  --temperature 2.0 `
  --eval-steps 500 `
  --save-steps 500 `
  --bf16
```

---

### Step 11 — Evaluate all models

```powershell
python -m chad.evaluate_model --model-dir runs/a1_ce --test-file data/splits/test.jsonl --output-dir runs/eval/a1_ce
python -m chad.evaluate_model --model-dir runs/a2_kd --test-file data/splits/test.jsonl --output-dir runs/eval/a2_kd
python -m chad.evaluate_model --model-dir runs/a4_entropy_gate --test-file data/splits/test.jsonl --output-dir runs/eval/a4_entropy
python -m chad.evaluate_model --model-dir runs/a5_semantic_gate --test-file data/splits/test.jsonl --output-dir runs/eval/a5_semantic
python -m chad.evaluate_model --model-dir runs/a6_chad --test-file data/splits/test.jsonl --output-dir runs/eval/a6_chad
```

---

## 10. Ablation Experiments

CHAD supports two ablation conditions testing whether the improvement comes from the counterfactual gate or whether simpler heuristics achieve the same result.

### A4: Entropy-Only Gate

Uses only teacher-side confidence features (entropy, max probability, margin, source/target length). If this ablation matches CHAD's performance, it would suggest simple confidence-based gating is sufficient — undermining CHAD's motivation. The expectation is that A4 underperforms A6.

```powershell
# 1. Train gate on entropy features only
python -m chad.train_gate `
  --probe-file data/probes/kd_usefulness_probe.jsonl `
  --output-file runs/gate/entropy_gate.joblib `
  --metrics-file runs/gate/entropy_metrics.json `
  --model-type logistic `
  --feature-columns teacher_entropy teacher_entropy_top5 teacher_max_prob teacher_min_prob source_len target_len

# 2. Score training set (no generation features needed)
python -m chad.score_gate `
  --input-file data/splits/train.jsonl `
  --teacher-model-name runs/teacher `
  --gate-file runs/gate/entropy_gate.joblib `
  --output-file data/gates/train_entropy_scores.jsonl `
  --no-generation-features

# 3. Train student with entropy-gated KD
python -m chad.train_student `
  --mode chad `
  --train-file data/splits/train.jsonl `
  --val-file data/splits/val.jsonl `
  --gate-scores-file data/gates/train_entropy_scores.jsonl `
  --model-name csebuetnlp/banglat5_small `
  --teacher-model-name runs/teacher `
  --output-dir runs/a4_entropy_gate `
  --epochs 13 --train-batch-size 4 --gradient-accumulation-steps 2 `
  --learning-rate 5e-5 --lambda-kd 0.5 --bf16
```

---

### A5: Semantic/ROUGE Agreement Gate

Uses only surface-level text agreement features (ROUGE-L between teacher summary and gold, compression ratio, novelty ratio). Tests whether teacher output quality alone predicts KD usefulness.

```powershell
# 1. Train gate on semantic/ROUGE features only
python -m chad.train_gate `
  --probe-file data/probes/kd_usefulness_probe.jsonl `
  --output-file runs/gate/semantic_gate.joblib `
  --metrics-file runs/gate/semantic_metrics.json `
  --model-type logistic `
  --feature-columns rouge_l_teacher_vs_target source_len target_len

# 2. Score training set
python -m chad.score_gate `
  --input-file data/splits/train.jsonl `
  --teacher-model-name runs/teacher `
  --gate-file runs/gate/semantic_gate.joblib `
  --output-file data/gates/train_semantic_scores.jsonl

# 3. Train student with semantic-gated KD
python -m chad.train_student `
  --mode chad `
  --train-file data/splits/train.jsonl `
  --val-file data/splits/val.jsonl `
  --gate-scores-file data/gates/train_semantic_scores.jsonl `
  --model-name csebuetnlp/banglat5_small `
  --teacher-model-name runs/teacher `
  --output-dir runs/a5_semantic_gate `
  --epochs 13 --train-batch-size 4 --gradient-accumulation-steps 2 `
  --learning-rate 5e-5 --lambda-kd 0.5 --bf16
```

---

## 11. Data Formats

### Raw BanSum JSON

Input data file — either a JSON array or newline-delimited JSON. Each record:
```json
{
  "ID": "abc123",
  "main": "...article text in Bangla...",
  "sum1": "...reference summary in Bangla..."
}
```

### Normalized JSONL Splits (train/val/test)

All internal processing uses a 3-field normalized format:
```json
{"id": "abc123", "source": "...article...", "target": "...summary..."}
```

### Probe JSONL

Output of `build_probe_labels.py`. One JSON object per line:
```json
{
  "id": "abc123",
  "source_chars": 1430.0,   "target_chars": 210.0,
  "source_words": 287.0,    "target_words": 42.0,
  "compression_ratio": 0.146, "novelty_ratio": 0.381,
  "teacher_entropy": 0.823,   "teacher_max_prob": 0.712,  "teacher_margin": 0.605,
  "student_ce_loss": 1.234,   "student_entropy": 1.102,   "student_teacher_kl": 0.342,
  "gold_teacher_rouge_l": 0.44,
  "semantic_agreement": null,
  "source": "...full article text...",
  "target": "...reference summary...",
  "grad_align_score": 0.032,
  "kd_useful": 1
}
```

### Gate Scores JSONL

Output of `score_gate.py`. One JSON object per training sample:
```json
{"id": "abc123", "gate_score": 0.72, "source_chars": 1430.0, ...other features...}
```

### Evaluation Output

`predictions.jsonl`:
```json
{"id": "abc123", "source": "...", "target": "...", "prediction": "..."}
```

`metrics.json`:
```json
{
  "model_dir": "runs/a6_chad",
  "test_file": "data/splits/test.jsonl",
  "count": 1000,
  "rouge_1": 0.412,
  "rouge_2": 0.198,
  "rouge_l": 0.376,
  "bleu": 14.2
}
```

---

## 12. Training Outputs and Artifacts

### Model Checkpoints (`runs/*/`)

Each trained model directory contains:
- `config.json` — HuggingFace model configuration
- `model.safetensors` — model weights (best checkpoint by validation ROUGE-L)
- `tokenizer.json`, `tokenizer_config.json` — tokenizer files
- `generation_config.json` — generation defaults
- `checkpoint-NNNNN/` — intermediate checkpoints (up to `save_total_limit = 2`)

### Gate Artifacts (`runs/gate/`)

| File | Description |
|------|-------------|
| `chad_gate.joblib` | CHAD gate sklearn Pipeline |
| `entropy_gate.joblib` | A4 ablation gate (entropy features only) |
| `semantic_gate.joblib` | A5 ablation gate (ROUGE/semantic features only) |
| `metrics.json` | CHAD gate validation metrics (accuracy/F1/AUC or MAE/R²) |
| `entropy_metrics.json` | A4 gate metrics |
| `semantic_metrics.json` | A5 gate metrics |

Each `.joblib` stores: `{"model": Pipeline, "feature_columns": [...], "gate_type": "classifier"|"regressor"}`.

---

## 13. Key Hyperparameters Reference

| Parameter | 20k pipeline | Full pipeline | Notes |
|-----------|-------------|---------------|-------|
| `--epochs` | 13 | 8 | Training epochs |
| `--learning-rate` | 5e-5 | 5e-5 | AdamW LR |
| `--train-batch-size` | 4 | 4 | Per-device batch size |
| `--gradient-accumulation-steps` | 2 | 2 | Effective batch = 4 × 2 = 8 |
| `--lambda-kd` | 0.5 | 0.5 | KD loss weight; must be consistent across probe, scoring, and training |
| `--temperature` | 2.0 | 0.5 | KD temperature; higher = softer distributions |
| `--probe-size` | 1000 | 5000 | Training samples labeled for the probe |
| `--val-probe-size` | 200 | 300 | Val samples for gradient alignment |
| `--max-source-length` | 512 | 768 | Source tokenizer truncation |
| `--max-target-length` | 128 | 128 | Target tokenizer truncation |
| `--eval-steps` | 500 | 2000 | Evaluation frequency |
| `--save-steps` | 500 | 2000 | Checkpoint save frequency |
| `--save-total-limit` | 2 | 2 | Maximum checkpoints retained |
| `--num-beams` | 4 | 4 | Beam search width for generation |
| `--threshold` | 0.0 | 0.0 | Cosine score threshold for binarizing probe labels |
| `--early-stopping-patience` | 0 | 5 | Stop if ROUGE-L does not improve for N evals |

---

## 14. Design Notes and Caveats

**Teacher must be fine-tuned.** Using the raw `csebuetnlp/banglat5` checkpoint as the teacher produces garbage KD signal. The teacher's soft labels must reflect learned summarization behaviour, not generic language modeling.

**Shared vocabulary required.** The KD loss in `losses.py` assumes student and teacher share the same tokenizer and vocabulary (identical vocab size and token IDs). This holds because both models are BanglaT5 variants. A `ValueError` is raised if logit shapes differ.

**Feature consistency between probe and scoring.** The gate is trained on features present in the probe file. When scoring the full training set, the exact same features must be computable. If `--no-generation-features` is used during probe building, it must also be used during scoring — otherwise `gold_teacher_rouge_l` will be absent from gate scoring and `score_gate.py` will raise a `ValueError` (unless `--allow-missing-features` is passed for median imputation).

**Lambda and temperature consistency.** The `--lambda-kd` and `--temperature` values used in `build_probe_labels.py` must match those used in `train_student.py --mode chad`. These hyperparameters define what KD loss gradient is measured during probing — if training uses different values, the labels are measuring a different objective than what is actually trained.

**Gradient alignment runs in `train()` mode.** Both the val gradient computation and the per-sample KD gradient computation call `student.train()`. This ensures batch normalization and dropout (if any) behave consistently between the two gradient directions.

**Gate score rescaling.** For regression gates, raw scores in `[-1, 1]` (the grad_align range) are rescaled to `[0, 1]` via `clip((score + 1) / 2, 0, 1)` before being used as `gate_weight`. This keeps the KD loss scale consistent regardless of gate type (classifier vs. regressor).

**Resumable training.** `train_student.py` checks for existing checkpoints in `--output-dir` using HuggingFace's `get_last_checkpoint`. If found, training resumes automatically. Both pipeline scripts are idempotent — stages are skipped if their output file already exists.

**CHAD mode on validation batches.** During evaluation, the HuggingFace trainer calls `compute_loss` on validation batches. These do not come from `attach_gate_scores` and therefore have no `gate_weight` key. `GatedKDTrainer` detects this and falls back to uniform weight `1.0` — equivalent to standard KD for validation, which is correct since CHAD only affects training.

**Bangla tokenization in ROUGE.** All ROUGE computation uses simple whitespace tokenization (`features.simple_tokens`). This is appropriate for Bangla text and avoids a dependency on language-specific tokenization libraries.

**BERTScore is optional.** The `--bertscore-model` argument in `evaluate_model.py` enables BERTScore computation. If the `bert_score` package is not installed, the exception is caught and logged in `metrics.json` without crashing evaluation.
