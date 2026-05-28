# EWAD + CPDP — Multilingual Knowledge Distillation for Abstractive Summarization

## Entropy-Weighted Agreement-Aware Distillation (EWAD) with Capacity-Proportional Divergence Preservation (CPDP) — Multilingual Extension

---

## 1. Project Overview

This project extends the **EWAD + CPDP** dual-teacher knowledge distillation framework to **multilingual abstractive summarization** using the [XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum) dataset across five languages:

- **Persian**, **Punjabi**, **Vietnamese**, **Marathi**, **Thai**

A compact **mT5-small** student is trained by distilling from two multilingual teachers:

| Teacher | Model | Role |
|---------|-------|------|
| **EWAD Teacher** | `google/mt5-base` (fine-tuned) | Confidence-weighted primary teacher |
| **CPDP Teacher** | `csebuetnlp/mT5_multilingual_XLSum` | Divergence-preservation regularizer |

---

## 2. Experimental Configurations

Three experiments are supported:

| # | Experiment | Description |
|---|-----------|-------------|
| 1 | `baseline` | Standard cross-entropy fine-tuning (no distillation) |
| 2 | `ewad` | Full EWAD dual-teacher distillation |
| 3 | `ewad_cpdp` | **Full proposed system:** EWAD + CPDP regularization |

---

## 3. Project Structure

```
ewad+cpdp multi language/
├── config_mt5.py           # Central configuration (models, paths, hyperparams)
├── prepare_xlsum.py        # Dataset preparation from XL-Sum (5 languages)
├── train_teacher.py        # Fine-tune the mt5-base EWAD teacher
├── train_mt5_kd.py         # Student training with EWAD/CPDP distillation
├── train_bt5_kd.py         # BanglaT5 KD variant
├── kd_results.json         # Experiment results
├── first6 results.txt      # Summary of first 6 experiment results
├── paper.bib               # Bibliography
└── compression_paper.bib   # Compression paper references
```

---

## 4. Dataset

**XL-Sum** — BBC multilingual summarization dataset.

| Property | Value |
|----------|-------|
| Languages | Persian, Punjabi, Vietnamese, Marathi, Thai |
| Samples per language | 1,500 |
| Train / Val / Test | 80% / 10% / 10% (1200 / 150 / 150) |
| Text field | `text` |
| Summary field | `summary` |
| Task prefix | `"summarize: "` |
| Random seed | 42 |

---

## 5. How to Run

### Step 1: Prepare Dataset

```bash
python prepare_xlsum.py
```

Outputs: `xlsum_5lang3_train.json`, `xlsum_5lang3_val.json`, `xlsum_5lang3_test.json`

### Step 2: Fine-tune EWAD Teacher (mt5-base)

```bash
python train_teacher.py --model mt5base
```

### Step 3: (Optional) Fine-tune CPDP Teacher

```bash
python train_teacher.py --model mt5xl
```

### Step 4: Run Experiments

```bash
# Baseline (no distillation)
python train_mt5_kd.py --experiment baseline

# EWAD distillation
python train_mt5_kd.py --experiment ewad

# Full EWAD + CPDP
python train_mt5_kd.py --experiment ewad_cpdp
```

---

## 6. Model Configuration

### Student — mT5-small

| Parameter | Value |
|-----------|-------|
| Model | `google/mt5-small` |
| Max input tokens | 512 |
| Max target tokens | 128 |

### Teacher Fine-tuning

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch size | 4 |
| Gradient accumulation | 4 (effective batch = 16) |
| Learning rate | 5e-4 |
| Warmup ratio | 10% |

---

## 7. Dependencies

```
torch >= 2.0
transformers >= 4.40
datasets
sentencepiece
rouge_score
nltk
numpy
tqdm
```

Install with:

```bash
pip install torch transformers datasets sentencepiece rouge_score nltk numpy tqdm
```

---

## License

Research use only.
