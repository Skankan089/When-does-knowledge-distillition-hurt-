# Dual-Teacher Knowledge Distillation for Bengali Abstractive Summarization

## EWAD + CPDP: Entropy-Weighted Agreement-Aware Distillation with Capacity-Proportional Divergence Preservation

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Why Knowledge Distillation?](#2-why-knowledge-distillation)
3. [Why Two Teachers Instead of One?](#3-why-two-teachers-instead-of-one)
4. [The Models We Use](#4-the-models-we-use)
5. [The Dataset: BanSum](#5-the-dataset-bansum)
6. [Dataset Quality Filtering](#6-dataset-quality-filtering)
7. [The Full Pipeline (Step by Step)](#7-the-full-pipeline-step-by-step)
8. [Phase 1: Teacher Scoring (Offline)](#8-phase-1-teacher-scoring-offline)
9. [Phase 2: Student Training with LoRA](#9-phase-2-student-training-with-lora)
10. [The Decoder-Only Format: How We Feed Data to the Model](#10-the-decoder-only-format-how-we-feed-data-to-the-model)
11. [The Causal Shift: Why logits[t] Predicts token[t+1]](#11-the-causal-shift-why-logitst-predicts-tokent1)
12. [EWAD Loss: The Novel Distillation Loss Function](#12-ewad-loss-the-novel-distillation-loss-function)
13. [CPDP Loss: The Regularizer](#13-cpdp-loss-the-regularizer)
14. [Combined Loss: Putting It All Together](#14-combined-loss-putting-it-all-together)
15. [The 8 Ablation Experiments](#15-the-8-ablation-experiments)
16. [Phase 3: Evaluation](#16-phase-3-evaluation)
17. [All Code Files Explained](#17-all-code-files-explained)
18. [Hyperparameters Reference](#18-hyperparameters-reference)
19. [Hardware Requirements](#19-hardware-requirements)
20. [How to Run Everything](#20-how-to-run-everything)
21. [What Makes This Novel?](#21-what-makes-this-novel)
22. [Glossary of Terms](#22-glossary-of-terms)

---

## 1. What Is This Project?

This project tackles a practical problem: **large language models (LLMs) are very good at summarizing Bengali text, but they are too big and expensive to use in real-world applications**. A 32-billion parameter model needs expensive GPU hardware just to respond to a single request.

Our solution: **teach a small model to behave like the large ones**. We take two large "teacher" models (32B and 14B parameters) and use their knowledge to train a small "student" model (3B parameters) that can produce similar quality summaries while being 10× smaller and much faster.

This process is called **Knowledge Distillation (KD)** — like a master chef teaching an apprentice their cooking techniques. The apprentice doesn't just learn the recipes (the gold labels); they also learn HOW the master thinks about cooking (the soft probability distributions over all possible outputs).

The twist in our approach: **we use TWO teachers instead of one**, and we invented two new loss functions (EWAD and CPDP) that intelligently combine knowledge from both teachers depending on how confident and how much they agree with each other at each individual token.

---

## 2. Why Knowledge Distillation?

### The Problem with Big Models

| Model | Parameters | GPU Memory Needed | Inference Speed |
|-------|-----------|-------------------|-----------------|
| Qwen2.5-32B | 32 billion | ~20 GB (4-bit) | Slow |
| Qwen2.5-14B | 14 billion | ~10 GB (4-bit) | Medium |
| Qwen2.5-3B | 3 billion | ~6 GB (full) | Fast |

A 32B model produces great summaries, but it's impractical for deployment. A 3B model is fast and cheap, but its summaries aren't as good when trained only on gold labels.

### How Knowledge Distillation Helps

In traditional training, the student sees:
- Input: "The president visited the capital city yesterday..." 
- Target: "রাষ্ট্রপতি গতকাল রাজধানী পরিদর্শন করেছেন" (The president visited the capital yesterday)
- Loss: Cross-entropy against the **one correct answer** (hard label)

In knowledge distillation, the student also sees:
- Teacher's full probability distribution: "রাষ্ট্রপতি" (45%), "প্রেসিডেন্ট" (30%), "সরকারপ্রধান" (15%), ...
- This tells the student: "রাষ্ট্রপতি is the best word, but প্রেসিডেন্ট is also reasonable"
- This extra information is called **soft labels** or **dark knowledge**

The soft labels contain rich information about word similarities, alternative phrasings, and contextual relationships that hard labels alone cannot provide.

---

## 3. Why Two Teachers Instead of One?

Most knowledge distillation uses a single teacher. We use two because:

1. **Different perspectives**: The 32B model might be very confident about formal Bengali vocabulary, while the 14B model might prefer simpler, more common words. Both are valid approaches.

2. **Error correction**: When one teacher makes a mistake, the other might be correct. By checking their agreement, we can detect unreliable predictions.

3. **Confidence calibration**: A larger model isn't always more reliable. Sometimes the smaller teacher is actually more confident (and correct) on certain tokens. Our system dynamically weighs each teacher based on confidence.

4. **Agreement as a quality signal**: When both teachers agree on a prediction, that prediction is very likely correct (use it for KD). When they disagree, the prediction is uncertain (fall back to the gold label).

---

## 4. The Models We Use

All three models belong to the **Qwen2.5 family** by Alibaba. This is important because they share the same tokenizer and architecture, which means:
- Token IDs mean the same thing across all models
- We can directly compare probability distributions without token alignment issues
- The vocabulary is the same (151,665 tokens)

### Teacher 1: Qwen2.5-32B-Instruct
- **Role**: Primary (larger) teacher
- **Parameters**: 32 billion
- **Type**: Instruction-tuned (fine-tuned to follow instructions)
- **Loading**: 4-bit NF4 quantization (fits in ~20 GB VRAM)
- **Compute dtype**: bfloat16 (16-bit brain floating point)

### Teacher 2: Qwen2.5-14B-Instruct
- **Role**: Secondary (smaller) teacher
- **Parameters**: 14 billion
- **Type**: Instruction-tuned
- **Loading**: 4-bit NF4 quantization (fits in ~10 GB VRAM)
- **Compute dtype**: bfloat16

### Student: Qwen2.5-3B
- **Role**: The model we are training
- **Parameters**: 3 billion (base model, not instruction-tuned)
- **Training method**: LoRA (Low-Rank Adaptation) — only trains ~120M parameters
- **Loading**: Full precision bfloat16 (3B is small enough)

### Why 4-bit Quantization for Teachers?

The teachers are only used for **scoring** (extracting probabilities), not for training their weights. 4-bit quantization reduces their memory footprint by ~8× with minimal quality loss. Specifically, we use:
- **NF4 (Normal Float 4-bit)**: A quantization format optimized for normally-distributed weights
- **Double quantization**: Quantizes the quantization constants themselves for extra savings
- **bfloat16 compute**: Dequantizes to bfloat16 for actual computations (maintains precision)

---

## 5. The Dataset: BanSum

**BanSum** is a Bengali abstractive summarization dataset. Each sample contains:
- `main`: The full Bengali article (news article, blog post, etc.)
- `sum2`: A human-written Bengali summary of the article

### Raw Dataset
- **Total samples**: ~141,000 article-summary pairs
- **Source file**: `bansum_lte_1000_tokens.json` (pre-filtered to articles with ≤1000 tokens)
- **Language**: Bengali (বাংলা)

### Dataset Split
After filtering (see next section), we split the data with a fixed random seed (42) for reproducibility:
- **Training set**: 80% of total samples
- **Validation set**: 10% of total samples  
- **Test set**: 10% of total samples

For our experiments with the filtered 20k dataset:
- **Training**: 16,000 samples
- **Validation**: 2,000 samples
- **Test**: 2,000 samples

---

## 6. Dataset Quality Filtering

Not all 141k samples are equally good for training. Some have very short summaries, some are near-duplicates, some contain garbled text. We built a quality filtering pipeline (`filter_dataset.py`) that applies two stages:

### Stage 1: Hard Filters (Remove Clearly Bad Samples)

These are pass/fail checks. If a sample fails any one of these, it is removed:

| Filter | Criterion | Why |
|--------|-----------|-----|
| 1. Deduplication | Remove articles with identical first 300 characters | Prevents data leakage and memorization |
| 2. Article length | ≥ 100 words | Too-short articles don't have enough content to summarize |
| 3. Summary length | 15–200 words | Too short = not informative; too long = not a summary |
| 4. Compression ratio | 0.05–0.70 (summary chars / article chars) | Below 0.05 = tiny summary for huge article; above 0.70 = barely any compression |
| 5. Word overlap (min) | ≥ 0.15 of summary words appear in article | Below this, the summary might be unrelated to the article |
| 6. Word overlap (max) | < 0.98 | Above this, the summary is basically a copy of the article (purely extractive) |
| 7. Bangla script ratio | ≥ 40% of summary characters are Bangla script | Filters out garbled, empty, or mostly-English summaries |
| 8. Not identical | Article ≠ Summary | Some samples have the article copied as the summary |

### Stage 2: Quality Scoring (Rank Remaining Samples)

After hard filters, each surviving sample gets a composite quality score (0 to 1) based on six dimensions. Each dimension is scored separately and then combined with weights:

| Dimension | Weight | Ideal Range | What It Measures |
|-----------|--------|-------------|------------------|
| Article length | 15% | 200–500 words | Whether the article is a good length for summarization (not too short, not too long) |
| Summary length | 15% | 30–100 words | Whether the summary is a good length (meaningful but concise) |
| Compression | 20% | 0.15–0.40 | How much the summary compresses the article (real summarization) |
| Abstractiveness | 20% | 0.15–0.50 novel unigrams | How much new phrasing the summary uses (not just copying words from the article) |
| Content relevance | 20% | 0.30–0.85 word overlap | Whether the summary is actually about the article's content |
| Bangla quality | 10% | ≥ 50% Bangla chars | How "clean" the Bangla text is |

**Final score** = 0.15 × article_length + 0.15 × summary_length + 0.20 × compression + 0.20 × abstractiveness + 0.20 × relevance + 0.10 × bangla_quality

We then sort all samples by their composite score (highest first) and select the top 20,000 samples. This ensures we train on the most representative, high-quality data.

---

## 7. The Full Pipeline (Step by Step)

The entire system runs in three phases, orchestrated by `run_all_experiments.py`:

```
Phase 1: Teacher Scoring (Offline)
    ├── Load Qwen2.5-32B-Instruct (4-bit)
    │   ├── Score train split (16,000 samples) → teacher_outputs/teacher_32b/train.jsonl
    │   ├── Score validation split (2,000 samples) → teacher_outputs/teacher_32b/validation.jsonl
    │   └── Score test split (2,000 samples) → teacher_outputs/teacher_32b/test.jsonl
    ├── Unload 32B, Load Qwen2.5-14B-Instruct (4-bit)
    │   ├── Score train split → teacher_outputs/teacher_14b/train.jsonl
    │   ├── Score validation split → teacher_outputs/teacher_14b/validation.jsonl
    │   └── Score test split → teacher_outputs/teacher_14b/test.jsonl
    └── Unload 14B

Phase 2: Student Training (8 Experiments)
    ├── Load Qwen2.5-3B + LoRA adapters
    ├── Load pre-computed teacher logprobs from Phase 1
    ├── For each of 8 experiment configurations:
    │   ├── Train for 2 epochs on train split (16,000 samples)
    │   ├── Validate after each epoch on validation split (2,000 samples)
    │   ├── Save best model (lowest validation loss)
    │   └── Save final model and training log
    └── Save model registry (paths to all 8 trained models)

Phase 3: Evaluation
    ├── For each of 8 trained models:
    │   ├── Load model (merge LoRA weights with base model)
    │   ├── Generate summaries for test set (2,000 samples)
    │   ├── Compute ROUGE-1, ROUGE-2, ROUGE-L
    │   ├── Compute BLEU-1, BLEU-2, BLEU-4
    │   ├── Compute BERTScore (semantic quality)
    │   ├── Compute Semantic Similarity (embedding-based)
    │   └── Save results and predictions
    └── Generate comparison table across all 8 experiments
```

---

## 8. Phase 1: Teacher Scoring (Offline)

### What Happens

For each article-summary pair in the dataset, we ask each teacher: **"If you were writing this exact gold summary for this article, how likely would you consider each possible token at each position?"**

This is called **teacher-forced scoring** — we don't let the teacher generate its own summary. Instead, we force-feed it the gold summary and capture its probability distribution at each token position.

### Why Teacher-Forced (Not Free Generation)?

| Approach | What It Does | Problems |
|----------|-------------|----------|
| Free generation | Let teacher write its own summary, then compare | Different length, different tokens — impossible to align with student's predictions |
| Teacher-forced | Show teacher the gold summary, extract probabilities | Perfect token alignment with student, deterministic, and 10-50× faster |

### How It Works (generate_teacher_outputs.py)

For each sample, the teacher receives:

```
[Prompt tokens] [Gold summary tokens]
```

The prompt is a Bengali instruction:
```
নিচের বাংলা প্রবন্ধটি সংক্ষিপ্তভাবে সারাংশ করুন:

{article text}

সারাংশ:
```

Translation: "Briefly summarize the following Bengali article:"

The teacher processes this as a causal language model (left-to-right). After the single forward pass, we extract:

For each summary token position `j`:
- The teacher's full probability distribution over all 151,665 tokens in the vocabulary
- We save only the **top 50 tokens** and their log-probabilities (to save disk space)
- This is stored as a list of `(token_id, log_probability)` pairs

### The Causal Shift in Teacher Scoring

In a causal language model, `logits[position]` predicts `token[position + 1]`. So:

```
Position:   0     1     2    ...  P-1    P     P+1   ...
Token:     [BOS] [tok1] [tok2] ... [prompt_last] [sum_0] [sum_1] ...
Logits[i]: predicts token at position i+1

To get P(sum_0 | prompt):  → use logits[P-1]
To get P(sum_1 | prompt, sum_0): → use logits[P]
To get P(sum_j | prompt, sum_0..j-1): → use logits[P-1+j]
```

Where `P` is the position where the gold summary starts (= length of prompt tokens).

### What Gets Saved

Each scored sample is saved as one JSON line in a `.jsonl` file:

```json
{
    "summary": "রাষ্ট্রপতি গতকাল রাজধানী পরিদর্শন করেছেন",
    "token_ids": [12345, 67890, 11111, ...],
    "top_k_logprobs": [
        [[12345, -0.35], [67890, -2.10], [11111, -3.45], ...],
        [[67890, -0.50], [12345, -1.80], ...],
        ...
    ]
}
```

- `token_ids`: The tokenized gold summary token IDs
- `top_k_logprobs[j]`: For position `j` in the summary, a list of the top 50 `(token_id, log_probability)` pairs from the teacher

### Batched Processing

To speed things up, we process multiple samples in a single batch:
- Batch size: 4 (configurable)
- Left-padding: Since sequences have different lengths, shorter ones are padded on the left with pad tokens (causal LM convention)
- Attention mask: 0 for padding positions, 1 for real tokens
- OOM fallback: If a batch causes out-of-memory, automatically falls back to processing samples one by one
- Resume support: If the script crashes, it can resume from where it left off (counts existing lines in the output file)

---

## 9. Phase 2: Student Training with LoRA

### What Is LoRA?

**LoRA (Low-Rank Adaptation)** is a technique to fine-tune large models efficiently. Instead of updating all 3 billion parameters, LoRA:

1. Freezes the original model weights (they don't change)
2. Adds tiny "adapter" matrices to specific layers
3. Only trains these adapter matrices

How LoRA works mathematically:

```
Original weight matrix: W (shape: d × d, e.g., 2048 × 2048)
LoRA adds: ΔW = B × A (where B is d × r and A is r × d)

With r = 64 (our setting):
  W: 2048 × 2048 = 4,194,304 parameters (frozen)
  B: 2048 × 64 = 131,072 parameters (trainable)
  A: 64 × 2048 = 131,072 parameters (trainable)
  ΔW total: 262,144 parameters (trainable) ← 16× smaller!
```

The scaling factor `alpha` controls how much the LoRA adapters influence the output:
- `alpha = 128`, `r = 64` → scaling = alpha/r = 2.0
- This means the LoRA update is scaled by 2.0 before being added to the original weights

### Which Layers Get LoRA?

We apply LoRA to **all attention and MLP (feed-forward) layers** in the model:

| Layer | Purpose |
|-------|---------|
| `q_proj` | Query projection in self-attention (what am I looking for?) |
| `k_proj` | Key projection in self-attention (what do I contain?) |
| `v_proj` | Value projection in self-attention (what information do I carry?) |
| `o_proj` | Output projection in self-attention (combine attention results) |
| `gate_proj` | Gating in MLP (SwiGLU activation) |
| `up_proj` | Up-projection in MLP (expand dimensions) |
| `down_proj` | Down-projection in MLP (compress back) |

This gives us about **120 million trainable parameters** out of 3.2 billion total (~3.7%).

### Training Configuration

| Setting | Value | Why |
|---------|-------|-----|
| Batch size | 4 | Fits in GPU memory |
| Gradient accumulation | 8 steps | Effective batch = 4 × 8 = 32 |
| Learning rate | 2 × 10⁻⁴ | Standard for LoRA fine-tuning |
| LR schedule | Cosine with warmup | Gradually decreases LR after warmup |
| Warmup ratio | 5% of total steps | Prevents early instability |
| Weight decay | 0.01 | Light L2 regularization |
| Max gradient norm | 1.0 | Clips gradients to prevent explosion |
| Precision | bfloat16 | Memory efficient, good for training |
| Gradient checkpointing | Enabled | Trades compute for memory (recomputes activations instead of storing them) |
| Epochs | 2 | Balance between training time and overfitting |

### Optimizer

We use **AdamW** — the standard Adam optimizer with decoupled weight decay:
- β₁ = 0.9, β₂ = 0.999 (momentum parameters)
- Weight decay = 0.01 (applies to all parameters)

### Learning Rate Schedule

**Cosine schedule with warmup**:

1. **Warmup phase** (first 5% of total steps): LR linearly increases from 0 to 2e-4
2. **Cosine decay phase** (remaining 95%): LR follows a smooth cosine curve down to near 0

This prevents the model from making too-large updates in the beginning (when it hasn't seen much data) and gradually reduces the learning rate as it converges.

---

## 10. The Decoder-Only Format: How We Feed Data to the Model

### Why This Matters

Qwen2.5-3B is a **decoder-only** (causal) language model, not an encoder-decoder model like BART or T5. This fundamentally changes how we feed data to it.

In an **encoder-decoder** model:
- Encoder processes the article → hidden states
- Decoder generates the summary using those hidden states
- Input and output are separate sequences

In a **decoder-only** model:
- There is only ONE sequence: everything is concatenated together
- The model processes left-to-right, predicting each token based on all previous tokens
- The article and summary must be part of the SAME sequence

### How We Format Data (in DistillationDataset.__getitem__)

For each training sample, we create:

```
Input sequence:  [article_token_0, article_token_1, ..., article_token_N, summary_token_0, summary_token_1, ..., summary_token_M]
                 ←─────────── article (with BOS) ──────────→ ←──────── summary (no BOS) ──────→

Labels:          [-100, -100, ..., -100, summary_token_0, summary_token_1, ..., summary_token_M]
                 ←── masked (no loss) ──→ ←──────── these are what we want to learn ──────→

Attention mask:  [1, 1, ..., 1, 1, 1, ..., 1]
                 ←── all positions attend to previous positions ──→
```

Key details:
- **Article tokens**: Tokenized with special tokens (BOS). Truncated to max 850 tokens.
- **Summary tokens**: Tokenized WITHOUT special tokens (raw continuation). Truncated to max 256 tokens.
- **Labels**: `-100` for all article positions (PyTorch's `cross_entropy` ignores these positions). Real token IDs for summary positions.
- **summary_start**: Tracks where the summary begins (= length of article tokens). This is crucial for extracting the right logits later.
- **summary_len**: How many summary tokens this sample has.

### Why -100 for Article Labels?

We only want the model to learn to GENERATE summaries, not to predict article tokens. By setting article labels to -100, PyTorch's cross-entropy loss function automatically ignores those positions. The loss is only computed on summary tokens.

### Left-Padding in the Collate Function

When we batch multiple samples together, they have different lengths. We need to pad them to the same length. **We pad on the LEFT** (not the right) because:

1. Causal LMs generate from left to right
2. Right-padding would place padding tokens in between article and summary for shorter sequences, breaking the causal flow
3. Left-padding keeps the meaningful tokens contiguous at the end

```
Sample 1 (long):  [art_0, art_1, art_2, art_3, sum_0, sum_1, sum_2]
Sample 2 (short): [PAD, PAD, art_0, art_1, sum_0, sum_1, sum_2]
                   └── left-padded ──┘
```

The `summary_start` for each sample is adjusted to account for the added padding:
```
new_summary_start = original_summary_start + pad_length
```

---

## 11. The Causal Shift: Why logits[t] Predicts token[t+1]

This is the most subtle but critical detail in the entire training pipeline. Getting this wrong produces garbage models (which is exactly what happened before we fixed it).

### The Fundamental Rule of Causal Language Models

In every causal (autoregressive) language model:

> **`logits[position t]` predicts the token at `position t+1`**

This is because the model is trained to predict the NEXT token given all PREVIOUS tokens. The logits at position `t` represent the model's prediction of what comes after position `t`.

### Example

```
Position:     0      1      2      3      4      5      6
Token:       [BOS]  [art]  [art]  [sum₀] [sum₁] [sum₂] [EOS]
Logits[0]:   predicts "art" (position 1)
Logits[1]:   predicts "art" (position 2)
Logits[2]:   predicts "sum₀" (position 3)  ← summary_start - 1
Logits[3]:   predicts "sum₁" (position 4)
Logits[4]:   predicts "sum₂" (position 5)
Logits[5]:   predicts "EOS" (position 6)
```

If `summary_start = 3` (the summary begins at position 3), then:
- The logit that predicts `sum₀` is at position `3 - 1 = 2`
- The logit that predicts `sum₁` is at position `3 - 1 + 1 = 3`
- The logit that predicts `sum_j` is at position `summary_start - 1 + j`

### How We Extract Summary Logits in Training

In the training loop (`train_student.py`), after the student model processes the full sequence:

```python
# For summary token j, the predicting logit is at position (summary_start - 1 + j)
logit_start = summary_start - 1
logit_end = logit_start + summary_length
summary_logits[i, :s_len, :] = all_logits[i, logit_start:logit_end, :]
```

This produces a tensor of shape `(batch, max_summary_len, vocab_size)` where `summary_logits[i, j]` predicts `summary_token[i, j]`.

### Why This Alignment Matters

The teacher logprobs are pre-computed using the exact same causal shift: `teacher_logprobs[j]` = P(summary_token_j | prompt, summary_0..j-1). 

So after extraction:
- `student_logits[i, j]` predicts summary token j for sample i
- `teacher_logprobs[i, j]` is the teacher's probability distribution for summary token j for sample i

They are now **perfectly aligned**, and we can directly compute KL divergence between them.

---

## 12. EWAD Loss: The Novel Distillation Loss Function

**EWAD** stands for **Entropy-Weighted Agreement-Aware Distillation**. It is the core novel contribution of this work.

### The Core Idea

Traditional KD just copies the teacher's distributions to the student. EWAD does something smarter — it looks at EACH TOKEN and asks:

1. **How confident is each teacher?** (Entropy-based confidence)
2. **Do the two teachers agree?** (JSD-based agreement)
3. **Based on answers 1 and 2, what should the student learn from?**

If teachers agree → Learn from teachers (weighted by confidence)
If teachers disagree → Don't trust teachers; learn from the gold label instead

### Step-by-Step Computation

#### Step 1: Teacher Confidence

For each token position `t` and each teacher `i` (32B or 14B):

$$C_i^t = 1 - \frac{H(p_i^t)}{\log|V|}$$

Where:
- $H(p_i^t) = -\sum_v p_i^t(v) \cdot \log p_i^t(v)$ is the entropy of the teacher's distribution
- $|V| = 151{,}665$ is the vocabulary size
- $\log|V|$ is the maximum possible entropy (uniform distribution)

**What this means**: 
- If the teacher puts 99% probability on one token → entropy is low → $C$ is close to 1 (very confident)
- If the teacher spreads probability evenly across many tokens → entropy is high → $C$ is close to 0 (not confident)

**Range**: $C \in [0, 1]$

#### Step 2: Confidence-Proportional Weights

We want to give more weight to the more confident teacher:

$$w_{32B}^t = \frac{\exp(C_{32B}^t / \tau_w)}{\exp(C_{32B}^t / \tau_w) + \exp(C_{14B}^t / \tau_w)}$$

$$w_{14B}^t = 1 - w_{32B}^t$$

This is just a **softmax** over the two confidence values, with temperature $\tau_w = 1.0$.

**What this means**:
- If 32B has confidence 0.9 and 14B has confidence 0.3 → $w_{32B} \approx 0.65$, $w_{14B} \approx 0.35$ (trust 32B more)
- If both have confidence 0.7 → $w_{32B} = w_{14B} = 0.5$ (trust equally)
- The temperature $\tau_w$ controls how sharply we differentiate (higher = more equal; lower = more winner-take-all)

#### Step 3: Teacher Agreement 

We measure how much the two teachers agree using **Jensen-Shannon Divergence (JSD)**:

$$A_t = 1 - JSD(p_{32B}^t \| p_{14B}^t)$$

Where JSD is computed as:

$$M = \frac{1}{2}(p_{32B}^t + p_{14B}^t)$$

$$JSD = \frac{1}{2} KL(p_{32B}^t \| M) + \frac{1}{2} KL(p_{14B}^t \| M)$$

And $KL(P \| Q) = \sum_v P(v) \cdot \log\frac{P(v)}{Q(v)}$ is the KL divergence.

We normalize JSD by $\log 2$ so it falls in $[0, 1]$.

**What is JSD?** JSD measures how different two probability distributions are:
- JSD = 0 → distributions are identical (teachers perfectly agree)
- JSD = 1 → distributions are completely different (teachers totally disagree)

So $A_t = 1 - JSD$:
- $A_t = 1$ → perfect agreement
- $A_t = 0$ → complete disagreement

**Why JSD and not just KL?** JSD is **symmetric** ($JSD(P \| Q) = JSD(Q \| P)$) and **bounded** ($[0, 1]$). KL divergence is neither — it can be infinite and the direction matters. JSD is a better measure of "how different are these two distributions?" without favoring either direction.

#### Step 4: Agreement Gate

We convert agreement into a smooth gate value:

$$\lambda_t = \sigma(k \cdot (A_t - \delta))$$

Where:
- $\sigma$ is the **sigmoid function**: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- $k = 5.0$ controls how sharp the transition is
- $\delta = 0.5$ is the threshold (when $A_t = \delta$, $\lambda_t = 0.5$)

**What this means**:
- When agreement is high ($A_t \gg \delta$) → $\lambda_t \approx 1$ (trust teachers fully)
- When agreement is low ($A_t \ll \delta$) → $\lambda_t \approx 0$ (don't trust teachers)
- The transition is smooth around $\delta = 0.5$

#### Step 5: KL Divergence (Teacher → Student)

For each teacher, compute how different the student is from the teacher:

$$KL(p_{32B}^t \| p_S^t) = \sum_v p_{32B}^t(v) \cdot (\log p_{32B}^t(v) - \log p_S^t(v))$$

$$KL(p_{14B}^t \| p_S^t) = \sum_v p_{14B}^t(v) \cdot (\log p_{14B}^t(v) - \log p_S^t(v))$$

This measures how far the student's distribution is from each teacher's distribution. The training goal is to minimize this — make the student's predictions more like the teachers.

We use this direction (teacher || student) because we want the student to cover all the high-probability tokens that teachers consider important.

#### Step 6: Weighted KD Loss

$$L_{KD}^t = w_{32B}^t \cdot KL(p_{32B}^t \| p_S^t) + w_{14B}^t \cdot KL(p_{14B}^t \| p_S^t)$$

This is the "teacher knowledge" loss — the student tries to match both teachers, weighted by their confidence.

#### Step 7: Gold Label CE Loss

$$L_{CE}^t = -\log p_S^t(y_t^*)$$

Where $y_t^*$ is the gold (correct) token at position $t$. This is standard cross-entropy loss — the student tries to predict the correct answer.

#### Step 8: Final EWAD Combination

$$L_{EWAD} = \frac{1}{T} \sum_{t=1}^{T} \left[ \lambda_t \cdot L_{KD}^t + (1 - \lambda_t) \cdot L_{CE}^t \right]$$

Where $T$ is the total number of summary tokens.

**The beautiful insight**: 
- When teachers agree ($\lambda_t \approx 1$): Loss ≈ $L_{KD}^t$ → learn from teachers
- When teachers disagree ($\lambda_t \approx 0$): Loss ≈ $L_{CE}^t$ → learn from gold label only
- In between: A smooth blend of both signals

This means the student gets the best of both worlds: rich soft knowledge from teachers when they're reliable, and falls back to the safe gold label when they're uncertain.

---

## 13. CPDP Loss: The Regularizer

**CPDP** stands for **Capacity-Proportional Divergence Preservation**. While EWAD tells the student WHAT to learn, CPDP tells it to maintain the right DISTANCE from each teacher.

### The Intuition

The 32B teacher is larger and has more "capacity" than the 14B teacher. The student (3B) should naturally be:
- Further from the 32B (harder to match a much larger model)
- Closer to the 14B (easier to match a model closer in size)

The gap between the student-to-32B distance and student-to-14B distance should be proportional to the gap between the teachers themselves (the teacher mutual divergence). CPDP enforces this geometric relationship.

### The Formula

$$L_{CPDP} = \left| \frac{KL(p_{32B} \| p_S)}{H(p_S)} - \frac{KL(p_{14B} \| p_S)}{H(p_S)} - \Delta^* \right|^2$$

Where:
- $KL(p_{32B} \| p_S)$ = distance from 32B teacher to student
- $KL(p_{14B} \| p_S)$ = distance from 14B teacher to student
- $H(p_S)$ = student's entropy (used for normalization, **detached** so CPDP doesn't push entropy up)
- $\Delta^* = KL(p_{32B} \| p_{14B})$ = the natural distance between the two teachers (NOT normalized by student entropy — this is a fixed property of the teachers)

### Why Normalize by Student Entropy?

Dividing by $H(p_S)$ makes the loss **capacity-aware**. When the student is very uncertain (high entropy), the raw KL values might be large, but they're less meaningful. Normalizing scales them relative to the student's current state.

**Important**: $H(p_S)$ is **detached** (treated as a constant during backpropagation). Without detaching, CPDP could cheat by making the student's entropy arbitrarily large to reduce the normalized KL values. Detaching prevents this pathological behavior.

### Why the Squared Difference?

The squared difference $|\cdot|^2$ means:
- If the student maintains the right proportional distances → loss = 0
- If the student drifts (e.g., becomes equally close to both teachers, or closer to 32B than 14B) → loss > 0
- The penalty grows quadratically with the error

### CPDP Weight

CPDP is a regularizer, not the main loss. It is weighted by $\mu = 0.05$:

$$L_{total} = L_{EWAD} + 0.05 \times L_{CPDP}$$

This small weight ensures CPDP gently guides the student's knowledge geometry without overwhelming the main distillation signal.

### Stability

The per-token CPDP loss is clamped at 100.0 to prevent numerical explosion during early training when the student's predictions are essentially random.

---

## 14. Combined Loss: Putting It All Together

### For the Full System (ewad_cpdp experiment):

$$L_{total} = L_{EWAD} + \mu \cdot L_{CPDP}$$

Expanding:

$$L_{total} = \underbrace{\frac{1}{T}\sum_t \left[\lambda_t \cdot (w_{32B}^t \cdot KL_{32B\to S} + w_{14B}^t \cdot KL_{14B\to S}) + (1-\lambda_t) \cdot CE_t\right]}_{EWAD} + \underbrace{0.05 \cdot \left|\frac{KL_{32B\to S}}{H_S} - \frac{KL_{14B\to S}}{H_S} - KL_{32B\to 14B}\right|^2}_{CPDP}$$

### For Fixed-Weight KD (no EWAD):

$$L_{total} = 0.5 \cdot (w_{32B} \cdot KL_{32B\to S} + w_{14B} \cdot KL_{14B\to S}) + 0.5 \cdot CE$$

Where $w_{32B}$ and $w_{14B}$ are fixed constants (e.g., 0.7 and 0.3).

### For Baseline (no distillation):

$$L_{total} = CE$$

Just standard cross-entropy against gold labels. The simplest baseline.

---

## 15. The 8 Ablation Experiments

We run 8 experiments to understand the contribution of each component. This is called **ablation study** — we systematically remove components to see how much each one matters.

### Experiment Table

| # | Name | What It Does | Purpose |
|---|------|-------------|---------|
| 1 | `baseline_no_distill` | Student trained on gold labels only (standard fine-tuning) | **Lower bound** — what you get without any distillation |
| 2 | `single_teacher_32b` | Only 32B teacher → student (standard single-teacher KD) | Tests whether the bigger teacher alone is enough |
| 3 | `single_teacher_14b` | Only 14B teacher → student (standard single-teacher KD) | Tests whether the smaller teacher alone is enough |
| 4 | `fixed_weights` | Both teachers with fixed weights (α=0.7 for 32B, β=0.3 for 14B) | **Naive dual-teacher** baseline — no dynamic weighting |
| 5 | `confidence_only` | EWAD with confidence weighting but NO agreement gate | Tests the contribution of confidence weighting alone |
| 6 | `agreement_only` | EWAD with agreement gate but NO confidence weighting (equal teacher weights) | Tests the contribution of agreement gating alone |
| 7 | `ewad_full` | Full EWAD (confidence + agreement) without CPDP | Tests EWAD without the geometric regularizer |
| 8 | `ewad_cpdp` | Full EWAD + CPDP (the complete system) | **The full proposed method** — should be the best |

### What We Expect

- Experiments 2-3 should beat experiment 1 (single-teacher KD should beat no KD)
- Experiment 4 should beat 2-3 (two teachers should beat one, even with fixed weights)
- Experiments 5-6 should beat 4 (dynamic components should beat fixed weights)
- Experiment 7 should beat 5-6 (combining both EWAD components should be better)
- Experiment 8 should be the best (full system with CPDP regularization)

If any of these expectations are violated, it reveals something interesting about the task or the data.

### How Each Experiment Handles the Loss

| Experiment | Uses Teachers? | KD Weights | Agreement Gate | CPDP |
|-----------|---------------|------------|---------------|------|
| baseline_no_distill | No | N/A | N/A | No |
| single_teacher_32b | 32B only | Fixed: 32B=1.0 | No | No |
| single_teacher_14b | 14B only | Fixed: 14B=1.0 | No | No |
| fixed_weights | Both | Fixed: 32B=0.7, 14B=0.3 | No | No |
| confidence_only | Both | Dynamic (softmax of confidence) | No (always 1.0) | No |
| agreement_only | Both | Fixed: 50/50 | Yes (JSD-based) | No |
| ewad_full | Both | Dynamic (softmax of confidence) | Yes (JSD-based) | No |
| ewad_cpdp | Both | Dynamic (softmax of confidence) | Yes (JSD-based) | Yes (μ=0.05) |

---

## 16. Phase 3: Evaluation

After training all 8 models, we evaluate each one on the held-out test set (2,000 samples).

### How Evaluation Works

1. **Load the trained model**: Load the LoRA adapter and merge it with the base Qwen2.5-3B model
2. **Generate summaries**: For each test article, generate a summary using beam search
3. **Compare with gold summaries**: Compute automatic metrics

### Generation Settings

| Setting | Value | Purpose |
|---------|-------|---------|
| Beam search | 4 beams | Explores 4 candidate sequences simultaneously |
| Max new tokens | 256 | Maximum summary length |
| Length penalty | 1.2 | Slightly favors longer (more informative) summaries |
| Early stopping | Yes | Stops when all beams produce an end token |
| Sampling | Disabled | Deterministic output for reproducibility |

### The 6 Evaluation Metrics

#### 1. ROUGE-1 (Recall-Oriented Understudy for Gisting Evaluation)
- **What it measures**: Unigram overlap between prediction and reference
- **How it works**: Counts how many individual words in the prediction match words in the reference
- **Range**: 0 to 1 (higher = better)
- **Tokenization**: Space-based (appropriate for Bangla, which is space-delimited)
- **Example**: Prediction "আমি বাড়ি যাই" vs Reference "আমি বাড়ি যাচ্ছি" → 2/3 overlap

#### 2. ROUGE-2
- **What it measures**: Bigram (2-word pairs) overlap
- **How it works**: Counts how many consecutive word pairs match
- **Why it matters**: Captures phrase-level similarity, not just individual words
- **Range**: 0 to 1 (higher = better)

#### 3. ROUGE-L
- **What it measures**: Longest common subsequence (LCS) ratio
- **How it works**: Finds the longest sequence of words that appear in the same order in both texts (not necessarily consecutive)
- **Why it matters**: Captures sentence-level structural similarity
- **Range**: 0 to 1 (higher = better)

#### 4. BLEU-1, BLEU-2, BLEU-4 (Bilingual Evaluation Understudy)
- **What it measures**: Precision-oriented n-gram overlap
- **BLEU-1**: 1-gram precision (what fraction of prediction words appear in reference?)
- **BLEU-2**: Average of 1-gram and 2-gram precision
- **BLEU-4**: Average of 1, 2, 3, and 4-gram precision
- **Smoothing**: Method 1 from NLTK (handles zero counts for higher n-grams)
- **Range**: 0 to 1 (higher = better)
- **Difference from ROUGE**: ROUGE measures recall (how much of the reference is captured), BLEU measures precision (how accurate is the prediction)

#### 5. BERTScore
- **What it measures**: Semantic similarity using BERT embeddings
- **How it works**: 
  1. Encode each token in prediction and reference using a pre-trained BERT model
  2. Compute pairwise cosine similarities between all token pairs
  3. Use greedy matching to find the best alignment
  4. Report precision, recall, and F1
- **Language**: Bengali (bn) — uses `bert-base-multilingual-cased`
- **Why it matters**: Captures meaning, not just surface-level word matching. "রাষ্ট্রপতি" and "প্রেসিডেন্ট" are different words but similar meaning.
- **Range**: 0 to 1 (higher = better)

#### 6. Semantic Similarity
- **What it measures**: Overall semantic similarity between prediction and reference at the sentence level
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2` (supports Bengali)
- **How it works**:
  1. Encode the entire prediction and reference into single 384-dimensional vectors
  2. Compute cosine similarity between the two vectors
- **Range**: -1 to 1 (higher = better; typically 0.3-0.9 for summarization)
- **Why it matters**: Captures whether the prediction and reference convey the same overall meaning, even if they use completely different words

### Comparison Table

After evaluating all 8 models, a comparison table is generated:

```
Experiment                ROUGE-1  ROUGE-2  ROUGE-L   BLEU-4  BERTScr   SemSim
─────────────────────────────────────────────────────────────────────────────────
baseline_no_distill        0.XXXX   0.XXXX   0.XXXX   0.XXXX   0.XXXX   0.XXXX
single_teacher_32b         0.XXXX   0.XXXX   0.XXXX   0.XXXX   0.XXXX   0.XXXX
single_teacher_14b         0.XXXX   0.XXXX   0.XXXX   0.XXXX   0.XXXX   0.XXXX
...
ewad_cpdp                  0.XXXX   0.XXXX   0.XXXX   0.XXXX   0.XXXX   0.XXXX
```

---

## 17. All Code Files Explained

### Main Pipeline Files

#### `config.py` — Central Configuration
Everything is configured in one place. Contains:
- Model names and paths (the 3 Qwen models)
- Dataset paths and split ratios (80/10/10)
- Teacher generation settings (tokenization, batch size, prompt template)
- Student training hyperparameters (LoRA, optimizer, scheduler)
- EWAD parameters (τ_w, k, δ, ε)
- CPDP parameters (μ, ε)
- Evaluation settings (beam search, max length)
- All 8 experiment configuration dictionaries
- Test mode overrides (1000 samples, 1 epoch, separate output dirs)

#### `generate_teacher_outputs.py` — Offline Teacher Scoring
- Loads teacher model in 4-bit quantization
- For each sample: concatenates prompt + gold summary, runs single forward pass
- Extracts top-50 log-probabilities at each summary token position
- Saves results as JSONL files (one line per sample)
- Supports resuming from partial runs
- Has OOM fallback (batch → single sample processing)
- Processes all splits: train, validation, test

#### `ewad_cpdp_loss.py` — Novel Loss Functions
Contains three main classes:
- `EWADLoss`: Entropy-Weighted Agreement-Aware Distillation (the main novel loss)
- `CPDPLoss`: Capacity-Proportional Divergence Preservation (the regularizer)
- `DualTeacherDistillationLoss`: Wrapper that handles all 8 experiment types

Also contains utility functions:
- `align_vocab_size()`: Truncates and renormalizes if student/teacher have different vocab sizes
- `sparse_topk_to_dense()`: Converts top-k (token_id, logprob) pairs to dense probability distribution
- `batch_sparse_to_dense()`: Batch version of the above

#### `train_student.py` — Student Training
The main training script. Contains:
- `DistillationDataset`: PyTorch Dataset that loads samples and teacher outputs, creates concatenated [article + summary] sequences
- `collate_fn`: Custom batching with left-padding, summary position tracking, and teacher logprob padding
- `train_student()`: Main training loop with:
  - Causal shift extraction (summary_start - 1 + j)
  - Gradient accumulation (effective batch = 32)
  - Periodic checkpointing
  - Validation after each epoch (CE loss only for fair comparison)
  - Best model saving (lowest validation loss)
  - Comprehensive logging

#### `evaluate.py` — Evaluation
- Loads trained model (detects LoRA and merges)
- Generates summaries with beam search
- Computes ROUGE-1, ROUGE-2, ROUGE-L (space-tokenized for Bangla)
- Computes BLEU-1, BLEU-2, BLEU-4 (with smoothing)
- Computes BERTScore (Bengali language model)
- Computes Semantic Similarity (multilingual sentence embeddings)
- Teacher disagreement analysis (optional)
- Saves results, predictions, and sample outputs

#### `run_all_experiments.py` — Pipeline Orchestrator
Runs the full pipeline:
1. Phase 1: Teacher generation (checks for existing outputs, skips if done)
2. Phase 2: Training all 8 experiments (sequential, saves model registry)
3. Phase 3: Evaluation and comparison table

CLI flags:
- `--skip-teachers`: Skip Phase 1 (assumes teacher outputs exist)
- `--skip-training`: Skip Phase 2 (only evaluate existing models)
- `--experiments name1 name2`: Run only specific experiments

### Support Files

#### `filter_dataset.py` — Dataset Quality Filtering
- Applies 8 hard filters to remove bad samples
- Computes 6-dimension quality score for remaining samples
- Sorts by score and selects top N samples
- Saves filtered dataset and filtering report

#### `run_test.py` — Quick Pipeline Test
- Sets test mode (1000 samples, 1 epoch)
- Runs 3 representative experiments for quick validation
- Useful for checking that the pipeline works before a long run

#### `test_losses.py` — Loss Function Unit Tests
- Creates synthetic data
- Tests EWADLoss, CPDPLoss, and DualTeacherDistillationLoss
- Verifies correct behavior under edge cases

---

## 18. Hyperparameters Reference

### Teacher Scoring

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TEACHER_MAX_INPUT_TOKENS` | 1024 | Maximum article tokens fed to teacher |
| `TEACHER_MAX_OUTPUT_TOKENS` | 256 | Maximum summary tokens scored |
| `TEACHER_BATCH_SIZE` | 4 | Samples per teacher scoring batch |
| `LOGIT_TOP_K` | 50 | Save top-50 tokens per position (saves disk space) |
| `TEACHER_QUANTIZATION` | 4bit (NF4) | Teacher model quantization level |
| `TEACHER_LOAD_DTYPE` | bfloat16 | Compute dtype for quantized teachers |

### Student Training

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LORA_R` | 64 | LoRA rank (higher = more capacity, more parameters) |
| `LORA_ALPHA` | 128 | LoRA scaling factor (alpha/r = 2.0) |
| `LORA_DROPOUT` | 0.05 | Dropout probability in LoRA layers |
| `LORA_TARGET_MODULES` | q, k, v, o, gate, up, down | All attention + MLP layers |
| `STUDENT_MAX_INPUT_TOKENS` | 850 | Maximum article tokens for student |
| `STUDENT_MAX_OUTPUT_TOKENS` | 256 | Maximum summary tokens for student |
| `STUDENT_BATCH_SIZE` | 4 | Samples per training batch |
| `STUDENT_GRADIENT_ACCUMULATION` | 8 | Accumulate gradients over 8 steps |
| `STUDENT_LEARNING_RATE` | 2e-4 | Peak learning rate |
| `STUDENT_NUM_EPOCHS` | 2 | Number of training epochs |
| `STUDENT_WARMUP_RATIO` | 0.05 | 5% of steps for LR warmup |
| `STUDENT_WEIGHT_DECAY` | 0.01 | L2 regularization weight |
| `STUDENT_MAX_GRAD_NORM` | 1.0 | Gradient clipping threshold |
| `STUDENT_BF16` | True | Use bfloat16 mixed precision |
| `STUDENT_GRADIENT_CHECKPOINTING` | True | Recompute activations to save memory |
| `STUDENT_SAVE_STEPS` | 500 | Save checkpoint every 500 optimizer steps |
| `STUDENT_LOGGING_STEPS` | 50 | Log metrics every 50 optimizer steps |

### EWAD Loss

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EWAD_TAU_W` | 1.0 | Temperature for confidence-weighted softmax |
| `EWAD_K` | 5.0 | Sigmoid sharpness for agreement gate |
| `EWAD_DELTA` | 0.5 | Agreement threshold |
| `EWAD_ENTROPY_EPS` | 1e-8 | Epsilon for entropy clamping |

### CPDP Loss

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CPDP_MU` | 0.05 | Weight of CPDP in total loss |
| `CPDP_EPS` | 1e-8 | Epsilon for numerical stability |

### Evaluation

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EVAL_BATCH_SIZE` | 8 | Samples per evaluation batch |
| `EVAL_NUM_BEAMS` | 4 | Beam search width |
| `EVAL_MAX_LENGTH` | 256 | Maximum generated summary length |
| `EVAL_LENGTH_PENALTY` | 1.2 | Penalty for shorter summaries |

---

## 19. Hardware Requirements

### Our Setup
- **GPU**: NVIDIA RTX 5090 (32 GB VRAM)
- **RAM**: 64 GB system memory
- **Storage**: SSD recommended for fast JSONL I/O

### Memory Usage During Each Phase

| Phase | What's Loaded | Approx. VRAM |
|-------|--------------|-------------|
| Teacher 32B scoring | Qwen2.5-32B-Instruct (4-bit) | ~20 GB |
| Teacher 14B scoring | Qwen2.5-14B-Instruct (4-bit) | ~10 GB |
| Student training | Qwen2.5-3B (bfloat16) + LoRA + optimizer states + teacher logprobs batch | ~18-24 GB |
| Evaluation | Qwen2.5-3B (bfloat16, merged) + beam search buffers | ~8-12 GB |

### Time Estimates (RTX 5090)

| Phase | Duration |
|-------|----------|
| Teacher 32B scoring (20k samples, 3 splits) | ~3-4 hours |
| Teacher 14B scoring (20k samples, 3 splits) | ~2-3 hours |
| Training 1 experiment (2 epochs, 16k samples) | ~2-3 hours |
| Training all 8 experiments | ~16-24 hours |
| Evaluation 1 model (2k test samples) | ~30-45 minutes |
| Evaluation all 8 models | ~4-6 hours |
| **Total pipeline** | **~24-36 hours** |

---

## 20. How to Run Everything

### Prerequisites

```bash
pip install torch transformers peft bitsandbytes accelerate
pip install rouge-score nltk bert-score sentence-transformers
pip install tqdm numpy
```

### Option 1: Full Pipeline (Everything from Scratch)

```bash
cd LMI
python run_all_experiments.py
```

This runs all 3 phases: teacher scoring → student training → evaluation.

### Option 2: Skip Teacher Scoring (If Already Done)

```bash
cd LMI
python run_all_experiments.py --skip-teachers
```

### Option 3: Only Evaluate Existing Models

```bash
cd LMI
python run_all_experiments.py --skip-training
```

### Option 4: Run Specific Experiments Only

```bash
cd LMI
python run_all_experiments.py --skip-teachers --experiments ewad_full ewad_cpdp
```

### Option 5: Quick Test (Verify Pipeline Works)

```bash
cd LMI
python run_test.py
```

Runs 3 experiments with 1000 samples, 1 epoch — takes about 15-20 minutes instead of 24+ hours.

### Option 6: Run Individual Components

```bash
# Score with 32B teacher
python generate_teacher_outputs.py --teacher 32b

# Score with 14B teacher
python generate_teacher_outputs.py --teacher 14b

# Train a specific experiment
python train_student.py --experiment ewad_cpdp

# Evaluate a specific model
python evaluate.py --model-dir "student_outputs/ewad_cpdp_TIMESTAMP/best_model" --analysis
```

### Dataset Filtering (Only Needed Once)

```bash
# Filter the raw 141k dataset to top 35k or 20k samples
python filter_dataset.py
```

---

## 21. What Makes This Novel?

### Novel Contributions

1. **EWAD Loss Function**: A new distillation loss that combines entropy-weighted confidence scoring with JSD-based agreement gating. Unlike existing multi-teacher methods that use fixed weights, EWAD dynamically adjusts weights at the TOKEN level based on each teacher's confidence AND their mutual agreement.

2. **CPDP Regularizer**: A new regularization loss that preserves the proportional divergence structure between teachers and student. It ensures the student maintains a geometrically consistent position relative to both teachers in probability space.

3. **Dual-Teacher Framework for Bengali NLP**: First application of dual-teacher knowledge distillation to Bengali abstractive summarization.

4. **Agreement-Gated Fallback**: The idea of using teacher agreement as a reliability signal — trusting teachers only when they agree, and falling back to gold labels when they disagree. This is different from simple averaging or voting.

### Comparison with Existing Approaches

| Method | Teacher Weights | Agreement Signal | Capacity Awareness |
|--------|---------------|-----------------|-------------------|
| Standard KD (Hinton et al., 2015) | Single teacher | N/A | No |
| Multi-teacher KD (You et al., 2017) | Fixed or averaged | No | No |
| TAKD (Mirzadeh et al., 2020) | Sequential (chain) | No | Partially |
| **EWAD+CPDP (Ours)** | Dynamic per-token | Yes (JSD-based) | Yes (CPDP regularizer) |

### Key Differences from Standard Multi-Teacher KD

1. **Token-level weighting**: Standard approaches use the same weights for all tokens. We compute different weights for each token based on teacher confidence.

2. **Agreement gating**: Standard approaches always blend teacher knowledge. We only use teacher knowledge when teachers agree; otherwise we fall back to gold labels.

3. **Two separate mechanisms**: Confidence weighting (who to trust MORE) and agreement gating (whether to trust AT ALL) are separate mechanisms that address different aspects of reliability.

4. **Divergence preservation**: CPDP adds a structural constraint that no existing dual-teacher method provides.

---

## 22. Glossary of Terms

| Term | Simple Explanation |
|------|-------------------|
| **Knowledge Distillation (KD)** | Training a small model to mimic a large model's behavior |
| **Teacher** | The large model whose knowledge we want to transfer |
| **Student** | The small model that learns from the teacher |
| **Soft labels** | The teacher's full probability distribution over all tokens (rich information) |
| **Hard labels** | The single correct answer (gold label) — just one token ID |
| **LoRA** | A technique to fine-tune large models by only training small adapter matrices |
| **Causal LM** | A language model that predicts left-to-right (each token based on all previous tokens) |
| **Decoder-only** | A model architecture with only a decoder (no separate encoder), like GPT or Qwen |
| **Encoder-decoder** | A model with separate encoder (reads input) and decoder (generates output), like BART or T5 |
| **Teacher-forced scoring** | Feeding the gold answer to the teacher and extracting its probabilities (no generation) |
| **Cross-entropy (CE)** | A loss function that measures how well predicted probabilities match the true label |
| **KL divergence** | A measure of how different two probability distributions are |
| **JSD (Jensen-Shannon Divergence)** | A symmetric, bounded version of KL divergence — measures distance between two distributions |
| **Entropy** | A measure of uncertainty in a probability distribution (high entropy = uncertain, spread out) |
| **Sigmoid** | A function that maps any number to the range [0, 1] — used for smooth gating |
| **Softmax** | A function that converts a vector of numbers into probabilities that sum to 1 |
| **BOS (Beginning of Sequence)** | A special token that marks the start of a text sequence |
| **EOS (End of Sequence)** | A special token that marks the end of a text sequence |
| **Quantization** | Storing model weights in fewer bits (e.g., 4-bit instead of 16-bit) to save memory |
| **NF4 (Normal Float 4-bit)** | A specific 4-bit format optimized for normally-distributed neural network weights |
| **bfloat16** | A 16-bit floating point format with the same range as 32-bit but less precision — good for training |
| **Gradient accumulation** | Summing gradients over multiple small batches before updating weights — simulates a larger batch |
| **Gradient checkpointing** | Saving memory by recomputing activations during backward pass instead of storing them |
| **Cosine schedule** | A learning rate schedule that follows a cosine curve from max to near-zero |
| **Warmup** | Gradually increasing the learning rate at the start of training to prevent instability |
| **Weight decay** | L2 regularization — adds a small penalty for large weights to prevent overfitting |
| **Beam search** | A generation strategy that explores multiple candidate sequences simultaneously |
| **ROUGE** | An evaluation metric that measures overlap between generated and reference text (recall-focused) |
| **BLEU** | An evaluation metric that measures n-gram precision of generated text |
| **BERTScore** | An evaluation metric that uses BERT embeddings for semantic similarity |
| **Ablation study** | Systematically removing components to measure each one's contribution |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch size** | Number of samples processed together in one forward pass |
| **Effective batch size** | batch_size × gradient_accumulation — the "real" batch size for optimization |
| **Logprobs** | Log-probabilities — logarithm of probability values (for numerical stability) |
| **Top-k** | Saving only the k highest-probability tokens (instead of all vocabulary tokens) |
| **Left-padding** | Adding padding tokens on the left side of a sequence (convention for causal LMs) |

---

## Project Structure

```
not more than limit/
├── bansum_lte_1000_tokens.json          # Raw BanSum dataset (141k samples)
├── bansum_filtered_35k.json             # Quality-filtered dataset (35k samples)
├── bansum_filtered_20k.json             # Top 20k quality samples (used for training)
│
├── LMI/                                 # Main project code
│   ├── config.py                        # Central configuration
│   ├── generate_teacher_outputs.py      # Phase 1: Offline teacher scoring
│   ├── ewad_cpdp_loss.py               # Novel loss functions (EWAD + CPDP)
│   ├── train_student.py                 # Phase 2: Student training with LoRA
│   ├── evaluate.py                      # Phase 3: Evaluation (6 metrics)
│   ├── run_all_experiments.py           # Pipeline orchestrator
│   ├── filter_dataset.py               # Dataset quality filtering
│   ├── run_test.py                      # Quick pipeline sanity check
│   ├── test_losses.py                   # Loss function unit tests
│   ├── README.md                        # Short project overview
│   ├── README_DETAILED.md              # This file (comprehensive docs)
│   │
│   ├── teacher_outputs/                 # Pre-computed teacher logprobs
│   │   ├── teacher_32b/
│   │   │   ├── train.jsonl              # 16k scored samples
│   │   │   ├── validation.jsonl         # 2k scored samples
│   │   │   └── test.jsonl              # 2k scored samples
│   │   └── teacher_14b/
│   │       ├── train.jsonl
│   │       ├── validation.jsonl
│   │       └── test.jsonl
│   │
│   ├── student_outputs/                 # Trained models (Phase 2 output)
│   │   ├── baseline_no_distill_TIMESTAMP/
│   │   │   ├── best_model/              # Best checkpoint (lowest val loss)
│   │   │   ├── final_model/             # Last epoch checkpoint
│   │   │   ├── checkpoints/             # Periodic checkpoints
│   │   │   ├── experiment_config.json   # All hyperparameters
│   │   │   └── training_log.json        # Loss curves
│   │   ├── single_teacher_32b_TIMESTAMP/
│   │   ├── ...
│   │   └── ewad_cpdp_TIMESTAMP/
│   │
│   └── eval_results/                    # Evaluation results (Phase 3 output)
│       └── comparison_results.json      # All 8 experiments compared
```

---

*This project was built for Bengali abstractive summarization research using the Qwen2.5 model family and novel dual-teacher knowledge distillation techniques.*
