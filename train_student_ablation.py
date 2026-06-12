"""
Enhanced Knowledge Distillation with Multi-Teacher Ensemble for Bangla Summarization
=====================================================================================

Novel contributions:
1. Heterogeneous Multi-Teacher KD: Logit-level from same-vocab teacher (BanglaT5) +
   sequence-level pseudo-labels from cross-architecture teachers (mT5-base, mT5-XLSum)
2. Confidence-Adaptive Temperature: Per-sample tau based on teacher uncertainty
3. Multi-Granularity Loss: Intermediate encoder hidden state alignment via learned projection
4. Comprehensive Ablation: Systematic evaluation of each component's contribution

Ablation Configurations:
  A1_baseline       : Fine-tune student alone (no distillation)
  A2_single_kd      : + Logit-level KD from BanglaT5 teacher (fixed temperature)
  A3_multi_teacher   : + Pseudo-labels from mT5-base and mT5-XLSum teachers
  A4_adaptive_temp   : + Confidence-adaptive temperature scaling
  A5_full_pipeline   : + Intermediate encoder hidden state matching

Usage:
  python train_student_ablation.py --config A1_baseline
  python train_student_ablation.py --config A5_full_pipeline --quick
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from rouge_score import rouge_scorer

# Fix PyTorch memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================================
# Ablation Configurations
# ============================================================================

ABLATION_CONFIGS = {
    "A1_baseline": {
        "description": "Baseline: Fine-tune student alone (no distillation)",
        "use_logit_kd": False,
        "use_pseudo_labels": False,
        "use_adaptive_temp": False,
        "use_intermediate_matching": False,
        "alpha_kd": 0.0,
        "alpha_inter": 0.0,
        "pseudo_prob": 0.0,
        "temperature": 1.0,
    },
    "A2_single_kd": {
        "description": "Single-teacher logit-level KD (BanglaT5, fixed temperature)",
        "use_logit_kd": True,
        "use_pseudo_labels": False,
        "use_adaptive_temp": False,
        "use_intermediate_matching": False,
        "alpha_kd": 0.01,
        "alpha_inter": 0.0,
        "pseudo_prob": 0.0,
        "temperature": 0.8,
    },
    "A3_multi_teacher": {
        "description": "Multi-teacher: Logit KD + pseudo-label augmentation from mT5",
        "use_logit_kd": True,
        "use_pseudo_labels": True,
        "use_adaptive_temp": False,
        "use_intermediate_matching": False,
        "alpha_kd": 0.01,
        "alpha_inter": 0.0,
        "pseudo_prob": 0.3,
        "temperature": 0.8,
    },
    "A4_adaptive_temp": {
        "description": "Multi-teacher + confidence-adaptive temperature",
        "use_logit_kd": True,
        "use_pseudo_labels": True,
        "use_adaptive_temp": True,
        "use_intermediate_matching": False,
        "alpha_kd": 0.01,
        "alpha_inter": 0.0,
        "pseudo_prob": 0.3,
        "tau_min": 0.5,
        "tau_max": 2.0,
    },
    "A5_full_pipeline": {
        "description": "Full: Multi-teacher + adaptive temp + intermediate matching",
        "use_logit_kd": True,
        "use_pseudo_labels": True,
        "use_adaptive_temp": True,
        "use_intermediate_matching": True,
        "alpha_kd": 0.01,
        "alpha_inter": 0.1,
        "pseudo_prob": 0.3,
        "tau_min": 0.5,
        "tau_max": 2.0,
    },
}

# ============================================================================
# Paths & Hyperparameters
# ============================================================================

TEACHER_MODEL_PATH = "./bangla_t5_teacher_finetuned_20251216_143715/final_model"
STUDENT_MODEL_NAME = "csebuetnlp/banglat5_small"
PSEUDO_LABEL_DIR = "data/pseudo_labels"

STUDENT_DIM = 512  # banglat5_small d_model
TEACHER_DIM = 768  # BanglaT5 d_model

TRAIN_FILE = "data/train.csv"
VAL_FILE = "data/val.csv"
TEST_FILE = "data/test.csv"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
NUM_EPOCHS = 5
EVAL_STEPS = 1500
SAVE_STEPS = 1500
SEED = 42


# ============================================================================
# Bangla ROUGE Tokenizer
# ============================================================================

class SpaceTokenizer:
    """Space-based tokenizer for Bangla ROUGE evaluation."""
    def tokenize(self, text):
        return text.split()


# ============================================================================
# Pseudo-Label Data Collator
# ============================================================================

class PseudoLabelDataCollator(DataCollatorForSeq2Seq):
    """
    Data collator that optionally swaps gold labels with mT5 pseudo-labels.

    When pseudo_prob > 0, each sample has that probability of using a
    randomly-selected pseudo-label instead of the gold summary. This
    implements sequence-level knowledge transfer from cross-architecture
    mT5 teachers (whose vocabularies differ from the student).
    """

    def __init__(self, pseudo_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pseudo_prob = pseudo_prob

    def __call__(self, features, return_tensors=None):
        for feature in features:
            # Collect & remove pseudo-label columns
            pseudo_options = []
            for key in ("mt5_base_labels", "mt5_xlsum_labels"):
                val = feature.pop(key, None)
                if val is not None and len(val) > 0:
                    pseudo_options.append(val)

            # Randomly substitute gold label with a pseudo-label
            if (
                self.pseudo_prob > 0
                and pseudo_options
                and random.random() < self.pseudo_prob
            ):
                feature["labels"] = random.choice(pseudo_options)

        return super().__call__(features, return_tensors=return_tensors)


# ============================================================================
# Enhanced Distillation Trainer
# ============================================================================

class EnhancedDistillationTrainer(Seq2SeqTrainer):
    """
    Trainer with multi-granularity knowledge distillation.

    Loss = alpha_hard * L_hard  +  alpha_kd * L_kd  +  alpha_inter * L_inter

    - L_hard:  standard CE with gold/pseudo labels (always on)
    - L_kd:    KL-div with BanglaT5 teacher logits (fixed or adaptive temp)
    - L_inter: normalised MSE between projected student & teacher encoder states

    Pseudo-label augmentation is handled by PseudoLabelDataCollator (data level),
    not here.
    """

    def __init__(self, teacher_model=None, distill_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cfg = distill_config or {}
        self.use_logit_kd = cfg.get("use_logit_kd", False)
        self.use_adaptive_temp = cfg.get("use_adaptive_temp", False)
        self.use_intermediate_matching = cfg.get("use_intermediate_matching", False)
        self.alpha_kd = cfg.get("alpha_kd", 0.01)
        self.alpha_inter = cfg.get("alpha_inter", 0.1)
        self.temperature = cfg.get("temperature", 0.8)
        self.tau_min = cfg.get("tau_min", 0.5)
        self.tau_max = cfg.get("tau_max", 2.0)

        # Teacher setup
        self.teacher = teacher_model
        if self.teacher is not None:
            self.teacher.eval()
            self.teacher.to(self.args.device)
            for p in self.teacher.parameters():
                p.requires_grad = False

        # Loss accumulation for logging
        self._loss_accum = {}
        self._loss_count = {}

    # ------------------------------------------------------------------ loss
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Student forward
        outputs = model(**inputs)

        # 1. Hard label loss (always present)
        hard_loss = outputs.loss
        loss_dict = {"hard_loss": hard_loss.item()}

        # Dynamic alpha_hard: give weight back if a component is disabled / no teacher
        alpha_hard = 1.0
        have_teacher = self.teacher is not None
        if self.use_logit_kd and have_teacher:
            alpha_hard -= self.alpha_kd
        if self.use_intermediate_matching and have_teacher:
            alpha_hard -= self.alpha_inter

        total_loss = alpha_hard * hard_loss

        # Teacher forward (shared by KD + intermediate)
        if (self.use_logit_kd or self.use_intermediate_matching) and have_teacher:
            with torch.no_grad():
                teacher_outputs = self.teacher(**inputs)

            # 2. Logit-level KD
            if self.use_logit_kd:
                s_logits = outputs.logits
                t_logits = teacher_outputs.logits
                if self.use_adaptive_temp:
                    kd_loss = self._adaptive_kd_loss(s_logits, t_logits, inputs.get("labels"))
                else:
                    kd_loss = self._fixed_kd_loss(s_logits, t_logits)
                total_loss = total_loss + self.alpha_kd * kd_loss
                loss_dict["kd_loss"] = kd_loss.item()

            # 3. Intermediate matching
            if self.use_intermediate_matching and hasattr(model, "encoder_projection"):
                s_hidden = outputs.encoder_last_hidden_state
                t_hidden = teacher_outputs.encoder_last_hidden_state
                inter_loss = self._intermediate_loss(
                    s_hidden, t_hidden, model.encoder_projection, inputs.get("attention_mask")
                )
                total_loss = total_loss + self.alpha_inter * inter_loss
                loss_dict["inter_loss"] = inter_loss.item()

        loss_dict["total_loss"] = total_loss.item()

        # Accumulate for logging
        for k, v in loss_dict.items():
            self._loss_accum[k] = self._loss_accum.get(k, 0) + v
            self._loss_count[k] = self._loss_count.get(k, 0) + 1

        return (total_loss, outputs) if return_outputs else total_loss

    # -------------------------------------------------- fixed temperature KD
    def _fixed_kd_loss(self, s_logits, t_logits):
        tau = self.temperature
        log_s = F.log_softmax(s_logits / tau, dim=-1)
        p_t = F.softmax(t_logits / tau, dim=-1)
        return F.kl_div(log_s, p_t, reduction="batchmean") * (tau ** 2)

    # ---------------------------------------------- adaptive temperature KD
    def _adaptive_kd_loss(self, s_logits, t_logits, labels=None):
        """
        Per-sample adaptive temperature based on teacher confidence (entropy).
        High confidence (low entropy) -> lower tau (sharper targets)
        Low confidence  (high entropy) -> higher tau (softer targets)
        """
        # Token-level teacher entropy
        p_t = F.softmax(t_logits, dim=-1)
        token_ent = -(p_t * torch.log(p_t + 1e-10)).sum(dim=-1)  # (B, S)

        # Mask padding
        if labels is not None:
            mask = (labels != -100).float()
            sample_ent = (token_ent * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        else:
            mask = None
            sample_ent = token_ent.mean(dim=-1)

        # Sigmoid mapping centred on batch mean
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(
            sample_ent - sample_ent.mean()
        )
        tau = tau.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

        log_s = F.log_softmax(s_logits / tau, dim=-1)
        p_t_scaled = F.softmax(t_logits / tau, dim=-1)

        kl = F.kl_div(log_s, p_t_scaled, reduction="none").sum(dim=-1)  # (B, S)
        kl_weighted = kl * (tau.squeeze(-1) ** 2)

        if mask is not None:
            return (kl_weighted * mask).sum() / mask.sum().clamp(min=1)
        return kl_weighted.mean()

    # ----------------------------------------- intermediate layer matching
    def _intermediate_loss(self, s_hidden, t_hidden, projection, attention_mask):
        """Normalised MSE between projected student and teacher encoder states."""
        s_proj = projection(s_hidden)
        s_norm = F.normalize(s_proj, p=2, dim=-1)
        t_norm = F.normalize(t_hidden.detach(), p=2, dim=-1)

        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            diff = (s_norm - t_norm) * m
            return (diff ** 2).sum() / (m.sum() * s_norm.size(-1))
        return F.mse_loss(s_norm, t_norm)

    # ------------------------------------------------- logging override
    def log(self, logs, *args, **kwargs):
        if self._loss_accum:
            for k in list(self._loss_accum.keys()):
                c = self._loss_count.get(k, 1)
                if c > 0:
                    logs[k] = round(self._loss_accum[k] / c, 6)
            self._loss_accum = {}
            self._loss_count = {}
        super().log(logs, *args, **kwargs)


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(eval_preds, tokenizer, rouge_obj):
    preds, labels = eval_preds
    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    r1, r2, rL = [], [], []
    for pred, ref in zip(decoded_preds, decoded_labels):
        pred, ref = pred.strip() or "।", ref.strip() or "।"
        sc = rouge_obj.score(ref, pred)
        r1.append(sc["rouge1"].fmeasure)
        r2.append(sc["rouge2"].fmeasure)
        rL.append(sc["rougeL"].fmeasure)

    return {
        "rouge1": np.mean(r1),
        "rouge2": np.mean(r2),
        "rougeL": np.mean(rL),
        "rougeLsum": np.mean(rL),
        "gen_len": np.mean([len(p.split()) for p in decoded_preds]),
    }


# ============================================================================
# Data Loading & Preprocessing
# ============================================================================

def load_and_prepare_data(tokenizer, config, quick=False):
    """Load CSVs, optionally attach pseudo-labels, tokenize."""

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    if quick:
        train_df = train_df.head(500)
        val_df = val_df.head(200)
        test_df = test_df.head(200)

    # Attach pseudo-labels to training data
    include_pseudo = config.get("use_pseudo_labels", False)
    if include_pseudo:
        for teacher_key in ("mt5_base", "mt5_xlsum"):
            pl_path = os.path.join(PSEUDO_LABEL_DIR, f"train_{teacher_key}.json")
            if os.path.exists(pl_path):
                with open(pl_path, "r", encoding="utf-8") as f:
                    pseudo = json.load(f)
                # Align by index; truncate or pad with empty string
                col = [pseudo[i] if i < len(pseudo) else "" for i in range(len(train_df))]
                train_df[f"{teacher_key}_summary"] = col
                print(f"  Loaded {len(pseudo)} pseudo-labels from {pl_path}")
            else:
                print(f"  WARNING: {pl_path} not found — skipping {teacher_key}")

    # Convert to HF datasets
    train_ds = Dataset.from_pandas(train_df[["text", "summary"] +
        ([c for c in train_df.columns if c.endswith("_summary") and c != "summary"]
         if include_pseudo else [])])
    val_ds = Dataset.from_pandas(val_df[["text", "summary"]])
    test_ds = Dataset.from_pandas(test_df[["text", "summary"]])

    # ----- tokenization functions -----
    def tok_train(examples):
        mi = tokenizer(examples["text"], max_length=MAX_INPUT_LENGTH,
                       truncation=True, padding=False)
        mi["labels"] = tokenizer(text_target=examples["summary"],
                                  max_length=MAX_TARGET_LENGTH,
                                  truncation=True, padding=False)["input_ids"]
        if include_pseudo:
            for tk in ("mt5_base", "mt5_xlsum"):
                col = f"{tk}_summary"
                if col in examples:
                    mi[f"{tk}_labels"] = tokenizer(
                        text_target=[str(s) if s else "" for s in examples[col]],
                        max_length=MAX_TARGET_LENGTH,
                        truncation=True, padding=False,
                    )["input_ids"]
        return mi

    def tok_eval(examples):
        mi = tokenizer(examples["text"], max_length=MAX_INPUT_LENGTH,
                       truncation=True, padding=False)
        mi["labels"] = tokenizer(text_target=examples["summary"],
                                  max_length=MAX_TARGET_LENGTH,
                                  truncation=True, padding=False)["input_ids"]
        return mi

    train_tok = train_ds.map(tok_train, batched=True,
                             remove_columns=train_ds.column_names, desc="Tok train")
    val_tok = val_ds.map(tok_eval, batched=True,
                         remove_columns=val_ds.column_names, desc="Tok val")
    test_tok = test_ds.map(tok_eval, batched=True,
                           remove_columns=test_ds.column_names, desc="Tok test")

    print(f"  Train: {len(train_tok)}, Val: {len(val_tok)}, Test: {len(test_tok)}")
    return train_tok, val_tok, test_tok


# ============================================================================
# Main Training Function
# ============================================================================

def train_ablation(config_name, quick=False, resume_dir=None):
    """Train student model with specified ablation configuration.
    
    Args:
        config_name: Name of ablation config (A1_baseline, etc.)
        quick: Quick test mode (500 samples, 1 epoch)
        resume_dir: If provided, resume from this directory instead of creating new one
    """
    config = ABLATION_CONFIGS[config_name]

    print("\n" + "=" * 80)
    print(f"ABLATION: {config_name}")
    print(f"  {config['description']}")
    print("=" * 80)
    for k, v in config.items():
        if k != "description":
            print(f"  {k}: {v}")

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Tokenizer (student's)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)

    # Data
    print("Loading & tokenizing data...")
    train_tok, val_tok, test_tok = load_and_prepare_data(tokenizer, config, quick)

    # Teacher (only when needed)
    teacher_model = None
    if config["use_logit_kd"] or config.get("use_intermediate_matching", False):
        print(f"\nLoading TEACHER from {TEACHER_MODEL_PATH}...")
        teacher_model = AutoModelForSeq2SeqLM.from_pretrained(TEACHER_MODEL_PATH)
        tp = sum(p.numel() for p in teacher_model.parameters())
        print(f"  Teacher params: {tp:,} ({tp/1e6:.1f}M)")

    # Student
    print(f"\nLoading STUDENT from {STUDENT_MODEL_NAME}...")
    student_model = AutoModelForSeq2SeqLM.from_pretrained(STUDENT_MODEL_NAME)
    sp = sum(p.numel() for p in student_model.parameters())
    print(f"  Student params: {sp:,} ({sp/1e6:.1f}M)")

    # Add projection for intermediate matching
    if config.get("use_intermediate_matching", False):
        print(f"  Adding encoder projection: Linear({STUDENT_DIM}, {TEACHER_DIM})")
        student_model.encoder_projection = nn.Linear(STUDENT_DIM, TEACHER_DIM)
        nn.init.xavier_uniform_(student_model.encoder_projection.weight)
        nn.init.zeros_(student_model.encoder_projection.bias)

    # Generation config — set on generation_config, not model.config (transformers v5+)
    student_model.generation_config.max_length = MAX_TARGET_LENGTH
    student_model.generation_config.num_beams = 6

    # ROUGE scorer
    rouge_obj = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=SpaceTokenizer(),
    )

    # Data collator
    data_collator = PseudoLabelDataCollator(
        pseudo_prob=config.get("pseudo_prob", 0.0),
        tokenizer=tokenizer,
        model=student_model,
        padding=True,
        label_pad_token_id=-100,
    )

    # Output directory - reuse existing or create new
    results_base = "ablation_results"
    os.makedirs(results_base, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find existing directory for this config (resume support)
    existing_dirs = sorted([d for d in os.listdir(results_base) 
                           if d.startswith(config_name + "_") and 
                           os.path.isdir(os.path.join(results_base, d))],
                          reverse=True)
    
    resume_from_checkpoint = None

    def _latest_checkpoint_path(run_dir):
        checkpoints = sorted(
            [
                d for d in os.listdir(run_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(run_dir, d))
            ],
            key=lambda x: int(x.split("-")[1])
        )
        return os.path.join(run_dir, checkpoints[-1]) if checkpoints else None

    if resume_dir:
        output_dir = resume_dir
        if not os.path.isdir(output_dir):
            raise ValueError(f"resume_dir does not exist: {resume_dir}")
        print(f"\n  Using explicit resume_dir: {output_dir}")
        resume_from_checkpoint = _latest_checkpoint_path(output_dir)
        if resume_from_checkpoint:
            print(f"  Resuming from {resume_from_checkpoint}")
        else:
            print("  No checkpoint found in resume_dir; starting from step 0")
    elif existing_dirs:
        output_dir = os.path.join(results_base, existing_dirs[0])
        resume_from_checkpoint = _latest_checkpoint_path(output_dir)
        if resume_from_checkpoint:
            print(f"\n  Resuming from {resume_from_checkpoint}")
        else:
            print(f"\n  Reusing directory {output_dir} (no checkpoints found)")
    else:
        output_dir = os.path.join(results_base, f"{config_name}_{timestamp}")
        print(f"\n  Creating new directory {output_dir}")

    # Keep metadata timestamp aligned with output directory when possible
    try:
        timestamp = os.path.basename(output_dir).split("_", 2)[-1]
    except Exception:
        pass
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1 if quick else NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=50 if quick else WARMUP_STEPS,
        bf16=True,
        eval_strategy="steps",
        eval_steps=100 if quick else EVAL_STEPS,
        save_strategy="steps",
        save_steps=100 if quick else SAVE_STEPS,
        logging_steps=10 if quick else 50,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    # Trainer
    trainer = EnhancedDistillationTrainer(
        teacher_model=teacher_model,
        distill_config=config,
        model=student_model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda ep: compute_metrics(ep, tokenizer, rouge_obj),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Train
    print("\n" + "=" * 80)
    print("TRAINING STARTED")
    print("=" * 80)
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save best model
    best_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    test_results = trainer.evaluate(test_tok, metric_key_prefix="test")

    print(f"\n  ROUGE-1:  {test_results.get('test_rouge1', 0):.4f}")
    print(f"  ROUGE-2:  {test_results.get('test_rouge2', 0):.4f}")
    print(f"  ROUGE-L:  {test_results.get('test_rougeL', 0):.4f}")
    print(f"  Gen len:  {test_results.get('test_gen_len', 0):.1f}")

    # Save results
    results = {
        "config_name": config_name,
        "quick_mode": bool(quick),
        "config": {k: v for k, v in config.items()},
        "train_samples": len(train_tok),
        "val_samples": len(val_tok),
        "test_samples": len(test_tok),
        "train_loss": train_result.training_loss,
        "train_runtime_sec": train_result.metrics.get("train_runtime", 0),
        "test_results": {k: float(v) if isinstance(v, (float, np.floating)) else v
                         for k, v in test_results.items()},
        "student_params": sp,
        "teacher_params": sum(p.numel() for p in teacher_model.parameters()) if teacher_model else 0,
        "timestamp": timestamp,
    }

    results_file = os.path.join(output_dir, "ablation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to {results_file}")
    print("=" * 80)

    return results


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for enhanced KD")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=list(ABLATION_CONFIGS.keys()),
        help="Ablation config to run",
    )
    parser.add_argument("--quick", action="store_true", help="Quick test (500 train samples, 1 epoch)")
    parser.add_argument("--resume_dir", type=str, default=None, help="Resume from a specific output directory")
    args = parser.parse_args()

    try:
        results = train_ablation(args.config, quick=args.quick, resume_dir=args.resume_dir)
        print(f"\nAblation {args.config} completed successfully!")
    except Exception as e:
        print(f"\nERROR in {args.config}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
