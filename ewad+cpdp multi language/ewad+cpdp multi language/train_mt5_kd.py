"""
MT5-Small Knowledge Distillation — EWAD + Encoder CPDP  (Multilingual)
=======================================================================
Student   : google/mt5-small
Teacher 1 : teacher_outputs/mt5base/best_model   (EWAD — same vocab → KL valid)
Teacher 2 : teacher_outputs/mt5xl/best_model     (CPDP — encoder distance)

All three models share the MT5 SentencePiece tokeniser (250 112-token vocab),
so a single tokeniser is used throughout and all masks are identical.

Three experiments (one per run):

  baseline   — fine-tune MT5-small on gold labels (CE only)
  ewad       — confidence-gated KD from MT5-base teacher + gold CE
  ewad_cpdp  — ewad + encoder-space CPDP regularisation (mT5_XLSum teacher)

EWAD:
  L = (1-w)*CE(student, gold) + w*KL(T1 || student)
  w = sigmoid(K*(max_prob_T1 - delta)) * (1 - CE_FLOOR)

Encoder CPDP:
  L_CPDP = mean( (d(S,T1) - d(S,T2) - d(T1,T2))^2 )
  d = cosine distance  |  T1/T2 encoders are frozen  |  d(T1,T2) detached

Usage:
  python train_mt5_kd.py --experiment baseline
  python train_mt5_kd.py --experiment ewad
  python train_mt5_kd.py --experiment ewad_cpdp
"""

import os, sys, json, argparse, math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_mt5 import *


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_splits():
    """Load pre-split JSON files produced by prepare_xlsum.py."""
    assert os.path.exists(TRAIN_JSON), (
        f"Missing {TRAIN_JSON}. Run  python prepare_xlsum.py  first.")
    assert os.path.exists(VAL_JSON), (
        f"Missing {VAL_JSON}. Run  python prepare_xlsum.py  first.")
    with open(TRAIN_JSON, 'r', encoding='utf-8') as f:
        train = json.load(f)
    with open(VAL_JSON, 'r', encoding='utf-8') as f:
        val = json.load(f)
    print(f"  train={len(train):,}  val={len(val):,}")
    return train, val


class SumDataset(Dataset):
    """
    Single-tokeniser dataset — all MT5 variants share the same vocabulary.
    Applies TASK_PREFIX to source text (best practice for MT5 family).
    """
    def __init__(self, records, tokenizer):
        self.records   = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r       = self.records[idx]
        text    = TASK_PREFIX + r[DATASET_TEXT_KEY]
        summary = r[DATASET_SUMMARY_KEY]

        src = self.tokenizer(text, truncation=True,
                             max_length=MAX_INPUT_TOKENS,
                             padding=False, return_tensors=None)
        tgt = self.tokenizer(summary, truncation=True,
                             max_length=MAX_TARGET_TOKENS,
                             padding=False, return_tensors=None)
        return {
            'input_ids':      src['input_ids'],
            'attention_mask': src['attention_mask'],
            'labels':         tgt['input_ids'],
        }


def _pad(seqs, pad_id):
    L = max(len(s) for s in seqs)
    return [list(s) + [pad_id] * (L - len(s)) for s in seqs]


def make_collate(pad_id):
    def collate(batch):
        return {
            'input_ids':      torch.tensor(_pad([b['input_ids']      for b in batch], pad_id)),
            'attention_mask': torch.tensor(_pad([b['attention_mask'] for b in batch], 0)),
            'labels':         torch.tensor(_pad([b['labels']         for b in batch], -100)),
        }
    return collate


# ── EWAD Loss ─────────────────────────────────────────────────────────────────

def ewad_loss(student_logits, teacher_logits, labels):
    """
    Confidence-gated KD (EWAD, single-teacher variant).

    gate    = sigmoid(K * (max_teacher_prob - DELTA))
    kd_w    = gate * (1 - CE_FLOOR)      in [0.0, 0.7]
    ce_w    = 1 - kd_w                   in [0.3, 1.0]
    loss    = mean over valid tokens of (kd_w * KL + ce_w * CE)

    Both logits: (B, T, V).  labels: (B, T), -100 = padding.
    """
    B, T, V = student_logits.shape
    s_lp = F.log_softmax(student_logits.float(), dim=-1)    # (B,T,V)
    t_p  = F.softmax(teacher_logits.float(),     dim=-1)    # (B,T,V)
    t_lp = F.log_softmax(teacher_logits.float(), dim=-1)    # (B,T,V)

    # KL(teacher || student) per token — clamp negatives from float rounding
    kl = (t_p * (t_lp - s_lp)).sum(dim=-1).clamp(min=0)    # (B,T)

    # Gold CE per token
    ce = F.cross_entropy(
        student_logits.float().view(-1, V),
        labels.view(-1),
        ignore_index=-100, reduction='none',
    ).view(B, T)

    valid = (labels != -100).float()                        # (B,T)

    # Confidence gate on teacher's most probable token
    gate  = torch.sigmoid(EWAD_K * (t_p.max(dim=-1).values - EWAD_DELTA))
    kd_w  = gate * (1.0 - EWAD_CE_FLOOR)
    ce_w  = 1.0 - kd_w

    per_tok = (kd_w * kl + ce_w * ce) * valid
    loss    = per_tok.sum() / valid.sum().clamp(min=1)

    n = valid.sum().clamp(min=1).item()
    diag = {
        'kl':   (kl   * valid).sum().item() / n,
        'ce':   (ce   * valid).sum().item() / n,
        'gate': (gate * valid).sum().item() / n,
        'conf': (t_p.max(dim=-1).values * valid).sum().item() / n,
        'kdw':  (kd_w * valid).sum().item() / n,
    }
    return loss, diag


# ── Encoder CPDP ──────────────────────────────────────────────────────────────

class EncoderCPDP(nn.Module):
    """
    Cross-architecture CPDP via projected mean-pooled encoder representations.

    Each model's encoder output is projected to a shared CPDP_PROJ_DIM-d
    unit-normalised space.  The student is regularised to preserve its
    relative cosine distance to both teachers, proportional to the teachers'
    own mutual distance (detached — teachers are frozen).

    L = mean( (d(S,T1) - d(S,T2) - d(T1,T2))^2 )
    d(a,b) = 1 - cosine_similarity(a, b)

    Dims:  student=512, T1=MT5-base=768, T2=mT5_XLSum=768  (both MT5-base based)
    All three encode the same tokens (shared tokeniser) → same attention mask.
    """

    def __init__(self, student_dim: int, t1_dim: int, t2_dim: int,
                 proj_dim: int = CPDP_PROJ_DIM):
        super().__init__()
        self.proj_s  = nn.Linear(student_dim, proj_dim, bias=False)
        self.proj_t1 = nn.Linear(t1_dim,      proj_dim, bias=False)
        self.proj_t2 = nn.Linear(t2_dim,      proj_dim, bias=False)
        for layer in (self.proj_s, self.proj_t1, self.proj_t2):
            nn.init.orthogonal_(layer.weight)

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mask-weighted mean pool. hidden:(B,L,D)  mask:(B,L) → (B,D)."""
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def forward(self, s_enc, mask, t1_enc, t2_enc):
        """
        All three encoders process the same tokens → share the same mask.
        s_enc, t1_enc, t2_enc : (B, L, D_i)
        mask                  : (B, L)
        """
        s  = F.normalize(self.proj_s( self._mean_pool(s_enc.float(),  mask)), dim=-1)
        t1 = F.normalize(self.proj_t1(self._mean_pool(t1_enc.float(), mask)), dim=-1)
        t2 = F.normalize(self.proj_t2(self._mean_pool(t2_enc.float(), mask)), dim=-1)

        d_s_t1  = 1.0 - (s * t1).sum(dim=-1)               # (B,)
        d_s_t2  = 1.0 - (s * t2).sum(dim=-1)               # (B,)
        d_t1_t2 = (1.0 - (t1 * t2).sum(dim=-1)).detach()   # (B,) — teachers frozen

        loss = ((d_s_t1 - d_s_t2 - d_t1_t2) ** 2).mean()
        return loss, {
            'd_s_t1':  d_s_t1.mean().item(),
            'd_s_t2':  d_s_t2.mean().item(),
            'd_t1_t2': d_t1_t2.mean().item(),
        }


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_eval(model, tokenizer, val_records, device, n):
    """
    Returns (overall_scores, per_lang_scores).
      overall_scores  : {'rouge1': float, 'rouge2': float, 'rougeL': float}
      per_lang_scores : {lang: {'rouge1': float, 'rouge2': float, 'rougeL': float}}
    n samples distributed proportionally (n // n_languages per language).
    """
    from rouge_score import rouge_scorer as _rs
    from collections import defaultdict

    class _SpaceTok:
        def tokenize(self, t): return t.split()

    scorer  = _rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], tokenizer=_SpaceTok())
    model.eval()

    by_lang = defaultdict(list)
    for r in val_records:
        by_lang[r['language']].append(r)

    per_n           = max(1, n // max(len(by_lang), 1))
    per_lang_scores = {}
    all_r1, all_r2, all_rl = [], [], []

    for lang in LANGUAGES:
        records = by_lang.get(lang, [])
        if not records:
            continue
        r1_list, r2_list, rl_list = [], [], []
        for item in records[:per_n]:
            text = TASK_PREFIX + item[DATASET_TEXT_KEY]
            enc  = tokenizer(text, max_length=MAX_INPUT_TOKENS,
                             truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=EVAL_MAX_NEW_TOKENS,
                    num_beams=EVAL_NUM_BEAMS,
                    early_stopping=True,
                )
            pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            ref  = item[DATASET_SUMMARY_KEY]
            s    = scorer.score(ref, pred)
            r1_list.append(s['rouge1'].fmeasure)
            r2_list.append(s['rouge2'].fmeasure)
            rl_list.append(s['rougeL'].fmeasure)
        per_lang_scores[lang] = {
            'rouge1': float(np.mean(r1_list)),
            'rouge2': float(np.mean(r2_list)),
            'rougeL': float(np.mean(rl_list)),
        }
        all_r1.extend(r1_list)
        all_r2.extend(r2_list)
        all_rl.extend(rl_list)

    model.train()
    overall = {
        'rouge1': float(np.mean(all_r1)) if all_r1 else 0.0,
        'rouge2': float(np.mean(all_r2)) if all_r2 else 0.0,
        'rougeL': float(np.mean(all_rl)) if all_rl else 0.0,
    }
    return overall, per_lang_scores


# ── Training ──────────────────────────────────────────────────────────────────

def train(experiment: str, num_epochs: int = NUM_EPOCHS):
    assert experiment in ('baseline', 'ewad', 'ewad_cpdp')
    use_ewad = experiment in ('ewad', 'ewad_cpdp')
    use_cpdp = experiment == 'ewad_cpdp'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir   = os.path.join(OUTPUT_DIR, f"{experiment}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Experiment : {experiment}")
    print(f"  Device     : {device}")
    print(f"  Output     : {out_dir}")
    print(f"{'='*60}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    # All three models (MT5-small, MT5-base, mT5_XLSum) share the same
    # SentencePiece vocabulary → one tokenizer handles everything
    print("\nLoading shared MT5 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"  vocab_size = {tokenizer.vocab_size:,}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading dataset ...")
    train_data, val_data = load_splits()
    train_ds = SumDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=make_collate(tokenizer.pad_token_id),
        pin_memory=(device.type == 'cuda'),
    )

    # ── Student model ─────────────────────────────────────────────────────────
    print(f"\nLoading student  ({STUDENT_MODEL}) ...")
    student = AutoModelForSeq2SeqLM.from_pretrained(STUDENT_MODEL).to(device)
    print(f"  d_model    = {student.config.d_model}")
    print(f"  vocab_size = {student.config.vocab_size:,}")
    print(f"  params     = {sum(p.numel() for p in student.parameters()):,}")

    ewad_teacher, cpdp_teacher, cpdp_module = None, None, None

    if use_ewad:
        assert os.path.exists(EWAD_TEACHER_MODEL), (
            f"EWAD teacher not found at: {EWAD_TEACHER_MODEL}\n"
            f"Run  python train_teacher.py --model mt5base  first.")
        print(f"\nLoading EWAD teacher  ({EWAD_TEACHER_MODEL}) ...")
        ewad_teacher = AutoModelForSeq2SeqLM.from_pretrained(
            EWAD_TEACHER_MODEL, torch_dtype=torch.bfloat16
        ).to(device)
        ewad_teacher.eval()
        for p in ewad_teacher.parameters():
            p.requires_grad_(False)
        print(f"  d_model    = {ewad_teacher.config.d_model}")
        print(f"  vocab_size = {ewad_teacher.config.vocab_size:,}")

        # Verify shared vocabulary — required for KL divergence
        assert student.config.vocab_size == ewad_teacher.config.vocab_size, (
            f"Student vocab ({student.config.vocab_size}) != "
            f"EWAD teacher vocab ({ewad_teacher.config.vocab_size}). "
            "Both must be MT5-family models with the same SentencePiece vocab."
        )

    if use_cpdp:
        is_local = os.path.exists(CPDP_TEACHER_MODEL)
        is_hub   = "/" in CPDP_TEACHER_MODEL and not os.sep in CPDP_TEACHER_MODEL
        assert is_local or is_hub, (
            f"CPDP teacher not found at: {CPDP_TEACHER_MODEL}\n"
            f"Run  python train_teacher.py --model mt5xl  first.")
        print(f"\nLoading CPDP teacher  ({CPDP_TEACHER_MODEL}) ...")
        cpdp_teacher = AutoModelForSeq2SeqLM.from_pretrained(
            CPDP_TEACHER_MODEL, torch_dtype=torch.bfloat16
        ).to(device)
        cpdp_teacher.eval()
        for p in cpdp_teacher.parameters():
            p.requires_grad_(False)
        print(f"  d_model    = {cpdp_teacher.config.d_model}")

        s_dim  = student.config.d_model       # MT5-small : 512
        t1_dim = ewad_teacher.config.d_model  # MT5-base  : 768
        t2_dim = cpdp_teacher.config.d_model  # mT5_XLSum : 768 (MT5-base based)
        cpdp_module = EncoderCPDP(s_dim, t1_dim, t2_dim).to(device)
        print(f"\n  CPDP projections: "
              f"S({s_dim})→{CPDP_PROJ_DIM}, "
              f"T1({t1_dim})→{CPDP_PROJ_DIM}, "
              f"T2({t2_dim})→{CPDP_PROJ_DIM}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    params = list(student.parameters())
    if cpdp_module:
        params += list(cpdp_module.parameters())

    optimizer    = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.01)
    total_steps  = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION) * num_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler('cuda', enabled=(device.type == 'cuda'))

    print(f"\nTraining config:")
    print(f"  Epochs          : {num_epochs}")
    print(f"  Batches/epoch   : {len(train_loader):,}")
    print(f"  Optimiser steps : {total_steps:,}  (warmup {warmup_steps})")
    print(f"  Effective batch : {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Eval every      : {QUICK_EVAL_STEPS} batches  "
          f"(quick={QUICK_EVAL_SAMPLES}, epoch={EPOCH_EVAL_SAMPLES})")

    # Save experiment config
    with open(os.path.join(out_dir, 'experiment_config.json'), 'w') as f:
        json.dump({
            'experiment':   experiment,
            'student':      STUDENT_MODEL,
            'ewad_teacher': EWAD_TEACHER_MODEL if use_ewad else None,
            'cpdp_teacher': CPDP_TEACHER_MODEL if use_cpdp else None,
            'languages':    LANGUAGES,
            'num_epochs':   num_epochs,
            'batch_size':   BATCH_SIZE,
            'grad_accum':   GRADIENT_ACCUMULATION,
            'lr':           LEARNING_RATE,
            'ewad_k':       EWAD_K if use_ewad else None,
            'ewad_delta':   EWAD_DELTA if use_ewad else None,
            'ewad_ce_floor':EWAD_CE_FLOOR if use_ewad else None,
            'cpdp_weight':  CPDP_WEIGHT if use_cpdp else None,
            'cpdp_proj_dim':CPDP_PROJ_DIM if use_cpdp else None,
            'timestamp':    timestamp,
        }, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_scores  = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    best_rl      = 0.0
    no_improve   = 0
    global_step  = 0
    training_log = []

    for epoch in range(num_epochs):
        student.train()
        if cpdp_module:
            cpdp_module.train()

        epoch_loss = 0.0
        epoch_diag: dict = {}
        nb = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels    = batch['labels'].to(device)

            with autocast('cuda', dtype=torch.bfloat16):

                # ── Student forward ──────────────────────────────────────
                student_out    = student(input_ids=input_ids,
                                         attention_mask=attn_mask,
                                         labels=labels)
                student_logits = student_out.logits                    # (B, T, V)
                student_enc    = student_out.encoder_last_hidden_state # (B, L, 512)

                if use_ewad:
                    # ── EWAD teacher forward (no grad) ───────────────────
                    with torch.no_grad():
                        t1_out = ewad_teacher(input_ids=input_ids,
                                              attention_mask=attn_mask,
                                              labels=labels)
                    t1_logits = t1_out.logits                          # (B, T, V)
                    t1_enc    = t1_out.encoder_last_hidden_state       # (B, L, 768)

                    loss, diag = ewad_loss(student_logits, t1_logits, labels)
                else:
                    loss  = student_out.loss
                    diag  = {'ce': student_out.loss.item()}
                    t1_enc = None

                # ── CPDP loss ────────────────────────────────────────────
                if use_cpdp:
                    with torch.no_grad():
                        t2_enc = cpdp_teacher.encoder(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                        ).last_hidden_state                            # (B, L, 768)

                    # All three share attn_mask (same tokeniser & same input)
                    cpdp_l, cpdp_diag = cpdp_module(
                        student_enc, attn_mask,
                        t1_enc,      t2_enc,
                    )
                    loss = loss + CPDP_WEIGHT * cpdp_l
                    diag.update(cpdp_diag)

            scaler.scale(loss / GRADIENT_ACCUMULATION).backward()

            epoch_loss += loss.item()
            nb += 1
            for k, v in diag.items():
                epoch_diag[k] = epoch_diag.get(k, 0.0) + v

            # ── Optimiser step ───────────────────────────────────────────
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg = epoch_loss / nb
                _d  = {k: f"{v/nb:.3f}" for k, v in epoch_diag.items()}
                pbar.set_postfix(loss=f"{avg:.4f}", **_d,
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")

            # ── Mid-epoch eval every QUICK_EVAL_STEPS batches ───────────
            if (batch_idx + 1) % QUICK_EVAL_STEPS == 0:
                sc, sc_langs = run_eval(student, tokenizer, val_data,
                                        device, QUICK_EVAL_SAMPLES)
                print(f"\n  [E{epoch+1} | step {global_step:>4} | "
                      f"batch {batch_idx+1:>4}]  loss={epoch_loss/nb:.4f}  "
                      f"overall: R1={sc['rouge1']:.4f}  "
                      f"R2={sc['rouge2']:.4f}  RL={sc['rougeL']:.4f}")
                for lang in LANGUAGES:
                    if lang in sc_langs:
                        ls = sc_langs[lang]
                        print(f"    {lang:<12}: R1={ls['rouge1']:.4f}  "
                              f"R2={ls['rouge2']:.4f}  RL={ls['rougeL']:.4f}")
                if cpdp_module:
                    cpdp_module.train()

                if sc['rougeL'] > best_rl:
                    best_rl     = sc['rougeL']
                    best_scores = sc
                    _save_best(out_dir, student, tokenizer, cpdp_module)
                    print(f"  *** NEW BEST  RL={best_rl:.4f} → best_model/")

                training_log.append({
                    'type': 'step', 'step': global_step,
                    'batch': batch_idx + 1, 'epoch': epoch + 1,
                    'loss': epoch_loss / nb, **sc, 'per_lang': sc_langs,
                })
                _write_log(out_dir, training_log)

        # ── End-of-epoch eval ────────────────────────────────────────────
        avg_loss = epoch_loss / max(nb, 1)
        sc_ep, sc_ep_langs = run_eval(student, tokenizer, val_data,
                                      device, EPOCH_EVAL_SAMPLES)

        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch+1}/{num_epochs}  avg_loss={avg_loss:.4f}  "
              f"overall: R1={sc_ep['rouge1']:.4f}  "
              f"R2={sc_ep['rouge2']:.4f}  RL={sc_ep['rougeL']:.4f}")
        for lang in LANGUAGES:
            if lang in sc_ep_langs:
                ls = sc_ep_langs[lang]
                print(f"    {lang:<12}: R1={ls['rouge1']:.4f}  "
                      f"R2={ls['rouge2']:.4f}  RL={ls['rougeL']:.4f}")

        # Epoch checkpoint
        ckpt_dir = os.path.join(out_dir, 'checkpoints', f'epoch_{epoch+1}')
        student.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        if cpdp_module:
            torch.save(cpdp_module.state_dict(),
                       os.path.join(ckpt_dir, 'cpdp_module.pt'))
        print(f"  Checkpoint saved → {ckpt_dir}")

        if sc_ep['rougeL'] > best_rl:
            best_rl     = sc_ep['rougeL']
            best_scores = sc_ep
            no_improve  = 0
            _save_best(out_dir, student, tokenizer, cpdp_module)
            print(f"  *** NEW BEST  RL={best_rl:.4f} → best_model/")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{EARLY_STOP_PATIENCE})  "
                  f"best RL={best_rl:.4f}")
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}.")
                break

        training_log.append({
            'type': 'epoch', 'epoch': epoch + 1,
            'avg_loss': avg_loss, **sc_ep, 'best_rl': best_rl,
            'per_lang': sc_ep_langs,
        })
        _write_log(out_dir, training_log)
        print(f"{'─'*60}")

    print(f"\n{'='*60}")
    print(f"  Experiment : {experiment}")
    print(f"  Best ROUGE-1 = {best_scores['rouge1']:.4f}")
    print(f"  Best ROUGE-2 = {best_scores['rouge2']:.4f}")
    print(f"  Best ROUGE-L = {best_scores['rougeL']:.4f}")
    print(f"  Saved to     : {out_dir}")
    print(f"{'='*60}")
    return out_dir


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_best(out_dir, model, tokenizer, cpdp_module=None):
    d = os.path.join(out_dir, 'best_model')
    model.save_pretrained(d)
    tokenizer.save_pretrained(d)
    if cpdp_module is not None:
        torch.save(cpdp_module.state_dict(),
                   os.path.join(d, 'cpdp_module.pt'))


def _write_log(out_dir, log):
    with open(os.path.join(out_dir, 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='MT5-small distillation: baseline / ewad / ewad_cpdp')
    parser.add_argument('--experiment', required=True,
                        choices=['baseline', 'ewad', 'ewad_cpdp'])
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Max training epochs (default {NUM_EPOCHS})')
    args = parser.parse_args()
    train(args.experiment, num_epochs=args.epochs)


if __name__ == '__main__':
    main()
