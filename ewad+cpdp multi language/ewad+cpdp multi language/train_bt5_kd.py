"""
BanglaT5-Small Knowledge Distillation — EWAD + Encoder CPDP
=============================================================
Three experiments (one per run):

  baseline   -- fine-tune BanglaT5-small on gold labels (CE only)
  ewad       -- confidence-gated KD from BanglaT5 teacher + gold CE
  ewad_cpdp  -- ewad + encoder-space CPDP regularisation (MT5-XL-Sum)

EWAD (single-teacher variant):
  L = (1 - kd_weight) * CE(student, gold)  +  kd_weight * KL(BT5 || student)
  kd_weight = gate * (1 - CE_FLOOR),  gate = sigmoid(K * (max_prob_teacher - delta))

Encoder CPDP:
  Project mean-pooled encoder outputs of all three models to a shared 256-d space.
  L_CPDP = mean( (d(S,T1) - d(S,T2) - d(T1,T2))^2 )
  where d = cosine distance and T1=BanglaT5, T2=MT5-XL-Sum.
  d(T1,T2) is detached (teachers are frozen).

Usage:
  python train_bt5_kd.py --experiment baseline
  python train_bt5_kd.py --experiment ewad
  python train_bt5_kd.py --experiment ewad_cpdp
  python train_bt5_kd.py --experiment ewad --epochs 4 --max-samples 2000  # smoke test
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
from config_bt5 import *


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_splits(max_samples=None, dataset_file=None):
    _file = dataset_file or DATASET_FILE
    with open(_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    np.random.seed(SEED)
    data = [data[i] for i in np.random.permutation(len(data))]
    n = len(data)
    train_end = int(TRAIN_SPLIT * n)
    val_end   = train_end + int(VAL_SPLIT * n)
    train, val = data[:train_end], data[train_end:val_end]
    if max_samples:
        train = train[:int(max_samples * TRAIN_SPLIT)]
        val   = val[:int(max_samples * VAL_SPLIT)]
    print(f"  train={len(train)}, val={len(val)}")
    return train, val


class SumDataset(Dataset):
    """
    Returns tokenised inputs for the BanglaT5 tokenizer (used for student +
    EWAD teacher) and, optionally, MT5 tokenizer (used for CPDP teacher encoder).
    """
    def __init__(self, data, bt5_tok, mt5_tok=None):
        self.data    = data
        self.bt5_tok = bt5_tok
        self.mt5_tok = mt5_tok

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item    = self.data[idx]
        text    = item[DATASET_TEXT_KEY]
        summary = item[DATASET_SUMMARY_KEY]

        bt5_src = self.bt5_tok(text, truncation=True,
                               max_length=MAX_INPUT_TOKENS,
                               padding=False, return_tensors=None)
        bt5_tgt = self.bt5_tok(summary, truncation=True,
                               max_length=MAX_TARGET_TOKENS,
                               padding=False, return_tensors=None)
        out = {
            'bt5_input_ids':      bt5_src['input_ids'],
            'bt5_attention_mask': bt5_src['attention_mask'],
            'bt5_labels':         bt5_tgt['input_ids'],
        }
        if self.mt5_tok is not None:
            mt5_src = self.mt5_tok(text, truncation=True,
                                   max_length=MAX_INPUT_TOKENS,
                                   padding=False, return_tensors=None)
            out['mt5_input_ids']      = mt5_src['input_ids']
            out['mt5_attention_mask'] = mt5_src['attention_mask']
        return out


def _pad(seqs, pad_id):
    """Right-pad list of lists to the same length, return 2-D list."""
    L = max(len(s) for s in seqs)
    return [list(s) + [pad_id] * (L - len(s)) for s in seqs]


def make_collate(bt5_pad_id, mt5_pad_id=None):
    def collate(batch):
        bt5_input  = torch.tensor(_pad([b['bt5_input_ids']      for b in batch], bt5_pad_id))
        bt5_mask   = torch.tensor(_pad([b['bt5_attention_mask'] for b in batch], 0))
        # Labels: pad with -100 so those positions are ignored in CE loss
        bt5_labels = torch.tensor(_pad([b['bt5_labels'] for b in batch], -100))
        result = {'bt5_input': bt5_input, 'bt5_mask': bt5_mask, 'bt5_labels': bt5_labels}
        if 'mt5_input_ids' in batch[0]:
            result['mt5_input'] = torch.tensor(
                _pad([b['mt5_input_ids']      for b in batch], mt5_pad_id))
            result['mt5_mask']  = torch.tensor(
                _pad([b['mt5_attention_mask'] for b in batch], 0))
        return result
    return collate


# ── EWAD Loss ─────────────────────────────────────────────────────────────────

def ewad_loss(student_logits, teacher_logits, labels):
    """
    Confidence-gated knowledge distillation (single-teacher EWAD).

    Gate = sigmoid(K * (max_teacher_prob - DELTA))
      * High teacher confidence  → gate→1 → KD weight high, CE weight low
      * Low teacher confidence   → gate→0 → KD weight≈0,  CE weight≈1

    kd_weight = gate * (1 - CE_FLOOR)         in [0, 0.7]
    ce_weight = 1 - kd_weight                 in [0.3, 1.0]

    Both logits: (B, T, V).  labels: (B, T), -100 = ignore.
    """
    B, T, V = student_logits.shape
    s_lp  = F.log_softmax(student_logits.float(), dim=-1)   # (B,T,V)
    t_p   = F.softmax(teacher_logits.float(),    dim=-1)    # (B,T,V)
    t_lp  = F.log_softmax(teacher_logits.float(), dim=-1)   # (B,T,V)

    # KL(teacher || student) per position — clamp to avoid negative numerical noise
    kl = (t_p * (t_lp - s_lp)).sum(dim=-1).clamp(min=0)    # (B,T)

    # Gold CE per position
    ce = F.cross_entropy(
        student_logits.float().view(-1, V),
        labels.view(-1),
        ignore_index=-100, reduction='none',
    ).view(B, T)

    valid = (labels != -100).float()                        # (B,T)

    # Confidence gate from teacher's max probability
    gate     = torch.sigmoid(EWAD_K * (t_p.max(dim=-1).values - EWAD_DELTA))   # (B,T)
    kd_w     = gate * (1.0 - EWAD_CE_FLOOR)                # in [0, 0.7]
    ce_w     = 1.0 - kd_w                                  # in [0.3, 1.0]

    per_tok  = (kd_w * kl + ce_w * ce) * valid
    loss     = per_tok.sum() / valid.sum().clamp(min=1)

    n = valid.sum().clamp(min=1).item()
    diag = {
        'kl':   (kl * valid).sum().item() / n,
        'ce':   (ce * valid).sum().item() / n,
        'gate': (gate * valid).sum().item() / n,
        'conf': (t_p.max(dim=-1).values * valid).sum().item() / n,
        'kdw':  (kd_w * valid).sum().item() / n,
    }
    return loss, diag


# ── Encoder CPDP ──────────────────────────────────────────────────────────────

class EncoderCPDP(nn.Module):
    """
    Cross-architecture CPDP via projected encoder representations.

    Each model's mean-pooled encoder output is projected to a shared
    CPDP_PROJ_DIM-dimensional unit-norm space.  The student is regularised to
    maintain its relative cosine distance to both teachers in proportion to the
    teachers' mutual distance (which is detached — teachers are frozen).

    L = mean( (d(S,T1) - d(S,T2) - d(T1,T2))^2 )
        where d(a,b) = 1 - cosine_similarity(a, b)

    Projections are initialised orthogonally and trained jointly with the student.
    """

    def __init__(self, student_dim: int, bt5_dim: int, mt5_dim: int,
                 proj_dim: int = CPDP_PROJ_DIM):
        super().__init__()
        self.proj_s  = nn.Linear(student_dim, proj_dim, bias=False)
        self.proj_t1 = nn.Linear(bt5_dim,     proj_dim, bias=False)
        self.proj_t2 = nn.Linear(mt5_dim,     proj_dim, bias=False)
        for layer in (self.proj_s, self.proj_t1, self.proj_t2):
            nn.init.orthogonal_(layer.weight)

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mask-weighted mean pool. hidden:(B,L,D), mask:(B,L) → (B,D)."""
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def forward(self, student_enc, student_mask,
                bt5_enc, bt5_mask,
                mt5_enc, mt5_mask):
        s  = F.normalize(self.proj_s( self._mean_pool(student_enc.float(), student_mask)), dim=-1)
        t1 = F.normalize(self.proj_t1(self._mean_pool(bt5_enc.float(),     bt5_mask)),     dim=-1)
        t2 = F.normalize(self.proj_t2(self._mean_pool(mt5_enc.float(),     mt5_mask)),     dim=-1)

        d_s_t1  = 1.0 - (s * t1).sum(dim=-1)               # (B,)
        d_s_t2  = 1.0 - (s * t2).sum(dim=-1)               # (B,)
        d_t1_t2 = (1.0 - (t1 * t2).sum(dim=-1)).detach()   # (B,) teachers fixed

        loss = ((d_s_t1 - d_s_t2 - d_t1_t2) ** 2).mean()
        return loss, {
            'd_s_t1':  d_s_t1.mean().item(),
            'd_s_t2':  d_s_t2.mean().item(),
            'd_t1_t2': d_t1_t2.mean().item(),
        }


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_eval(model, tokenizer, val_data, device, n=QUICK_EVAL_SAMPLES):
    from rouge_score import rouge_scorer as _rs

    class _Tok:
        def tokenize(self, t): return t.split()

    scorer = _rs.RougeScorer(['rougeL'], tokenizer=_Tok())
    model.eval()
    preds, refs = [], []

    for item in val_data[:n]:
        enc = tokenizer(
            item[DATASET_TEXT_KEY],
            max_length=MAX_INPUT_TOKENS, truncation=True,
            return_tensors='pt',
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                num_beams=EVAL_NUM_BEAMS,
                early_stopping=True,
            )
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        preds.append(pred)
        refs.append(item[DATASET_SUMMARY_KEY])

    rl = float(np.mean([scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(preds, refs)]))
    model.train()
    return rl


# ── Training ──────────────────────────────────────────────────────────────────

def train(experiment: str, num_epochs: int = NUM_EPOCHS, max_samples=None,
          dataset_file: str = None):
    assert experiment in ('baseline', 'ewad', 'ewad_cpdp')
    use_ewad = experiment in ('ewad', 'ewad_cpdp')
    use_cpdp = experiment == 'ewad_cpdp'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir   = os.path.join(OUTPUT_DIR, f"{experiment}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nExperiment : {experiment}")
    print(f"Device     : {device}")
    print(f"Output     : {out_dir}")

    # ── Dataset path override ────────────────────────────────────────────────
    _dataset_file = dataset_file or DATASET_FILE

    # ── Tokenizers ──────────────────────────────────────────────────────────
    print("\nLoading tokenizers...")
    # Load student tokenizer from EWAD teacher path to guarantee vocab alignment
    bt5_tok = AutoTokenizer.from_pretrained(EWAD_TEACHER_MODEL)
    mt5_tok = AutoTokenizer.from_pretrained(CPDP_TEACHER_MODEL) if use_cpdp else None

    # Sanity-check vocab compatibility for KL divergence (compare model configs, not raw tokenizer)
    from transformers import AutoConfig
    student_cfg = AutoConfig.from_pretrained(STUDENT_MODEL)
    teacher_cfg = AutoConfig.from_pretrained(EWAD_TEACHER_MODEL)
    assert student_cfg.vocab_size == teacher_cfg.vocab_size, (
        f"Student vocab_size={student_cfg.vocab_size} != BanglaT5 vocab_size={teacher_cfg.vocab_size}. "
        "EWAD KL requires identical model vocabulary."
    )
    print(f"  Vocab size (shared): {student_cfg.vocab_size}  tokenizer: {bt5_tok.vocab_size}")

    # ── Data ────────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    print(f"  file: {_dataset_file}")
    train_data, val_data = load_splits(max_samples or MAX_SAMPLES, dataset_file=_dataset_file)
    train_ds = SumDataset(train_data, bt5_tok, mt5_tok)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=make_collate(
            bt5_tok.pad_token_id,
            mt5_tok.pad_token_id if mt5_tok else None,
        ),
        pin_memory=(device.type == 'cuda'),
    )

    # ── Models ──────────────────────────────────────────────────────────────
    print("\nLoading student (BanglaT5-small)...")
    student = AutoModelForSeq2SeqLM.from_pretrained(STUDENT_MODEL).to(device)
    # Student vocab_size is 32128 (same as teacher) — do NOT resize

    bt5_teacher, mt5_teacher, cpdp_module = None, None, None

    if use_ewad:
        print("Loading EWAD teacher (BanglaT5 fine-tuned)...")
        bt5_teacher = AutoModelForSeq2SeqLM.from_pretrained(
            EWAD_TEACHER_MODEL, dtype=torch.bfloat16
        ).to(device)
        bt5_teacher.eval()
        for p in bt5_teacher.parameters():
            p.requires_grad_(False)

    if use_cpdp:
        print("Loading CPDP teacher (MT5-XL-Sum fine-tuned)...")
        mt5_teacher = AutoModelForSeq2SeqLM.from_pretrained(
            CPDP_TEACHER_MODEL, dtype=torch.bfloat16
        ).to(device)
        mt5_teacher.eval()
        for p in mt5_teacher.parameters():
            p.requires_grad_(False)

        s_dim  = student.config.d_model       # 512
        t1_dim = bt5_teacher.config.d_model   # 768
        t2_dim = mt5_teacher.config.d_model   # 768
        cpdp_module = EncoderCPDP(s_dim, t1_dim, t2_dim).to(device)
        print(f"  CPDP projections: {s_dim}→{CPDP_PROJ_DIM}, "
              f"{t1_dim}→{CPDP_PROJ_DIM}, {t2_dim}→{CPDP_PROJ_DIM}")

    # ── Optimizer ───────────────────────────────────────────────────────────
    params = list(student.parameters())
    if cpdp_module:
        params += list(cpdp_module.parameters())

    optimizer  = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION) * num_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler  = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler     = GradScaler('cuda', enabled=(device.type == 'cuda'))

    print(f"\nTraining config:")
    print(f"  Batches/epoch    : {len(train_loader)}")
    print(f"  Optimizer steps  : {total_steps}  (warmup {warmup_steps})")
    print(f"  Effective batch  : {BATCH_SIZE * GRADIENT_ACCUMULATION}")

    # Save experiment metadata
    with open(os.path.join(out_dir, 'experiment_config.json'), 'w') as f:
        json.dump({
            'experiment': experiment,
            'student': STUDENT_MODEL,
            'dataset': _dataset_file,
            'ewad_teacher': EWAD_TEACHER_MODEL if use_ewad else None,
            'cpdp_teacher': CPDP_TEACHER_MODEL if use_cpdp else None,
            'num_epochs': num_epochs, 'batch_size': BATCH_SIZE,
            'grad_accum': GRADIENT_ACCUMULATION, 'lr': LEARNING_RATE,
            'ewad_ce_floor': EWAD_CE_FLOOR, 'cpdp_weight': CPDP_WEIGHT if use_cpdp else None,
            'timestamp': timestamp,
        }, f, indent=2)

    # ── Training loop ────────────────────────────────────────────────────────
    best_rl       = 0.0
    no_improve    = 0
    global_step   = 0
    training_log  = []

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
            bt5_input  = batch['bt5_input'].to(device)
            bt5_mask   = batch['bt5_mask'].to(device)
            bt5_labels = batch['bt5_labels'].to(device)

            with autocast('cuda', dtype=torch.bfloat16):
                # ── Student forward (always needed) ──────────────────────
                student_out    = student(input_ids=bt5_input,
                                         attention_mask=bt5_mask,
                                         labels=bt5_labels)
                student_logits = student_out.logits   # (B, T, V)
                student_enc    = student_out.encoder_last_hidden_state  # (B, L, 512)

                if use_ewad:
                    # ── EWAD teacher forward (no grad) ────────────────────
                    with torch.no_grad():
                        bt5_out    = bt5_teacher(input_ids=bt5_input,
                                                  attention_mask=bt5_mask,
                                                  labels=bt5_labels)
                    teacher_logits = bt5_out.logits            # (B, T, V) bfloat16→cast in ewad_loss
                    bt5_enc        = bt5_out.encoder_last_hidden_state  # reuse for CPDP

                    loss, diag = ewad_loss(student_logits, teacher_logits, bt5_labels)
                else:
                    # Baseline: model's own CE loss
                    loss = student_out.loss
                    diag = {'ce': student_out.loss.item()}
                    bt5_enc = None

                # ── CPDP loss ─────────────────────────────────────────────
                if use_cpdp:
                    mt5_input = batch['mt5_input'].to(device)
                    mt5_mask  = batch['mt5_mask'].to(device)

                    with torch.no_grad():
                        mt5_enc_out = mt5_teacher.encoder(
                            input_ids=mt5_input,
                            attention_mask=mt5_mask,
                        )
                    mt5_enc = mt5_enc_out.last_hidden_state   # (B, L_mt5, 768)

                    cpdp_l, cpdp_diag = cpdp_module(
                        student_enc, bt5_mask,       # student: (B,L,512), same tokenizer as bt5
                        bt5_enc,     bt5_mask,       # bt5: (B,L,768)
                        mt5_enc,     mt5_mask,       # mt5: (B,L_mt5,768)
                    )
                    loss = loss + CPDP_WEIGHT * cpdp_l
                    diag.update(cpdp_diag)

            scaler.scale(loss / GRADIENT_ACCUMULATION).backward()

            epoch_loss += loss.item()
            nb += 1
            for k, v in diag.items():
                epoch_diag[k] = epoch_diag.get(k, 0.0) + v

            # ── Optimizer step ───────────────────────────────────────────
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg  = epoch_loss / nb
                _d   = {k: f"{v/nb:.3f}" for k, v in epoch_diag.items()}
                pbar.set_postfix(loss=f"{avg:.4f}", **_d,
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")

            # ── Quick eval every QUICK_EVAL_STEPS batches ───────────────
            if (batch_idx + 1) % QUICK_EVAL_STEPS == 0:
                rl = run_eval(student, bt5_tok, val_data, device, QUICK_EVAL_SAMPLES)
                print(f"\n  [Eval @ batch {batch_idx+1}] ROUGE-L: {rl:.4f}")
                pbar.set_postfix(loss=f"{epoch_loss/nb:.4f}", RL=f"{rl:.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")
                if cpdp_module:
                    cpdp_module.train()

                if rl > best_rl:
                    best_rl = rl
                    _save_best(out_dir, student, bt5_tok, cpdp_module)
                    print(f"  [NEW BEST RL={rl:.4f}] -> best_model/")

                training_log.append({'step': global_step, 'batch': batch_idx + 1,
                                     'epoch': epoch + 1, 'rouge_l': rl,
                                     'loss': epoch_loss / nb})
                _write_log(out_dir, training_log)

        # ── End-of-epoch eval ────────────────────────────────────────────
        avg_loss = epoch_loss / max(nb, 1)
        rl_epoch = run_eval(student, bt5_tok, val_data, device, EPOCH_EVAL_SAMPLES)
        print(f"\nEpoch {epoch+1}/{num_epochs}  loss={avg_loss:.4f}  ROUGE-L={rl_epoch:.4f}")

        # Save epoch checkpoint always
        ckpt_dir = os.path.join(out_dir, 'checkpoints', f'epoch_{epoch+1}')
        student.save_pretrained(ckpt_dir)
        bt5_tok.save_pretrained(ckpt_dir)
        if cpdp_module:
            torch.save(cpdp_module.state_dict(),
                       os.path.join(ckpt_dir, 'cpdp_module.pt'))

        if rl_epoch > best_rl:
            best_rl = rl_epoch
            no_improve = 0
            _save_best(out_dir, student, bt5_tok, cpdp_module)
            print(f"  [NEW BEST RL={rl_epoch:.4f}] -> best_model/")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{EARLY_STOP_PATIENCE}). Best: {best_rl:.4f}")
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch+1}.")
                break

        training_log.append({'epoch': epoch + 1, 'loss': avg_loss,
                              'rouge_l': rl_epoch, 'best_rl': best_rl})
        _write_log(out_dir, training_log)

    print(f"\nFinished. Best ROUGE-L: {best_rl:.4f}  |  {out_dir}")
    return out_dir


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_best(out_dir, model, tokenizer, cpdp_module=None):
    d = os.path.join(out_dir, 'best_model')
    model.save_pretrained(d)
    tokenizer.save_pretrained(d)
    if cpdp_module is not None:
        torch.save(cpdp_module.state_dict(), os.path.join(d, 'cpdp_module.pt'))


def _write_log(out_dir, log):
    with open(os.path.join(out_dir, 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BanglaT5-small distillation: baseline / ewad / ewad_cpdp")
    parser.add_argument('--experiment', required=True,
                        choices=['baseline', 'ewad', 'ewad_cpdp'])
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Max epochs (default {NUM_EPOCHS}, early stop after '
                             f'{EARLY_STOP_PATIENCE} no-improvement epochs)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Cap dataset size per split for smoke tests')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override dataset file path (default: DATASET_FILE in config)')
    args = parser.parse_args()
    train(args.experiment, num_epochs=args.epochs, max_samples=args.max_samples,
          dataset_file=args.dataset)


if __name__ == '__main__':
    main()
