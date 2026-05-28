"""
Fine-tune a teacher model on the 5-language XL-Sum subset
==========================================================
Trains either google/mt5-base  (--model mt5base)
           or  csebuetnlp/mT5_XLSum  (--model mt5xl)
on xlsum_5lang_train.json for TEACHER_EPOCHS epochs.

Logs ROUGE-1, ROUGE-2, ROUGE-L to console every TEACHER_EVAL_STEPS batches.
Best model (highest val ROUGE-L) is saved to:
    teacher_outputs/mt5base/best_model/   or
    teacher_outputs/mt5xl/best_model/

Usage:
    python train_teacher.py --model mt5base
    python train_teacher.py --model mt5xl
    python train_teacher.py --model mt5base --epochs 3
"""

import os, sys, json, argparse, math
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_mt5 import *

MODEL_MAP = {
    'mt5base': EWAD_TEACHER_HUB,
    'mt5xl':   CPDP_TEACHER_HUB,
}

# Fixed canonical output paths — train_mt5_kd.py reads from here
CANONICAL_OUTPUT = {
    'mt5base': os.path.join(TEACHER_OUTPUT_DIR, 'mt5base'),
    'mt5xl':   os.path.join(TEACHER_OUTPUT_DIR, 'mt5xl'),
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class SumDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records   = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r   = self.records[idx]
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


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, val_records, device, n):
    """
    Returns (overall_scores, per_lang_scores).
      overall_scores  : {'rouge1': float, 'rouge2': float, 'rougeL': float}
      per_lang_scores : {lang: {'rouge1': float, 'rouge2': float, 'rougeL': float}}
    n samples are distributed proportionally (n // n_languages per language).
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
                    max_new_tokens=TEACHER_EVAL_MAX_NEW,
                    num_beams=TEACHER_EVAL_BEAMS,
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

def train_teacher(model_key: str, num_epochs: int = TEACHER_EPOCHS):
    assert model_key in MODEL_MAP, f"model must be one of {list(MODEL_MAP)}"
    hub_name   = MODEL_MAP[model_key]
    out_dir    = CANONICAL_OUTPUT[model_key]
    best_dir   = os.path.join(out_dir, 'best_model')
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  Teacher  : {hub_name}")
    print(f"  Device   : {device}")
    print(f"  Out dir  : {out_dir}")
    print(f"{'='*60}")

    # ── Data ────────────────────────────────────────────────────────────────
    assert os.path.exists(TRAIN_JSON), (
        f"Missing {TRAIN_JSON}. Run  python prepare_xlsum.py  first.")
    assert os.path.exists(VAL_JSON), (
        f"Missing {VAL_JSON}. Run  python prepare_xlsum.py  first.")

    with open(TRAIN_JSON, 'r', encoding='utf-8') as f:
        train_records = json.load(f)
    with open(VAL_JSON, 'r', encoding='utf-8') as f:
        val_records = json.load(f)
    print(f"\n  Train samples : {len(train_records):,}")
    print(f"  Val samples   : {len(val_records):,}")

    # ── Model & tokenizer ────────────────────────────────────────────────────
    print(f"\nLoading {hub_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(hub_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(hub_name).to(device)
    print(f"  d_model    : {model.config.d_model}")
    print(f"  vocab_size : {model.config.vocab_size:,}")
    print(f"  params     : {sum(p.numel() for p in model.parameters()):,}")

    pad_id       = tokenizer.pad_token_id
    train_ds     = SumDataset(train_records, tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=TEACHER_BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=make_collate(pad_id),
        pin_memory=(device.type == 'cuda'),
    )

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimizer    = torch.optim.AdamW(model.parameters(),
                                     lr=TEACHER_LR, weight_decay=0.01)
    total_steps  = math.ceil(len(train_loader) / TEACHER_GRAD_ACCUM) * num_epochs
    warmup_steps = int(total_steps * TEACHER_WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler('cuda', enabled=(device.type == 'cuda'))

    print(f"\nTraining:")
    print(f"  Epochs         : {num_epochs}")
    print(f"  Batches/epoch  : {len(train_loader):,}")
    print(f"  Optimiser steps: {total_steps:,}  (warmup {warmup_steps})")
    print(f"  Effective batch: {TEACHER_BATCH_SIZE * TEACHER_GRAD_ACCUM}")
    print(f"  Eval every     : {TEACHER_EVAL_STEPS} batches  "
          f"(quick={TEACHER_QUICK_EVAL_N}, epoch={TEACHER_EPOCH_EVAL_N})")

    best_rl      = 0.0
    no_improve   = 0
    global_step  = 0
    training_log = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        nb         = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels    = batch['labels'].to(device)

            with autocast('cuda', dtype=torch.bfloat16):
                out  = model(input_ids=input_ids,
                             attention_mask=attn_mask,
                             labels=labels)
                loss = out.loss

            scaler.scale(loss / TEACHER_GRAD_ACCUM).backward()
            epoch_loss += loss.item()
            nb += 1

            # ── Optimiser step ───────────────────────────────────────────
            if (batch_idx + 1) % TEACHER_GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TEACHER_MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_postfix(
                    loss=f"{epoch_loss/nb:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            # ── Mid-epoch evaluation every TEACHER_EVAL_STEPS batches ───
            if (batch_idx + 1) % TEACHER_EVAL_STEPS == 0:
                sc, sc_langs = evaluate(model, tokenizer, val_records,
                                        device, TEACHER_QUICK_EVAL_N)
                print(f"\n  [E{epoch+1} | step {global_step:>4} | "
                      f"batch {batch_idx+1:>4}]  loss={epoch_loss/nb:.4f}  "
                      f"overall: R1={sc['rouge1']:.4f}  "
                      f"R2={sc['rouge2']:.4f}  RL={sc['rougeL']:.4f}")
                for lang in LANGUAGES:
                    if lang in sc_langs:
                        ls = sc_langs[lang]
                        print(f"    {lang:<12}: R1={ls['rouge1']:.4f}  "
                              f"R2={ls['rouge2']:.4f}  RL={ls['rougeL']:.4f}")

                if sc['rougeL'] > best_rl:
                    best_rl = sc['rougeL']
                    _save_best(best_dir, model, tokenizer)
                    print(f"  *** NEW BEST  RL={best_rl:.4f} → {best_dir}")

                training_log.append({
                    'type': 'step', 'epoch': epoch + 1,
                    'global_step': global_step, 'batch': batch_idx + 1,
                    'loss': epoch_loss / nb, **sc, 'per_lang': sc_langs,
                })
                _write_log(out_dir, training_log)

        # ── End-of-epoch evaluation ──────────────────────────────────────
        avg_loss = epoch_loss / max(nb, 1)
        sc_ep, sc_ep_langs = evaluate(model, tokenizer, val_records,
                                      device, TEACHER_EPOCH_EVAL_N)

        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch+1}/{num_epochs}  avg_loss={avg_loss:.4f}  "
              f"overall: R1={sc_ep['rouge1']:.4f}  "
              f"R2={sc_ep['rouge2']:.4f}  RL={sc_ep['rougeL']:.4f}")
        for lang in LANGUAGES:
            if lang in sc_ep_langs:
                ls = sc_ep_langs[lang]
                print(f"    {lang:<12}: R1={ls['rouge1']:.4f}  "
                      f"R2={ls['rouge2']:.4f}  RL={ls['rougeL']:.4f}")

        # Save epoch checkpoint
        ckpt_dir = os.path.join(out_dir, 'checkpoints', f'epoch_{epoch+1}')
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"  Checkpoint saved → {ckpt_dir}")

        if sc_ep['rougeL'] > best_rl:
            best_rl    = sc_ep['rougeL']
            no_improve = 0
            _save_best(best_dir, model, tokenizer)
            print(f"  *** NEW BEST  RL={best_rl:.4f} → {best_dir}")
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
    print(f"  Finished.  Best ROUGE-L = {best_rl:.4f}")
    print(f"  Best model saved to: {best_dir}")
    if model_key == 'mt5base':
        print(f"\n  config_mt5.py already points EWAD_TEACHER_MODEL to:")
        print(f"    \"{best_dir}\"")
    else:
        print(f"\n  config_mt5.py already points CPDP_TEACHER_MODEL to:")
        print(f"    \"{best_dir}\"")
    print(f"{'='*60}")
    return out_dir


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_best(best_dir, model, tokenizer):
    os.makedirs(best_dir, exist_ok=True)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)


def _write_log(out_dir, log):
    with open(os.path.join(out_dir, 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune teacher model on the 5-language XL-Sum subset')
    parser.add_argument('--model', required=True, choices=['mt5base', 'mt5xl'],
                        help='mt5base → google/mt5-base  |  mt5xl → csebuetnlp/mT5_XLSum')
    parser.add_argument('--epochs', type=int, default=TEACHER_EPOCHS,
                        help=f'Training epochs (default {TEACHER_EPOCHS})')
    args = parser.parse_args()
    train_teacher(args.model, args.epochs)


if __name__ == '__main__':
    main()
