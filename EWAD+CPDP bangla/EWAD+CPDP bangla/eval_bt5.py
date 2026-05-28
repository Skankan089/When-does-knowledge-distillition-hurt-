"""
Evaluate a saved BanglaT5-small (student) model on the held-out TEST split.

Metrics: ROUGE-1/2/L, BLEU, BERTScore (F1), Semantic Similarity

Usage:
    python eval_bt5.py --model student_outputs_bt5/ewad_cpdp_20260516_121514/best_model
    python eval_bt5.py --model <path> --dataset <json> --samples 0   # 0 = all test samples
    python eval_bt5.py --model <path> --batch-size 16 --beams 4
    python eval_bt5.py --model <path> --skip-bertscore --skip-semsim  # ROUGE+BLEU only
"""

import os, sys, json, argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_bt5 import (
    SEED, TRAIN_SPLIT, VAL_SPLIT,
    MAX_INPUT_TOKENS, MAX_TARGET_TOKENS,
    EVAL_MAX_NEW_TOKENS, EVAL_NUM_BEAMS,
    DATASET_FILE, DATASET_TEXT_KEY, DATASET_SUMMARY_KEY,
)


def get_test_split(dataset_file, max_samples=None):
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    np.random.seed(SEED)
    data = [data[i] for i in np.random.permutation(len(data))]
    n = len(data)
    train_end = int(TRAIN_SPLIT * n)
    val_end   = train_end + int(VAL_SPLIT * n)
    test = data[val_end:]
    if max_samples and max_samples > 0:
        test = test[:max_samples]
    return test


# ── Extra metric helpers ──────────────────────────────────────────────────────

def _compute_bleu(predictions, references):
    """Corpus BLEU via sacrebleu with whitespace tokenisation."""
    import sacrebleu
    preds_tok = [' '.join(p.split()) for p in predictions]
    refs_tok  = [' '.join(r.split()) for r in references]
    result = sacrebleu.corpus_bleu(preds_tok, [refs_tok], tokenize='none', force=True)
    return result.score   # 0-100


def _compute_bertscore(predictions, references, device):
    """BERTScore (P/R/F1) using xlm-roberta-base — multilingual, handles Bangla."""
    from bert_score import score as bs_score
    print("\nComputing BERTScore (xlm-roberta-base) …")
    P, R, F1 = bs_score(
        predictions, references,
        model_type='xlm-roberta-base',
        device=str(device),
        batch_size=64,
        verbose=True,
    )
    return float(F1.mean()), float(P.mean()), float(R.mean())


# Models that need a text prefix for best results (e.g. "query: ", "passage: ")
_E5_MODELS = {'multilingual-e5-large', 'multilingual-e5-base', 'multilingual-e5-small'}


def _compute_semsim(predictions, references, device,
                    model_name='BAAI/bge-m3'):
    """
    Mean cosine similarity between prediction and reference embeddings.

    Default model: BAAI/bge-m3
      - 100+ languages including Bangla, SOTA on MTEB multilingual benchmarks,
        industry-grade dense embedding model.
    Other good options:
      - sentence-transformers/LaBSE   (Google, 109 langs, strong on Bengali)
      - intfloat/multilingual-e5-large (Microsoft SOTA, needs 'query: ' prefix)
    """
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F

    short_name = model_name.split('/')[-1]
    print(f"\nComputing Semantic Similarity ({short_name}) …")

    # e5 models require a task prefix for best performance
    add_prefix = any(m in model_name for m in _E5_MODELS)
    if add_prefix:
        predictions = ['query: ' + p for p in predictions]
        references  = ['query: ' + r for r in references]

    # bge-m3 is ~570 M params — use smaller batch to avoid OOM
    batch_size = 16 if 'bge-m3' in model_name else 64

    smodel = SentenceTransformer(model_name, device=str(device))
    pred_embs = smodel.encode(
        predictions, batch_size=batch_size, show_progress_bar=True,
        convert_to_tensor=True, normalize_embeddings=True,
    )
    ref_embs = smodel.encode(
        references, batch_size=batch_size, show_progress_bar=True,
        convert_to_tensor=True, normalize_embeddings=True,
    )
    # With normalized embeddings cosine sim == dot product
    cos_sims = (pred_embs * ref_embs).sum(dim=-1)
    return float(cos_sims.mean().cpu())


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(model_dir, dataset_file, max_samples, batch_size, num_beams,
             skip_bertscore=False, skip_semsim=False,
             semsim_model='BAAI/bge-m3'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")
    print(f"Model  : {model_dir}")
    print(f"Dataset: {dataset_file}")

    # ── Tokenizer + model ────────────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir, dtype=torch.bfloat16
    ).to(device).eval()

    # ── Test split ───────────────────────────────────────────────────────────
    test = get_test_split(dataset_file, max_samples)
    print(f"Test samples : {len(test)}\n")

    # ── Inference ────────────────────────────────────────────────────────────
    from rouge_score import rouge_scorer as _rs

    class _Tok:
        def tokenize(self, t): return t.split()

    scorer = _rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], tokenizer=_Tok())

    all_r1, all_r2, all_rl = [], [], []
    all_preds, all_refs = [], []

    pbar = tqdm(range(0, len(test), batch_size), desc="Generating")
    for i in pbar:
        batch = test[i : i + batch_size]
        texts = [s[DATASET_TEXT_KEY] for s in batch]
        refs  = [s[DATASET_SUMMARY_KEY] for s in batch]

        enc = tok(
            texts,
            max_length=MAX_INPUT_TOKENS,
            padding=True,
            truncation=True,
            return_tensors='pt',
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                num_beams=num_beams,
                early_stopping=True,
            )

        preds = tok.batch_decode(out_ids, skip_special_tokens=True)

        for pred, ref in zip(preds, refs):
            scores = scorer.score(ref, pred)
            all_r1.append(scores['rouge1'].fmeasure)
            all_r2.append(scores['rouge2'].fmeasure)
            all_rl.append(scores['rougeL'].fmeasure)
            all_preds.append(pred)
            all_refs.append(ref)

        pbar.set_postfix({'RL': f"{np.mean(all_rl):.4f}", 'n': len(all_rl)})

    # Free GPU memory before loading BERTScore / SentenceTransformer models
    model.cpu()
    del model
    torch.cuda.empty_cache()

    # ── BLEU (fast, CPU) ─────────────────────────────────────────────────────
    print("\nComputing BLEU …")
    bleu = _compute_bleu(all_preds, all_refs)

    # ── BERTScore ────────────────────────────────────────────────────────────
    bert_f1 = bert_p = bert_r = None
    if not skip_bertscore:
        bert_f1, bert_p, bert_r = _compute_bertscore(all_preds, all_refs, device)
        torch.cuda.empty_cache()

    # ── Semantic Similarity ──────────────────────────────────────────────────
    semsim = None
    if not skip_semsim:
        semsim = _compute_semsim(all_preds, all_refs, device, model_name=semsim_model)
        torch.cuda.empty_cache()

    # ── Aggregate ────────────────────────────────────────────────────────────
    results = {
        'model': model_dir,
        'dataset': dataset_file,
        'n_samples': len(test),
        'num_beams': num_beams,
        'rouge1': float(np.mean(all_r1)),
        'rouge2': float(np.mean(all_r2)),
        'rougeL': float(np.mean(all_rl)),
        'bleu': float(bleu),
    }
    if bert_f1 is not None:
        results['bertscore_f1']        = bert_f1
        results['bertscore_precision'] = bert_p
        results['bertscore_recall']    = bert_r
    if semsim is not None:
        results['semantic_similarity'] = semsim
        results['semsim_model'] = semsim_model

    print("\n── Test Results ──────────────────────────────────────")
    print(f"  ROUGE-1       : {results['rouge1']:.4f}")
    print(f"  ROUGE-2       : {results['rouge2']:.4f}")
    print(f"  ROUGE-L       : {results['rougeL']:.4f}")
    print(f"  BLEU          : {bleu:.2f}")
    if bert_f1 is not None:
        print(f"  BERTScore F1  : {bert_f1:.4f}  (P={bert_p:.4f}, R={bert_r:.4f})")
    if semsim is not None:
        short = semsim_model.split('/')[-1]
        print(f"  Sem. Sim.     : {semsim:.4f}  ({short})")
    print(f"  n             : {results['n_samples']}")
    print("─────────────────────────────────────────────────────\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_base  = model_dir if os.path.isdir(model_dir) else os.path.dirname(model_dir)
    res_path  = os.path.join(out_base, 'test_results.json')
    pred_path = os.path.join(out_base, 'test_predictions.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    sample_preds = [{'prediction': p, 'reference': r}
                    for p, r in zip(all_preds[:500], all_refs[:500])]
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(sample_preds, f, indent=2, ensure_ascii=False)
    print(f"Results saved     -> {res_path}")
    print(f"Predictions (500) -> {pred_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='Path to saved model directory (best_model/ or any HF dir)')
    parser.add_argument('--dataset', default=None,
                        help=f'Dataset JSON (default: DATASET_FILE from config_bt5.py)')
    parser.add_argument('--samples', type=int, default=0,
                        help='Number of test samples to evaluate (0 = all, default: 0)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--beams', type=int, default=EVAL_NUM_BEAMS)
    parser.add_argument('--skip-bertscore', action='store_true',
                        help='Skip BERTScore (saves ~15 min on 14k samples)')
    parser.add_argument('--skip-semsim', action='store_true',
                        help='Skip Semantic Similarity')
    parser.add_argument('--semsim-model', default='BAAI/bge-m3',
                        help='Sentence-transformer model for semantic similarity '
                             '(default: BAAI/bge-m3). '
                             'Other options: sentence-transformers/LaBSE, '
                             'intfloat/multilingual-e5-large')
    args = parser.parse_args()

    evaluate(
        model_dir=args.model,
        dataset_file=args.dataset or DATASET_FILE,
        max_samples=args.samples,
        batch_size=args.batch_size,
        num_beams=args.beams,
        skip_bertscore=args.skip_bertscore,
        skip_semsim=args.skip_semsim,
        semsim_model=args.semsim_model,
    )


if __name__ == '__main__':
    main()
