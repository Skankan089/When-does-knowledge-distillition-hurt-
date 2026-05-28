from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sacrebleu import corpus_bleu
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import load_jsonl, move_to_device, save_jsonl
from .features import batched, rouge_l_f1, rouge_n_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate summaries and evaluate ROUGE-L/BLEU.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bertscore-model", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    records = load_jsonl(args.test_file, limit=args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for chunk in tqdm(list(batched(records, args.batch_size)), desc="generating"):
            enc = tokenizer(
                [row["source"] for row in chunk],
                max_length=args.max_source_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = move_to_device(enc, device)
            generated = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for record, prediction in zip(chunk, texts):
                predictions.append(
                    {
                        "id": record["id"],
                        "source": record["source"],
                        "target": record["target"],
                        "prediction": prediction,
                    }
                )

    refs = [row["target"] for row in predictions]
    preds = [row["prediction"] for row in predictions]
    rouge_1 = float(np.mean([rouge_n_f1(pred, ref, 1) for pred, ref in zip(preds, refs)])) if preds else 0.0
    rouge_2 = float(np.mean([rouge_n_f1(pred, ref, 2) for pred, ref in zip(preds, refs)])) if preds else 0.0
    rouge_l = float(np.mean([rouge_l_f1(pred, ref) for pred, ref in zip(preds, refs)])) if preds else 0.0
    bleu = float(corpus_bleu(preds, [refs]).score) if preds else 0.0
    metrics = {
        "model_dir": args.model_dir,
        "test_file": args.test_file,
        "count": len(predictions),
        "rouge_1": rouge_1,
        "rouge_2": rouge_2,
        "rouge_l": rouge_l,
        "bleu": bleu,
    }

    if args.bertscore_model:
        try:
            from bert_score import score as bert_score

            _, _, f1 = bert_score(preds, refs, model_type=args.bertscore_model, lang="bn", verbose=True)
            metrics["bertscore_f1"] = float(f1.mean().item())
        except Exception as exc:
            metrics["bertscore_error"] = str(exc)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(predictions, output_dir / "predictions.jsonl")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
