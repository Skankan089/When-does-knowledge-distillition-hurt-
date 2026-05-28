from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import read_bansum_records, save_jsonl, split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test JSONL splits for BanSum.")
    parser.add_argument("--data-file", default="bansum_filtered_20k.json")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--text-field", default="main")
    parser.add_argument("--summary-field", default="sum1")
    parser.add_argument("--id-field", default="ID")
    parser.add_argument("--val-size", type=float, default=1000)
    parser.add_argument("--test-size", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_bansum_records(
        args.data_file,
        text_field=args.text_field,
        summary_field=args.summary_field,
        id_field=args.id_field,
        limit=args.limit,
    )
    train, val, test = split_records(records, args.val_size, args.test_size, args.seed)

    output_dir = Path(args.output_dir)
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(val, output_dir / "val.jsonl")
    save_jsonl(test, output_dir / "test.jsonl")

    metadata = {
        "data_file": args.data_file,
        "text_field": args.text_field,
        "summary_field": args.summary_field,
        "seed": args.seed,
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
