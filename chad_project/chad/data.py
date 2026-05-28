from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


def _as_split_size(value: float | int, total: int) -> int:
    if isinstance(value, float) and 0 < value < 1:
        return int(round(total * value))
    return int(value)


def read_bansum_records(
    path: str | Path,
    text_field: str = "main",
    summary_field: str = "sum1",
    id_field: str = "ID",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    records: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        source = str(item.get(text_field, "")).strip()
        target = str(item.get(summary_field, "")).strip()
        if not source or not target:
            continue
        records.append(
            {
                "id": str(item.get(id_field, idx)),
                "source": source,
                "target": target,
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    return records


def save_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_records(
    records: list[dict[str, Any]],
    val_size: float | int,
    test_size: float | int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    n_val = _as_split_size(val_size, total)
    n_test = _as_split_size(test_size, total)
    if n_val + n_test >= total:
        raise ValueError("Validation and test sizes leave no training examples.")

    val = shuffled[:n_val]
    test = shuffled[n_val : n_val + n_test]
    train = shuffled[n_val + n_test :]
    return train, val, test


def attach_gate_scores(
    records: list[dict[str, Any]],
    gate_scores_file: str | Path,
    default_score: float = 0.0,
) -> list[dict[str, Any]]:
    score_rows = load_jsonl(gate_scores_file)
    scores = {str(row["id"]): float(row["gate_score"]) for row in score_rows}
    merged = []
    for record in records:
        item = dict(record)
        item["gate_score"] = scores.get(str(item["id"]), default_score)
        merged.append(item)
    return merged


class Seq2SeqRecordDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


@dataclass
class Seq2SeqCollator:
    tokenizer: Any
    max_source_length: int = 512
    max_target_length: int = 128

    def _encode_targets(self, targets: list[str]) -> dict[str, torch.Tensor]:
        try:
            return self.tokenizer(
                text_target=targets,
                max_length=self.max_target_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
        except TypeError:
            with self.tokenizer.as_target_tokenizer():
                return self.tokenizer(
                    targets,
                    max_length=self.max_target_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        sources = [item["source"] for item in batch]
        targets = [item["target"] for item in batch]

        inputs = self.tokenizer(
            sources,
            max_length=self.max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = self._encode_targets(targets)["input_ids"]
        labels = labels.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        if any("gate_score" in item for item in batch):
            inputs["gate_weight"] = torch.tensor(
                [float(item.get("gate_score", 0.0)) for item in batch],
                dtype=torch.float,
            )

        return inputs


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
