from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .data import Seq2SeqCollator, move_to_device


def simple_tokens(text: str) -> list[str]:
    return [tok for tok in text.strip().split() if tok]


def novelty_ratio(source: str, target: str) -> float:
    source_set = set(simple_tokens(source))
    target_tokens = simple_tokens(target)
    if not target_tokens:
        return 0.0
    novel = sum(1 for tok in target_tokens if tok not in source_set)
    return novel / len(target_tokens)


def rouge_n_f1(candidate: str, reference: str, n: int) -> float:
    cand = simple_tokens(candidate)
    ref = simple_tokens(reference)
    if len(cand) < n or len(ref) < n:
        return 0.0
    cand_ngrams: dict[tuple, int] = {}
    for i in range(len(cand) - n + 1):
        g = tuple(cand[i : i + n])
        cand_ngrams[g] = cand_ngrams.get(g, 0) + 1
    ref_ngrams: dict[tuple, int] = {}
    for i in range(len(ref) - n + 1):
        g = tuple(ref[i : i + n])
        ref_ngrams[g] = ref_ngrams.get(g, 0) + 1
    overlap = sum(min(cand_ngrams.get(g, 0), cnt) for g, cnt in ref_ngrams.items())
    precision = overlap / (len(cand) - n + 1)
    recall = overlap / (len(ref) - n + 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(candidate: str, reference: str) -> float:
    cand = simple_tokens(candidate)
    ref = simple_tokens(reference)
    if not cand or not ref:
        return 0.0

    prev = [0] * (len(ref) + 1)
    for c_tok in cand:
        curr = [0]
        for j, r_tok in enumerate(ref, start=1):
            if c_tok == r_tok:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[j]))
        prev = curr

    lcs = prev[-1]
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def basic_text_features(source: str, target: str) -> dict[str, float]:
    source_words = simple_tokens(source)
    target_words = simple_tokens(target)
    source_len = len(source_words)
    target_len = len(target_words)
    return {
        "source_chars": float(len(source)),
        "target_chars": float(len(target)),
        "source_words": float(source_len),
        "target_words": float(target_len),
        "compression_ratio": float(target_len / max(source_len, 1)),
        "novelty_ratio": float(novelty_ratio(source, target)),
    }


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def student_distribution_features(
    student_model: Any,
    teacher_model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 4,
    max_source_length: int = 512,
    max_target_length: int = 128,
    desc: str = "student stats",
) -> list[dict[str, float]]:
    """Per-sample student CE loss, entropy, and student-teacher KL divergence."""
    collator = Seq2SeqCollator(tokenizer, max_source_length, max_target_length)
    stats: list[dict[str, float]] = []
    student_model.eval()
    teacher_model.eval()

    total = math.ceil(len(records) / batch_size) if records else 0
    with torch.no_grad():
        for chunk in tqdm(batched(records, batch_size), total=total, desc=desc):
            batch = move_to_device(collator(chunk), device)
            labels = batch["labels"]
            mask = labels.ne(-100)

            student_out = student_model(**batch)
            teacher_out = teacher_model(**batch)

            s_logits = student_out.logits.float()
            t_logits = teacher_out.logits.float()

            # Per-token CE loss of student
            shift_labels = labels.clone()
            shift_labels[~mask] = 0
            s_log_probs = F.log_softmax(s_logits, dim=-1)
            s_ce = -s_log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T)

            # Student entropy
            s_probs = F.softmax(s_logits, dim=-1)
            s_entropy = -(s_probs * s_log_probs).sum(dim=-1)  # (B, T)

            # Student–teacher KL: KL(student || teacher) per token
            t_log_probs = F.log_softmax(t_logits, dim=-1)
            kl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)  # (B, T)

            for i in range(labels.size(0)):
                row_mask = mask[i].float()
                denom = row_mask.sum().clamp_min(1).item()
                stats.append({
                    "student_ce_loss": float((s_ce[i] * row_mask).sum().item() / denom),
                    "student_entropy": float((s_entropy[i] * row_mask).sum().item() / denom),
                    "student_teacher_kl": float((kl[i] * row_mask).sum().item() / denom),
                })
    return stats


def teacher_distribution_features(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 4,
    max_source_length: int = 512,
    max_target_length: int = 128,
    desc: str = "teacher stats",
) -> list[dict[str, float]]:
    collator = Seq2SeqCollator(tokenizer, max_source_length, max_target_length)
    stats: list[dict[str, float]] = []
    model.eval()

    total = math.ceil(len(records) / batch_size) if records else 0
    with torch.no_grad():
        for chunk in tqdm(batched(records, batch_size), total=total, desc=desc):
            batch = move_to_device(collator(chunk), device)
            labels = batch["labels"]
            outputs = model(**batch)
            probs = F.softmax(outputs.logits.float(), dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            top2 = probs.topk(k=2, dim=-1).values
            max_prob = top2[..., 0]
            margin = top2[..., 0] - top2[..., 1]
            mask = labels.ne(-100)

            for row_idx in range(labels.size(0)):
                row_mask = mask[row_idx]
                denom = row_mask.sum().clamp_min(1)
                stats.append(
                    {
                        "teacher_entropy": float((entropy[row_idx] * row_mask).sum().item() / denom.item()),
                        "teacher_max_prob": float((max_prob[row_idx] * row_mask).sum().item() / denom.item()),
                        "teacher_margin": float((margin[row_idx] * row_mask).sum().item() / denom.item()),
                    }
                )
    return stats


def generate_teacher_summaries(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 4,
    max_source_length: int = 512,
    max_new_tokens: int = 128,
    num_beams: int = 4,
    desc: str = "teacher summaries",
) -> list[str]:
    summaries: list[str] = []
    model.eval()

    total = math.ceil(len(records) / batch_size) if records else 0
    with torch.no_grad():
        for chunk in tqdm(batched(records, batch_size), total=total, desc=desc):
            texts = [item["source"] for item in chunk]
            enc = tokenizer(
                texts,
                max_length=max_source_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = move_to_device(enc, device)
            generated = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            summaries.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return summaries


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def semantic_similarity_scores(
    model_name: str,
    pairs: list[tuple[str, str]],
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 256,
) -> list[float]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    scores: list[float] = []

    total = math.ceil(len(pairs) / batch_size) if pairs else 0
    with torch.no_grad():
        for chunk in tqdm(batched(pairs, batch_size), total=total, desc="semantic agreement"):
            left = [item[0] for item in chunk]
            right = [item[1] for item in chunk]
            left_enc = tokenizer(left, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            right_enc = tokenizer(right, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            left_enc = move_to_device(left_enc, device)
            right_enc = move_to_device(right_enc, device)
            left_out = model(**left_enc)
            right_out = model(**right_enc)
            left_vec = _mean_pool(left_out.last_hidden_state, left_enc["attention_mask"])
            right_vec = _mean_pool(right_out.last_hidden_state, right_enc["attention_mask"])
            sims = F.cosine_similarity(left_vec, right_vec, dim=-1)
            scores.extend(float(x) for x in sims.detach().cpu())
    return scores


def build_feature_rows(
    records: list[dict[str, Any]],
    teacher_stats: list[dict[str, float]] | None = None,
    teacher_summaries: list[str] | None = None,
    semantic_scores: list[float] | None = None,
    student_stats: list[dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    teacher_stats = teacher_stats or [{} for _ in records]
    teacher_summaries = teacher_summaries or ["" for _ in records]
    semantic_scores = semantic_scores or [math.nan for _ in records]
    student_stats = student_stats or [{} for _ in records]

    for record, tstats, teacher_summary, semantic_score, sstats in zip(
        records, teacher_stats, teacher_summaries, semantic_scores, student_stats
    ):
        row: dict[str, Any] = {
            "id": record["id"],
            **basic_text_features(record["source"], record["target"]),
            **tstats,
            **sstats,
            "gold_teacher_rouge_l": rouge_l_f1(teacher_summary, record["target"]) if teacher_summary else math.nan,
            "semantic_agreement": float(semantic_score) if not np.isnan(semantic_score) else math.nan,
        }
        rows.append(row)
    return rows
