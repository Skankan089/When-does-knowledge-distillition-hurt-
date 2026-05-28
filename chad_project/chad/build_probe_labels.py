from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import Seq2SeqCollator, Seq2SeqRecordDataset, load_jsonl, move_to_device, save_jsonl
from .features import (
    build_feature_rows,
    generate_teacher_summaries,
    semantic_similarity_scores,
    student_distribution_features,
    teacher_distribution_features,
)
from .losses import kd_loss_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create counterfactual KD-helpfulness labels.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--student-model-name", default="csebuetnlp/banglat5_small")
    parser.add_argument("--teacher-model-name", required=True)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--output-file", default="data/probes/kd_usefulness_probe.jsonl")
    parser.add_argument(
        "--method",
        choices=["simulation", "grad_align"],
        default="grad_align",
        help=(
            "'simulation': one-step lookahead (restore weights, CE step, eval; restore, KD step, eval; Δ). "
            "'grad_align': cosine similarity between val CE gradient and per-sample KD gradient. "
            "grad_align is faster, lower-variance, and directly citable via TracIn."
        ),
    )
    parser.add_argument("--probe-size", type=int, default=1000)
    parser.add_argument("--val-probe-size", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=1e-5,
                        help="Learning rate for simulation mode only.")
    parser.add_argument("--lambda-kd", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--semantic-model-name", default=None)
    parser.add_argument("--no-generation-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Gradient-alignment helpers ────────────────────────────────────────────────

def _flat_grad(model: Any) -> torch.Tensor:
    """Return a 1-D tensor of all current .grad values (params that have grad)."""
    parts = [p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None]
    if not parts:
        raise RuntimeError("No gradients found on model parameters.")
    return torch.cat(parts)


def compute_val_gradient(
    student: Any,
    val_loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Accumulate CE gradients over the val probe set; return mean flat gradient vector.

    The vector is computed once and reused for all probe samples, so mini-val
    sampling noise affects all labels equally (rather than each sample independently).
    """
    student.train()
    student.zero_grad(set_to_none=True)
    n_batches = 0
    for batch in val_loader:
        batch = move_to_device(batch, device)
        outputs = student(**batch)
        outputs.loss.backward()
        n_batches += 1
    if n_batches > 1:
        with torch.no_grad():
            for param in student.parameters():
                if param.grad is not None:
                    param.grad.div_(n_batches)
    g_val = _flat_grad(student)
    student.zero_grad(set_to_none=True)
    return g_val


def compute_kd_gradient(
    student: Any,
    teacher: Any,
    batch: dict[str, torch.Tensor],
    lambda_kd: float,
    temperature: float,
) -> torch.Tensor:
    """Gradient of the KD loss component only for a single training sample."""
    student.train()
    student.zero_grad(set_to_none=True)
    outputs = student(**batch)
    with torch.no_grad():
        teacher_outputs = teacher(**batch)
    kd = kd_loss_mean(outputs.logits, teacher_outputs.logits, batch["labels"], temperature)
    (lambda_kd * kd).backward()
    g_kd = _flat_grad(student)
    student.zero_grad(set_to_none=True)
    return g_kd


def cosine_similarity_score(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity in [-1, 1].  Positive → vectors point the same direction."""
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b / denom).item())


def eval_ce_loss(
    model: Any,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            outputs = model(**batch)
            batch_size = batch["input_ids"].size(0)
            total_loss += float(outputs.loss.detach().cpu()) * batch_size
            total_examples += batch_size
    return total_loss / max(total_examples, 1)


def manual_sgd_step(model: Any, lr: float) -> None:
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.add_(param.grad, alpha=-lr)
    model.zero_grad(set_to_none=True)


def train_one_probe_step(
    student: Any,
    teacher: Any,
    batch: dict[str, torch.Tensor],
    use_kd: bool,
    lr: float,
    lambda_kd: float,
    temperature: float,
) -> float:
    student.train()
    student.zero_grad(set_to_none=True)
    outputs = student(**batch)
    loss = outputs.loss

    if use_kd:
        teacher.eval()
        with torch.no_grad():
            teacher_outputs = teacher(**batch)
        kd = kd_loss_mean(outputs.logits, teacher_outputs.logits, batch["labels"], temperature)
        loss = loss + lambda_kd * kd

    loss.backward()
    manual_sgd_step(student, lr)
    return float(loss.detach().cpu())


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    tokenizer_name = args.tokenizer_name or args.student_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    student = AutoModelForSeq2SeqLM.from_pretrained(args.student_model_name).to(device)
    teacher = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_model_name).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    train_records = load_jsonl(args.train_file)
    val_records = load_jsonl(args.val_file)
    rng = random.Random(args.seed)
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    probe_records = train_records[: args.probe_size]
    val_probe_records = val_records[: args.val_probe_size]

    collator = Seq2SeqCollator(tokenizer, args.max_source_length, args.max_target_length)
    val_loader = DataLoader(
        Seq2SeqRecordDataset(val_probe_records),
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    print("Computing teacher-side probe features...")
    teacher_stats = teacher_distribution_features(
        teacher,
        tokenizer,
        probe_records,
        device,
        batch_size=args.feature_batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    print("Computing student-side probe features...")
    student_stats = student_distribution_features(
        student,
        teacher,
        tokenizer,
        probe_records,
        device,
        batch_size=args.feature_batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    teacher_summaries = None
    if not args.no_generation_features:
        teacher_summaries = generate_teacher_summaries(
            teacher,
            tokenizer,
            probe_records,
            device,
            batch_size=args.feature_batch_size,
            max_source_length=args.max_source_length,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
    semantic_scores = None
    if args.semantic_model_name and teacher_summaries:
        pairs = list(zip(teacher_summaries, [row["target"] for row in probe_records]))
        semantic_scores = semantic_similarity_scores(args.semantic_model_name, pairs, device)

    feature_rows = build_feature_rows(probe_records, teacher_stats, teacher_summaries, semantic_scores, student_stats)

    output_rows: list[dict[str, Any]] = []

    if args.method == "grad_align":
        # ── Gradient-alignment labeling ────────────────────────────────────────
        # Step 1: compute mean val CE gradient once (shared across all probe samples).
        # This eliminates per-sample val-batch sampling variance entirely.
        print("Computing validation gradient (grad_align method)...")
        g_val = compute_val_gradient(student, val_loader, device)
        g_val_norm = g_val.norm().item()
        print(f"  val gradient L2 norm: {g_val_norm:.4f}")

        for idx, record in enumerate(tqdm(probe_records, desc="gradient alignment probes")):
            batch = move_to_device(collator([record]), device)

            # Step 2: gradient of KD loss for this sample.
            g_kd = compute_kd_gradient(
                student,
                teacher,
                batch,
                lambda_kd=args.lambda_kd,
                temperature=args.temperature,
            )

            # Step 3: cosine(g_val, g_kd).
            # Positive → KD gradient aligns with reducing val loss → helpful.
            # Negative → KD gradient opposes val improvement → harmful.
            score = cosine_similarity_score(g_val, g_kd)

            row = {
                **feature_rows[idx],
                "source": record["source"],
                "target": record["target"],
                "grad_align_score": score,
                "kd_useful": int(score > args.threshold),
            }
            output_rows.append(row)

    else:
        # ── One-step simulation labeling (original method) ─────────────────────
        base_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}

        def restore_student() -> None:
            student.load_state_dict(base_state, strict=True)
            student.to(device)
            student.zero_grad(set_to_none=True)

        for idx, record in enumerate(tqdm(probe_records, desc="simulation probes")):
            batch = move_to_device(collator([record]), device)

            restore_student()
            train_one_probe_step(
                student, teacher, batch,
                use_kd=False, lr=args.probe_lr,
                lambda_kd=args.lambda_kd, temperature=args.temperature,
            )
            ce_step_val_loss = eval_ce_loss(student, val_loader, device)

            restore_student()
            train_one_probe_step(
                student, teacher, batch,
                use_kd=True, lr=args.probe_lr,
                lambda_kd=args.lambda_kd, temperature=args.temperature,
            )
            kd_step_val_loss = eval_ce_loss(student, val_loader, device)

            delta = ce_step_val_loss - kd_step_val_loss
            row = {
                **feature_rows[idx],
                "source": record["source"],
                "target": record["target"],
                "ce_step_val_loss": ce_step_val_loss,
                "kd_step_val_loss": kd_step_val_loss,
                "delta_val_loss": delta,
                "kd_useful": int(delta > args.threshold),
            }
            output_rows.append(row)

    save_jsonl(output_rows, args.output_file)
    helpful = sum(row["kd_useful"] for row in output_rows)
    summary = {
        "output_file": args.output_file,
        "method": args.method,
        "probe_size": len(output_rows),
        "helpful": helpful,
        "harmful_or_neutral": len(output_rows) - helpful,
        "helpful_ratio": helpful / max(len(output_rows), 1),
        "threshold": args.threshold,
    }
    Path(args.output_file).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
