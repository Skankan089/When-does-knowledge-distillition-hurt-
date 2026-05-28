from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import load_jsonl, save_jsonl
from .features import (
    build_feature_rows,
    generate_teacher_summaries,
    semantic_similarity_scores,
    student_distribution_features,
    teacher_distribution_features,
)
from .train_gate import build_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score samples with the trained CHAD gate.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--gate-file", required=True)
    parser.add_argument("--teacher-model-name", required=True)
    parser.add_argument("--student-model-name", default=None,
                        help="Student model dir for student-side features. Required when gate uses student_* features.")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--output-file", default="data/gates/train_gate_scores.jsonl")
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--semantic-model-name", default=None)
    parser.add_argument("--no-generation-features", action="store_true")
    parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help="Impute features that are absent during scoring. By default this is an error.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def has_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not np.isnan(float(value))


def missing_feature_columns(rows: list[dict[str, object]], feature_columns: list[str]) -> list[str]:
    missing = []
    for column in feature_columns:
        if not any(has_number(row.get(column)) for row in rows):
            missing.append(column)
    return missing


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    artifact = joblib.load(args.gate_file)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    records = load_jsonl(args.input_file, limit=args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.teacher_model_name)
    teacher = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_model_name).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    # Load student model if gate uses student-side features
    student = None
    if args.student_model_name:
        student = AutoModelForSeq2SeqLM.from_pretrained(args.student_model_name).to(device)
        student.eval()
        for param in student.parameters():
            param.requires_grad_(False)

    needs_student = any(c.startswith("student_") for c in feature_columns)
    if needs_student and student is None:
        raise ValueError(
            "Gate uses student_* features but --student-model-name was not provided."
        )

    teacher_stats = teacher_distribution_features(
        teacher,
        tokenizer,
        records,
        device,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    student_stats = None
    if needs_student and student is not None:
        student_stats = student_distribution_features(
            student,
            teacher,
            tokenizer,
            records,
            device,
            batch_size=args.batch_size,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            desc="student stats",
        )
    teacher_summaries = None
    if not args.no_generation_features:
        teacher_summaries = generate_teacher_summaries(
            teacher,
            tokenizer,
            records,
            device,
            batch_size=args.batch_size,
            max_source_length=args.max_source_length,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
    semantic_scores = None
    if args.semantic_model_name and teacher_summaries:
        pairs = list(zip(teacher_summaries, [row["target"] for row in records]))
        semantic_scores = semantic_similarity_scores(args.semantic_model_name, pairs, device)

    rows = build_feature_rows(records, teacher_stats, teacher_summaries, semantic_scores, student_stats)
    missing = missing_feature_columns(rows, feature_columns)
    if missing and not args.allow_missing_features:
        raise ValueError(
            "The gate expects features that were not computed while scoring: "
            f"{missing}. Retrain the gate without label/debug-only features, pass the "
            "needed feature options such as --semantic-model-name, or explicitly pass "
            "--allow-missing-features to use imputation."
        )

    x = build_matrix(rows, feature_columns)
    # Regression gate: predict() returns continuous score directly.
    # Classification gate (legacy): use predict_proba.
    artifact_type = artifact.get("gate_type", "classifier")
    if artifact_type == "regressor":
        raw_scores = model.predict(x).astype(float)
        # Clip to [0,1] — grad_align is in [-1,1] but we only care about direction/magnitude
        scores = np.clip((raw_scores + 1.0) / 2.0, 0.0, 1.0)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)[:, 1]
    else:
        raw = model.decision_function(x)
        scores = 1 / (1 + np.exp(-raw))

    output_rows = []
    for record, feature_row, score in zip(records, rows, scores):
        output_rows.append(
            {
                "id": record["id"],
                "gate_score": float(score),
                **feature_row,
            }
        )

    save_jsonl(output_rows, args.output_file)
    summary = {
        "input_file": args.input_file,
        "output_file": args.output_file,
        "count": len(output_rows),
        "mean_gate_score": float(np.mean(scores)) if len(scores) else 0.0,
        "median_gate_score": float(np.median(scores)) if len(scores) else 0.0,
    }
    Path(args.output_file).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
