from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, EarlyStoppingCallback, EvalPrediction, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_callback import ProgressCallback
from transformers.trainer_utils import get_last_checkpoint


class RougeProgressCallback(ProgressCallback):
    """ProgressCallback that shows the latest eval ROUGE-L in the tqdm bar postfix."""

    def __init__(self) -> None:
        super().__init__()
        self._best_rouge_l: float = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs=logs, **kwargs)
        if state.is_local_process_zero and self.training_bar is not None and logs:
            rl = logs.get("eval_rouge_l")
            if rl is not None:
                if rl > self._best_rouge_l:
                    self._best_rouge_l = rl
                self.training_bar.set_postfix(
                    rl=f"{rl:.4f}",
                    best=f"{self._best_rouge_l:.4f}",
                )


def _trainer_tokenizer_kwarg(tokenizer: Any) -> dict[str, Any]:
    """Return the correct keyword for the tokenizer argument across transformers versions.

    transformers >= 4.46 uses ``processing_class``; older versions use ``tokenizer``.
    """
    sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in sig.parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}

from .data import (
    Seq2SeqCollator,
    Seq2SeqRecordDataset,
    attach_gate_scores,
    load_jsonl,
)
from .features import rouge_l_f1, rouge_n_f1
from .losses import kd_loss_per_sample


def make_compute_metrics(tokenizer: Any):
    """Return a compute_metrics fn that reports ROUGE-1/2/L on generated text."""
    pad_id = tokenizer.pad_token_id

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        predictions, labels = eval_pred
        # predictions from generate() may be padded with -100; replace before decode
        predictions = np.where(predictions < 0, pad_id, predictions)
        labels = np.where(labels != -100, labels, pad_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        r1 = float(np.mean([rouge_n_f1(p, r, 1) for p, r in zip(decoded_preds, decoded_labels)]))
        r2 = float(np.mean([rouge_n_f1(p, r, 2) for p, r in zip(decoded_preds, decoded_labels)]))
        rl = float(np.mean([rouge_l_f1(p, r) for p, r in zip(decoded_preds, decoded_labels)]))
        return {"rouge_1": r1, "rouge_2": r2, "rouge_l": rl}

    return compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CE, KD, or CHAD-gated KD student.")
    parser.add_argument("--mode", choices=["ce", "kd", "chad"], required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--gate-scores-file", default=None)
    parser.add_argument("--model-name", default="csebuetnlp/banglat5_small")
    parser.add_argument("--teacher-model-name", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lambda-kd", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop training if val rouge_l does not improve for this many evals. 0 = disabled.")
    return parser.parse_args()


class GatedKDTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args: Any,
        teacher_model: Any | None = None,
        mode: str = "ce",
        lambda_kd: float = 0.5,
        temperature: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.mode = mode
        self.lambda_kd = lambda_kd
        self.temperature = temperature

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        gate_weight = inputs.pop("gate_weight", None)
        outputs = model(**inputs)
        loss = outputs.loss

        if self.mode in {"kd", "chad"}:
            if self.teacher_model is None:
                raise ValueError("KD modes require a teacher model.")
            teacher_device = next(self.teacher_model.parameters()).device
            student_device = outputs.logits.device
            if teacher_device != student_device:
                self.teacher_model.to(student_device)
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
            per_sample_kd = kd_loss_per_sample(
                outputs.logits,
                teacher_outputs.logits,
                inputs["labels"],
                temperature=self.temperature,
            )
            if self.mode == "chad":
                if gate_weight is None:
                    # Eval batches from the val set have no gate_weight; use uniform weights.
                    weights = torch.ones_like(per_sample_kd)
                else:
                    weights = gate_weight.to(per_sample_kd.device).float().clamp(0.0, 1.0)
            else:
                weights = torch.ones_like(per_sample_kd)
            kd_loss = (per_sample_kd * weights).mean()
            loss = loss + self.lambda_kd * kd_loss

        return (loss, outputs) if return_outputs else loss


def training_args_from_namespace(args: argparse.Namespace) -> Seq2SeqTrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "predict_with_generate": True,
        "generation_max_length": args.max_target_length,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "remove_unused_columns": False,
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "rouge_l",
        "greater_is_better": True,
        "logging_strategy": "steps",
        "save_strategy": "steps",
    }
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    return Seq2SeqTrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    if args.mode == "chad" and not args.gate_scores_file:
        raise ValueError("--gate-scores-file is required for CHAD mode.")
    if args.mode in {"kd", "chad"} and not args.teacher_model_name:
        raise ValueError("--teacher-model-name is required for KD and CHAD modes.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name)
    student = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    teacher = None
    if args.mode in {"kd", "chad"}:
        teacher = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_model_name)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

    train_records = load_jsonl(args.train_file)
    val_records = load_jsonl(args.val_file)
    if args.gate_scores_file:
        train_records = attach_gate_scores(train_records, args.gate_scores_file)

    collator = Seq2SeqCollator(tokenizer, args.max_source_length, args.max_target_length)
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = GatedKDTrainer(
        model=student,
        args=training_args_from_namespace(args),
        train_dataset=Seq2SeqRecordDataset(train_records),
        eval_dataset=Seq2SeqRecordDataset(val_records),
        data_collator=collator,
        compute_metrics=make_compute_metrics(tokenizer),
        **_trainer_tokenizer_kwarg(tokenizer),
        teacher_model=teacher,
        mode=args.mode,
        lambda_kd=args.lambda_kd,
        temperature=args.temperature,
        callbacks=callbacks or None,
    )
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(RougeProgressCallback())

    last_checkpoint = get_last_checkpoint(args.output_dir) if Path(args.output_dir).is_dir() else None
    if last_checkpoint:
        print(f"[INFO] Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
