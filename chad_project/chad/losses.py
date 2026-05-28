from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_loss_per_sample(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Student and teacher logits must have the same shape. "
            "This implementation assumes a shared tokenizer/vocabulary."
        )

    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    mask = labels.ne(-100)
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    token_kl = token_kl * (temperature**2)

    token_counts = mask.sum(dim=1).clamp_min(1)
    return (token_kl * mask).sum(dim=1) / token_counts


def kd_loss_mean(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    return kd_loss_per_sample(student_logits, teacher_logits, labels, temperature).mean()
