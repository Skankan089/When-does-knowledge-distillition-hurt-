"""
EWAD + CPDP Loss Functions
============================
Entropy-Weighted Agreement-Aware Distillation (EWAD) with
Capacity-Proportional Divergence Preservation (CPDP)

Novel dual-teacher knowledge distillation losses for Bengali summarization.

Mathematical Formulation:
-------------------------

EWAD Loss (per token t):
    C_i^t = 1 - H(p_i^t) / log|V|              (teacher confidence)
    w_32B^t = softmax(C_32B^t / tau_w)          (confidence-proportional weight)
    A_t = 1 - JSD(p_32B^t || p_14B^t)           (teacher agreement)
    λ_t = sigmoid(k * (A_t - δ))                (agreement gate)
    
    L_EWAD = (1/T) Σ_t [
        λ_t * (w_32B^t * KL(p_32B^t || p_S^t) + w_14B^t * KL(p_14B^t || p_S^t))
        + (1 - λ_t) * CE(y_t*, p_S^t)
    ]

CPDP Loss:
    Δ* = KL(p_32B || p_14B)                      (teacher mutual divergence, unnormalized)
    L_CPDP = |KL(p_32B||p_S)/H(p_S) - KL(p_14B||p_S)/H(p_S) - Δ*|²

Combined:
    L_total = L_EWAD + μ * L_CPDP

All operations use standard PyTorch — no custom CUDA kernels.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    EWAD_TAU_W, EWAD_K, EWAD_DELTA, EWAD_ENTROPY_EPS,
    CPDP_MU, CPDP_EPS, LOGIT_TOP_K
)


def align_vocab_size(student_logprobs, teacher_logprobs):
    """
    Align vocab dimension between student and teacher tensors.
    Truncates the larger to match the smaller (extra tokens are rare special tokens).
    After truncation, renormalizes the log-probabilities.
    """
    sv = student_logprobs.size(-1)
    tv = teacher_logprobs.size(-1)
    if sv == tv:
        return student_logprobs, teacher_logprobs
    min_v = min(sv, tv)
    if sv > min_v:
        student_logprobs = student_logprobs[..., :min_v]
        student_logprobs = student_logprobs - torch.logsumexp(student_logprobs, dim=-1, keepdim=True)
    if tv > min_v:
        teacher_logprobs = teacher_logprobs[..., :min_v]
        teacher_logprobs = teacher_logprobs - torch.logsumexp(teacher_logprobs, dim=-1, keepdim=True)
    return student_logprobs, teacher_logprobs


def _masked_mean(values, mask):
    if mask is None:
        return values.mean()
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def is_sparse_teacher(teacher_logprobs):
    return isinstance(teacher_logprobs, dict) and {
        "indices", "logprobs", "mask"
    }.issubset(teacher_logprobs.keys())


def _sparse_components(teacher_logprobs, vocab_size, device):
    """
    Reconstruct the same distribution previously produced by
    batch_sparse_to_dense(), but without allocating (B, T, V).

    Saved top-k probabilities are kept as-is; all unseen tokens receive a
    uniform 1/V prior, then the distribution is renormalized.
    """
    indices = teacher_logprobs["indices"].to(device=device, dtype=torch.long)
    raw_logprobs = teacher_logprobs["logprobs"].to(device=device, dtype=torch.float32)
    mask = teacher_logprobs["mask"].to(device=device, dtype=torch.bool)
    valid = mask & (indices >= 0) & (indices < vocab_size)
    safe_indices = indices.clamp(min=0, max=max(vocab_size - 1, 0))

    neg_inf = torch.full_like(raw_logprobs, -float("inf"))
    masked_raw = torch.where(valid, raw_logprobs, neg_inf)
    topk_mass = torch.exp(masked_raw).sum(dim=-1)

    topk_count = valid.sum(dim=-1).to(torch.float32)
    tail_count = (float(vocab_size) - topk_count).clamp(min=0.0)
    tail_mass = tail_count / float(vocab_size)

    normalizer = (topk_mass + tail_mass).clamp(min=1e-30)
    log_z = normalizer.log()

    topk_logprobs = raw_logprobs - log_z.unsqueeze(-1)
    topk_probs = torch.where(valid, topk_logprobs.exp(), torch.zeros_like(topk_logprobs))
    tail_logprob = -math.log(vocab_size) - log_z
    tail_prob = tail_logprob.exp()

    return {
        "indices": safe_indices,
        "valid": valid,
        "topk_logprobs": topk_logprobs,
        "topk_probs": topk_probs,
        "tail_count": tail_count,
        "tail_logprob": tail_logprob,
        "tail_prob": tail_prob,
    }


def sparse_teacher_entropy(teacher_logprobs, vocab_size, log_vocab, eps):
    comp = _sparse_components(
        teacher_logprobs,
        vocab_size=vocab_size,
        device=teacher_logprobs["logprobs"].device,
    )
    topk_entropy = -(comp["topk_probs"] * comp["topk_logprobs"]).sum(dim=-1)
    tail_entropy = -(comp["tail_count"] * comp["tail_prob"] * comp["tail_logprob"])
    return (topk_entropy + tail_entropy).clamp(min=eps, max=log_vocab)


def sparse_teacher_kl_to_student(teacher_logprobs, student_logprobs):
    """
    Exact KL against the old sparse+uniform-tail teacher approximation, without
    constructing the dense teacher tensor.
    """
    vocab_size = student_logprobs.size(-1)
    comp = _sparse_components(
        teacher_logprobs,
        vocab_size=vocab_size,
        device=student_logprobs.device,
    )

    student_at_topk = torch.gather(student_logprobs, dim=-1, index=comp["indices"])
    topk_kl = (
        comp["topk_probs"]
        * (comp["topk_logprobs"] - student_at_topk)
        * comp["valid"].to(student_logprobs.dtype)
    ).sum(dim=-1)

    # Tail contribution: every token not present in top-k has identical teacher
    # probability. We only need the sum of student log-probs over those tokens.
    student_sum_all = student_logprobs.sum(dim=-1)
    student_sum_topk = (
        student_at_topk * comp["valid"].to(student_logprobs.dtype)
    ).sum(dim=-1)
    student_sum_tail = student_sum_all - student_sum_topk
    tail_kl = comp["tail_prob"] * (
        comp["tail_count"] * comp["tail_logprob"] - student_sum_tail
    )

    return (topk_kl + tail_kl).clamp(min=0.0)


def dense_teacher_kl_to_student(teacher_logprobs, student_logprobs):
    student_logprobs, teacher_logprobs = align_vocab_size(student_logprobs, teacher_logprobs)
    probs = teacher_logprobs.exp()
    return (probs * (teacher_logprobs - student_logprobs)).sum(dim=-1).clamp(min=0.0)


def teacher_kl_to_student(teacher_logprobs, student_logprobs):
    if is_sparse_teacher(teacher_logprobs):
        return sparse_teacher_kl_to_student(teacher_logprobs, student_logprobs)
    return dense_teacher_kl_to_student(teacher_logprobs, student_logprobs)


def sparse_teacher_kl_between(source_logprobs, target_logprobs, vocab_size):
    """
    KL(source || target) using the sparse+uniform-tail approximation.

    This is used for CPDP only. It intentionally avoids dense vocab tensors;
    for non-overlapping source top-k tokens the target contributes its uniform
    tail probability.
    """
    device = source_logprobs["logprobs"].device
    source = _sparse_components(source_logprobs, vocab_size=vocab_size, device=device)
    target = _sparse_components(target_logprobs, vocab_size=vocab_size, device=device)

    B, T, _ = source["indices"].shape
    out = torch.zeros((B, T), device=device, dtype=torch.float32)

    # B is 1 in the full run. This small Python loop avoids allocating
    # per-token dense maps of size vocab.
    for b in range(B):
        for t in range(T):
            s_valid = source["valid"][b, t]
            s_ids = source["indices"][b, t][s_valid]
            s_lp = source["topk_logprobs"][b, t][s_valid]
            s_prob = source["topk_probs"][b, t][s_valid]
            s_id_set = set(s_ids.tolist())

            t_valid = target["valid"][b, t]
            t_ids = target["indices"][b, t][t_valid]
            t_lp = target["topk_logprobs"][b, t][t_valid]

            target_lookup = {int(tok): lp for tok, lp in zip(t_ids.tolist(), t_lp)}
            terms = []

            if s_ids.numel() > 0:
                target_for_source = torch.stack([
                    target_lookup.get(int(tok), target["tail_logprob"][b, t])
                    for tok in s_ids.tolist()
                ])
                terms.append(
                    (s_prob * (s_lp - target_for_source.to(device=device, dtype=s_lp.dtype))).sum()
                )

            # Source-tail tokens that are explicit top-k tokens in the target.
            target_only_lps = [
                lp for tok, lp in target_lookup.items()
                if tok not in s_id_set
            ]
            if target_only_lps:
                target_only_lp = torch.stack(target_only_lps).to(device=device, dtype=torch.float32)
                terms.append(
                    (source["tail_prob"][b, t] * (source["tail_logprob"][b, t] - target_only_lp)).sum()
                )

            union_count = len(s_id_set.union(set(target_lookup.keys())))
            shared_tail_count = max(vocab_size - union_count, 0)
            if shared_tail_count > 0:
                terms.append(
                    shared_tail_count
                    * source["tail_prob"][b, t]
                    * (source["tail_logprob"][b, t] - target["tail_logprob"][b, t])
                )

            if terms:
                out[b, t] = torch.stack([term.to(device=device, dtype=torch.float32) for term in terms]).sum().clamp(min=0.0)

    return out


class EWADLoss(nn.Module):
    """
    Entropy-Weighted Agreement-Aware Distillation Loss.
    
    Performs token-level adaptive weighting between two teachers,
    with agreement-gated fallback to gold labels when teachers disagree.
    
    Args:
        vocab_size: Vocabulary size for entropy normalization
        tau_w: Temperature for confidence-weighted softmax (default: 1.0)
        k: Sigmoid sharpness for agreement gate (default: 5.0)  
        delta: Agreement threshold (default: 0.5)
        eps: Epsilon for numerical stability (default: 1e-8)
        mode: "full" | "confidence_only" | "agreement_only" for ablations
    """
    
    def __init__(
        self,
        vocab_size: int,
        tau_w: float = EWAD_TAU_W,
        k: float = EWAD_K,
        delta: float = EWAD_DELTA,
        eps: float = EWAD_ENTROPY_EPS,
        mode: str = "full"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tau_w = tau_w
        self.k = k
        self.delta = delta
        self.eps = eps
        self.mode = mode  # "full", "confidence_only", "agreement_only"
        self.log_vocab = math.log(vocab_size)
    
    def compute_entropy(self, logprobs):
        """
        Compute entropy from log-probabilities.
        H(p) = -Σ p(v) * log p(v)
        
        Args:
            logprobs: (batch, seq_len, vocab) log-probabilities
        Returns:
            entropy: (batch, seq_len) per-token entropy, clamped for stability
        """
        if is_sparse_teacher(logprobs):
            return sparse_teacher_entropy(
                logprobs,
                vocab_size=self.vocab_size,
                log_vocab=self.log_vocab,
                eps=self.eps,
            )

        probs = logprobs.exp()
        entropy = -(probs * logprobs).sum(dim=-1)
        # Clamp for stability (reviewer fix)
        entropy = entropy.clamp(min=self.eps, max=self.log_vocab)
        return entropy
    
    def compute_confidence(self, logprobs):
        """
        Normalized confidence: C = 1 - H(p) / log|V|
        Bounded in [0, 1]. High = teacher is confident.
        
        Args:
            logprobs: (batch, seq_len, vocab)
        Returns:
            confidence: (batch, seq_len)
        """
        entropy = self.compute_entropy(logprobs)
        confidence = 1.0 - entropy / self.log_vocab
        return confidence
    
    def compute_agreement(self, logprobs_32b, logprobs_14b):
        """
        Teacher agreement via Jensen-Shannon Divergence.
        A_t = 1 - JSD(p_32B || p_14B)
        
        JSD is symmetric, bounded in [0, 1] (for log base e: [0, log2]),
        so we normalize by log(2).
        
        Args:
            logprobs_32b: (batch, seq_len, vocab)
            logprobs_14b: (batch, seq_len, vocab)
        Returns:
            agreement: (batch, seq_len) in [0, 1]
        """
        # Align teacher vocab sizes if they differ
        logprobs_32b, logprobs_14b = align_vocab_size(logprobs_32b, logprobs_14b)
        probs_32b = logprobs_32b.exp()
        probs_14b = logprobs_14b.exp()
        
        # Mixture distribution M = 0.5 * (P + Q)
        m = 0.5 * (probs_32b + probs_14b)
        log_m = (m + self.eps).log()
        
        # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        # Using explicit KL formula: KL(P||M) = Σ P * (log P - log M)
        # (Avoids F.kl_div sign/direction ambiguity)
        kl_p_m = (probs_32b * (logprobs_32b - log_m)).sum(dim=-1)
        kl_q_m = (probs_14b * (logprobs_14b - log_m)).sum(dim=-1)
        jsd = 0.5 * (kl_p_m + kl_q_m)
        
        # Normalize by log(2) so JSD ∈ [0, 1]
        jsd = jsd / math.log(2)
        jsd = jsd.clamp(min=0.0, max=1.0)
        
        agreement = 1.0 - jsd
        return agreement
    
    def forward(
        self,
        student_logits,      # (batch, seq_len, vocab) — raw student logits
        teacher_32b_logprobs, # (batch, seq_len, vocab) — teacher 32B log-probs
        teacher_14b_logprobs, # (batch, seq_len, vocab) — teacher 14B log-probs
        gold_labels,          # (batch, seq_len) — gold token IDs
        teacher_32b_gold_mask=None, # (batch, seq_len) — 1 if gold was in 32B top-k
        teacher_14b_gold_mask=None, # (batch, seq_len) — 1 if gold was in 14B top-k
        attention_mask=None   # (batch, seq_len) — mask for valid positions
    ):
        """
        Compute EWAD loss.
        
        Returns:
            loss: scalar tensor
            diagnostics: dict with intermediate values for logging
        """
        # Cast student logits to float32 once for stable gradients
        student_logits = student_logits.to(torch.float32)
        student_logprobs = F.log_softmax(student_logits, dim=-1)

        valid_mask = (gold_labels != -100).float()
        kd_valid_mask = valid_mask
        if attention_mask is not None:
            # The caller passes teacher availability here. It should gate KD
            # positions only; gold CE must still train every labeled token,
            # including the appended EOS token.
            kd_valid_mask = valid_mask * attention_mask.to(
                device=student_logits.device,
                dtype=valid_mask.dtype,
            )

        has_32b = teacher_32b_logprobs is not None
        has_14b = teacher_14b_logprobs is not None
        if not has_32b and not has_14b:
            ce_loss = F.cross_entropy(
                student_logits.reshape(-1, student_logits.size(-1)),
                gold_labels.reshape(-1),
                reduction='none',
                ignore_index=-100
            ).reshape(student_logits.size(0), student_logits.size(1))
            loss = _masked_mean(ce_loss, valid_mask)
            return loss, {
                'ewad_loss': loss.item(),
                'kd_loss_mean': 0.0,
                'ce_loss_mean': loss.item(),
                'gate_mean': 0.0,
                'teacher_support_mean': 0.0,
                'kd_weight_mean': 0.0,
                'agreement_mean': -1.0,
                'conf_32b_mean': -1.0,
                'conf_14b_mean': -1.0,
                'w_32b_mean': 0.0,
                'w_14b_mean': 0.0,
            }

        if has_32b:
            conf_32b = self.compute_confidence(teacher_32b_logprobs)
            kl_32b = teacher_kl_to_student(teacher_32b_logprobs, student_logprobs)
        else:
            conf_32b = torch.zeros_like(gold_labels, dtype=student_logits.dtype)
            kl_32b = torch.zeros_like(conf_32b)

        if has_14b:
            conf_14b = self.compute_confidence(teacher_14b_logprobs)
            kl_14b = teacher_kl_to_student(teacher_14b_logprobs, student_logprobs)
        else:
            conf_14b = torch.zeros_like(gold_labels, dtype=student_logits.dtype)
            kl_14b = torch.zeros_like(conf_14b)

        if has_32b and has_14b and self.mode in ("full", "confidence_only"):
            confs = torch.stack([conf_32b / self.tau_w, conf_14b / self.tau_w], dim=-1)
            weights = F.softmax(confs, dim=-1)
            w_32b = weights[..., 0]
            w_14b = weights[..., 1]
        elif has_32b and has_14b:
            w_32b = torch.full_like(conf_32b, 0.5)
            w_14b = torch.full_like(conf_14b, 0.5)
        elif has_32b:
            w_32b = torch.ones_like(conf_32b)
            w_14b = torch.zeros_like(conf_32b)
        else:
            w_32b = torch.zeros_like(conf_14b)
            w_14b = torch.ones_like(conf_14b)

        if has_32b and has_14b and self.mode in ("full", "agreement_only"):
            if is_sparse_teacher(teacher_32b_logprobs) or is_sparse_teacher(teacher_14b_logprobs):
                s32 = teacher_32b_gold_mask.to(student_logits.device, dtype=student_logits.dtype) if teacher_32b_gold_mask is not None else torch.ones_like(w_32b)
                s14 = teacher_14b_gold_mask.to(student_logits.device, dtype=student_logits.dtype) if teacher_14b_gold_mask is not None else torch.ones_like(w_14b)
                agreement = 1.0 - (s32 - s14).abs().clamp(0.0, 1.0)
            else:
                agreement = self.compute_agreement(teacher_32b_logprobs, teacher_14b_logprobs)
        elif self.mode in ("full", "agreement_only"):
            # Cross-tokenizer case: agreement is measured from gold-token
            # support masks instead of invalid full-vocab JSD.
            s32 = teacher_32b_gold_mask.to(student_logits.device, dtype=student_logits.dtype) if teacher_32b_gold_mask is not None else torch.ones_like(w_32b)
            s14 = teacher_14b_gold_mask.to(student_logits.device, dtype=student_logits.dtype) if teacher_14b_gold_mask is not None else torch.ones_like(w_14b)
            agreement = 1.0 - (s32 - s14).abs().clamp(0.0, 1.0)
        else:
            agreement = None

        if agreement is not None:
            gate = torch.sigmoid(self.k * (agreement - self.delta))
        else:
            gate = torch.ones_like(w_32b)

        kd_loss = w_32b * kl_32b + w_14b * kl_14b

        # Teacher support gate: if an instruct teacher did not rank the gold
        # token in its top-k, its KD signal is treated as unreliable for that
        # position and the loss falls back toward CE.
        if teacher_32b_gold_mask is not None:
            support_32b = teacher_32b_gold_mask.to(device=student_logits.device, dtype=kd_loss.dtype)
        else:
            support_32b = torch.ones_like(kd_loss) if has_32b else torch.zeros_like(kd_loss)
        if teacher_14b_gold_mask is not None:
            support_14b = teacher_14b_gold_mask.to(device=student_logits.device, dtype=kd_loss.dtype)
        else:
            support_14b = torch.ones_like(kd_loss) if has_14b else torch.zeros_like(kd_loss)
        teacher_support = (w_32b * support_32b + w_14b * support_14b).clamp(0.0, 1.0)
        if attention_mask is not None:
            teacher_support = teacher_support * attention_mask.to(
                device=student_logits.device,
                dtype=teacher_support.dtype,
            )
        
        # ===== Step 5: Gold Label CE Loss =====
        ce_loss = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            gold_labels.reshape(-1),
            reduction='none',
            ignore_index=-100
        ).reshape(student_logits.size(0), student_logits.size(1))  # (B, T)
        
        # ===== Step 6: Agreement-Gated Combination =====
        # CE floor of 0.3: when two same-family teachers always agree (gate≈1),
        # the raw formula gives only ~9% CE supervision, causing severe overfitting.
        # A 30% floor ensures gold labels always anchor the student.
        CE_FLOOR = 0.3
        kd_weight = gate * teacher_support * (1.0 - CE_FLOOR)   # scales in [0, 0.7]
        ce_weight = 1.0 - kd_weight            # scales in [0.3, 1.0]
        per_token_loss = kd_weight * kd_loss + ce_weight * ce_loss  # (B, T)
        
        # ===== Apply Mask =====
        per_token_loss = per_token_loss * valid_mask
        num_tokens = valid_mask.sum().clamp(min=1.0)
        
        loss = per_token_loss.sum() / num_tokens
        
        # ===== Diagnostics =====
        diagnostics = {
            'ewad_loss': loss.item(),
            'kd_loss_mean': _masked_mean(kd_loss, kd_valid_mask).item(),
            'ce_loss_mean': _masked_mean(ce_loss, valid_mask).item(),
            'gate_mean': _masked_mean(gate, kd_valid_mask).item(),
            'teacher_support_mean': _masked_mean(teacher_support, valid_mask).item(),
            'kd_weight_mean': _masked_mean(kd_weight, valid_mask).item(),
            'agreement_mean': _masked_mean(agreement, kd_valid_mask).item() if agreement is not None else -1.0,
            'conf_32b_mean': _masked_mean(conf_32b, kd_valid_mask).item() if has_32b else -1.0,
            'conf_14b_mean': _masked_mean(conf_14b, kd_valid_mask).item() if has_14b else -1.0,
            'w_32b_mean': _masked_mean(w_32b, kd_valid_mask).item(),
            'w_14b_mean': _masked_mean(w_14b, kd_valid_mask).item(),
        }
        
        return loss, diagnostics


class CPDPLoss(nn.Module):
    """
    Capacity-Proportional Divergence Preservation Loss.
    
    Regularizes the student to maintain the relative distance between itself
    and each teacher, proportional to the teachers' mutual divergence.
    
    L_CPDP = |KL(p_32B||p_S)/H(p_S) - KL(p_14B||p_S)/H(p_S) - Δ*|²
    
    where Δ* = KL(p_32B || p_14B) (unnormalized, computed online).
    Student entropy is detached to prevent pathological optimization.
    
    Args:
        eps: Epsilon for numerical stability
    """
    
    def __init__(self, eps: float = CPDP_EPS):
        super().__init__()
        self.eps = eps
    
    def forward(
        self, 
        student_logits,       # (batch, seq_len, vocab)
        teacher_32b_logprobs, # (batch, seq_len, vocab)
        teacher_14b_logprobs, # (batch, seq_len, vocab)
        attention_mask=None   # (batch, seq_len)
    ):
        """
        Compute CPDP regularization loss.
        
        Returns:
            loss: scalar
            diagnostics: dict 
        """
        # Cast student logits to float32 once
        student_logits = student_logits.to(torch.float32)
        student_logprobs = F.log_softmax(student_logits, dim=-1)

        student_probs = student_logprobs.exp()
        
        # Student entropy for normalization (detached to prevent CPDP from
        # artificially pushing entropy high — it should only shape KL geometry)
        student_entropy = -(student_probs * student_logprobs).sum(dim=-1)  # (B, T)
        student_entropy = student_entropy.clamp(min=self.eps).detach()
        
        kl_32b_s = teacher_kl_to_student(teacher_32b_logprobs, student_logprobs)
        kl_14b_s = teacher_kl_to_student(teacher_14b_logprobs, student_logprobs)
        
        # KL(p_32B || p_14B) — teacher mutual divergence per token
        if is_sparse_teacher(teacher_32b_logprobs) and is_sparse_teacher(teacher_14b_logprobs):
            kl_32b_14b = sparse_teacher_kl_between(
                teacher_32b_logprobs,
                teacher_14b_logprobs,
                vocab_size=student_logprobs.size(-1),
            )
        elif not is_sparse_teacher(teacher_32b_logprobs) and not is_sparse_teacher(teacher_14b_logprobs):
            teacher_32b_lp_aligned, teacher_14b_lp_aligned = align_vocab_size(teacher_32b_logprobs, teacher_14b_logprobs)
            probs_32b_aligned = teacher_32b_lp_aligned.exp()
            kl_32b_14b = (probs_32b_aligned * (teacher_32b_lp_aligned - teacher_14b_lp_aligned)).sum(dim=-1)
            kl_32b_14b = kl_32b_14b.clamp(min=0.0)
        else:
            kl_32b_14b = torch.zeros_like(kl_32b_s)
        
        # Normalize student-teacher KLs by student entropy (capacity-aware scaling)
        # But Δ* (teacher-teacher KL) is NOT normalized by student entropy —
        # it's a fixed property of the teachers, not the student.
        normalized_gap = (kl_32b_s / student_entropy) - (kl_14b_s / student_entropy)
        
        # CPDP loss: squared difference (Δ* stays unnormalized)
        per_token_cpdp = (normalized_gap - kl_32b_14b).pow(2)  # (B, T)
        
        # Clamp to prevent explosion during early training
        per_token_cpdp = per_token_cpdp.clamp(max=100.0)
        
        # Apply mask
        if attention_mask is not None:
            per_token_cpdp = per_token_cpdp * attention_mask
            num_tokens = attention_mask.sum().clamp(min=1.0)
        else:
            num_tokens = torch.tensor(per_token_cpdp.numel(), dtype=per_token_cpdp.dtype, device=per_token_cpdp.device)
        
        loss = per_token_cpdp.sum() / num_tokens
        
        diagnostics = {
            'cpdp_loss': loss.item(),
            'kl_32b_student_mean': kl_32b_s.mean().item(),
            'kl_14b_student_mean': kl_14b_s.mean().item(),
            'kl_teacher_mutual_mean': kl_32b_14b.mean().item(),
        }
        
        return loss, diagnostics


class DualTeacherDistillationLoss(nn.Module):
    """
    Combined loss: L_total = L_EWAD + μ * L_CPDP
    
    Supports different modes for ablation experiments:
    - "no_distill": Only CE loss (baseline)
    - "fixed_weights": Fixed α/β teacher weights, no EWAD
    - "confidence_only": EWAD confidence weighting only
    - "agreement_only": EWAD agreement gating only  
    - "ewad_full": Full EWAD
    - "ewad_cpdp": EWAD + CPDP
    
    Args:
        vocab_size: Vocabulary size
        config: Experiment config dict from config.py EXPERIMENTS
        cpdp_mu: Weight of CPDP loss (default: 0.05)
    """
    
    def __init__(
        self,
        vocab_size: int,
        experiment_config: dict,
        cpdp_mu: float = CPDP_MU,
    ):
        super().__init__()
        self.experiment_config = experiment_config
        self.use_distillation = experiment_config.get("use_distillation", False)
        self.use_cpdp = experiment_config.get("use_cpdp", False)
        self.cpdp_mu = cpdp_mu
        
        if self.use_distillation:
            ewad_mode_raw = experiment_config.get("use_ewad", False)
            
            if ewad_mode_raw is False:
                # Fixed-weight KD (no EWAD)
                self.ewad = None
                self.fixed_weights = experiment_config.get("teacher_weights", {"32b": 0.7, "14b": 0.3})
            else:
                if ewad_mode_raw is True:
                    ewad_mode = "full"
                elif ewad_mode_raw == "confidence_only":
                    ewad_mode = "confidence_only"
                elif ewad_mode_raw == "agreement_only":
                    ewad_mode = "agreement_only"
                else:
                    ewad_mode = "full"
                
                self.ewad = EWADLoss(vocab_size=vocab_size, mode=ewad_mode)
                self.fixed_weights = None
            
            if self.use_cpdp:
                self.cpdp = CPDPLoss()
            else:
                self.cpdp = None
        else:
            self.ewad = None
            self.cpdp = None
            self.fixed_weights = None
    
    def forward(
        self,
        student_logits,         # (B, T, V) — raw student logits
        gold_labels,            # (B, T) — gold token IDs
        teacher_32b_logprobs=None,  # (B, T, V) — teacher 32B log-probs
        teacher_14b_logprobs=None,  # (B, T, V) — teacher 14B log-probs
        teacher_32b_gold_mask=None, # (B, T) — 1 if gold was in 32B top-k
        teacher_14b_gold_mask=None, # (B, T) — 1 if gold was in 14B top-k
        cpdp_teacher_32b_logprobs=None, # optional Qwen-compatible CPDP teacher A
        cpdp_teacher_14b_logprobs=None, # optional Qwen-compatible CPDP teacher B
        attention_mask=None     # (B, T) — valid position mask
    ):
        """
        Compute combined distillation loss.
        
        Returns:
            loss: scalar
            diagnostics: dict of intermediate values for logging
        """
        diagnostics = {}
        
        # ===== Case 1: No distillation (baseline) =====
        if not self.use_distillation:
            ce_loss = F.cross_entropy(
                student_logits.reshape(-1, student_logits.size(-1)),
                gold_labels.reshape(-1),
                ignore_index=-100
            )
            diagnostics['total_loss'] = ce_loss.item()
            diagnostics['ce_loss'] = ce_loss.item()
            return ce_loss, diagnostics
        
        # ===== Case 2: Fixed-weight KD (no EWAD) =====
        if self.ewad is None:
            student_logits = student_logits.to(torch.float32)
            student_logprobs = F.log_softmax(student_logits, dim=-1)

            w_32b = self.fixed_weights["32b"]
            w_14b = self.fixed_weights["14b"]
            available_weight = 0.0
            if w_32b > 0 and teacher_32b_logprobs is not None:
                available_weight += w_32b
            if w_14b > 0 and teacher_14b_logprobs is not None:
                available_weight += w_14b
            if available_weight > 0:
                w_32b = w_32b / available_weight
                w_14b = w_14b / available_weight
            
            # Compute valid token mask (excludes padding)
            valid_mask = (gold_labels != -100).float()
            num_valid = valid_mask.sum().clamp(min=1.0)
            
            kd_loss = torch.tensor(0.0, device=student_logits.device)
            
            if w_32b > 0 and teacher_32b_logprobs is not None:
                kl_32b_per_token = teacher_kl_to_student(teacher_32b_logprobs, student_logprobs)
                support = (
                    teacher_32b_gold_mask.to(device=student_logits.device, dtype=valid_mask.dtype)
                    if teacher_32b_gold_mask is not None else torch.ones_like(valid_mask)
                )
                diagnostics['teacher_32b_support_mean'] = (support * valid_mask).sum().item() / num_valid.item()
                kl_32b_per_token = (kl_32b_per_token.clamp(min=0.0) * valid_mask * support).sum() / num_valid
                kd_loss = kd_loss + w_32b * kl_32b_per_token
                diagnostics['kl_32b'] = kl_32b_per_token.item()
            
            if w_14b > 0 and teacher_14b_logprobs is not None:
                kl_14b_per_token = teacher_kl_to_student(teacher_14b_logprobs, student_logprobs)
                support = (
                    teacher_14b_gold_mask.to(device=student_logits.device, dtype=valid_mask.dtype)
                    if teacher_14b_gold_mask is not None else torch.ones_like(valid_mask)
                )
                diagnostics['teacher_14b_support_mean'] = (support * valid_mask).sum().item() / num_valid.item()
                kl_14b_per_token = (kl_14b_per_token.clamp(min=0.0) * valid_mask * support).sum() / num_valid
                kd_loss = kd_loss + w_14b * kl_14b_per_token
                diagnostics['kl_14b'] = kl_14b_per_token.item()
            
            # Also add CE with gold labels (as anchor)
            ce_loss = F.cross_entropy(
                student_logits.reshape(-1, student_logits.size(-1)),
                gold_labels.reshape(-1),
                ignore_index=-100
            )
            # KL starts ~2x larger than CE at init, so 0.3/0.7 roughly
            # equalises gradient contributions. CE-heavy ratio also reduces
            # the impact of corrupted teacher targets near position 0.
            if available_weight > 0:
                loss = 0.3 * kd_loss + 0.7 * ce_loss
            else:
                loss = ce_loss
                diagnostics['kd_skipped_no_compatible_teacher'] = 1.0
            
            diagnostics['total_loss'] = loss.item()
            diagnostics['ce_loss'] = ce_loss.item()
            diagnostics['kd_loss'] = kd_loss.item()
            return loss, diagnostics
        
        # ===== Case 3: EWAD (with or without CPDP) =====
        ewad_loss, ewad_diag = self.ewad(
            student_logits, teacher_32b_logprobs, teacher_14b_logprobs,
            gold_labels, teacher_32b_gold_mask, teacher_14b_gold_mask,
            attention_mask
        )
        diagnostics.update(ewad_diag)
        
        total_loss = ewad_loss
        
        if self.cpdp is not None:
            cpdp_32b = cpdp_teacher_32b_logprobs if cpdp_teacher_32b_logprobs is not None else teacher_32b_logprobs
            cpdp_14b = cpdp_teacher_14b_logprobs if cpdp_teacher_14b_logprobs is not None else teacher_14b_logprobs
            if cpdp_32b is not None and cpdp_14b is not None:
                cpdp_loss, cpdp_diag = self.cpdp(
                    student_logits, cpdp_32b, cpdp_14b,
                    attention_mask
                )
                diagnostics.update(cpdp_diag)
                total_loss = total_loss + self.cpdp_mu * cpdp_loss
                diagnostics['cpdp_weighted'] = (self.cpdp_mu * cpdp_loss).item()
            else:
                diagnostics['cpdp_loss'] = 0.0
                diagnostics['cpdp_weighted'] = 0.0
                diagnostics['cpdp_skipped_cross_tokenizer'] = 1.0
        
        diagnostics['total_loss'] = total_loss.item()
        return total_loss, diagnostics


# ============================================================================
# Utility: Convert sparse teacher top-k logprobs to dense distribution
# ============================================================================
def sparse_topk_to_dense(top_k_logprobs, vocab_size, device='cpu'):
    """
    Convert sparse top-k (token_id, logprob) list to dense log-probability distribution.
    Missing entries get uniform prior (not -inf!), then we renormalize.
    
    Args:
        top_k_logprobs: list of (token_id, logprob) tuples for one position
        vocab_size: total vocabulary size
        device: torch device
    Returns:
        logprobs: (vocab_size,) — normalized log-probability distribution
    """
    # Fill with uniform prior (not -inf!) to avoid NaN and entropy distortion
    # when only top-k tokens are available. This gives missing tokens a small
    # but non-zero probability mass.
    uniform_logprob = -math.log(vocab_size)
    logprobs = torch.full((vocab_size,), uniform_logprob, device=device)
    
    for token_id, logprob in top_k_logprobs:
        if 0 <= token_id < vocab_size:
            logprobs[token_id] = logprob
    
    # Renormalize in log-space so distribution sums to 1
    logprobs = logprobs - torch.logsumexp(logprobs, dim=0)
    
    return logprobs


def batch_sparse_to_dense(batch_top_k_logprobs, vocab_size, device='cpu'):
    """
    Convert a batch of sequences of sparse top-k logprobs to dense tensors.
    
    Args:
        batch_top_k_logprobs: list[list[list[(token_id, logprob)]]]
            Shape: [batch_size][seq_len][top_k entries]
        vocab_size: total vocab size
        device: torch device
    Returns:
        dense_logprobs: (batch, max_seq_len, vocab_size)
    """
    batch_size = len(batch_top_k_logprobs)
    max_seq_len = max(len(seq) for seq in batch_top_k_logprobs)
    
    # Fill with uniform prior (not -inf!) to avoid NaN and entropy distortion
    uniform_logprob = -math.log(vocab_size)
    dense = torch.full(
        (batch_size, max_seq_len, vocab_size), 
        uniform_logprob, 
        device=device
    )
    
    for b, seq in enumerate(batch_top_k_logprobs):
        for t, top_k in enumerate(seq):
            for token_id, logprob in top_k:
                if 0 <= token_id < vocab_size:
                    dense[b, t, token_id] = logprob
            # Renormalize so distribution sums to 1
            dense[b, t] = dense[b, t] - torch.logsumexp(dense[b, t], dim=0)
    
    return dense
