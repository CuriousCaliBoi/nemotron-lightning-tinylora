"""Transparent GRPO and truncated-importance-sampling objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass
class GRPOBatch:
    rewards: Tensor
    group_ids: Tensor
    old_logprobs: Tensor
    rollout_logprobs: Tensor
    response_mask: Tensor


def compute_group_advantages(
    rewards: Tensor,
    group_ids: Tensor,
    *,
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> Tensor:
    """Compute scalar group-relative advantages without Python-side detaches."""
    if rewards.ndim != 1 or group_ids.shape != rewards.shape:
        raise ValueError("rewards and group_ids must be one-dimensional and equal-sized")
    advantages = torch.empty_like(rewards, dtype=torch.float32)
    for group_id in torch.unique(group_ids, sorted=True):
        mask = group_ids == group_id
        values = rewards[mask].float()
        centered = values - values.mean()
        if normalize and values.numel() > 1:
            centered = centered / (values.std(unbiased=True) + epsilon)
        advantages[mask] = centered
    return advantages


def truncated_importance_weights(
    trainer_logprobs: Tensor,
    rollout_logprobs: Tensor,
    response_mask: Tensor,
    *,
    mode: Literal["none", "token_clip", "token_mask", "sequence_clip", "sequence_mask"] = "token_clip",
    minimum: float = 0.1,
    maximum: float = 10.0,
) -> Tensor:
    """Correct the PyTorch/vLLM sampling mismatch with bounded IS weights."""
    if not (trainer_logprobs.shape == rollout_logprobs.shape == response_mask.shape):
        raise ValueError("logprob and mask shapes must match")
    if mode == "none":
        return torch.ones_like(trainer_logprobs)
    log_ratio = (trainer_logprobs - rollout_logprobs).float()
    if mode.startswith("sequence"):
        lengths = response_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        log_ratio = (log_ratio * response_mask).sum(dim=-1, keepdim=True) / lengths
    ratio = torch.exp(log_ratio.clamp(max=20.0))
    if mode.endswith("clip"):
        ratio = ratio.clamp(min=minimum, max=maximum)
    elif mode.endswith("mask"):
        ratio = ratio.masked_fill((ratio < minimum) | (ratio > maximum), 0.0)
    return ratio.expand_as(trainer_logprobs)


def grpo_policy_loss(
    current_logprobs: Tensor,
    advantages: Tensor,
    batch: GRPOBatch,
    *,
    clip_epsilon: float = 0.2,
    tis_mode: Literal["none", "token_clip", "token_mask", "sequence_clip", "sequence_mask"] = "token_clip",
    tis_minimum: float = 0.1,
    tis_maximum: float = 10.0,
    reduction: Literal["sample_mean", "token_mean"] = "sample_mean",
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the clipped GRPO loss and detached diagnostic metrics."""
    if current_logprobs.shape != batch.old_logprobs.shape:
        raise ValueError("current and old logprobs must have the same shape")
    if advantages.ndim != 1 or advantages.shape[0] != current_logprobs.shape[0]:
        raise ValueError("advantages must contain one scalar per trajectory")
    mask = batch.response_mask.to(current_logprobs.dtype)
    policy_ratio = torch.exp((current_logprobs - batch.old_logprobs).float().clamp(-20.0, 20.0))
    unclipped = policy_ratio * advantages[:, None]
    clipped = policy_ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages[:, None]
    surrogate = torch.minimum(unclipped, clipped)
    tis = truncated_importance_weights(
        batch.old_logprobs,
        batch.rollout_logprobs,
        batch.response_mask,
        mode=tis_mode,
        minimum=tis_minimum,
        maximum=tis_maximum,
    )
    per_token = -surrogate * tis * mask
    if reduction == "sample_mean":
        loss = (per_token.sum(-1) / mask.sum(-1).clamp_min(1)).mean()
    elif reduction == "token_mean":
        loss = per_token.sum() / mask.sum().clamp_min(1)
    else:
        raise ValueError(f"unknown reduction: {reduction}")

    with torch.no_grad():
        rollout_log_ratio = (batch.old_logprobs - batch.rollout_logprobs).float()
        if tis_mode.startswith("sequence"):
            lengths = mask.sum(dim=-1, keepdim=True).clamp_min(1)
            rollout_log_ratio = (rollout_log_ratio * mask).sum(dim=-1, keepdim=True) / lengths
            rollout_log_ratio = rollout_log_ratio.expand_as(mask)
        raw_rollout_ratio = torch.exp(rollout_log_ratio.clamp(max=20.0))
        selected_rollout_ratios = raw_rollout_ratio[mask.bool()]
        outside_tis_bounds = (
            (selected_rollout_ratios < tis_minimum)
            | (selected_rollout_ratios > tis_maximum)
        )
        metrics = {
            "policy_ratio_mean": (policy_ratio * mask).sum() / mask.sum().clamp_min(1),
            "clip_fraction": (((policy_ratio - 1.0).abs() > clip_epsilon) * mask).sum()
            / mask.sum().clamp_min(1),
            "tis_mean": (tis * mask).sum() / mask.sum().clamp_min(1),
            "rollout_ratio_mean": selected_rollout_ratios.mean(),
            "rollout_ratio_min": selected_rollout_ratios.min(),
            "rollout_ratio_max": selected_rollout_ratios.max(),
            "tis_truncated_fraction": outside_tis_bounds.float().mean(),
            "approx_kl": (((batch.old_logprobs - current_logprobs).float()) * mask).sum()
            / mask.sum().clamp_min(1),
        }
    return loss, metrics


def selected_completion_logprobs(
    logits: Tensor,
    input_ids: Tensor,
    completion_mask: Tensor,
) -> Tensor:
    """Select next-token log-probabilities at completion positions."""
    if logits.shape[:2] != input_ids.shape or input_ids.shape != completion_mask.shape:
        raise ValueError("batch/sequence dimensions must match")
    next_logits = logits[:, :-1].float()
    next_tokens = input_ids[:, 1:].unsqueeze(-1)
    selected = next_logits.gather(dim=-1, index=next_tokens).squeeze(-1)
    log_normalizer = torch.logsumexp(next_logits, dim=-1)
    logprobs = selected - log_normalizer
    # Position t's token is predicted by logits at t-1; prepend a masked slot to
    # retain the same [batch, sequence] layout as completion_mask.
    return torch.cat((torch.zeros_like(logprobs[:, :1]), logprobs), dim=1) * completion_mask
