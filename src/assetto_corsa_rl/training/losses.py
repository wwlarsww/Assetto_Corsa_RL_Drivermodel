"""Supervised warm-start and auxiliary objectives."""

from __future__ import annotations

import torch
from torch import Tensor

from assetto_corsa_rl.models.bezier import decode_bezier_trajectory


def behavioural_cloning_loss(action_mean: Tensor, expert_action_delta: Tensor) -> Tensor:
    """Mean squared error for the policy warm start from expert laps."""

    if action_mean.shape != expert_action_delta.shape:
        raise ValueError("policy and expert action tensors must have identical shape")
    return torch.nn.functional.mse_loss(action_mean, expert_action_delta)


def trajectory_prediction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Generic same-representation loss retained for custom predictor codecs."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target tensors must have identical shape")
    return torch.nn.functional.smooth_l1_loss(prediction, target)


def bezier_trajectory_prediction_loss(
    flat_prediction: Tensor,
    target_ego_xy_m: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Fit decoded Bézier paths to actually observed ego-frame future positions."""

    if target_ego_xy_m.ndim != 3 or target_ego_xy_m.shape[-1] != 2:
        raise ValueError("target_ego_xy_m must have shape (batch, horizon, 2)")
    decoded = decode_bezier_trajectory(
        flat_prediction,
        sample_count=target_ego_xy_m.shape[1],
    )
    if decoded.shape != target_ego_xy_m.shape:
        raise ValueError("prediction batch and target batch must have matching shapes")
    per_step = torch.nn.functional.smooth_l1_loss(
        decoded,
        target_ego_xy_m,
        reduction="none",
    ).mean(dim=(-2, -1))
    if valid_mask is not None:
        if valid_mask.shape != per_step.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be a boolean tensor with shape (batch,)")
        per_step = per_step[valid_mask]
        if per_step.numel() == 0:
            raise ValueError("valid_mask must select at least one prediction target")
    return per_step.mean()
