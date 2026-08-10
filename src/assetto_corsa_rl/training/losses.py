"""Supervised warm-start and auxiliary objectives."""

from __future__ import annotations

import numpy.typing as npt
import torch
from torch import Tensor

from assetto_corsa_rl.models.bezier import decode_bezier_trajectory


def behavioural_cloning_loss(
    action_mean: Tensor,
    expert_action: Tensor,
    *,
    output_weights: Tensor | None = None,
    element_weights: Tensor | None = None,
) -> Tensor:
    """Mean squared error for the policy warm start from expert laps."""

    if action_mean.shape != expert_action.shape:
        raise ValueError("policy and expert action tensors must have identical shape")
    squared_error = (action_mean - expert_action).square()
    if output_weights is not None and element_weights is not None:
        raise ValueError("provide output_weights or element_weights, not both")
    if element_weights is not None:
        if element_weights.shape != action_mean.shape:
            raise ValueError("element_weights must match the action tensor shape")
        if not bool(torch.all(torch.isfinite(element_weights) & (element_weights > 0.0))):
            raise ValueError("element_weights must be finite and positive")
        return (squared_error * element_weights).sum().div(element_weights.sum())
    if output_weights is None:
        return squared_error.mean()
    if output_weights.ndim != 1 or output_weights.shape[0] != action_mean.shape[-1]:
        raise ValueError("output_weights must match the action output dimension")
    if not bool(torch.all(torch.isfinite(output_weights) & (output_weights > 0.0))):
        raise ValueError("output_weights must be finite and positive")
    return (squared_error * output_weights).sum(dim=-1).div(output_weights.sum()).mean()


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


def normalized_bezier_trajectory_loss_per_sample(
    flat_prediction: Tensor,
    target_ego_xy_m: Tensor,
    target_scale_m: npt.ArrayLike,
) -> Tensor:
    """Return target-scaled Smooth-L1 values while decoding predictions in metres."""

    if target_ego_xy_m.ndim != 3 or target_ego_xy_m.shape[-1] != 2:
        raise ValueError("target_ego_xy_m must have shape (batch, horizon, 2)")
    decoded_m = decode_bezier_trajectory(
        flat_prediction,
        sample_count=target_ego_xy_m.shape[1],
    )
    if decoded_m.shape != target_ego_xy_m.shape:
        raise ValueError("prediction batch and target batch must have matching shapes")
    scale_m = torch.as_tensor(
        target_scale_m,
        dtype=decoded_m.dtype,
        device=decoded_m.device,
    )
    if scale_m.shape != target_ego_xy_m.shape[1:]:
        raise ValueError("target_scale_m must have shape (horizon, 2)")
    if not bool(torch.all(torch.isfinite(scale_m) & (scale_m > 0.0))):
        raise ValueError("target_scale_m must be finite and positive")
    return torch.nn.functional.smooth_l1_loss(
        decoded_m / scale_m,
        target_ego_xy_m / scale_m,
        reduction="none",
    ).mean(dim=(-2, -1))


def masked_normalized_bezier_trajectory_loss(
    flat_prediction: Tensor,
    target_ego_xy_m: Tensor,
    valid_mask: Tensor,
    target_scale_m: npt.ArrayLike,
) -> Tensor:
    """Evaluate normalized trajectory loss only where a complete future exists."""

    if valid_mask.shape != (target_ego_xy_m.shape[0],) or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be a boolean tensor with shape (batch,)")
    if bool(valid_mask.any()):
        return normalized_bezier_trajectory_loss_per_sample(
            flat_prediction[valid_mask],
            target_ego_xy_m[valid_mask],
            target_scale_m,
        ).mean()
    return flat_prediction.sum() * 0.0
