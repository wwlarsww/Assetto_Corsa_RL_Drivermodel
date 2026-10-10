"""Local trajectory targets and the masked auxiliary objective."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PredictorConfig:
    enabled: bool
    layers: tuple[int, ...]
    prediction_points: int
    output_dimensions: int
    loss_weight: float
    horizons_seconds: tuple[float, ...]


def horizon_indices(horizons: tuple[float, ...], control_interval: float) -> tuple[int, ...]:
    """Require exact control-grid horizons rather than silently rounding them."""
    if not isfinite(control_interval) or control_interval <= 0:
        raise ValueError("control_interval must be finite and positive.")
    if not horizons or any(not isfinite(value) or value <= 0 for value in horizons):
        raise ValueError("Predictor horizons must be finite and positive.")
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError("Predictor horizons must be strictly increasing.")
    offsets = np.asarray(horizons, dtype=np.float64) / control_interval
    rounded = np.rint(offsets)
    if not np.allclose(offsets, rounded, rtol=0.0, atol=1e-6) or np.any(rounded < 1):
        raise ValueError("Predictor horizons must be integer multiples of control_interval.")
    return tuple(int(value) for value in rounded)


def build_trajectory_targets(
    positions: NDArray[np.float64], forwards: NDArray[np.float64],
    next_positions: NDArray[np.float64], dones: NDArray[np.bool_],
    offsets: tuple[int, ...],
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Build targets from transitions in one rollout, including terminal endpoints.

    Shapes are (steps, environments, 3) for poses. next_positions contains the
    actual endpoint of each transition, before any automatic reset. A terminal
    endpoint is valid; every horizon beyond it is masked. The rollout's final
    transition can also supply an endpoint, without looking into another rollout.
    """
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("Vehicle positions must have shape (steps, environments, 3).")
    if forwards.shape != positions.shape or next_positions.shape != positions.shape or dones.shape != positions.shape[:2]:
        raise ValueError("Rollout poses and episode boundaries must have matching shapes.")
    if not offsets or any(type(value) is not int or value <= 0 for value in offsets):
        raise ValueError("Horizon offsets must be positive integers.")
    if not all(np.isfinite(values).all() for values in (positions, forwards, next_positions)):
        raise ValueError("Rollout poses must be finite.")
    direction = forwards[..., (0, 2)]
    lengths = np.linalg.norm(direction, axis=-1, keepdims=True)
    if np.any(lengths <= 1e-9):
        raise ValueError("Vehicle forward must have a non-zero horizontal direction.")
    direction = direction / lengths
    left = np.stack((direction[..., 1], -direction[..., 0]), axis=-1)
    steps, environments = positions.shape[:2]
    targets = np.zeros((steps, environments, len(offsets), 2), dtype=np.float32)
    mask = np.zeros((steps, environments, len(offsets)), dtype=np.bool_)
    for index, offset in enumerate(offsets):
        count = steps - offset + 1
        if count <= 0:
            continue
        displacement = next_positions[offset - 1:, ..., (0, 2)] - positions[:count, ..., (0, 2)]
        targets[:count, :, index, 0] = np.sum(displacement * direction[:count], axis=-1)
        targets[:count, :, index, 1] = np.sum(displacement * left[:count], axis=-1)
        for start in range(count):
            mask[start, :, index] = ~np.any(dones[start:start + offset - 1], axis=0)
    targets[~mask] = 0.0
    return targets, mask


def masked_predictor_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean SmoothL1 over coordinates of valid points only, retaining the graph."""
    if prediction.shape != target.shape or mask.shape != prediction.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError("Prediction, targets and per-horizon boolean mask must match.")
    if not mask.any():
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction[mask], target[mask], reduction="mean")
