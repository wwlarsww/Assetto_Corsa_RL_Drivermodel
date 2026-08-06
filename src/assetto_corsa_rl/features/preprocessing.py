"""Shared preprocessing rules for BC demonstrations and live observations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.types import FloatArray

LOW_SPEED_SLIP_ZERO_MPS = 1.0
LOW_SPEED_SLIP_FULL_MPS = 5.0
LOW_SPEED_SLIP_PREPROCESSING_NAME = "linear_speed_weight_1_to_5_mps_v1"
PREVIOUS_ACTION_IDENTITY_FEATURE_NAMES = (
    "action.previous_combined_pedal",
    "action.previous_steering",
)
PREVIOUS_ACTION_NORMALIZATION_NAME = "identity_previous_actions_v1"


def low_speed_slip_weight(speed_mps: npt.ArrayLike) -> FloatArray:
    """Return a linear slip reliability weight for absolute vehicle speed.

    Tyre slip ratios are poorly conditioned close to standstill because the
    vehicle speed in their denominator approaches zero. The weight is zero at
    and below 1 m/s, rises linearly, and reaches one at 5 m/s.
    """

    speed = np.asarray(speed_mps, dtype=np.float32)
    if not np.all(np.isfinite(speed)):
        raise ValueError("speed_mps must contain only finite values")
    if np.any(speed < 0.0):
        raise ValueError("speed_mps must be non-negative")

    scale = LOW_SPEED_SLIP_FULL_MPS - LOW_SPEED_SLIP_ZERO_MPS
    return np.asarray(
        np.clip((speed - LOW_SPEED_SLIP_ZERO_MPS) / scale, 0.0, 1.0),
        dtype=np.float32,
    )


def stabilise_low_speed_slip(
    speed_mps: npt.ArrayLike,
    slip_features: npt.ArrayLike,
) -> FloatArray:
    """Apply the same low-speed slip preprocessing to BC and live features.

    ``slip_features`` must use the final-axis order
    ``front_ratio, rear_ratio, front_angle_rad, rear_angle_rad``.
    """

    values = np.asarray(slip_features, dtype=np.float32)
    speed = np.asarray(speed_mps, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] != 4:
        raise ValueError("slip_features must have final dimension 4")
    if values.shape[:-1] != speed.shape:
        raise ValueError("speed_mps shape must match slip_features without its final dimension")
    if not np.all(np.isfinite(values)):
        raise ValueError("slip_features must contain only finite values")

    weight = low_speed_slip_weight(speed)
    return np.asarray(values * weight[..., None], dtype=np.float32)
