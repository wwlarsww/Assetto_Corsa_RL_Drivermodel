"""Utilities for decoding a flat auxiliary head into Bézier trajectories."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from assetto_corsa_rl.models.trajectory_contract import (
    PREDICTOR_CONTROL_POINT_COUNT,
    PREDICTOR_COORDINATE_DIMENSION,
    PREDICTOR_COORDINATE_SCALE_M,
    PREDICTOR_WAYPOINT_COUNT,
)


def decode_control_points(
    flat_prediction: Tensor,
    *,
    coordinate_dimension: int = PREDICTOR_COORDINATE_DIMENSION,
) -> Tensor:
    """Decode five free, dimensionless ego-frame control points."""

    if coordinate_dimension <= 0:
        raise ValueError("coordinate_dimension must be positive")
    if flat_prediction.shape[-1] % coordinate_dimension:
        raise ValueError("prediction width must be divisible by coordinate_dimension")
    control_points = flat_prediction.reshape(
        *flat_prediction.shape[:-1],
        flat_prediction.shape[-1] // coordinate_dimension,
        coordinate_dimension,
    )
    if control_points.shape[-2] != PREDICTOR_CONTROL_POINT_COUNT:
        raise ValueError("trajectory head must decode to exactly five free control points")
    return control_points


def anchor_control_points(free_control_points: Tensor) -> Tensor:
    """Prepend the current vehicle position ``[0, 0]`` to free points."""

    if free_control_points.ndim < 2 or free_control_points.shape[-1] != 2:
        raise ValueError("free_control_points must end in (points, 2)")
    origin = torch.zeros_like(free_control_points[..., :1, :])
    return torch.cat((origin, free_control_points), dim=-2)


def _sample_bezier_at(control_points: Tensor, time: Tensor) -> Tensor:
    point_count = control_points.shape[-2]
    degree = point_count - 1
    basis = torch.stack(
        [
            math.comb(degree, index) * (1.0 - time) ** (degree - index) * time**index
            for index in range(point_count)
        ],
        dim=-1,
    )
    return torch.einsum("sp,...pc->...sc", basis, control_points)


def sample_bezier(
    control_points: Tensor,
    sample_count: int = PREDICTOR_WAYPOINT_COUNT,
) -> Tensor:
    """Sample a Bézier curve for every leading batch dimension."""

    if control_points.ndim < 2:
        raise ValueError("control_points must end in (points, coordinates)")
    if control_points.shape[-2] < 2:
        raise ValueError("at least two control points are required")
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")

    time = torch.linspace(
        0.0,
        1.0,
        sample_count,
        dtype=control_points.dtype,
        device=control_points.device,
    )
    return _sample_bezier_at(control_points, time)


def _validate_predictor_sample_count(sample_count: int) -> None:
    if sample_count != PREDICTOR_WAYPOINT_COUNT:
        raise ValueError(
            f"predictor trajectory must contain exactly {PREDICTOR_WAYPOINT_COUNT} future waypoints"
        )


def decode_bezier_trajectory_normalized(
    flat_prediction: Tensor,
    *,
    sample_count: int = PREDICTOR_WAYPOINT_COUNT,
) -> Tensor:
    """Decode ten future points in the dimensionless local vehicle frame.

    The complete curve starts at the vehicle origin for Bézier parameter zero.
    Returned future points use ``0.1, 0.2, ..., 1.0`` and represent the fixed
    predictor times ``t + 0.1 s`` through ``t + 1.0 s``.
    """

    _validate_predictor_sample_count(sample_count)
    control_points = anchor_control_points(decode_control_points(flat_prediction))
    time = torch.arange(
        1,
        sample_count + 1,
        dtype=control_points.dtype,
        device=control_points.device,
    ) / float(sample_count)
    return _sample_bezier_at(control_points, time)


def decode_bezier_trajectory(
    flat_prediction: Tensor,
    *,
    sample_count: int = PREDICTOR_WAYPOINT_COUNT,
) -> Tensor:
    """Decode ten future path points in local vehicle coordinates and metres."""

    return (
        decode_bezier_trajectory_normalized(
            flat_prediction,
            sample_count=sample_count,
        )
        * PREDICTOR_COORDINATE_SCALE_M
    )


def decode_bezier_curve_with_origin(
    flat_prediction: Tensor,
    *,
    future_sample_count: int = PREDICTOR_WAYPOINT_COUNT,
) -> Tensor:
    """Decode the complete metre curve including vehicle origin ``[0, 0]``."""

    _validate_predictor_sample_count(future_sample_count)
    control_points_m = (
        anchor_control_points(decode_control_points(flat_prediction)) * PREDICTOR_COORDINATE_SCALE_M
    )
    return sample_bezier(
        control_points_m,
        sample_count=future_sample_count + 1,
    )
