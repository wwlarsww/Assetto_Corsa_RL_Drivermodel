"""Utilities for decoding a flat auxiliary head into Bézier trajectories."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def decode_control_points(
    flat_prediction: Tensor,
    *,
    coordinate_dimension: int = 2,
) -> Tensor:
    """Interpret the 10-output head with the public ``bezier-5x2`` codec.

    The thesis fixes the head width to 10 but does not expose the private
    target-encoding implementation. This named, replaceable public variant
    treats the vector as five ego-centric 2D control points.
    """

    if coordinate_dimension <= 0:
        raise ValueError("coordinate_dimension must be positive")
    if flat_prediction.shape[-1] % coordinate_dimension:
        raise ValueError("prediction width must be divisible by coordinate_dimension")
    return flat_prediction.reshape(
        *flat_prediction.shape[:-1],
        flat_prediction.shape[-1] // coordinate_dimension,
        coordinate_dimension,
    )


def sample_bezier(control_points: Tensor, sample_count: int = 20) -> Tensor:
    """Sample a Bézier curve for every leading batch dimension."""

    if control_points.ndim < 2:
        raise ValueError("control_points must end in (points, coordinates)")
    if control_points.shape[-2] < 2:
        raise ValueError("at least two control points are required")
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")

    point_count = control_points.shape[-2]
    degree = point_count - 1
    time = torch.linspace(
        0.0,
        1.0,
        sample_count,
        dtype=control_points.dtype,
        device=control_points.device,
    )
    basis = torch.stack(
        [
            math.comb(degree, index) * (1.0 - time) ** (degree - index) * time**index
            for index in range(point_count)
        ],
        dim=-1,
    )
    return torch.einsum("sp,...pc->...sc", basis, control_points)


def decode_bezier_trajectory(
    flat_prediction: Tensor,
    *,
    sample_count: int = 20,
) -> Tensor:
    """Decode the named public predictor representation to ego-frame path points."""

    return sample_bezier(
        decode_control_points(flat_prediction, coordinate_dimension=2),
        sample_count=sample_count,
    )
