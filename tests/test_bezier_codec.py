from __future__ import annotations

import pytest
import torch

from assetto_corsa_rl.models.bezier import (
    decode_bezier_curve_with_origin,
    decode_bezier_trajectory,
    decode_bezier_trajectory_normalized,
    decode_control_points,
)
from assetto_corsa_rl.models.trajectory_contract import (
    PREDICTOR_COORDINATE_SCALE_M,
    PREDICTOR_FORWARD_INCREMENT_MAX_M,
    PREDICTOR_LATERAL_LIMIT_M,
)


def test_anchored_bezier_curve_starts_at_vehicle_origin() -> None:
    flat_prediction = torch.tensor(
        [
            [
                0.01,
                0.10,
                0.02,
                0.25,
                -0.01,
                0.50,
                0.00,
                0.75,
                0.03,
                1.00,
            ]
        ],
        dtype=torch.float32,
    )

    complete_curve = decode_bezier_curve_with_origin(
        flat_prediction,
        future_sample_count=10,
    )
    future_points = decode_bezier_trajectory(
        flat_prediction,
        sample_count=10,
    )

    assert complete_curve.shape == (1, 11, 2)
    assert future_points.shape == (1, 10, 2)
    torch.testing.assert_close(
        complete_curve[:, 0],
        torch.zeros((1, 2)),
    )
    torch.testing.assert_close(
        complete_curve[:, 1:],
        future_points,
    )
    control_points = decode_control_points(flat_prediction)
    torch.testing.assert_close(
        future_points[:, -1],
        control_points[:, -1] * PREDICTOR_COORDINATE_SCALE_M,
    )


def test_normalized_and_metre_decoders_use_fixed_scale() -> None:
    flat_prediction = torch.randn(
        (3, 10),
        generator=torch.Generator().manual_seed(11),
    )

    normalized = decode_bezier_trajectory_normalized(flat_prediction)
    metres = decode_bezier_trajectory(flat_prediction)

    torch.testing.assert_close(
        metres,
        normalized * PREDICTOR_COORDINATE_SCALE_M,
    )


def test_zero_prediction_produces_straight_forward_prior() -> None:
    prediction = decode_bezier_trajectory(torch.zeros((2, 10)))
    expected_forward = torch.arange(1, 11, dtype=torch.float32) * 5.0

    torch.testing.assert_close(prediction[..., 0], torch.zeros((2, 10)))
    torch.testing.assert_close(
        prediction[..., 1],
        expected_forward.expand(2, -1),
    )


def test_decoder_enforces_bounded_monotonic_geometry() -> None:
    raw = torch.tensor(
        [[50.0, -50.0, -50.0, 50.0, 50.0, -10.0, -50.0, 10.0, 50.0, 50.0]],
        dtype=torch.float32,
    )
    control_points = decode_control_points(raw) * PREDICTOR_COORDINATE_SCALE_M
    prediction = decode_bezier_trajectory(raw)

    assert torch.all(control_points[..., 0].abs() <= PREDICTOR_LATERAL_LIMIT_M + 1e-5)
    assert torch.all(control_points[..., 1] >= 0.0)
    assert torch.all(torch.diff(control_points[..., 1], dim=-1) >= -1e-6)
    assert torch.all(torch.diff(prediction[..., 1], dim=-1) >= -1e-5)
    assert torch.all(prediction[..., 1] <= PREDICTOR_FORWARD_INCREMENT_MAX_M * 5 + 1e-4)


def test_decoder_rejects_non_contract_waypoint_count() -> None:
    with pytest.raises(ValueError, match="exactly 10 future waypoints"):
        decode_bezier_trajectory(torch.zeros((1, 10)), sample_count=20)
