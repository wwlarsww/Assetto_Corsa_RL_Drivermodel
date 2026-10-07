"""Tests for the RL observation builder."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from assetto_corsa_ai.observation_builder import ObservationBuilder


@pytest.fixture
def telemetry():
    """Provide representative telemetry values for observation tests."""

    return SimpleNamespace(
        speed_kmh=72.0,
        acceleration_g=(0.5, 0.1, 0.25),
        local_angular_velocity=(0.01, 0.30, 0.02),
        steering_angle=-0.2,
        gas=0.6,
        brake=0.0,
    )


def test_build_observation(telemetry: SimpleNamespace) -> None:
    """Build an observation from telemetry and derived track features."""

    builder = ObservationBuilder()

    left_boundary = (
        (5.0, -4.0),
        (10.0, -4.2),
    )
    right_boundary = (
        (5.0, 4.0),
        (10.0, 4.2),
    )
    reference_waypoints = (
        (5.0, 0.0),
        (10.0, 0.5),
    )

    observation = builder.build(
        telemetry=cast(Any, telemetry),
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        reference_waypoints=reference_waypoints,
        front_slip_ratio=0.10,
        rear_slip_ratio=0.20,
        front_slip_angle=0.03,
        rear_slip_angle=0.04,
    )

    # Vehicle dynamics
    assert observation.speed == pytest.approx(20.0)
    assert observation.longitudinal_acceleration == pytest.approx(0.25)
    assert observation.lateral_acceleration == pytest.approx(0.5)
    assert observation.yaw_rate == pytest.approx(0.30)

    # Tyre dynamics
    assert observation.front_slip_ratio == pytest.approx(0.10)
    assert observation.rear_slip_ratio == pytest.approx(0.20)
    assert observation.front_slip_angle == pytest.approx(0.03)
    assert observation.rear_slip_angle == pytest.approx(0.04)

    # Track perception
    assert observation.left_boundary == left_boundary
    assert observation.right_boundary == right_boundary
    assert observation.reference_waypoints == reference_waypoints

    # Applied controls
    assert observation.steering == pytest.approx(-0.2)
    assert observation.longitudinal_control == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("throttle", "brake", "expected"),
    [
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.25),
        (1.0, 0.0, 1.0),
        (0.0, 0.25, -0.25),
        (0.0, 1.0, -1.0),
    ],
)
def test_longitudinal_control_mapping(
    telemetry: SimpleNamespace,
    throttle: float,
    brake: float,
    expected: float,
) -> None:
    """Map throttle and brake to one signed longitudinal control."""

    telemetry.gas = throttle
    telemetry.brake = brake

    observation = ObservationBuilder().build(
        telemetry=cast(Any, telemetry),
        left_boundary=(),
        right_boundary=(),
        reference_waypoints=(),
    )

    assert observation.longitudinal_control == pytest.approx(expected)


def test_speed_is_converted_from_kmh_to_mps(telemetry: SimpleNamespace) -> None:
    """Convert simulator speed from km/h to m/s."""

    telemetry.speed_kmh = 36.0

    observation = ObservationBuilder().build(
        telemetry=cast(Any, telemetry),
        left_boundary=(),
        right_boundary=(),
        reference_waypoints=(),
    )

    assert observation.speed == pytest.approx(10.0)


def test_validated_vehicle_dynamics_axes_are_used(
    telemetry: SimpleNamespace,
) -> None:
    """Use the empirically validated AC axis mapping."""

    telemetry.acceleration_g = (
        -0.4,  # lateral
        9.99,  # unused vertical component
        -1.2,  # longitudinal
    )

    telemetry.local_angular_velocity = (
        8.88,
        -0.35,  # yaw rate
        7.77,
    )

    observation = ObservationBuilder().build(
        telemetry=cast(Any, telemetry),
        left_boundary=(),
        right_boundary=(),
        reference_waypoints=(),
    )

    assert observation.lateral_acceleration == pytest.approx(-0.4)
    assert observation.longitudinal_acceleration == pytest.approx(-1.2)
    assert observation.yaw_rate == pytest.approx(-0.35)
