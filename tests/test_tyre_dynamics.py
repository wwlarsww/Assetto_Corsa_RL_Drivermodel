"""Tests for tyre dynamics calculations."""

import math
from types import SimpleNamespace
from typing import Any

import pytest

from assetto_corsa_ai.tyre_dynamics import TyreDynamicsCalculator

FRONT_AXLE_DISTANCE = 1.2
REAR_AXLE_DISTANCE = 1.4

TYRE_RADIUS = (0.30, 0.30, 0.30, 0.30)


@pytest.fixture
def calculator() -> TyreDynamicsCalculator:
    """Create a tyre dynamics calculator with known vehicle geometry."""

    return TyreDynamicsCalculator(
        front_axle_distance=FRONT_AXLE_DISTANCE,
        rear_axle_distance=REAR_AXLE_DISTANCE,
        minimum_speed=1.0,
    )


def make_telemetry(
    longitudinal_velocity: float = 10.0,
    lateral_velocity: float = 0.0,
    yaw_rate: float = 0.0,
    wheel_angular_speed: tuple[float, float, float, float] = (
        10.0 / 0.30,
        10.0 / 0.30,
        10.0 / 0.30,
        10.0 / 0.30,
    ),
):
    """Create minimal telemetry required by TyreDynamicsCalculator."""

    return SimpleNamespace(
        # AC local velocity:
        # [0] vertical
        # [1] longitudinal
        # [2] lateral
        local_velocity=(
            0.0,
            longitudinal_velocity,
            lateral_velocity,
        ),
        # Validated AC angular velocity:
        # [1] yaw rate
        local_angular_velocity=(
            0.0,
            yaw_rate,
            0.0,
        ),
        wheel_angular_speed=wheel_angular_speed,
    )


@pytest.fixture
def static_info():
    """Create minimal static vehicle information."""

    return SimpleNamespace(
        tyre_radius=TYRE_RADIUS,
    )


# ---------------------------------------------------------------------------
# Slip ratio
# ---------------------------------------------------------------------------


def test_zero_slip_ratio_when_wheel_speed_matches_vehicle_speed(
    calculator: Any,
    static_info: Any,
) -> None:
    """Pure rolling should result in approximately zero slip."""

    telemetry = make_telemetry(
        longitudinal_velocity=10.0,
        wheel_angular_speed=(
            10.0 / 0.30,
            10.0 / 0.30,
            10.0 / 0.30,
            10.0 / 0.30,
        ),
    )

    result = calculator.calculate(telemetry, static_info)

    assert result.front_slip_ratio == pytest.approx(0.0)
    assert result.rear_slip_ratio == pytest.approx(0.0)


def test_positive_slip_ratio_when_wheels_spin_faster(
    calculator: Any,
    static_info: Any,
) -> None:
    """Wheel spin during acceleration should produce positive slip."""

    vehicle_speed = 10.0
    wheel_speed = 11.0

    telemetry = make_telemetry(
        longitudinal_velocity=vehicle_speed,
        wheel_angular_speed=(
            wheel_speed / 0.30,
            wheel_speed / 0.30,
            wheel_speed / 0.30,
            wheel_speed / 0.30,
        ),
    )

    result = calculator.calculate(telemetry, static_info)

    assert result.front_slip_ratio == pytest.approx(0.10)
    assert result.rear_slip_ratio == pytest.approx(0.10)


def test_negative_slip_ratio_when_wheels_rotate_slower(
    calculator: Any,
    static_info: Any,
) -> None:
    """Braking should produce negative longitudinal slip."""

    vehicle_speed = 10.0
    wheel_speed = 8.0

    telemetry = make_telemetry(
        longitudinal_velocity=vehicle_speed,
        wheel_angular_speed=(
            wheel_speed / 0.30,
            wheel_speed / 0.30,
            wheel_speed / 0.30,
            wheel_speed / 0.30,
        ),
    )

    result = calculator.calculate(telemetry, static_info)

    assert result.front_slip_ratio == pytest.approx(-0.20)
    assert result.rear_slip_ratio == pytest.approx(-0.20)


def test_front_and_rear_slip_are_calculated_independently(
    calculator: Any,
    static_info: Any,
) -> None:
    """Front and rear axle slip should be averaged independently."""

    vehicle_speed = 10.0

    telemetry = make_telemetry(
        longitudinal_velocity=vehicle_speed,
        wheel_angular_speed=(
            10.0 / 0.30,  # FL -> 0 %
            10.0 / 0.30,  # FR -> 0 %
            12.0 / 0.30,  # RL -> +20 %
            12.0 / 0.30,  # RR -> +20 %
        ),
    )

    result = calculator.calculate(telemetry, static_info)

    assert result.front_slip_ratio == pytest.approx(0.0)
    assert result.rear_slip_ratio == pytest.approx(0.20)


def test_left_and_right_wheel_slip_are_averaged(
    calculator: Any,
    static_info: Any,
) -> None:
    """Slip ratios of both wheels on an axle should be averaged."""

    telemetry = make_telemetry(
        longitudinal_velocity=10.0,
        wheel_angular_speed=(
            11.0 / 0.30,  # FL -> +10 %
            13.0 / 0.30,  # FR -> +30 %
            10.0 / 0.30,
            10.0 / 0.30,
        ),
    )

    result = calculator.calculate(telemetry, static_info)

    assert result.front_slip_ratio == pytest.approx(0.20)
    assert result.rear_slip_ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Slip angle
# ---------------------------------------------------------------------------


def test_zero_slip_angle_during_straight_driving(
    calculator: Any,
    static_info: Any,
) -> None:
    """Straight driving without lateral velocity should give zero angle."""

    telemetry = make_telemetry(
        longitudinal_velocity=10.0,
        lateral_velocity=0.0,
        yaw_rate=0.0,
    )

    result = calculator.calculate(telemetry, static_info)

    assert result.front_slip_angle == pytest.approx(0.0)
    assert result.rear_slip_angle == pytest.approx(0.0)


def test_lateral_velocity_affects_front_and_rear_equally_without_yaw(
    calculator: Any,
    static_info: Any,
) -> None:
    """Without yaw, both axle centres have the same velocity angle."""

    telemetry = make_telemetry(
        longitudinal_velocity=10.0,
        lateral_velocity=1.0,
        yaw_rate=0.0,
    )

    result = calculator.calculate(telemetry, static_info)

    expected = math.atan2(1.0, 10.0)

    assert result.front_slip_angle == pytest.approx(expected)
    assert result.rear_slip_angle == pytest.approx(expected)


def test_yaw_rate_changes_front_and_rear_angles_differently(
    calculator: Any,
    static_info: Any,
) -> None:
    """Yaw adds lateral velocity at the front and subtracts it at the rear."""

    telemetry = make_telemetry(
        longitudinal_velocity=10.0,
        lateral_velocity=1.0,
        yaw_rate=0.2,
    )

    result = calculator.calculate(telemetry, static_info)

    expected_front = math.atan2(
        1.0 + 0.2 * FRONT_AXLE_DISTANCE,
        10.0,
    )

    expected_rear = math.atan2(
        1.0 - 0.2 * REAR_AXLE_DISTANCE,
        10.0,
    )

    assert result.front_slip_angle == pytest.approx(expected_front)
    assert result.rear_slip_angle == pytest.approx(expected_rear)


def test_pure_yaw_produces_opposite_front_and_rear_angles(
    calculator: Any,
    static_info: Any,
) -> None:
    """Pure yaw produces opposite lateral velocities at both axles."""

    telemetry = make_telemetry(
        longitudinal_velocity=10.0,
        lateral_velocity=0.0,
        yaw_rate=0.5,
    )

    result = calculator.calculate(telemetry, static_info)

    expected_front = math.atan2(
        0.5 * FRONT_AXLE_DISTANCE,
        10.0,
    )

    expected_rear = math.atan2(
        -0.5 * REAR_AXLE_DISTANCE,
        10.0,
    )

    assert result.front_slip_angle == pytest.approx(expected_front)
    assert result.rear_slip_angle == pytest.approx(expected_rear)


def test_reversing_yaw_rate_reverses_slip_angle_signs(
    calculator: Any,
    static_info: Any,
) -> None:
    """Changing turn direction should reverse the yaw contribution."""

    left_turn = make_telemetry(
        longitudinal_velocity=10.0,
        lateral_velocity=0.0,
        yaw_rate=0.3,
    )

    right_turn = make_telemetry(
        longitudinal_velocity=10.0,
        lateral_velocity=0.0,
        yaw_rate=-0.3,
    )

    left = calculator.calculate(left_turn, static_info)
    right = calculator.calculate(right_turn, static_info)

    assert right.front_slip_angle == pytest.approx(-left.front_slip_angle)

    assert right.rear_slip_angle == pytest.approx(-left.rear_slip_angle)


# ---------------------------------------------------------------------------
# Low-speed behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "longitudinal_velocity",
    [
        0.0,
        0.1,
        0.5,
        0.99,
        -0.5,
    ],
)
def test_slip_values_are_zero_below_minimum_speed(
    calculator: Any,
    static_info: Any,
    longitudinal_velocity: float,
) -> None:
    """Slip calculations should be disabled close to standstill."""

    telemetry = make_telemetry(
        longitudinal_velocity=longitudinal_velocity,
        lateral_velocity=1.0,
        yaw_rate=0.5,
        wheel_angular_speed=(
            100.0,
            100.0,
            100.0,
            100.0,
        ),
    )

    result = calculator.calculate(telemetry, static_info)

    assert result.front_slip_ratio == pytest.approx(0.0)
    assert result.rear_slip_ratio == pytest.approx(0.0)
    assert result.front_slip_angle == pytest.approx(0.0)
    assert result.rear_slip_angle == pytest.approx(0.0)
