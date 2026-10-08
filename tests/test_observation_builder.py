"""Tests for construction of the RL observation."""

from types import SimpleNamespace
from typing import Any

import pytest

from assetto_corsa_ai.observation_builder import ObservationBuilder
from assetto_corsa_ai.track_perception import TrackPerception
from assetto_corsa_ai.tyre_dynamics import TyreDynamics


class FakeTyreDynamicsCalculator:
    """Return deterministic tyre dynamics for builder tests."""

    def __init__(self) -> None:
        self.telemetry = None
        self.static_info = None

    def calculate(
        self,
        telemetry: Any,
        static_info: Any,
    ) -> TyreDynamics:

        self.telemetry = telemetry
        self.static_info = static_info

        return TyreDynamics(
            front_slip_ratio=0.10,
            rear_slip_ratio=0.20,
            front_slip_angle=0.03,
            rear_slip_angle=0.04,
        )


class FakeTrackPerceptionCalculator:
    """Return deterministic track perception for builder tests."""

    def __init__(self) -> None:
        self.telemetry = None
        self.track_geometry = None
        self.current_s = None
        self.track_length = None

    def calculate(
        self,
        telemetry: Any,
        track_geometry: Any,
        current_s: float,
        track_length: float,
    ) -> TrackPerception:

        self.telemetry = telemetry
        self.track_geometry = track_geometry
        self.current_s = current_s
        self.track_length = track_length

        return TrackPerception(
            left_boundary=(
                (5.0, 4.0),
                (10.0, 4.1),
            ),
            right_boundary=(
                (5.0, -4.0),
                (10.0, -4.1),
            ),
            reference_waypoints=(),
        )


@pytest.fixture
def tyre_dynamics_calculator() -> FakeTyreDynamicsCalculator:
    return FakeTyreDynamicsCalculator()


@pytest.fixture
def track_perception_calculator() -> FakeTrackPerceptionCalculator:
    return FakeTrackPerceptionCalculator()


@pytest.fixture
def builder(
    tyre_dynamics_calculator: FakeTyreDynamicsCalculator,
    track_perception_calculator: FakeTrackPerceptionCalculator,
) -> ObservationBuilder:
    return ObservationBuilder(
        tyre_dynamics_calculator=tyre_dynamics_calculator,  # type: ignore[arg-type]
        track_perception_calculator=track_perception_calculator,  # type: ignore[arg-type]
    )


@pytest.fixture
def telemetry() -> Any:
    return SimpleNamespace(
        speed_kmh=72.0,
        # Validated AC layout:
        # [0] lateral
        # [2] longitudinal
        acceleration_g=(
            0.50,
            0.00,
            0.25,
        ),
        # [1] = yaw rate
        local_angular_velocity=(
            0.00,
            0.30,
            0.00,
        ),
        steering_angle=-0.20,
        gas=0.60,
        brake=0.00,
    )


@pytest.fixture
def static_info() -> Any:
    return SimpleNamespace(
        tyre_radius=(
            0.30,
            0.30,
            0.30,
            0.30,
        ),
        track_spline_length=5000.0,
    )


@pytest.fixture
def track_geometry() -> Any:
    return SimpleNamespace()


def test_builds_complete_observation(
    builder: ObservationBuilder,
    telemetry: Any,
    static_info: Any,
    track_geometry: Any,
) -> None:
    """All observation fields should be populated correctly."""

    observation = builder.build(
        telemetry=telemetry,
        static_info=static_info,
        track_geometry=track_geometry,
        current_s=100.0,
    )

    assert observation.speed == pytest.approx(20.0)

    assert observation.longitudinal_acceleration == pytest.approx(0.25)

    assert observation.lateral_acceleration == pytest.approx(0.50)

    assert observation.yaw_rate == pytest.approx(0.30)

    assert observation.front_slip_ratio == pytest.approx(0.10)

    assert observation.rear_slip_ratio == pytest.approx(0.20)

    assert observation.front_slip_angle == pytest.approx(0.03)

    assert observation.rear_slip_angle == pytest.approx(0.04)

    assert observation.left_boundary == (
        (5.0, 4.0),
        (10.0, 4.1),
    )

    assert observation.right_boundary == (
        (5.0, -4.0),
        (10.0, -4.1),
    )

    assert observation.steering == pytest.approx(-0.20)

    assert observation.longitudinal_control == pytest.approx(0.60)


def test_speed_is_converted_from_kmh_to_mps(
    builder: ObservationBuilder,
    telemetry: Any,
    static_info: Any,
    track_geometry: Any,
) -> None:
    """Vehicle speed should be converted from km/h to m/s."""

    telemetry.speed_kmh = 36.0

    observation = builder.build(
        telemetry=telemetry,
        static_info=static_info,
        track_geometry=track_geometry,
        current_s=0.0,
    )

    assert observation.speed == pytest.approx(10.0)


def test_validated_acceleration_axes_are_used(
    builder: ObservationBuilder,
    telemetry: Any,
    static_info: Any,
    track_geometry: Any,
) -> None:
    """Builder should use the validated AC acceleration indices."""

    telemetry.acceleration_g = (
        1.23,  # lateral
        9.99,  # unrelated axis
        -4.56,  # longitudinal
    )

    observation = builder.build(
        telemetry=telemetry,
        static_info=static_info,
        track_geometry=track_geometry,
        current_s=0.0,
    )

    assert observation.lateral_acceleration == pytest.approx(1.23)

    assert observation.longitudinal_acceleration == pytest.approx(-4.56)


def test_validated_yaw_rate_axis_is_used(
    builder: ObservationBuilder,
    telemetry: Any,
    static_info: Any,
    track_geometry: Any,
) -> None:
    """Builder should use local angular velocity index 1 as yaw rate."""

    telemetry.local_angular_velocity = (
        8.0,
        -0.42,
        9.0,
    )

    observation = builder.build(
        telemetry=telemetry,
        static_info=static_info,
        track_geometry=track_geometry,
        current_s=0.0,
    )

    assert observation.yaw_rate == pytest.approx(-0.42)


@pytest.mark.parametrize(
    (
        "gas",
        "brake",
        "expected",
    ),
    [
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.25),
        (0.50, 0.0, 0.50),
        (1.00, 0.0, 1.00),
        (0.0, 0.25, -0.25),
        (0.0, 0.50, -0.50),
        (0.0, 1.00, -1.00),
    ],
)
def test_longitudinal_control_combines_gas_and_brake(
    builder: ObservationBuilder,
    telemetry: Any,
    static_info: Any,
    track_geometry: Any,
    gas: float,
    brake: float,
    expected: float,
) -> None:
    """Throttle should be positive and braking negative."""

    telemetry.gas = gas
    telemetry.brake = brake

    observation = builder.build(
        telemetry=telemetry,
        static_info=static_info,
        track_geometry=track_geometry,
        current_s=0.0,
    )

    assert observation.longitudinal_control == pytest.approx(expected)


def test_tyre_dynamics_are_forwarded_to_observation(
    builder: ObservationBuilder,
    telemetry: Any,
    static_info: Any,
    track_geometry: Any,
) -> None:
    """Tyre dynamics returned by the calculator should be preserved."""

    observation = builder.build(
        telemetry=telemetry,
        static_info=static_info,
        track_geometry=track_geometry,
        current_s=0.0,
    )

    assert observation.front_slip_ratio == pytest.approx(0.10)
    assert observation.rear_slip_ratio == pytest.approx(0.20)

    assert observation.front_slip_angle == pytest.approx(0.03)
    assert observation.rear_slip_angle == pytest.approx(0.04)


def test_track_perception_is_forwarded_to_observation(
    builder: ObservationBuilder,
    telemetry: Any,
    static_info: Any,
    track_geometry: Any,
) -> None:
    """Track boundary points should be preserved in the observation."""

    observation = builder.build(
        telemetry=telemetry,
        static_info=static_info,
        track_geometry=track_geometry,
        current_s=0.0,
    )

    assert observation.left_boundary == (
        (5.0, 4.0),
        (10.0, 4.1),
    )

    assert observation.right_boundary == (
        (5.0, -4.0),
        (10.0, -4.1),
    )


def test_builder_passes_correct_inputs_to_tyre_calculator(
    builder: ObservationBuilder,
    telemetry: Any,
    static_info: Any,
    track_geometry: Any,
    tyre_dynamics_calculator: FakeTyreDynamicsCalculator,
) -> None:
    """The builder should pass telemetry and static info unchanged."""

    builder.build(
        telemetry=telemetry,
        static_info=static_info,
        track_geometry=track_geometry,
        current_s=100.0,
    )

    assert tyre_dynamics_calculator.telemetry is telemetry
    assert tyre_dynamics_calculator.static_info is static_info


def test_builder_passes_correct_inputs_to_track_perception(
    builder: ObservationBuilder,
    telemetry: Any,
    static_info: Any,
    track_geometry: Any,
    track_perception_calculator: FakeTrackPerceptionCalculator,
) -> None:
    """Track perception should receive the current projection position."""

    builder.build(
        telemetry=telemetry,
        static_info=static_info,
        track_geometry=track_geometry,
        current_s=123.4,
    )

    assert track_perception_calculator.telemetry is telemetry
    assert track_perception_calculator.track_geometry is track_geometry

    assert track_perception_calculator.current_s == pytest.approx(123.4)

    assert track_perception_calculator.track_length == pytest.approx(5000.0)
