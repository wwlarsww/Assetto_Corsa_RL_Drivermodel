"""Verify episode termination using a real projector and synthetic telemetry."""

import pytest

from assetto_corsa_ai.telemetry import TelemetryFrame
from assetto_corsa_ai.termination import (
    EpisodeTerminationChecker,
    TerminationReason,
    TerminationResult,
)
from assetto_corsa_ai.track_geometry import TrackGeometry, TrackPoint
from assetto_corsa_ai.track_projection import TrackProjector


@pytest.fixture
def projector() -> TrackProjector:
    track = TrackGeometry([
        TrackPoint(
            position=(0.0, 0.0, z),
            length=-z,
            side_left=5.0,
            side_right=5.0,
            normal=(0.0, 1.0, 0.0),
            forward=(0.0, 0.0, -1.0),
        )
        for z in (0.0, -100.0)
    ])
    return TrackProjector(track, closed=False)


@pytest.fixture
def checker(projector: TrackProjector) -> EpisodeTerminationChecker:
    return EpisodeTerminationChecker(projector, stuck_time=4.0, minimum_progress=2.0)


@pytest.fixture
def telemetry() -> TelemetryFrame:
    return TelemetryFrame(
        speed_kmh=36.0,
        gas=0.0,
        brake=0.0,
        fuel=10.0,
        gear=2,
        rpm=2000,
        steering_angle=0.0,
        acceleration_g=(0.0, 0.0, 0.0),
        wheel_slip=(0.0, 0.0, 0.0, 0.0),
        wheel_angular_speed=(0.0, 0.0, 0.0, 0.0),
        tyre_contact_points=(
            (-1.0, 0.0, -20.0),
            (1.0, 0.0, -20.0),
            (-1.0, 0.0, -22.0),
            (1.0, 0.0, -22.0),
        ),
        local_velocity=(0.0, 0.0, 0.0),
        local_angular_velocity=(0.0, 0.0, 0.0),
        car_coordinates=(0.0, 0.0, -21.0),
        normalized_track_position=0.21,
        car_damage=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


@pytest.mark.parametrize("wheel", [0, 1, 2, 3])
@pytest.mark.parametrize("x", [-5.01, 5.01])
def test_any_single_tyre_outside_either_boundary_terminates(
    checker: EpisodeTerminationChecker,
    telemetry: TelemetryFrame,
    wheel: int,
    x: float,
) -> None:
    points = list(telemetry.tyre_contact_points)
    points[wheel] = (x, 0.0, -20.0)
    telemetry.tyre_contact_points = (points[0], points[1], points[2], points[3])

    assert checker.check(telemetry, delta_s=3.0, dt=1.0) == TerminationResult(
        True, TerminationReason.OFF_TRACK
    )


@pytest.mark.parametrize("x", [-5.0, 5.0])
def test_tyre_exactly_on_boundary_is_allowed(
    checker: EpisodeTerminationChecker,
    telemetry: TelemetryFrame,
    x: float,
) -> None:
    telemetry.tyre_contact_points = (
        (x, 0.0, -20.0),
        *telemetry.tyre_contact_points[1:],
    )
    assert checker.check(telemetry, delta_s=3.0, dt=1.0) == TerminationResult(False, None)


@pytest.mark.parametrize("count", [0, 3, 5])
def test_invalid_tyre_count_is_rejected(
    checker: EpisodeTerminationChecker,
    telemetry: TelemetryFrame,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> None:
    # Deliberately violate the model to exercise the runtime validation.
    monkeypatch.setattr(telemetry, "tyre_contact_points", ((0.0, 0.0, -20.0),) * count)
    with pytest.raises(ValueError, match="exactly four wheels"):
        checker.check(telemetry, delta_s=3.0, dt=1.0)


def test_collision_terminates_immediately_during_initial_window(
    checker: EpisodeTerminationChecker,
    telemetry: TelemetryFrame,
) -> None:
    assert checker.check(
        telemetry, delta_s=0.0, dt=0.1, strong_collision=True
    ) == TerminationResult(True, TerminationReason.STRONG_COLLISION)


@pytest.mark.parametrize("delta_s", [0.0, -1.0])
def test_standing_still_and_reverse_driving_terminate_after_full_window(
    checker: EpisodeTerminationChecker,
    telemetry: TelemetryFrame,
    delta_s: float,
) -> None:
    for dt in (1.5, 1.5, 0.5):
        assert checker.check(telemetry, delta_s, dt) == TerminationResult(False, None)
    assert checker.check(telemetry, delta_s, 0.5) == TerminationResult(
        True, TerminationReason.STUCK_OR_WRONG_WAY
    )


@pytest.mark.parametrize(
    "progress, expected",
    [(1.99, True), (2.0, False), (2.01, False)],
)
def test_progress_threshold_is_inclusive(
    checker: EpisodeTerminationChecker,
    telemetry: TelemetryFrame,
    progress: float,
    expected: bool,
) -> None:
    assert not checker.check(telemetry, progress / 2.0, 2.0).terminated
    result = checker.check(telemetry, progress / 2.0, 2.0)
    assert result == TerminationResult(
        expected, TerminationReason.STUCK_OR_WRONG_WAY if expected else None
    )


def test_sustained_forward_progress_keeps_episode_running(
    checker: EpisodeTerminationChecker,
    telemetry: TelemetryFrame,
) -> None:
    for _ in range(12):
        assert checker.check(telemetry, delta_s=1.0, dt=1.0) == TerminationResult(False, None)


def test_progress_before_current_time_window_no_longer_prevents_termination(
    checker: EpisodeTerminationChecker,
    telemetry: TelemetryFrame,
) -> None:
    assert not checker.check(telemetry, delta_s=10.0, dt=1.0).terminated
    for _ in range(3):
        assert not checker.check(telemetry, delta_s=0.0, dt=1.0).terminated

    # At t=5 the four-second window covers (1, 5]; progress from (0, 1] expired.
    assert checker.check(telemetry, delta_s=0.0, dt=1.0) == TerminationResult(
        True, TerminationReason.STUCK_OR_WRONG_WAY
    )


def test_backwards_progress_cancels_forward_progress_in_window(
    checker: EpisodeTerminationChecker,
    telemetry: TelemetryFrame,
) -> None:
    assert not checker.check(telemetry, delta_s=5.0, dt=2.0).terminated
    assert checker.check(telemetry, delta_s=-4.0, dt=2.0) == TerminationResult(
        True, TerminationReason.STUCK_OR_WRONG_WAY
    )


def test_reset_restarts_initial_window(
    checker: EpisodeTerminationChecker,
    telemetry: TelemetryFrame,
) -> None:
    assert checker.check(telemetry, delta_s=0.0, dt=4.0).terminated
    checker.reset()
    assert checker.check(telemetry, delta_s=0.0, dt=1.0) == TerminationResult(False, None)


def test_reset_discards_progress_from_previous_episode(
    checker: EpisodeTerminationChecker,
    telemetry: TelemetryFrame,
) -> None:
    assert not checker.check(telemetry, delta_s=100.0, dt=1.0).terminated
    checker.reset()
    for _ in range(3):
        assert not checker.check(telemetry, delta_s=0.0, dt=1.0).terminated
    assert checker.check(telemetry, delta_s=0.0, dt=1.0) == TerminationResult(
        True, TerminationReason.STUCK_OR_WRONG_WAY
    )


@pytest.mark.parametrize(
    "off_track, collision, expected_reason",
    [
        (True, True, TerminationReason.OFF_TRACK),
        (True, False, TerminationReason.OFF_TRACK),
        (False, True, TerminationReason.STRONG_COLLISION),
    ],
)
def test_current_reason_priority_when_conditions_coincide(
    checker: EpisodeTerminationChecker,
    telemetry: TelemetryFrame,
    off_track: bool,
    collision: bool,
    expected_reason: TerminationReason,
) -> None:
    for _ in range(3):
        assert not checker.check(telemetry, delta_s=0.0, dt=1.0).terminated
    if off_track:
        telemetry.tyre_contact_points = (
            (-6.0, 0.0, -20.0),
            *telemetry.tyre_contact_points[1:],
        )
    assert checker.check(
        telemetry, delta_s=0.0, dt=1.0, strong_collision=collision
    ) == TerminationResult(True, expected_reason)


@pytest.mark.parametrize("dt", [0.0, -1.0])
def test_nonpositive_step_duration_is_rejected_without_advancing_state(
    checker: EpisodeTerminationChecker,
    telemetry: TelemetryFrame,
    dt: float,
) -> None:
    with pytest.raises(ValueError, match="dt"):
        checker.check(telemetry, delta_s=100.0, dt=dt)
    assert checker.check(telemetry, delta_s=0.0, dt=4.0).terminated


@pytest.mark.parametrize(
    "kwargs, parameter",
    [
        ({"stuck_time": 0.0}, "stuck_time"),
        ({"stuck_time": -1.0}, "stuck_time"),
        ({"minimum_progress": -1.0}, "minimum_progress"),
    ],
)
def test_invalid_termination_configuration_is_rejected(
    projector: TrackProjector,
    kwargs: dict[str, float],
    parameter: str,
) -> None:
    with pytest.raises(ValueError, match=parameter):
        EpisodeTerminationChecker(projector, **kwargs)


def test_zero_minimum_progress_allows_standing_still(
    projector: TrackProjector,
    telemetry: TelemetryFrame,
) -> None:
    checker = EpisodeTerminationChecker(projector, minimum_progress=0.0)
    assert checker.check(telemetry, delta_s=0.0, dt=4.0) == TerminationResult(False, None)
