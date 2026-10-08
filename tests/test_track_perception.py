"""Tests for vehicle-relative track perception."""

from types import SimpleNamespace
from typing import Any

import pytest

from assetto_corsa_ai.track_perception import TrackPerceptionCalculator


def make_track_point(
    length: float,
    left_boundary: tuple[float, float, float],
    right_boundary: tuple[float, float, float],
    *,
    position: tuple[float, float, float] | None = None,
) -> SimpleNamespace:
    """Create a minimal track point required by TrackPerceptionCalculator."""

    # Synthetic tracks default to a centred AI line; an explicit position
    # also allows testing reference lines that are offset from the centre.
    if position is None:
        position = (
            (left_boundary[0] + right_boundary[0]) / 2.0,
            (left_boundary[1] + right_boundary[1]) / 2.0,
            (left_boundary[2] + right_boundary[2]) / 2.0,
        )

    return SimpleNamespace(
        length=length,
        position=position,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
    )


def make_vehicle_tyres(
    origin_x: float = 0.0,
    origin_z: float = 0.0,
    forward: tuple[float, float] = (0.0, 1.0),
    left: tuple[float, float] = (-1.0, 0.0),
    half_wheelbase: float = 1.0,
    half_track_width: float = 1.0,
):
    """Create four tyre contact points from a vehicle coordinate frame.

    Wheel order:
        0 = front left
        1 = front right
        2 = rear left
        3 = rear right
    """

    fx, fz = forward
    lx, lz = left

    front_center = (
        origin_x + fx * half_wheelbase,
        origin_z + fz * half_wheelbase,
    )

    rear_center = (
        origin_x - fx * half_wheelbase,
        origin_z - fz * half_wheelbase,
    )

    fl = (
        front_center[0] + lx * half_track_width,
        0.0,
        front_center[1] + lz * half_track_width,
    )

    fr = (
        front_center[0] - lx * half_track_width,
        0.0,
        front_center[1] - lz * half_track_width,
    )

    rl = (
        rear_center[0] + lx * half_track_width,
        0.0,
        rear_center[1] + lz * half_track_width,
    )

    rr = (
        rear_center[0] - lx * half_track_width,
        0.0,
        rear_center[1] - lz * half_track_width,
    )

    return (fl, fr, rl, rr)


def make_telemetry(
    tyre_contact_points: tuple[tuple[float, float, float], ...] | None = None,
) -> Any:
    """Create minimal telemetry for track perception tests."""

    if tyre_contact_points is None:
        tyre_contact_points = make_vehicle_tyres()

    return SimpleNamespace(
        tyre_contact_points=tyre_contact_points,
    )


def calculator_input(value: object) -> Any:
    """Treat a lightweight test double as the calculator's input type."""

    return value


def test_straight_track_boundaries_are_vehicle_relative() -> None:
    """Straight boundaries should appear ahead and on the correct side."""

    calculator = TrackPerceptionCalculator(
        lookahead_distances=(5.0, 10.0),
        reference_distances=(5.0, 10.0),
    )

    # Vehicle points along +z.
    #
    # left  = negative world x
    # right = positive world x
    track_geometry = SimpleNamespace(
        points=(
            make_track_point(
                0.0,
                (-4.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
            ),
            make_track_point(
                10.0,
                (-4.0, 0.0, 10.0),
                (4.0, 0.0, 10.0),
            ),
            make_track_point(
                20.0,
                (-4.0, 0.0, 20.0),
                (4.0, 0.0, 20.0),
            ),
        )
    )

    result = calculator.calculate(
        telemetry=calculator_input(make_telemetry()),
        track_geometry=calculator_input(track_geometry),
        current_s=0.0,
        track_length=20.0,
    )

    assert result.left_boundary[0] == pytest.approx((5.0, 4.0))

    assert result.left_boundary[1] == pytest.approx((10.0, 4.0))

    assert result.right_boundary[0] == pytest.approx((5.0, -4.0))

    assert result.right_boundary[1] == pytest.approx((10.0, -4.0))

    assert result.reference_waypoints[0] == pytest.approx((5.0, 0.0))
    assert result.reference_waypoints[1] == pytest.approx((10.0, 0.0))


def test_boundary_points_are_interpolated() -> None:
    """Boundary positions between track points should be interpolated."""

    calculator = TrackPerceptionCalculator(
        lookahead_distances=(5.0,),
        reference_distances=(5.0,),
    )

    # Boundary width changes from 4 m to 6 m.
    track_geometry = SimpleNamespace(
        points=(
            make_track_point(
                0.0,
                (-4.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                position=(1.0, 0.0, 0.0),
            ),
            make_track_point(
                10.0,
                (-6.0, 0.0, 10.0),
                (6.0, 0.0, 10.0),
                position=(3.0, 0.0, 10.0),
            ),
        )
    )

    result = calculator.calculate(
        telemetry=calculator_input(make_telemetry()),
        track_geometry=calculator_input(track_geometry),
        current_s=0.0,
        track_length=10.0,
    )

    # Halfway between 4 m and 6 m width -> 5 m.
    assert result.left_boundary[0] == pytest.approx((5.0, 5.0))

    assert result.right_boundary[0] == pytest.approx((5.0, -5.0))

    # Interpolate the AI line itself, rather than the midpoint of the edges.
    assert result.reference_waypoints[0] == pytest.approx((5.0, -2.0))


def test_vehicle_rotation_is_handled_correctly() -> None:
    """World coordinates should be transformed into the vehicle frame."""

    calculator = TrackPerceptionCalculator(
        lookahead_distances=(5.0,),
        reference_distances=(5.0,),
    )

    # Vehicle points along +x.
    #
    # Vehicle left is now +z.
    tyre_contact_points = make_vehicle_tyres(
        forward=(1.0, 0.0),
        left=(0.0, 1.0),
    )

    telemetry = make_telemetry(
        tyre_contact_points=tyre_contact_points,
    )

    track_geometry = SimpleNamespace(
        points=(
            make_track_point(
                0.0,
                (0.0, 0.0, 4.0),
                (0.0, 0.0, -4.0),
            ),
            make_track_point(
                10.0,
                (10.0, 0.0, 4.0),
                (10.0, 0.0, -4.0),
            ),
        )
    )

    result = calculator.calculate(
        telemetry=calculator_input(telemetry),
        track_geometry=calculator_input(track_geometry),
        current_s=0.0,
        track_length=10.0,
    )

    assert result.left_boundary[0] == pytest.approx((5.0, 4.0))

    assert result.right_boundary[0] == pytest.approx((5.0, -4.0))

    assert result.reference_waypoints[0] == pytest.approx((5.0, 0.0))


def test_current_s_is_added_to_lookahead_distance() -> None:
    """Lookahead distance should be relative to the current track position."""

    calculator = TrackPerceptionCalculator(
        lookahead_distances=(5.0,),
        reference_distances=(5.0,),
    )

    track_geometry = SimpleNamespace(
        points=(
            make_track_point(
                0.0,
                (-4.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
            ),
            make_track_point(
                10.0,
                (-4.0, 0.0, 10.0),
                (4.0, 0.0, 10.0),
            ),
            make_track_point(
                20.0,
                (-4.0, 0.0, 20.0),
                (4.0, 0.0, 20.0),
            ),
        )
    )

    result = calculator.calculate(
        telemetry=calculator_input(make_telemetry()),
        track_geometry=calculator_input(track_geometry),
        current_s=5.0,
        track_length=20.0,
    )

    # current_s=5 + lookahead=5 -> s=10
    assert result.left_boundary[0] == pytest.approx((10.0, 4.0))

    assert result.right_boundary[0] == pytest.approx((10.0, -4.0))

    assert result.reference_waypoints[0] == pytest.approx((10.0, 0.0))


def test_track_position_wraps_around_start_finish() -> None:
    """Lookahead should wrap around when crossing the lap boundary."""

    calculator = TrackPerceptionCalculator(
        lookahead_distances=(7.0,),
        reference_distances=(7.0,),
    )

    # The vehicle is placed at world z=100.
    #
    # After wrapping:
    # current_s = 98
    # + 7 m
    # = 105
    # % 100
    # = s=5
    #
    # s=5 corresponds to world z=105,
    # which is 5 m ahead of the vehicle.
    telemetry = make_telemetry(
        tyre_contact_points=make_vehicle_tyres(
            origin_z=100.0,
        )
    )

    track_geometry = SimpleNamespace(
        points=(
            make_track_point(
                0.0,
                (-4.0, 0.0, 100.0),
                (4.0, 0.0, 100.0),
            ),
            make_track_point(
                10.0,
                (-4.0, 0.0, 110.0),
                (4.0, 0.0, 110.0),
            ),
            make_track_point(
                50.0,
                (-4.0, 0.0, 150.0),
                (4.0, 0.0, 150.0),
            ),
            make_track_point(
                90.0,
                (-4.0, 0.0, 190.0),
                (4.0, 0.0, 190.0),
            ),
        )
    )

    result = calculator.calculate(
        telemetry=calculator_input(telemetry),
        track_geometry=calculator_input(track_geometry),
        current_s=98.0,
        track_length=100.0,
    )

    assert result.left_boundary[0] == pytest.approx((5.0, 4.0))

    assert result.right_boundary[0] == pytest.approx((5.0, -4.0))

    assert result.reference_waypoints[0] == pytest.approx((5.0, 0.0))


def test_returns_one_boundary_point_per_lookahead_distance() -> None:
    """Each configured lookahead distance should produce one point per side."""

    calculator = TrackPerceptionCalculator(
        lookahead_distances=(
            5.0,
            10.0,
            20.0,
        ),
    )

    track_geometry = SimpleNamespace(
        points=(
            make_track_point(
                0.0,
                (-4.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
            ),
            make_track_point(
                10.0,
                (-4.0, 0.0, 10.0),
                (4.0, 0.0, 10.0),
            ),
            make_track_point(
                20.0,
                (-4.0, 0.0, 20.0),
                (4.0, 0.0, 20.0),
            ),
            make_track_point(
                30.0,
                (-4.0, 0.0, 30.0),
                (4.0, 0.0, 30.0),
            ),
        )
    )

    result = calculator.calculate(
        telemetry=calculator_input(make_telemetry()),
        track_geometry=calculator_input(track_geometry),
        current_s=0.0,
        track_length=30.0,
    )

    assert len(result.left_boundary) == 3
    assert len(result.right_boundary) == 3

    assert isinstance(result.left_boundary, tuple)
    assert isinstance(result.right_boundary, tuple)

    assert len(result.reference_waypoints) == len(
        TrackPerceptionCalculator.DEFAULT_REFERENCE_DISTANCES
    )
    assert isinstance(result.reference_waypoints, tuple)


def test_rejects_invalid_number_of_tyres() -> None:
    """Exactly four tyre contact points are required."""

    calculator = TrackPerceptionCalculator(
        lookahead_distances=(5.0,),
    )

    telemetry = make_telemetry(
        tyre_contact_points=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    )

    track_geometry = SimpleNamespace(
        points=(
            make_track_point(
                0.0,
                (-4.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
            ),
            make_track_point(
                10.0,
                (-4.0, 0.0, 10.0),
                (4.0, 0.0, 10.0),
            ),
        )
    )

    with pytest.raises(ValueError):
        calculator.calculate(
            telemetry=calculator_input(telemetry),
            track_geometry=calculator_input(track_geometry),
            current_s=0.0,
            track_length=10.0,
        )


def test_rejects_track_with_less_than_two_points() -> None:
    """Interpolation requires at least two track points."""

    calculator = TrackPerceptionCalculator(
        lookahead_distances=(5.0,),
    )

    track_geometry = SimpleNamespace(
        points=(
            make_track_point(
                0.0,
                (-4.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
            ),
        )
    )

    with pytest.raises(ValueError):
        calculator.calculate(
            telemetry=calculator_input(make_telemetry()),
            track_geometry=calculator_input(track_geometry),
            current_s=0.0,
            track_length=10.0,
        )


def test_rejects_zero_length_vehicle_forward_vector() -> None:
    """Front and rear axle centres must not occupy the same position."""

    calculator = TrackPerceptionCalculator(
        lookahead_distances=(5.0,),
    )

    tyre_contact_points = (
        (-1.0, 0.0, 0.0),  # FL
        (1.0, 0.0, 0.0),  # FR
        (-1.0, 0.0, 0.0),  # RL
        (1.0, 0.0, 0.0),  # RR
    )

    track_geometry = SimpleNamespace(
        points=(
            make_track_point(
                0.0,
                (-4.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
            ),
            make_track_point(
                10.0,
                (-4.0, 0.0, 10.0),
                (4.0, 0.0, 10.0),
            ),
        )
    )

    with pytest.raises(ValueError):
        calculator.calculate(
            telemetry=calculator_input(make_telemetry(tyre_contact_points)),
            track_geometry=calculator_input(track_geometry),
            current_s=0.0,
            track_length=10.0,
        )
