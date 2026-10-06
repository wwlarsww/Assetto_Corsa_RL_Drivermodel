"""Verify projection with analytic geometry, without a running simulator."""

import math

import pytest

from assetto_corsa_ai.track_geometry import TrackGeometry, TrackPoint, Vector3
from assetto_corsa_ai.track_projection import TrackProjector


def point(position: Vector3, *, left: float = 5.0) -> TrackPoint:
    return TrackPoint(position, 0.0, left, 4.0, (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))


@pytest.mark.parametrize("offset", [-6.0, 0.0, 2.0, 7.0])
def test_straight_track_signed_offsets(offset: float) -> None:
    track = TrackGeometry([point((0.0, 0.0, 0.0)), point((0.0, 0.0, -10.0))])
    result = TrackProjector(track, closed=False).project((-offset, 3.0, -4.0))
    assert result.position == pytest.approx((0.0, 0.0, -4.0))
    assert result.segment_fraction == pytest.approx(0.4)
    assert result.distance_to_spline == pytest.approx(math.hypot(offset, 3.0))
    assert result.lateral_offset == pytest.approx(offset)
    assert result.left_clearance == pytest.approx(5.0 - offset)
    assert result.right_clearance == pytest.approx(4.0 + offset)
    assert result.distance_along_track == pytest.approx(4.0)
    assert result.normalized_track_position == pytest.approx(0.4)


def test_widths_are_interpolated() -> None:
    track = TrackGeometry([
        point((0.0, 0.0, 0.0), left=2.0),
        point((0.0, 0.0, -10.0), left=6.0),
    ])
    result = TrackProjector(track, closed=False).project((-1.0, 0.0, -5.0))
    assert result.left_clearance == pytest.approx(3.0)


def test_corner_selects_nearest_segment() -> None:
    track = TrackGeometry([
        point((0.0, 0.0, 0.0)), point((0.0, 0.0, -10.0)),
        point((10.0, 0.0, -10.0)),
    ])
    result = TrackProjector(track, closed=False).project((6.0, 0.0, -8.0))
    assert result.segment_index == 1
    assert result.position == pytest.approx((6.0, 0.0, -10.0))
    assert result.lateral_offset == pytest.approx(-2.0)
    assert result.distance_along_track == pytest.approx(16.0)


def test_closed_track_includes_closing_segment_and_wraps_progress() -> None:
    track = TrackGeometry([
        point((0.0, 0.0, 0.0)), point((0.0, 0.0, -10.0)),
        point((10.0, 0.0, -10.0)), point((10.0, 0.0, 0.0)),
    ])
    projector = TrackProjector(track)
    result = projector.project((5.0, 0.0, 1.0))
    assert projector.track_length == pytest.approx(40.0)
    assert result.segment_index == 3
    assert result.position == pytest.approx((5.0, 0.0, 0.0))
    assert result.normalized_track_position == pytest.approx(0.875)
    assert projector.project((0.0, 0.0, 0.0)).normalized_track_position == 0.0
    assert projector.project((0.01, 0.0, 0.0)).normalized_track_position > 0.99
    assert projector.project((0.0, 0.0, -0.01)).normalized_track_position < 0.01


@pytest.mark.parametrize("z, fraction", [(5.0, 0.0), (-15.0, 1.0)])
def test_open_track_clamps_to_endpoints(z: float, fraction: float) -> None:
    track = TrackGeometry([point((0.0, 0.0, 0.0)), point((0.0, 0.0, -10.0))])
    result = TrackProjector(track, closed=False).project((0.0, 0.0, z))
    assert result.segment_fraction == fraction
    assert result.normalized_track_position == fraction
    assert result.lateral_offset == 0.0


def test_duplicate_points_are_skipped_and_keep_original_segment_index() -> None:
    track = TrackGeometry([
        point((0.0, 0.0, 0.0)), point((0.0, 0.0, 0.0)),
        point((0.0, 0.0, -10.0)),
    ])
    result = TrackProjector(track, closed=False).project((0.0, 0.0, -5.0))
    assert result.segment_index == 1
    assert result.distance_along_track == pytest.approx(5.0)


def test_elevation_distinguishes_overlapping_segments() -> None:
    track = TrackGeometry([
        point((0.0, 0.0, 0.0)), point((0.0, 0.0, -10.0)),
        point((0.0, 10.0, -10.0)), point((0.0, 10.0, 0.0)),
    ])
    result = TrackProjector(track, closed=False).project((0.0, 9.0, -5.0))
    assert result.segment_index == 2
    assert result.distance_to_spline == pytest.approx(1.0)


@pytest.mark.parametrize("positions", [
    [(0.0, 0.0, 0.0)],
    [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    [(0.0, 0.0, 0.0), (math.nan, 0.0, 0.0)],
])
def test_invalid_geometry_is_rejected(positions: list[Vector3]) -> None:
    with pytest.raises(ValueError):
        TrackProjector(TrackGeometry([point(p) for p in positions]))


def test_non_finite_query_is_rejected() -> None:
    track = TrackGeometry([point((0.0, 0.0, 0.0)), point((0.0, 0.0, -10.0))])
    with pytest.raises(ValueError):
        TrackProjector(track).project((math.inf, 0.0, 0.0))


def test_banked_surface_separates_height_from_lateral_offset() -> None:
    normal = (1.0, 1.0, 0.0)
    track = TrackGeometry([
        TrackPoint((0.0, 0.0, 0.0), 0.0, 5.0, 4.0, normal, (0.0, 0.0, -1.0)),
        TrackPoint((0.0, 0.0, -10.0), 10.0, 5.0, 4.0, normal, (0.0, 0.0, -1.0)),
    ])
    # Two metres along the banked left axis, three along the surface normal.
    scale = 1.0 / math.sqrt(2.0)
    result = TrackProjector(track, closed=False).project((scale, 5.0 * scale, -5.0))
    assert result.lateral_offset == pytest.approx(2.0)
    assert result.distance_to_spline == pytest.approx(math.sqrt(13.0))


def test_exact_tie_selects_first_segment() -> None:
    track = TrackGeometry([
        point((0.0, 0.0, 0.0)), point((0.0, 0.0, -10.0)),
        point((0.0, 0.0, 0.0)),
    ])
    assert TrackProjector(track).project((1.0, 0.0, -5.0)).segment_index == 0


@pytest.mark.parametrize("normal, left", [
    ((0.0, 0.0, 0.0), 5.0), ((0.0, 1.0, 0.0), -1.0),
])
def test_invalid_normals_and_widths_are_rejected(normal: Vector3, left: float) -> None:
    track = TrackGeometry([
        TrackPoint((0.0, 0.0, 0.0), 0.0, left, 4.0, normal, (0.0, 0.0, -1.0)),
        point((0.0, 0.0, -10.0)),
    ])
    with pytest.raises(ValueError):
        TrackProjector(track)


def test_parallel_normal_and_tangent_cannot_define_lateral_axis() -> None:
    track = TrackGeometry([
        TrackPoint((0.0, 0.0, 0.0), 0.0, 5.0, 4.0, (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        TrackPoint((0.0, 0.0, -10.0), 10.0, 5.0, 4.0, (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
    ])
    with pytest.raises(ValueError, match="lateral direction"):
        TrackProjector(track, closed=False).project((0.0, 0.0, -5.0))
