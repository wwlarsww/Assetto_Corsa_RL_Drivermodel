import pytest

from assetto_corsa_ai.track_geometry import TrackGeometry, TrackPoint


def test_track_point_boundaries() -> None:
    """Calculate left and right boundaries for a simple straight track."""

    point = TrackPoint(
        position=(0.0, 0.0, 0.0),
        length=0.0,
        side_left=5.0,
        side_right=4.0,
        normal=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, -1.0),
    )

    # The left boundary should be 5 units to the left of the position, which is at (-5, 0, 0)
    assert point.left_boundary == (-5.0, 0.0, 0.0)
    # The right boundary should be 4 units to the right of the position, which is at (4, 0, 0)
    assert point.right_boundary == (4.0, 0.0, 0.0)

def test_track_point_boundaries_normalize_direction() -> None:
    """Boundary distances should not depend on vector magnitude."""

    point = TrackPoint(
        position=(10.0, 2.0, 20.0),
        length=100.0,
        side_left=5.0,
        side_right=4.0,
        normal=(0.0, 2.0, 0.0),
        forward=(0.0, 0.0, -3.0),
    )

    # The left and right boundaries should be calculated based on the normalized direction vectors, so the result should be the same as in the previous test.
    assert point.left_boundary == pytest.approx((5.0, 2.0, 20.0))

    # The right boundary should also be calculated based on the normalized direction vectors, so the result should be the same as in the previous test.
    assert point.right_boundary == pytest.approx((14.0, 2.0, 20.0))


def test_track_geometry_contains_points() -> None:
    """Store track points in a track geometry."""

    point = TrackPoint(
        position=(0.0, 0.0, 0.0),
        length=0.0,
        side_left=5.0,
        side_right=4.0,
        normal=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, -1.0),
    )

    track = TrackGeometry([point])

    assert len(track) == 1 # Check that the track geometry contains one point
    assert track.points[0] == point # Check that the point in the track geometry is the same as the original point


def test_track_geometry_rejects_empty_track() -> None:
    """Reject track geometry without any points."""

    with pytest.raises(ValueError):
        TrackGeometry([])


