"""Represent simulator-independent race track geometry."""

from dataclasses import dataclass
import numpy as np

Vector3 = tuple[float, float, float] # Alias


def _normalize(vector: Vector3 | np.ndarray) -> np.ndarray:
    """Return a normalized 3D vector."""

    array = np.asarray(vector, dtype=float)
    length = np.linalg.norm(array)

    if length == 0.0:
        raise ValueError("Cannot normalize a zero vector.")

    return array / length


@dataclass(frozen=True) # Make TrackPoint immutable
class TrackPoint:
    """Represent one sampled point along a race track. Can be used to reconstruct the track geometry, 
    left and right boundaries."""

    position: Vector3
    length: float

    side_left: float
    side_right: float

    normal: Vector3
    forward: Vector3
    reference_speed: float | None = None # Raw AI-file speed; unit must be validated by the caller.

    @property
    def left_boundary(self) -> Vector3:
        """Return the left track boundary position."""

        left_direction = _normalize(np.cross(self.normal, self.forward))

        position = np.asarray(self.position, dtype=float)

        boundary = position + left_direction * self.side_left

        return (
            float(boundary[0]),
            float(boundary[1]),
            float(boundary[2]),
        )

    @property
    def right_boundary(self) -> Vector3:
        """Return the right track boundary position."""

        left_direction = _normalize(np.cross(self.normal, self.forward))

        position = np.asarray(self.position, dtype=float)

        boundary = position - left_direction * self.side_right

        return (
            float(boundary[0]),
            float(boundary[1]),
            float(boundary[2]),
        )


class TrackGeometry:
    """Represent the geometry of a complete race track."""

    def __init__(self, points: list[TrackPoint]) -> None:
        if not points:
            raise ValueError("Track geometry requires at least one point.")

        self._points = tuple(points) # Store points as an immutable tuple to prevent accidental modification

    @property
    def points(self) -> tuple[TrackPoint, ...]:
        """Return all sampled track points."""
        return self._points

    def __len__(self) -> int:
        """Return the number of sampled track points."""
        return len(self._points)
