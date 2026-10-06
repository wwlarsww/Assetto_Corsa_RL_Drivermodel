"""Project world positions onto simulator-independent track geometry."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from assetto_corsa_ai.track_geometry import TrackGeometry, Vector3


@dataclass(frozen=True)
class TrackProjection:
    """Describe the nearest position on the sampled AI spline.

    Distances are in metres. Lateral offset is positive to the left in
    sample order. Clearances are signed lateral margins, not Euclidean
    distances to boundary curves; negative means beyond that boundary.
    """

    position: Vector3
    segment_index: int # Index of the segment in the track geometry
    segment_fraction: float # Fraction along the segment, from 0.0 to 1.0
    distance_to_spline: float # Euclidean distance from the query position to the nearest point on the spline
    distance_along_track: float # Distance along the track from the start to the projected point
    normalized_track_position: float # Normalized position along the track, from 0.0 to 1.0
    lateral_offset: float # Lateral offset from the spline, positive to the left in sample order
    left_clearance: float
    right_clearance: float


class TrackProjector:
    """Find the nearest point on a 3D polyline using cached segment arrays.

    Closed tracks include the last-to-first segment. Zero-length segments
    are skipped. Exact distance ties select the first segment in sample
    order. The projector is stateless and performs a global O(N) search.
    """

    def __init__(self, track: TrackGeometry, *, closed: bool = True) -> None:
        points = track.points
        if len(points) < 2:
            raise ValueError("Projection requires at least two track points.")

        # Precompute arrays for efficient projection calculations.
        positions = np.asarray([p.position for p in points], dtype=np.float64)
        normals = np.asarray([p.normal for p in points], dtype=np.float64)
        widths = np.asarray(
            [(p.side_left, p.side_right) for p in points], dtype=np.float64
        )
        # Validate that the arrays contain finite values and meet the required conditions.
        if not all(np.isfinite(a).all() for a in (positions, normals, widths)):
            raise ValueError("Track positions, normals and widths must be finite.")
        if np.any(widths < 0):
            raise ValueError("Track widths must be non-negative.")
        normal_lengths = np.linalg.norm(normals, axis=1)
        if np.any(normal_lengths == 0):
            raise ValueError("Track surface normals must be non-zero.")
        normals = normals / normal_lengths[:, None]

        # Compute segment vectors, lengths, and offsets for projection calculations.
        starts = np.arange(len(points) if closed else len(points) - 1)
        ends = (starts + 1) % len(points)
        vectors = positions[ends] - positions[starts]
        lengths = np.linalg.norm(vectors, axis=1)
        valid = lengths > 0
        if not np.any(valid):
            raise ValueError("Track must contain a non-zero-length segment.")

        self._closed = closed
        self._track_length = float(lengths.sum())
        self._indices = starts[valid]
        self._starts = positions[starts[valid]]
        self._vectors = vectors[valid]
        self._lengths = lengths[valid]
        self._offsets = np.concatenate(([0.0], np.cumsum(lengths)[:-1]))[valid]
        self._start_normals = normals[starts[valid]]
        self._end_normals = normals[ends[valid]]
        self._start_widths = widths[starts[valid]]
        self._end_widths = widths[ends[valid]]

    @property
    def track_length(self) -> float:
        """Return the total 3D polyline length in metres."""
        return self._track_length

    def project(self, position: Vector3) -> TrackProjection:
        """Project onto the nearest segment and interpolate local track widths.

        The lateral axis uses the surface normal and segment tangent, with
        the same cross-product convention as TrackPoint.left_boundary.
        Height and longitudinal endpoint residuals are excluded from lateral
        offset, but included in distance_to_spline.
        """
        query: NDArray[np.float64] = np.asarray(position, dtype=np.float64)
        if query.shape != (3,) or not np.isfinite(query).all():
            raise ValueError("Position must contain three finite coordinates.")

        fractions = np.clip(
            np.sum((query - self._starts) * self._vectors, axis=1)
            / (self._lengths * self._lengths),
            0.0,
            1.0,
        )
        candidates = self._starts + fractions[:, None] * self._vectors
        residuals = query - candidates
        squared_distances = np.sum(residuals * residuals, axis=1)
        index = int(np.argmin(squared_distances))
        fraction = float(fractions[index])
        projected = candidates[index]

        normal = (
            (1.0 - fraction) * self._start_normals[index]
            + fraction * self._end_normals[index]
        )
        tangent = self._vectors[index] / self._lengths[index]
        left = np.cross(normal, tangent)
        left_length = float(np.linalg.norm(left))
        if left_length <= 1e-12:
            raise ValueError("Surface normal does not define a lateral direction.")
        lateral_offset = float(np.dot(residuals[index], left / left_length))
        widths = (
            (1.0 - fraction) * self._start_widths[index]
            + fraction * self._end_widths[index]
        )
        distance_along = float(
            self._offsets[index] + fraction * self._lengths[index]
        )
        if self._closed:
            distance_along %= self._track_length

        return TrackProjection(
            position=(float(projected[0]), float(projected[1]), float(projected[2])),
            segment_index=int(self._indices[index]),
            segment_fraction=fraction,
            distance_to_spline=float(np.sqrt(squared_distances[index])),
            distance_along_track=distance_along,
            normalized_track_position=distance_along / self._track_length,
            lateral_offset=lateral_offset,
            left_clearance=float(widths[0]) - lateral_offset,
            right_clearance=float(widths[1]) + lateral_offset,
        )
