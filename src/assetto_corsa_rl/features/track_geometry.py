"""Deterministic closed-track geometry shared by offline and live adapters."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.types import TRACK_PREVIEW_DISTANCES_M, FloatArray

if TYPE_CHECKING:
    from assetto_corsa_rl.ac_integration.ai_spline import AssettoCorsaAISpline


def periodic_interpolate(
    query_distance_m: npt.ArrayLike,
    distance_m: FloatArray,
    values: FloatArray,
    track_length_m: float,
) -> FloatArray:
    """Interpolate values on one closed lap using its configured arc coordinate."""

    query = np.mod(np.asarray(query_distance_m, dtype=np.float64), track_length_m)
    nodes = np.append(np.asarray(distance_m, dtype=np.float64), track_length_m)
    periodic_values = np.append(np.asarray(values, dtype=np.float64), float(values[0]))
    return np.asarray(
        np.interp(query.reshape(-1), nodes, periodic_values).reshape(query.shape),
        dtype=np.float32,
    )


def world_to_ego(
    world_points_xy_m: npt.ArrayLike,
    origin_xy_m: npt.ArrayLike,
    heading_rad: npt.ArrayLike,
) -> FloatArray:
    """Transform batched world points to ``x=left, z=forward`` vehicle frames."""

    points = np.asarray(world_points_xy_m, dtype=np.float32)
    origins = np.asarray(origin_xy_m, dtype=np.float32)
    headings = np.asarray(heading_rad, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("world_points_xy_m must have shape (samples, points, 2)")
    if origins.shape != (points.shape[0], 2):
        raise ValueError("origin_xy_m must have shape (samples, 2)")
    if headings.shape != (points.shape[0],):
        raise ValueError("heading_rad must have shape (samples,)")
    if not all(np.all(np.isfinite(array)) for array in (points, origins, headings)):
        raise ValueError("world-to-ego inputs must contain only finite values")

    displacement = points - origins[:, None, :]
    cosine = np.cos(headings)[:, None]
    sine = np.sin(headings)[:, None]
    forward = displacement[..., 0] * cosine + displacement[..., 1] * sine
    left = -displacement[..., 0] * sine + displacement[..., 1] * cosine
    return np.asarray(np.stack((left, forward), axis=-1), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class TrackGeometry:
    """Closed left/right boundaries parameterised by the baseline arc coordinate."""

    distance_m: FloatArray
    track_length_m: float
    left_xy_m: FloatArray
    right_xy_m: FloatArray

    def __post_init__(self) -> None:
        distance = np.asarray(self.distance_m, dtype=np.float32)
        left = np.asarray(self.left_xy_m, dtype=np.float32)
        right = np.asarray(self.right_xy_m, dtype=np.float32)
        if distance.ndim != 1 or distance.size < 3:
            raise ValueError("track geometry must contain at least three points")
        if left.shape != (distance.size, 2) or right.shape != (distance.size, 2):
            raise ValueError("track boundaries must have shape (points, 2)")
        if not np.isfinite(self.track_length_m) or self.track_length_m <= 0:
            raise ValueError("track_length_m must be positive and finite")
        if not np.isclose(distance[0], 0.0) or np.any(np.diff(distance) <= 0):
            raise ValueError("track geometry distances must start at zero and increase")
        if distance[-1] >= self.track_length_m:
            raise ValueError("track geometry distances must lie below track_length_m")
        if not all(np.all(np.isfinite(array)) for array in (distance, left, right)):
            raise ValueError("track geometry contains non-finite values")
        object.__setattr__(self, "distance_m", distance)
        object.__setattr__(self, "left_xy_m", left)
        object.__setattr__(self, "right_xy_m", right)

    @classmethod
    def from_csv(cls, path: str | Path, track_length_m: float) -> Self:
        """Load the pinned AssettoCorsaGym boundary CSV without pickle support."""

        required = {
            "left_border_x",
            "left_border_y",
            "right_border_x",
            "right_border_y",
            "pos_x",
            "pos_y",
        }
        with Path(path).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required <= set(reader.fieldnames):
                raise ValueError("track geometry CSV misses required boundary columns")
            rows = list(reader)
        if len(rows) < 3:
            raise ValueError("track geometry must contain at least three points")

        def columns(x_name: str, y_name: str) -> FloatArray:
            return np.asarray(
                [[float(row[x_name]), float(row[y_name])] for row in rows],
                dtype=np.float32,
            )

        centre = columns("pos_x", "pos_y")
        segment_length = np.linalg.norm(np.roll(centre, -1, axis=0) - centre, axis=1)
        measured_length = float(segment_length.sum())
        if not np.isfinite(measured_length) or measured_length <= 0:
            raise ValueError("track geometry has invalid cumulative length")
        distance = np.concatenate(([0.0], np.cumsum(segment_length[:-1], dtype=np.float64)))
        distance *= track_length_m / measured_length
        return cls(
            distance_m=np.asarray(distance, dtype=np.float32),
            track_length_m=track_length_m,
            left_xy_m=columns("left_border_x", "left_border_y"),
            right_xy_m=columns("right_border_x", "right_border_y"),
        )

    @classmethod
    def from_ai_spline(cls, spline: AssettoCorsaAISpline) -> Self:
        """Build deterministic boundary geometry from AC's loaded AI-line data."""

        left, right = spline.boundaries_xy_m()
        return cls(
            distance_m=spline.distance_m,
            track_length_m=spline.track_length_m,
            left_xy_m=left,
            right_xy_m=right,
        )

    def preview_world(self, progress_m: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Sample both world-space boundaries at the seven observation distances."""

        progress = np.asarray(progress_m, dtype=np.float32)
        if progress.ndim != 1 or not np.all(np.isfinite(progress)):
            raise ValueError("progress_m must be a finite one-dimensional array")
        query = progress[:, None] + TRACK_PREVIEW_DISTANCES_M[None, :]
        left = np.stack(
            (
                periodic_interpolate(
                    query, self.distance_m, self.left_xy_m[:, 0], self.track_length_m
                ),
                periodic_interpolate(
                    query, self.distance_m, self.left_xy_m[:, 1], self.track_length_m
                ),
            ),
            axis=-1,
        )
        right = np.stack(
            (
                periodic_interpolate(
                    query, self.distance_m, self.right_xy_m[:, 0], self.track_length_m
                ),
                periodic_interpolate(
                    query, self.distance_m, self.right_xy_m[:, 1], self.track_length_m
                ),
            ),
            axis=-1,
        )
        return np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)

    def preview_ego(
        self,
        progress_m: float,
        position_xy_m: npt.ArrayLike,
        heading_rad: float,
    ) -> tuple[FloatArray, FloatArray]:
        """Return one frame of left/right boundary points in vehicle coordinates."""

        progress = np.asarray([progress_m], dtype=np.float32)
        position = np.asarray(position_xy_m, dtype=np.float32)
        if position.shape != (2,):
            raise ValueError("position_xy_m must have shape (2,)")
        left_world, right_world = self.preview_world(progress)
        origins = position[None, :]
        headings = np.asarray([heading_rad], dtype=np.float32)
        return (
            world_to_ego(left_world, origins, headings)[0],
            world_to_ego(right_world, origins, headings)[0],
        )
