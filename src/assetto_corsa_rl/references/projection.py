"""Deterministic progress geometry for actual and predicted trajectories."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.types import FloatArray


def wrapped_progress_delta(
    previous_progress_m: float,
    current_progress_m: float,
    track_length_m: float,
) -> float:
    """Return the signed shortest arc delta on a closed reference line."""

    values = np.asarray(
        [previous_progress_m, current_progress_m, track_length_m],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("progress values and track length must be finite")
    if track_length_m <= 0:
        raise ValueError("track_length_m must be positive")
    raw_delta = current_progress_m - previous_progress_m
    if abs(raw_delta) <= 0.5 * track_length_m:
        return float(raw_delta)
    wrapped = (raw_delta + 0.5 * track_length_m) % track_length_m
    return float(wrapped - 0.5 * track_length_m)


def project_points_to_polyline_progress(
    points_xz_m: npt.ArrayLike,
    reference_polyline_xz_m: npt.ArrayLike,
    reference_progress_m: npt.ArrayLike | None = None,
) -> FloatArray:
    """Project points onto an open polyline and return its progress coordinate."""

    points = np.asarray(points_xz_m, dtype=np.float32)
    reference = np.asarray(reference_polyline_xz_m, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
        raise ValueError("points_xz_m must have shape (points>=1, 2)")
    if reference.ndim != 2 or reference.shape[1] != 2 or reference.shape[0] < 2:
        raise ValueError("reference_polyline_xz_m must have shape (points>=2, 2)")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(reference)):
        raise ValueError("projection geometry contains non-finite values")

    segment_start = reference[:-1]
    segment_vector = reference[1:] - segment_start
    segment_length_squared = np.sum(segment_vector * segment_vector, axis=1)
    if np.any(segment_length_squared <= 1e-12):
        raise ValueError("reference polyline contains a zero-length segment")
    segment_length = np.sqrt(segment_length_squared)
    if reference_progress_m is None:
        progress_coordinate = np.concatenate(
            (
                np.zeros(1, dtype=np.float32),
                np.cumsum(segment_length, dtype=np.float32),
            )
        )
    else:
        progress_coordinate = np.asarray(reference_progress_m, dtype=np.float32)
        if (
            progress_coordinate.shape != (reference.shape[0],)
            or not np.all(np.isfinite(progress_coordinate))
            or np.any(np.diff(progress_coordinate) <= 0)
        ):
            raise ValueError(
                "reference_progress_m must be finite, strictly increasing, and match the polyline"
            )

    point_to_start = points[:, None, :] - segment_start[None, :, :]
    fraction = np.sum(point_to_start * segment_vector[None, :, :], axis=-1)
    fraction /= segment_length_squared[None, :]
    fraction = np.clip(fraction, 0.0, 1.0)
    projected = segment_start[None, :, :] + fraction[..., None] * segment_vector[None, :, :]
    distance_squared = np.sum((points[:, None, :] - projected) ** 2, axis=-1)
    closest_segment = np.argmin(distance_squared, axis=1)
    row = np.arange(points.shape[0])
    progress = (
        progress_coordinate[closest_segment]
        + fraction[row, closest_segment] * np.diff(progress_coordinate)[closest_segment]
    )
    return np.asarray(progress, dtype=np.float32)


def project_points_to_closed_reference_progress(
    points_xy_m: npt.ArrayLike,
    reference_xy_m: npt.ArrayLike,
    reference_distance_m: npt.ArrayLike,
    track_length_m: float,
) -> FloatArray:
    """Project world points onto a closed reference's arc coordinate."""

    points = np.asarray(points_xy_m, dtype=np.float32)
    reference = np.asarray(reference_xy_m, dtype=np.float32)
    distance = np.asarray(reference_distance_m, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
        raise ValueError("points_xy_m must have shape (points>=1, 2)")
    if reference.ndim != 2 or reference.shape[1] != 2 or reference.shape[0] < 3:
        raise ValueError("reference_xy_m must have shape (points>=3, 2)")
    if distance.shape != (reference.shape[0],):
        raise ValueError("reference_distance_m must match reference_xy_m")
    if (
        not np.all(np.isfinite(points))
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(distance))
        or not np.isfinite(track_length_m)
    ):
        raise ValueError("closed-reference projection contains non-finite values")
    if (
        track_length_m <= 0
        or not np.isclose(distance[0], 0.0, atol=1e-5)
        or np.any(np.diff(distance) <= 0)
        or distance[-1] >= track_length_m
    ):
        raise ValueError("reference distances must increase in [0, track_length_m)")

    segment_start = reference
    segment_vector = np.roll(reference, -1, axis=0) - reference
    segment_length_squared = np.sum(segment_vector * segment_vector, axis=1)
    if np.any(segment_length_squared <= 1e-12):
        raise ValueError("closed reference contains a zero-length segment")
    reference_delta = np.diff(np.append(distance.astype(np.float64), float(track_length_m))).astype(
        np.float32
    )

    point_to_start = points[:, None, :] - segment_start[None, :, :]
    fraction = np.sum(point_to_start * segment_vector[None, :, :], axis=-1)
    fraction /= segment_length_squared[None, :]
    fraction = np.clip(fraction, 0.0, 1.0)
    projected = segment_start[None, :, :] + fraction[..., None] * segment_vector[None, :, :]
    distance_squared = np.sum((points[:, None, :] - projected) ** 2, axis=-1)
    closest_segment = np.argmin(distance_squared, axis=1)
    row = np.arange(points.shape[0])
    progress = (
        distance[closest_segment]
        + fraction[row, closest_segment] * reference_delta[closest_segment]
    )
    return np.asarray(np.mod(progress, track_length_m), dtype=np.float32)
