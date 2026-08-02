"""Reproducible future-trajectory targets for the auxiliary prediction task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.references.fixed_reference import ReferenceTrajectory
from assetto_corsa_rl.types import REFERENCE_TIME_OFFSETS_S, FloatArray


@dataclass(frozen=True, slots=True)
class FutureTrajectoryTargets:
    """Ego-frame paths and a mask excluding incomplete episode horizons."""

    trajectory_xz_m: FloatArray
    valid_mask: npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ReferenceProjectionPaths:
    """Reference geometry and exact relative arc coordinate for reward shaping."""

    points_xz_m: FloatArray
    progress_m: FloatArray
    valid_mask: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xz_m, dtype=np.float32)
        progress = np.asarray(self.progress_m, dtype=np.float32)
        valid = np.asarray(self.valid_mask, dtype=np.bool_)
        if points.ndim != 3 or points.shape[-1] != 2 or points.shape[1] < 2:
            raise ValueError("points_xz_m must have shape (steps, points>=2, 2)")
        if progress.shape != points.shape[:2]:
            raise ValueError("progress_m must match the reference path points")
        if valid.shape != (points.shape[0],):
            raise ValueError("valid_mask must have shape (steps,)")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(progress)):
            raise ValueError("reference projection paths contain non-finite values")
        if np.any(np.abs(progress[valid, 0]) > 1e-6) or np.any(
            np.diff(progress[valid], axis=1) <= 0
        ):
            raise ValueError("valid reference progress must start at zero and increase")
        object.__setattr__(self, "points_xz_m", points)
        object.__setattr__(self, "progress_m", progress)
        object.__setattr__(self, "valid_mask", valid)


def _validate_pose_series(
    positions_xy_m: npt.ArrayLike,
    heading_rad: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    positions = np.asarray(positions_xy_m, dtype=np.float32)
    headings = np.asarray(heading_rad, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions_xy_m must have shape (steps, 2)")
    if headings.shape != (positions.shape[0],):
        raise ValueError("heading_rad must have shape (steps,)")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(headings)):
        raise ValueError("pose series contains non-finite values")
    return positions, headings


def _world_points_to_ego(
    world_points: npt.NDArray[np.float64] | FloatArray,
    origin_xy_m: FloatArray,
    heading_rad: float,
) -> FloatArray:
    displacement = np.asarray(world_points, dtype=np.float64) - origin_xy_m.astype(np.float64)
    cosine = np.cos(heading_rad)
    sine = np.sin(heading_rad)
    world_to_ego = np.asarray([[-sine, cosine], [cosine, sine]], dtype=np.float64)
    return np.asarray(displacement @ world_to_ego.T, dtype=np.float32)


def build_observed_trajectory_targets(
    *,
    positions_xy_m: npt.ArrayLike,
    heading_rad: npt.ArrayLike,
    simulation_time_s: npt.ArrayLike,
    trajectory_id: npt.ArrayLike,
    time_offsets_s: npt.ArrayLike = REFERENCE_TIME_OFFSETS_S,
) -> FutureTrajectoryTargets:
    """Interpolate actually observed future positions without crossing episodes."""

    positions, headings = _validate_pose_series(positions_xy_m, heading_rad)
    times = np.asarray(simulation_time_s, dtype=np.float64)
    trajectory_ids = np.asarray(trajectory_id)
    offsets = np.asarray(time_offsets_s, dtype=np.float64)
    step_count = positions.shape[0]
    if times.shape != (step_count,) or trajectory_ids.shape != (step_count,):
        raise ValueError("simulation_time_s and trajectory_id must have shape (steps,)")
    if (
        offsets.ndim != 1
        or offsets.size < 2
        or not np.all(np.isfinite(offsets))
        or np.any(offsets <= 0)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("time_offsets_s must be finite, positive and strictly increasing")
    if not np.all(np.isfinite(times)):
        raise ValueError("simulation_time_s contains non-finite values")

    trajectories = np.zeros((step_count, offsets.size, 2), dtype=np.float32)
    valid = np.zeros(step_count, dtype=np.bool_)
    segment_start = 0
    while segment_start < step_count:
        segment_end = segment_start
        while (
            segment_end + 1 < step_count
            and trajectory_ids[segment_end + 1] == trajectory_ids[segment_start]
        ):
            segment_end += 1
        segment_times = times[segment_start : segment_end + 1]
        if segment_times.size > 1 and np.any(np.diff(segment_times) <= 0):
            raise ValueError("simulation time must increase strictly within each episode")

        for origin_index in range(segment_start, segment_end + 1):
            query_times = times[origin_index] + offsets
            if query_times[-1] > times[segment_end] + 1e-9:
                continue
            interpolation_times = times[origin_index : segment_end + 1]
            future_x = np.asarray(
                np.interp(
                    query_times,
                    interpolation_times,
                    positions[origin_index : segment_end + 1, 0],
                ),
                dtype=np.float64,
            )
            future_y = np.asarray(
                np.interp(
                    query_times,
                    interpolation_times,
                    positions[origin_index : segment_end + 1, 1],
                ),
                dtype=np.float64,
            )
            future_world = np.asarray(np.column_stack((future_x, future_y)), dtype=np.float64)
            trajectories[origin_index] = _world_points_to_ego(
                future_world,
                positions[origin_index],
                float(headings[origin_index]),
            )
            valid[origin_index] = True
        segment_start = segment_end + 1

    return FutureTrajectoryTargets(trajectory_xz_m=trajectories, valid_mask=valid)


def build_reference_trajectory_targets(
    *,
    reference: ReferenceTrajectory,
    progress_m: npt.ArrayLike,
    positions_xy_m: npt.ArrayLike,
    heading_rad: npt.ArrayLike,
    valid_mask: npt.ArrayLike,
    time_offsets_s: npt.ArrayLike = REFERENCE_TIME_OFFSETS_S,
) -> FutureTrajectoryTargets:
    """Build reference paths in exactly the same ego frames as observed targets."""

    projection_paths = build_reference_projection_paths(
        reference=reference,
        progress_m=progress_m,
        positions_xy_m=positions_xy_m,
        heading_rad=heading_rad,
        valid_mask=valid_mask,
        time_offsets_s=time_offsets_s,
    )
    valid = np.asarray(valid_mask, dtype=np.bool_)
    return FutureTrajectoryTargets(
        trajectory_xz_m=projection_paths.points_xz_m[:, 1:],
        valid_mask=valid.copy(),
    )


def build_reference_projection_paths(
    *,
    reference: ReferenceTrajectory,
    progress_m: npt.ArrayLike,
    positions_xy_m: npt.ArrayLike,
    heading_rad: npt.ArrayLike,
    valid_mask: npt.ArrayLike,
    time_offsets_s: npt.ArrayLike = REFERENCE_TIME_OFFSETS_S,
) -> ReferenceProjectionPaths:
    """Build current-plus-future reference polylines for progress projection."""

    positions, headings = _validate_pose_series(positions_xy_m, heading_rad)
    progress = np.asarray(progress_m, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if progress.shape != (positions.shape[0],) or valid.shape != (positions.shape[0],):
        raise ValueError("progress_m and valid_mask must have shape (steps,)")
    if not np.all(np.isfinite(progress)):
        raise ValueError("progress_m contains non-finite values")

    offsets = np.asarray(time_offsets_s, dtype=np.float32)
    trajectories = np.zeros((positions.shape[0], offsets.size + 1, 2), dtype=np.float32)
    reference_progress = np.zeros(
        (positions.shape[0], offsets.size + 1),
        dtype=np.float32,
    )
    for index in np.flatnonzero(valid):
        trajectories[index] = reference.ego_path_with_origin(
            float(progress[index]),
            positions[index],
            float(headings[index]),
            offsets,
        )
        current_progress = float(progress[index] % reference.track_length_m)
        future_progress = reference.future_distances(float(progress[index]), offsets)
        wrapped_nodes = np.concatenate(
            (
                np.asarray([current_progress], dtype=np.float32),
                future_progress,
            )
        )
        reference_progress[index, 1:] = np.cumsum(
            np.mod(np.diff(wrapped_nodes), reference.track_length_m),
            dtype=np.float32,
        )
    return ReferenceProjectionPaths(
        points_xz_m=trajectories,
        progress_m=reference_progress,
        valid_mask=valid.copy(),
    )
