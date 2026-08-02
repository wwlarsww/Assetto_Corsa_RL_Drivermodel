from __future__ import annotations

import numpy as np
import pytest

from assetto_corsa_rl.references.fixed_reference import ReferenceTrajectory
from assetto_corsa_rl.training.targets import (
    build_observed_trajectory_targets,
    build_reference_projection_paths,
    build_reference_trajectory_targets,
)


def test_observed_targets_are_ego_centric_and_do_not_cross_episodes() -> None:
    positions = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
        dtype=np.float32,
    )
    targets = build_observed_trajectory_targets(
        positions_xy_m=positions,
        heading_rad=np.zeros(5, dtype=np.float32),
        simulation_time_s=np.asarray([0.0, 0.5, 1.0, 0.0, 0.5]),
        trajectory_id=np.asarray([0, 0, 0, 1, 1]),
        time_offsets_s=[0.5, 1.0],
    )
    np.testing.assert_array_equal(targets.valid_mask, [True, False, False, False, False])
    np.testing.assert_allclose(targets.trajectory_xz_m[0], [[0.0, 1.0], [0.0, 2.0]])


def test_observed_targets_rotate_world_motion_into_ego_frame() -> None:
    targets = build_observed_trajectory_targets(
        positions_xy_m=np.asarray([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]),
        heading_rad=np.full(3, np.pi / 2),
        simulation_time_s=[0.0, 0.5, 1.0],
        trajectory_id=[0, 0, 0],
        time_offsets_s=[0.5, 1.0],
    )
    np.testing.assert_allclose(
        targets.trajectory_xz_m[0],
        [[0.0, 1.0], [0.0, 2.0]],
        atol=1e-6,
    )


def test_reference_targets_share_observed_target_frames() -> None:
    reference = ReferenceTrajectory.constant(track_length_m=100.0, speed_mps=10.0)
    targets = build_reference_trajectory_targets(
        reference=reference,
        progress_m=[0.0, 1.0],
        positions_xy_m=[[0.0, 1.0], [1.0, 1.0]],
        heading_rad=[0.0, 0.0],
        valid_mask=[True, False],
        time_offsets_s=[0.5, 1.0],
    )
    np.testing.assert_array_equal(targets.valid_mask, [True, False])
    np.testing.assert_allclose(targets.trajectory_xz_m[0], [[-1.0, 5.0], [-1.0, 10.0]])
    np.testing.assert_array_equal(targets.trajectory_xz_m[1], np.zeros((2, 2)))
    projection_paths = build_reference_projection_paths(
        reference=reference,
        progress_m=[0.0, 1.0],
        positions_xy_m=[[0.0, 1.0], [1.0, 1.0]],
        heading_rad=[0.0, 0.0],
        valid_mask=[True, False],
        time_offsets_s=[0.5, 1.0],
    )
    np.testing.assert_allclose(
        projection_paths.points_xz_m[0],
        [[-1.0, 0.0], [-1.0, 5.0], [-1.0, 10.0]],
    )
    np.testing.assert_allclose(projection_paths.progress_m[0], [0.0, 5.0, 10.0])


def test_target_builder_rejects_invalid_pose_and_episode_time() -> None:
    with pytest.raises(ValueError, match="positions_xy"):
        build_observed_trajectory_targets(
            positions_xy_m=[0.0, 1.0],
            heading_rad=[0.0],
            simulation_time_s=[0.0],
            trajectory_id=[0],
            time_offsets_s=[0.5, 1.0],
        )
    with pytest.raises(ValueError, match="increase strictly"):
        build_observed_trajectory_targets(
            positions_xy_m=[[0.0, 0.0], [1.0, 0.0]],
            heading_rad=[0.0, 0.0],
            simulation_time_s=[0.0, 0.0],
            trajectory_id=[0, 0],
            time_offsets_s=[0.5, 1.0],
        )
