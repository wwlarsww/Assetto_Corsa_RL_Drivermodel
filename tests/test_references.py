from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import FastLaneFactory

from assetto_corsa_rl.config import ProjectConfig, ReferenceConfig
from assetto_corsa_rl.references.factory import load_reference_sampler
from assetto_corsa_rl.references.fixed_reference import (
    FixedReferenceSampler,
    ReferenceTrajectory,
)
from assetto_corsa_rl.references.projection import (
    project_points_to_closed_reference_progress,
    project_points_to_polyline_progress,
    wrapped_progress_delta,
)


def test_fixed_reference_sampler_and_validation() -> None:
    reference = ReferenceTrajectory.constant(track_length_m=100.0, speed_mps=20.0)
    sampler = FixedReferenceSampler(reference)
    assert sampler.sample(np.random.default_rng()) is reference
    assert sampler.mean() is reference
    with pytest.raises(ValueError, match="at least two"):
        ReferenceTrajectory.constant(
            track_length_m=100.0,
            speed_mps=20.0,
            sample_count=1,
        )


def test_preview_uses_speed_profile_time_and_wrap_safe_heading() -> None:
    reference = ReferenceTrajectory(
        track_distance_m=np.asarray([0.0, 25.0, 50.0, 75.0], dtype=np.float32),
        lateral_offset_m=np.zeros(4, dtype=np.float32),
        heading_rad=np.asarray([3.0, -3.0, -2.8, 2.8], dtype=np.float32),
        speed_mps=np.asarray([10.0, 10.0, 20.0, 20.0], dtype=np.float32),
        track_length_m=100.0,
        world_position_xy_m=np.column_stack(
            (np.asarray([0.0, 25.0, 50.0, 75.0]), np.zeros(4))
        ).astype(np.float32),
    )
    assert reference.future_distances(0.0, [3.0])[0] == pytest.approx(32.5)
    times = np.full(20, 2.5, dtype=np.float32)
    heading = reference.preview(0.0, times).heading_rad[0]
    assert abs(float(heading)) >= 3.0


def test_reference_ego_preview_uses_vehicle_pose() -> None:
    reference = ReferenceTrajectory.constant(track_length_m=100.0, speed_mps=20.0)
    preview = reference.ego_path_preview(
        0.0,
        vehicle_position_xy_m=[0.0, 1.0],
        vehicle_heading_rad=0.0,
    )
    np.testing.assert_allclose(preview[0], [-1.0, 5.0], atol=1e-5)
    with_origin = reference.ego_path_with_origin(
        0.0,
        vehicle_position_xy_m=[0.0, 1.0],
        vehicle_heading_rad=0.0,
    )
    assert with_origin.shape == (21, 2)
    np.testing.assert_allclose(with_origin[:2], [[-1.0, 0.0], [-1.0, 5.0]], atol=1e-5)


def test_progress_projection_uses_reference_arc_length_and_wraps() -> None:
    reference = np.asarray([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    points = np.asarray([[2.0, 5.0], [5.0, 12.0]])
    np.testing.assert_allclose(
        project_points_to_polyline_progress(points, reference),
        [5.0, 15.0],
    )
    assert wrapped_progress_delta(99.0, 1.0, 100.0) == pytest.approx(2.0)
    assert wrapped_progress_delta(1.0, 99.0, 100.0) == pytest.approx(-2.0)
    assert wrapped_progress_delta(99.0, 101.0, 100.0) == pytest.approx(2.0)

    closed_reference = np.asarray(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        dtype=np.float32,
    )
    distance = np.asarray([0.0, 10.0, 20.0, 30.0], dtype=np.float32)
    np.testing.assert_allclose(
        project_points_to_closed_reference_progress(
            [[5.0, 1.0], [-1.0, 5.0]],
            closed_reference,
            distance,
            40.0,
        ),
        [5.0, 35.0],
    )
    square = ReferenceTrajectory(
        track_distance_m=distance,
        lateral_offset_m=np.zeros(4, dtype=np.float32),
        heading_rad=np.zeros(4, dtype=np.float32),
        speed_mps=np.ones(4, dtype=np.float32),
        track_length_m=40.0,
        world_position_xy_m=closed_reference,
    )
    np.testing.assert_allclose(square.project_world_positions([[10.0, 4.0]]), [14.0])


def test_reference_prediction_geometry_and_time_are_validated() -> None:
    without_geometry = ReferenceTrajectory(
        track_distance_m=np.asarray([0.0, 50.0], dtype=np.float32),
        lateral_offset_m=np.zeros(2, dtype=np.float32),
        heading_rad=np.zeros(2, dtype=np.float32),
        speed_mps=np.ones(2, dtype=np.float32),
        track_length_m=100.0,
    )
    with pytest.raises(ValueError, match="world_position"):
        without_geometry.ego_path_preview(0.0, [0.0, 0.0], 0.0)
    with pytest.raises(ValueError, match="positive"):
        without_geometry.future_distances(0.0, [0.0])
    with pytest.raises(ValueError, match="start at zero"):
        ReferenceTrajectory(
            track_distance_m=np.asarray([1.0, 50.0], dtype=np.float32),
            lateral_offset_m=np.zeros(2, dtype=np.float32),
            heading_rad=np.zeros(2, dtype=np.float32),
            speed_mps=np.ones(2, dtype=np.float32),
            track_length_m=100.0,
        )


def test_reference_npz_round_trip_rejects_object_arrays(tmp_path: Path) -> None:
    reference = ReferenceTrajectory.constant(track_length_m=100.0, speed_mps=20.0)
    path = tmp_path / "reference.npz"
    reference.to_npz(path, extra_arrays={"label": np.asarray("fixed")})
    second_path = tmp_path / "reference_second.npz"
    reference.to_npz(second_path, extra_arrays={"label": np.asarray("fixed")})
    assert path.read_bytes() == second_path.read_bytes()
    loaded = ReferenceTrajectory.from_npz(path)
    np.testing.assert_allclose(loaded.as_feature_matrix(), reference.as_feature_matrix())
    assert loaded.world_position_xy_m is not None

    with pytest.raises(ValueError, match="object dtype"):
        reference.to_npz(
            tmp_path / "unsafe.npz",
            extra_arrays={"unsafe": np.asarray([object()], dtype=object)},
        )


def test_load_fixed_reference_sampler_checks_ai_spline_source(
    tmp_path: Path,
    fast_lane_factory: FastLaneFactory,
) -> None:
    installation = tmp_path / "assettocorsa"
    spline = fast_lane_factory(installation, 100.0, "monza")
    path = tmp_path / "reference.npz"
    reference = ReferenceTrajectory.constant(
        track_length_m=100.0,
        speed_mps=20.0,
        source_track_sha256=spline.source_sha256,
    )
    reference.to_npz(path)
    base = ProjectConfig(reference=ReferenceConfig(artifact_path=str(path)))
    config = replace(
        base,
        assetto_corsa=replace(
            base.assetto_corsa,
            installation_directory=str(installation),
        ),
    )
    assert load_reference_sampler(config).mean().track_length_m == 100.0

    ReferenceTrajectory.constant(
        track_length_m=100.0,
        speed_mps=20.0,
        source_track_sha256="0" * 64,
    ).to_npz(path)
    with pytest.raises(ValueError, match="source does not match"):
        load_reference_sampler(config)

    ReferenceTrajectory.constant(
        track_length_m=101.0,
        speed_mps=20.0,
        source_track_sha256=spline.source_sha256,
    ).to_npz(path)
    with pytest.raises(ValueError, match="track lengths differ"):
        load_reference_sampler(config)
