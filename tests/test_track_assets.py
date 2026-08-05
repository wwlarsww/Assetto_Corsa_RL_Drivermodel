"""Focused tests for automatic Assetto Corsa track assets."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from assetto_corsa_rl.ac_integration import track_assets
from assetto_corsa_rl.config import ProjectConfig


def _write_fast_lane(path: Path) -> None:
    points = [
        (0.0, 0.0, 0.0, 0.0, 0),
        (10.0, 0.0, 0.0, 10.0, 1),
        (10.0, 0.0, 10.0, 20.0, 2),
        (0.0, 0.0, 10.0, 30.0, 3),
    ]
    extras = [
        (
            20.0,
            1.0,
            0.0,
            0.0,
            10.0,
            5.0,
            6.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            float(index * 10),
            *forward,
            0.0,
            0.0,
        )
        for index, forward in enumerate(
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
        )
    ]
    payload = bytearray(struct.pack("<4i", 7, len(points), 90_000, len(points)))
    for point in points:
        payload.extend(struct.pack("<3ffi", *point))
    payload.extend(struct.pack("<i", len(extras)))
    for extra in extras:
        payload.extend(struct.pack("<9f3ff3f2f", *extra))
    payload.extend(struct.pack("<i", 0))
    path.write_bytes(payload)


def _config(tmp_path: Path) -> ProjectConfig:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'''
[dataset]
track_id = "test_track"
track_layout = ""
car_id = "test_car"
root_directory = "data"
steering_lock_degrees = 320.0
steering_lock_normalized = 0.7

[track_assets]
source = "assetto_corsa_fast_lane"
auto_build = true
cache_directory = "{(tmp_path / 'cache').as_posix()}"
reference_speed_source = "ai"
human_reference_artifact_path = ""
minimum_reference_speed_mps = 5.0

[reference]
mode = "fixed"
artifact_path = "auto"
''',
        encoding="utf-8",
    )
    return ProjectConfig.from_toml(config_path)


def test_parse_fast_lane_v7(tmp_path: Path) -> None:
    source = tmp_path / "fast_lane.ai"
    _write_fast_lane(source)

    parsed = track_assets.parse_fast_lane(source)

    assert parsed.version == 7
    assert parsed.positions_xyz_m.shape == (4, 3)
    np.testing.assert_allclose(parsed.side_left_m, 5.0)
    np.testing.assert_allclose(parsed.side_right_m, 6.0)
    np.testing.assert_allclose(parsed.speed_mps, 20.0)


def test_native_xyz_is_mapped_to_telemetry_world_xy() -> None:
    native = np.asarray([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
    converted = track_assets.assetto_corsa_xyz_to_world_xy(native)
    np.testing.assert_allclose(converted, [[3.0, 1.0], [6.0, -4.0]])


def test_ensure_track_assets_builds_telemetry_aligned_geometry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "fast_lane.ai"
    _write_fast_lane(source)
    config = _config(tmp_path)
    monkeypatch.setattr(
        track_assets,
        "resolve_configured_fast_lane_path",
        lambda _config: source,
    )

    bundle = track_assets.ensure_track_assets(config)
    geometry = track_assets.load_npz_arrays(bundle.paths.geometry_path)
    reference = track_assets.load_npz_arrays(bundle.paths.reference_path)

    # Native AC horizontal [x, z] points are exported as the same [z, x]
    # frame used by live telemetry and the original dataset.
    np.testing.assert_allclose(
        geometry["centerline_world_xy_m"],
        [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]],
    )
    np.testing.assert_allclose(
        geometry["left_boundary_world_xy_m"][0],
        [5.0, 0.0],
    )
    np.testing.assert_allclose(
        geometry["right_boundary_world_xy_m"][0],
        [-6.0, 0.0],
    )
    np.testing.assert_allclose(
        geometry["heading_rad"],
        [np.pi / 2.0, 0.0, -np.pi / 2.0, np.pi],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        reference["world_position_xy_m"],
        geometry["centerline_world_xy_m"],
    )
    np.testing.assert_allclose(reference["speed_mps"], 20.0)
    metadata = json.loads(str(geometry["metadata_json"].reshape(-1)[0]))
    assert metadata["asset_schema"] == 4
    assert metadata["coordinate_frame"] == "assetto_corsa_gym_world_xy_v1"
    assert metadata["coordinate_transform"] == "world_xy=[ac_z,ac_x]"


def test_closed_path_projection_and_wrapped_delta() -> None:
    path = np.asarray(
        [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]],
        dtype=np.float32,
    )
    distance = np.asarray([0.0, 10.0, 20.0, 30.0], dtype=np.float32)

    projection = track_assets.project_world_position(
        path,
        distance,
        40.0,
        np.asarray([1.0, 4.0]),
    )

    np.testing.assert_allclose(projection.point_xy_m, [0.0, 4.0])
    assert projection.progress_m == 4.0
    assert projection.distance_m == 1.0
    assert track_assets.wrapped_progress_delta(1.0, 39.0, 40.0) == 2.0
    assert track_assets.wrapped_progress_delta(39.0, 1.0, 40.0) == -2.0


def test_cached_schema_four_assets_are_reused(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "fast_lane.ai"
    _write_fast_lane(source)
    config = _config(tmp_path)
    monkeypatch.setattr(
        track_assets,
        "resolve_configured_fast_lane_path",
        lambda _config: source,
    )

    first = track_assets.ensure_track_assets(config)
    second = track_assets.ensure_track_assets(config)

    assert first.rebuilt_geometry
    assert first.rebuilt_reference
    assert not second.rebuilt_geometry
    assert not second.rebuilt_reference
    assert first.geometry_sha256 == second.geometry_sha256


def test_generated_assets_do_not_overwrite_configured_fixed_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "fast_lane.ai"
    _write_fast_lane(source)
    config_path = tmp_path / "config_fixed.toml"
    fixed_reference = tmp_path / "human_fixed_reference.npz"
    fixed_reference.write_bytes(b"do-not-overwrite")
    config_path.write_text(
        f'''
[dataset]
track_id = "test_track"
track_layout = ""
car_id = "test_car"
root_directory = "data"
steering_lock_degrees = 320.0
steering_lock_normalized = 0.7

[track_assets]
source = "assetto_corsa_fast_lane"
auto_build = true
cache_directory = "{(tmp_path / 'cache_fixed').as_posix()}"
reference_speed_source = "ai"
human_reference_artifact_path = ""
minimum_reference_speed_mps = 5.0

[reference]
mode = "fixed"
artifact_path = "{fixed_reference.as_posix()}"
''',
        encoding="utf-8",
    )
    config = ProjectConfig.from_toml(config_path)
    monkeypatch.setattr(
        track_assets,
        "resolve_configured_fast_lane_path",
        lambda _config: source,
    )

    track_assets.ensure_track_assets(config, force=True)

    assert fixed_reference.read_bytes() == b"do-not-overwrite"
