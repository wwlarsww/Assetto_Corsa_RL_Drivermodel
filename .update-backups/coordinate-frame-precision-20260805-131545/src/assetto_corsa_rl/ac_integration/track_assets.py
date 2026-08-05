"""Build reusable track geometry and reference artefacts from Assetto Corsa data.

The installed track's ``ai/fast_lane.ai`` is the source of truth for static
geometry. Generated artefacts are cached by track/layout and invalidated by the
SHA-256 digest of that file. No geometry is reconstructed from normalized model
observations.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from assetto_corsa_rl.ac_integration.launcher import configured_fast_lane_path
from assetto_corsa_rl.config import ProjectConfig

_AI_POINT_STRUCT = struct.Struct("<3ffi")
_AI_POINT_EXTRA_STRUCT = struct.Struct("<9f3ff3f2f")
_SUPPORTED_AI_VERSION = 7
_ASSET_SCHEMA = 3
CANONICAL_COORDINATE_FRAME = "assetto_corsa_gym_world_xz_v1"
SOURCE_COORDINATE_FRAME = "assetto_corsa_native_world_xz"


class TrackAssetBuildError(RuntimeError):
    """Raised when installed Assetto Corsa data cannot produce valid assets."""


@dataclass(frozen=True, slots=True)
class ParsedFastLane:
    """Raw arrays decoded from one Assetto Corsa ``fast_lane.ai`` file."""

    version: int
    lap_time_ms: int
    sample_count: int
    positions_xyz_m: np.ndarray
    point_length_m: np.ndarray
    point_ids: np.ndarray
    speed_mps: np.ndarray
    gas: np.ndarray
    brake: np.ndarray
    radius_m: np.ndarray
    side_left_m: np.ndarray
    side_right_m: np.ndarray
    camber: np.ndarray
    direction: np.ndarray
    normal_xyz: np.ndarray
    extra_length_m: np.ndarray
    forward_vector_xyz: np.ndarray
    tag: np.ndarray
    grade: np.ndarray


@dataclass(frozen=True, slots=True)
class TrackProjection:
    """Nearest point and progress on one closed world XZ polyline."""

    progress_m: float
    point_xy_m: np.ndarray
    distance_m: float
    segment_index: int
    segment_fraction: float


@dataclass(frozen=True, slots=True)
class TrackAssetPaths:
    """Stable generated paths for one track/layout."""

    directory: Path
    geometry_path: Path
    reference_path: Path
    metadata_path: Path


@dataclass(frozen=True, slots=True)
class TrackAssetBundle:
    """Resolved source and generated files used by one run."""

    source_fast_lane_path: Path
    source_sha256: str
    paths: TrackAssetPaths
    geometry_sha256: str
    reference_sha256: str
    rebuilt_geometry: bool
    rebuilt_reference: bool


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest for ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_fast_lane(path: str | Path) -> ParsedFastLane:
    """Decode Assetto Corsa AI spline version 7 without external dependencies.

    The binary layout follows the structures used by AcTools: each base point
    contains XYZ position, cumulative length and point ID; each extra point
    contains target speed, left/right spline widths, normal and forward vector.
    Grid data after the point arrays is intentionally ignored because static
    geometry and reference generation do not need it.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Assetto Corsa fast lane does not exist: {source}")

    data = memoryview(source.read_bytes())
    offset = 0

    def read_int32(label: str) -> int:
        nonlocal offset
        if offset + 4 > len(data):
            raise TrackAssetBuildError(
                f"truncated fast_lane.ai while reading {label}: {source}"
            )
        value = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        return int(value)

    version = read_int32("version")
    if version != _SUPPORTED_AI_VERSION:
        raise TrackAssetBuildError(
            f"unsupported fast_lane.ai version {version}; expected "
            f"{_SUPPORTED_AI_VERSION}: {source}"
        )

    point_count = read_int32("point count")
    lap_time_ms = read_int32("lap time")
    sample_count = read_int32("sample count")
    if point_count < 2 or point_count > 10_000_000:
        raise TrackAssetBuildError(
            f"implausible AI spline point count {point_count}: {source}"
        )

    positions = np.empty((point_count, 3), dtype=np.float32)
    point_length = np.empty(point_count, dtype=np.float32)
    point_ids = np.empty(point_count, dtype=np.int32)
    required = point_count * _AI_POINT_STRUCT.size
    if offset + required > len(data):
        raise TrackAssetBuildError(
            f"truncated base point array in fast_lane.ai: {source}"
        )
    for index in range(point_count):
        x, y, z, length, point_id = _AI_POINT_STRUCT.unpack_from(data, offset)
        offset += _AI_POINT_STRUCT.size
        positions[index] = (x, y, z)
        point_length[index] = length
        point_ids[index] = point_id

    extra_count = read_int32("extra point count")
    if extra_count != point_count:
        raise TrackAssetBuildError(
            "fast_lane.ai base and extra point counts differ: "
            f"{point_count} != {extra_count} ({source})"
        )

    extras = np.empty((extra_count, 18), dtype=np.float32)
    required = extra_count * _AI_POINT_EXTRA_STRUCT.size
    if offset + required > len(data):
        raise TrackAssetBuildError(
            f"truncated extra point array in fast_lane.ai: {source}"
        )
    for index in range(extra_count):
        values = _AI_POINT_EXTRA_STRUCT.unpack_from(data, offset)
        offset += _AI_POINT_EXTRA_STRUCT.size
        extras[index] = values

    _ = read_int32("grid-present flag")

    return ParsedFastLane(
        version=version,
        lap_time_ms=lap_time_ms,
        sample_count=sample_count,
        positions_xyz_m=positions,
        point_length_m=point_length,
        point_ids=point_ids,
        speed_mps=extras[:, 0].copy(),
        gas=extras[:, 1].copy(),
        brake=extras[:, 2].copy(),
        radius_m=extras[:, 4].copy(),
        side_left_m=extras[:, 5].copy(),
        side_right_m=extras[:, 6].copy(),
        camber=extras[:, 7].copy(),
        direction=extras[:, 8].copy(),
        normal_xyz=extras[:, 9:12].copy(),
        extra_length_m=extras[:, 12].copy(),
        forward_vector_xyz=extras[:, 13:16].copy(),
        tag=extras[:, 16].copy(),
        grade=extras[:, 17].copy(),
    )



def resolve_configured_fast_lane_path(config: ProjectConfig) -> Path:
    """Resolve the track/layout AI spline from the installed game data."""

    layout = config.dataset.track_layout.strip()
    if layout:
        track_root = (
            Path(config.assetto_corsa.installation_directory)
            / "content"
            / "tracks"
            / config.dataset.track_id
        )
        candidates = (
            track_root / layout / "ai" / "fast_lane.ai",
            track_root / "ai" / layout / "fast_lane.ai",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        expected = " or ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"configured track layout {layout!r} has no fast_lane.ai; expected {expected}"
        )
    return configured_fast_lane_path(config)

def resolve_track_asset_paths(
    config: ProjectConfig,
    *,
    fast_lane_path: Path | None = None,
) -> TrackAssetPaths:
    """Return deterministic cache paths for the configured track and layout."""

    source = fast_lane_path or configured_fast_lane_path(config)
    layout = config.dataset.track_layout.strip() or _layout_from_source(
        source,
        config.dataset.track_id,
    )
    safe_track = _safe_component(config.dataset.track_id)
    safe_layout = _safe_component(layout or "default")
    directory = Path(config.track_assets.cache_directory) / safe_track / safe_layout
    return TrackAssetPaths(
        directory=directory,
        geometry_path=directory / "track_geometry.npz",
        reference_path=directory / "reference.npz",
        metadata_path=directory / "metadata.json",
    )


def ensure_track_assets(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> TrackAssetBundle:
    """Build or reuse geometry/reference assets for the configured AC track."""

    if config.track_assets.source != "assetto_corsa_fast_lane":
        raise TrackAssetBuildError(
            "track_assets.source must be 'assetto_corsa_fast_lane'"
        )

    source = resolve_configured_fast_lane_path(config)
    if not source.is_file():
        raise FileNotFoundError(f"configured Assetto Corsa fast lane is missing: {source}")
    source_digest = sha256_file(source)
    paths = resolve_track_asset_paths(config, fast_lane_path=source)
    paths.directory.mkdir(parents=True, exist_ok=True)

    geometry_valid = (
        not force
        and _npz_metadata_matches(
            paths.geometry_path,
            source_digest=source_digest,
            asset_kind="track_geometry",
        )
    )
    if not geometry_valid:
        parsed = parse_fast_lane(source)
        _write_track_geometry(
            parsed,
            destination=paths.geometry_path,
            source_path=source,
            source_sha256=source_digest,
            config=config,
        )

    reference_valid = (
        not force
        and _npz_metadata_matches(
            paths.reference_path,
            source_digest=source_digest,
            asset_kind="reference",
        )
    )
    if not reference_valid:
        _write_reference(
            geometry_path=paths.geometry_path,
            destination=paths.reference_path,
            source_path=source,
            source_sha256=source_digest,
            config=config,
        )

    geometry_digest = sha256_file(paths.geometry_path)
    reference_digest = sha256_file(paths.reference_path)
    metadata = {
        "asset_schema": _ASSET_SCHEMA,
        "source": "assetto_corsa_fast_lane",
        "source_fast_lane_path": str(source.resolve()),
        "source_fast_lane_sha256": source_digest,
        "track_id": config.dataset.track_id,
        "track_layout": config.dataset.track_layout or _layout_from_source(
            source,
            config.dataset.track_id,
        ),
        "geometry_path": str(paths.geometry_path),
        "geometry_sha256": geometry_digest,
        "reference_path": str(paths.reference_path),
        "reference_sha256": reference_digest,
        "reference_speed_source": config.track_assets.reference_speed_source,
    }
    _write_json_atomically(paths.metadata_path, metadata)

    configured_reference = Path(config.reference.artifact_path)
    if configured_reference != paths.reference_path:
        configured_reference.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.reference_path, configured_reference)

    return TrackAssetBundle(
        source_fast_lane_path=source,
        source_sha256=source_digest,
        paths=paths,
        geometry_sha256=geometry_digest,
        reference_sha256=reference_digest,
        rebuilt_geometry=not geometry_valid,
        rebuilt_reference=not reference_valid,
    )


def load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load a non-pickled NPZ artefact into ordinary arrays."""

    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def assetto_corsa_xyz_to_world_xz(points_xyz_m: np.ndarray) -> np.ndarray:
    """Map native AC XYZ positions to the project's horizontal world XZ plane.

    The live telemetry adapter and the reference artefacts use native Assetto
    Corsa X as the first horizontal coordinate and native Z as the second.
    Earlier track assets accidentally stored ``[AC z, AC x]`` and therefore
    appeared transposed relative to the vehicle position.
    """

    points = np.asarray(points_xyz_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("native Assetto Corsa points must have shape (N, >=3)")
    return points[:, [0, 2]].copy()


def assetto_corsa_xyz_to_world_xy(points_xyz_m: np.ndarray) -> np.ndarray:
    """Backward compatible alias for :func:`assetto_corsa_xyz_to_world_xz`."""

    return assetto_corsa_xyz_to_world_xz(points_xyz_m)


def predictor_xz_to_vehicle_fl(points_xz_m: np.ndarray) -> np.ndarray:
    """Convert model local ``[lateral X, forward Z]`` into ``[forward, left]``.

    The model and existing Bézier codec use lateral X first and forward Z
    second.  Viewer and diagnostics use the semantically explicit vehicle frame
    ``[forward, left]``.
    """

    points = np.asarray(points_xz_m, dtype=np.float64)
    if points.ndim < 2 or points.shape[-1] != 2:
        raise ValueError("local predictor points must end with shape (..., 2)")
    return np.stack((points[..., 1], points[..., 0]), axis=-1).astype(np.float32)


def vehicle_fl_to_predictor_xz(points_fl_m: np.ndarray) -> np.ndarray:
    """Convert semantic vehicle ``[forward, left]`` into model local XZ."""

    points = np.asarray(points_fl_m, dtype=np.float64)
    if points.ndim < 2 or points.shape[-1] != 2:
        raise ValueError("vehicle-frame points must end with shape (..., 2)")
    return np.stack((points[..., 1], points[..., 0]), axis=-1).astype(np.float32)


def world_xz_to_local_xz(
    points_world_xz_m: np.ndarray,
    vehicle_world_xz_m: np.ndarray,
    heading_rad: float,
) -> np.ndarray:
    """Convert world XZ points into model local lateral-X/forward-Z points."""

    points = np.asarray(points_world_xz_m, dtype=np.float64)
    vehicle = np.asarray(vehicle_world_xz_m, dtype=np.float64).reshape(2)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("world points must have shape (N, 2)")
    delta = points - vehicle[None, :]
    forward = np.asarray([math.cos(heading_rad), math.sin(heading_rad)])
    left = np.asarray([-math.sin(heading_rad), math.cos(heading_rad)])
    lateral_x = delta @ left
    forward_z = delta @ forward
    return np.column_stack((lateral_x, forward_z)).astype(np.float32)


def world_xz_to_vehicle_fl(
    points_world_xz_m: np.ndarray,
    vehicle_world_xz_m: np.ndarray,
    heading_rad: float,
) -> np.ndarray:
    """Convert world XZ points into semantic vehicle ``[forward, left]``."""

    return predictor_xz_to_vehicle_fl(
        world_xz_to_local_xz(
            points_world_xz_m,
            vehicle_world_xz_m,
            heading_rad,
        )
    )


def vehicle_fl_to_world_xz(
    points_fl_m: np.ndarray,
    vehicle_world_xz_m: np.ndarray,
    heading_rad: float,
) -> np.ndarray:
    """Convert semantic vehicle ``[forward, left]`` points into world XZ."""

    return local_xz_to_world_xz(
        vehicle_fl_to_predictor_xz(points_fl_m),
        vehicle_world_xz_m,
        heading_rad,
    )



def geometry_centerline_world_xz(geometry_arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return a geometry centerline in canonical world XZ order."""

    key = (
        "centerline_world_xz_m"
        if "centerline_world_xz_m" in geometry_arrays
        else "centerline_world_xy_m"
    )
    return np.asarray(geometry_arrays[key], dtype=np.float32)


def reference_line_world_xz(reference_arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return a reference line in canonical world XZ order."""

    key = (
        "world_position_xz_m"
        if "world_position_xz_m" in reference_arrays
        else "world_position_xy_m"
    )
    return np.asarray(reference_arrays[key], dtype=np.float32)

def project_world_position(
    path_xy_m: np.ndarray,
    track_distance_m: np.ndarray,
    track_length_m: float,
    position_xy_m: np.ndarray,
) -> TrackProjection:
    """Project one world position onto a closed polyline without normalisation.

    The returned progress uses the same distance origin as ``track_distance_m``.
    The closing segment from the last point back to the first point is included.
    """

    path = np.asarray(path_xy_m, dtype=np.float64)
    distance = np.asarray(track_distance_m, dtype=np.float64).reshape(-1)
    position = np.asarray(position_xy_m, dtype=np.float64).reshape(2)
    length = float(track_length_m)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 2:
        raise ValueError("path_xy_m must have shape (N, 2) with N >= 2")
    if distance.shape != (path.shape[0],):
        raise ValueError("track_distance_m must match the path length")
    if not np.isfinite(path).all() or not np.isfinite(distance).all():
        raise ValueError("projection path contains non-finite values")
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("track_length_m must be positive and finite")

    segment_start = path
    segment_end = np.roll(path, -1, axis=0)
    segment_vector = segment_end - segment_start
    segment_length_squared = np.einsum("ij,ij->i", segment_vector, segment_vector)
    valid = segment_length_squared > 1e-12
    if not np.any(valid):
        raise ValueError("projection path contains no non-degenerate segment")

    offset = position[None, :] - segment_start
    fraction = np.zeros(path.shape[0], dtype=np.float64)
    fraction[valid] = np.clip(
        np.einsum("ij,ij->i", offset[valid], segment_vector[valid])
        / segment_length_squared[valid],
        0.0,
        1.0,
    )
    projected = segment_start + fraction[:, None] * segment_vector
    squared_distance = np.einsum(
        "ij,ij->i",
        projected - position[None, :],
        projected - position[None, :],
    )
    squared_distance[~valid] = np.inf
    index = int(np.argmin(squared_distance))

    if index == path.shape[0] - 1:
        segment_progress = max(length - float(distance[index]), 0.0)
    else:
        segment_progress = max(float(distance[index + 1] - distance[index]), 0.0)
    progress = (float(distance[index]) + fraction[index] * segment_progress) % length
    return TrackProjection(
        progress_m=progress,
        point_xy_m=projected[index].astype(np.float32),
        distance_m=float(math.sqrt(squared_distance[index])),
        segment_index=index,
        segment_fraction=float(fraction[index]),
    )


def wrapped_progress_delta(
    current_progress_m: float,
    previous_progress_m: float,
    track_length_m: float,
) -> float:
    """Return the signed shortest progress change on a closed track."""

    length = float(track_length_m)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("track_length_m must be positive and finite")
    delta = float(current_progress_m) - float(previous_progress_m)
    return float((delta + 0.5 * length) % length - 0.5 * length)


def circular_interpolate(
    distance_m: np.ndarray,
    values: np.ndarray,
    query_m: np.ndarray,
    track_length_m: float,
) -> np.ndarray:
    """Interpolate scalar or vector arrays on a closed track."""

    distance = np.asarray(distance_m, dtype=np.float64).reshape(-1)
    data = np.asarray(values, dtype=np.float64)
    query = np.mod(np.asarray(query_m, dtype=np.float64), track_length_m)
    extended_distance = np.concatenate((distance, [track_length_m]))
    if data.ndim == 1:
        return np.interp(
            query,
            extended_distance,
            np.concatenate((data, [data[0]])),
        )
    return np.column_stack(
        [
            np.interp(
                query,
                extended_distance,
                np.concatenate((data[:, axis], [data[0, axis]])),
            )
            for axis in range(data.shape[1])
        ]
    )


def reference_preview_world(
    reference_arrays: Mapping[str, np.ndarray],
    progress_m: float,
    time_offsets_s: np.ndarray,
) -> np.ndarray:
    """Sample an unnormalised world-space reference preview."""

    distance = np.asarray(reference_arrays["track_distance_m"], dtype=np.float64)
    position_key = (
        "world_position_xz_m"
        if "world_position_xz_m" in reference_arrays
        else "world_position_xy_m"
    )
    position = np.asarray(reference_arrays[position_key], dtype=np.float64)
    speed = np.asarray(reference_arrays["speed_mps"], dtype=np.float64)
    track_length = float(np.asarray(reference_arrays["track_length_m"]).reshape(-1)[0])
    targets = np.asarray(time_offsets_s, dtype=np.float64)
    current_progress = float(progress_m) % track_length
    current_time = 0.0
    query_progress: list[float] = []
    for target_time in targets:
        while current_time + 1e-12 < float(target_time):
            dt = min(0.01, float(target_time) - current_time)
            current_speed = float(
                circular_interpolate(
                    distance,
                    speed,
                    np.asarray([current_progress]),
                    track_length,
                )[0]
            )
            current_progress = (
                current_progress + max(current_speed, 0.0) * dt
            ) % track_length
            current_time += dt
        query_progress.append(current_progress)
    return circular_interpolate(
        distance,
        position,
        np.asarray(query_progress),
        track_length,
    ).astype(np.float32)


def boundary_preview_world(
    geometry_arrays: Mapping[str, np.ndarray],
    progress_m: float,
    preview_distances_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample left and right AC AI-spline boundaries in world coordinates."""

    distance = np.asarray(geometry_arrays["track_distance_m"], dtype=np.float64)
    left_key = (
        "left_boundary_world_xz_m"
        if "left_boundary_world_xz_m" in geometry_arrays
        else "left_boundary_world_xy_m"
    )
    right_key = (
        "right_boundary_world_xz_m"
        if "right_boundary_world_xz_m" in geometry_arrays
        else "right_boundary_world_xy_m"
    )
    left = np.asarray(geometry_arrays[left_key], dtype=np.float64)
    right = np.asarray(geometry_arrays[right_key], dtype=np.float64)
    track_length = float(np.asarray(geometry_arrays["track_length_m"]).reshape(-1)[0])
    query = float(progress_m) + np.asarray(preview_distances_m, dtype=np.float64)
    return (
        circular_interpolate(distance, left, query, track_length).astype(np.float32),
        circular_interpolate(distance, right, query, track_length).astype(np.float32),
    )


def local_xz_to_world_xz(
    points_xz_m: np.ndarray,
    position_xz_m: np.ndarray,
    heading_rad: float,
) -> np.ndarray:
    """Convert model local lateral-X/forward-Z points into world XZ."""

    points = np.asarray(points_xz_m, dtype=np.float64)
    position = np.asarray(position_xz_m, dtype=np.float64).reshape(2)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("local points must have shape (N, 2)")
    forward = np.asarray([math.cos(heading_rad), math.sin(heading_rad)])
    left = np.asarray([-math.sin(heading_rad), math.cos(heading_rad)])
    return (
        position[None, :]
        + points[:, 0:1] * left[None, :]
        + points[:, 1:2] * forward[None, :]
    ).astype(np.float32)


def local_xz_to_world_xy(
    points_xz_m: np.ndarray,
    position_xy_m: np.ndarray,
    heading_rad: float,
) -> np.ndarray:
    """Backward compatible alias for :func:`local_xz_to_world_xz`."""

    return local_xz_to_world_xz(points_xz_m, position_xy_m, heading_rad)


def _write_track_geometry(
    parsed: ParsedFastLane,
    *,
    destination: Path,
    source_path: Path,
    source_sha256: str,
    config: ProjectConfig,
) -> None:
    # The live telemetry, the original dataset and the fixed reference all use
    # native Assetto Corsa horizontal coordinates [x, z].  Keep that order
    # unchanged.  A previous implementation swapped the axes to [z, x], which
    # put the vehicle far away from an otherwise plausible track shape.
    native_position_xz = assetto_corsa_xyz_to_world_xz(parsed.positions_xyz_m)
    distance = _normalised_track_distance(parsed, native_position_xz)
    track_length = _track_length(distance, native_position_xz)
    native_forward_xz = _normalised_forward_vectors(
        parsed.forward_vector_xyz,
        native_position_xz,
    )
    native_left = np.column_stack(
        (-native_forward_xz[:, 1], native_forward_xz[:, 0])
    )
    native_right = -native_left

    side_left = np.asarray(parsed.side_left_m, dtype=np.float64)
    side_right = np.asarray(parsed.side_right_m, dtype=np.float64)
    if not _valid_side_widths(side_left, side_right):
        raise TrackAssetBuildError(
            "fast_lane.ai does not contain usable SideLeft/SideRight values; "
            "track borders cannot be generated without inventing geometry"
        )

    native_left_boundary_xz = native_position_xz + native_left * side_left[:, None]
    native_right_boundary_xz = native_position_xz + native_right * side_right[:, None]

    position_xz = native_position_xz
    left_boundary_xz = native_left_boundary_xz
    right_boundary_xz = native_right_boundary_xz
    forward_xz = native_forward_xz
    heading = np.arctan2(forward_xz[:, 1], forward_xz[:, 0])
    curvature = _curvature(distance, heading, track_length)

    speed = np.asarray(parsed.speed_mps, dtype=np.float64)
    if not np.isfinite(speed).all() or np.nanmax(speed) <= 0.0:
        raise TrackAssetBuildError(
            "fast_lane.ai does not contain a usable AI speed profile"
        )

    metadata = {
        "asset_kind": "track_geometry",
        "asset_schema": _ASSET_SCHEMA,
        "source": "assetto_corsa_fast_lane",
        "source_fast_lane_path": str(source_path.resolve()),
        "source_fast_lane_sha256": source_sha256,
        "ai_file_version": parsed.version,
        "ai_lap_time_ms": parsed.lap_time_ms,
        "ai_sample_count": parsed.sample_count,
        "track_id": config.dataset.track_id,
        "track_layout": config.dataset.track_layout,
        "coordinate_frame": CANONICAL_COORDINATE_FRAME,
        "source_coordinate_frame": SOURCE_COORDINATE_FRAME,
        "coordinate_transform": "world_xz=[ac_x,ac_z]",
        "boundary_definition": "fast_lane.ai SideLeft/SideRight",
    }
    _save_npz_atomically(
        destination,
        track_distance_m=distance.astype(np.float32),
        centerline_world_xz_m=position_xz.astype(np.float32),
        left_boundary_world_xz_m=left_boundary_xz.astype(np.float32),
        right_boundary_world_xz_m=right_boundary_xz.astype(np.float32),
        # Legacy aliases remain readable by old tools, but contain XZ data.
        centerline_world_xy_m=position_xz.astype(np.float32),
        left_boundary_world_xy_m=left_boundary_xz.astype(np.float32),
        right_boundary_world_xy_m=right_boundary_xz.astype(np.float32),
        heading_rad=heading.astype(np.float32),
        forward_vector_xz=forward_xz.astype(np.float32),
        forward_vector_xy=forward_xz.astype(np.float32),
        side_left_m=side_left.astype(np.float32),
        side_right_m=side_right.astype(np.float32),
        curvature_1pm=curvature.astype(np.float32),
        ai_speed_mps=speed.astype(np.float32),
        ai_radius_m=np.asarray(parsed.radius_m, dtype=np.float32),
        ai_camber=np.asarray(parsed.camber, dtype=np.float32),
        ai_grade=np.asarray(parsed.grade, dtype=np.float32),
        track_length_m=np.asarray(track_length, dtype=np.float64),
        coordinate_frame=np.asarray(CANONICAL_COORDINATE_FRAME),
        coordinate_transform=np.asarray("world_xz=[ac_x,ac_z]"),
        source_fast_lane_sha256=np.asarray(source_sha256),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _write_reference(
    *,
    geometry_path: Path,
    destination: Path,
    source_path: Path,
    source_sha256: str,
    config: ProjectConfig,
) -> None:
    geometry = load_npz_arrays(geometry_path)
    distance = np.asarray(geometry["track_distance_m"], dtype=np.float64)
    position = np.asarray(
        geometry.get("centerline_world_xz_m", geometry["centerline_world_xy_m"]),
        dtype=np.float64,
    )
    heading = np.asarray(geometry["heading_rad"], dtype=np.float64)
    ai_speed = np.asarray(geometry["ai_speed_mps"], dtype=np.float64)
    track_length = float(np.asarray(geometry["track_length_m"]).reshape(-1)[0])

    speed = ai_speed
    selected_speed_source = "assetto_corsa_ai_speed"
    human_path_text = config.track_assets.human_reference_artifact_path.strip()
    if human_path_text.casefold() == "auto":
        human_path_text = str(
            Path("data/processed")
            / f"{config.dataset.car_id}_{config.dataset.track_id}"
            / "fixed_reference.npz"
        )
    if (
        config.track_assets.reference_speed_source
        == "human_if_available_else_ai"
        and human_path_text
    ):
        human_path = Path(human_path_text)
        human_speed = _compatible_human_speed(
            human_path,
            target_distance_m=distance,
            target_position_xy_m=position,
            track_length_m=track_length,
            source_sha256=source_sha256,
        )
        if human_speed is not None:
            speed = human_speed
            selected_speed_source = "compatible_human_reference_speed"

    minimum_speed = config.track_assets.minimum_reference_speed_mps
    speed = np.maximum(np.asarray(speed, dtype=np.float64), minimum_speed)
    metadata = {
        "asset_kind": "reference",
        "asset_schema": _ASSET_SCHEMA,
        "method_variant": "assetto_corsa_fast_lane_reference_v2",
        "line_source": "assetto_corsa_fast_lane",
        "speed_source": selected_speed_source,
        "source_fast_lane_path": str(source_path.resolve()),
        "source_fast_lane_sha256": source_sha256,
        "track_id": config.dataset.track_id,
        "track_layout": config.dataset.track_layout,
        "car_id": config.dataset.car_id,
        "coordinate_frame": CANONICAL_COORDINATE_FRAME,
        "coordinate_transform": "world_xz=[ac_x,ac_z]",
    }
    _save_npz_atomically(
        destination,
        track_distance_m=distance.astype(np.float32),
        world_position_xz_m=position.astype(np.float32),
        world_position_xy_m=position.astype(np.float32),
        world_heading_rad=heading.astype(np.float32),
        heading_rad=heading.astype(np.float32),
        speed_mps=speed.astype(np.float32),
        lateral_offset_m=np.zeros(distance.size, dtype=np.float32),
        track_length_m=np.asarray(track_length, dtype=np.float64),
        coordinate_frame=np.asarray(CANONICAL_COORDINATE_FRAME),
        coordinate_transform=np.asarray("world_xz=[ac_x,ac_z]"),
        source_track_sha256=np.asarray(source_sha256),
        source_fast_lane_sha256=np.asarray(source_sha256),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _compatible_human_speed(
    path: Path,
    *,
    target_distance_m: np.ndarray,
    target_position_xy_m: np.ndarray,
    track_length_m: float,
    source_sha256: str,
) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        arrays = load_npz_arrays(path)
        candidate_hash = _string_scalar(
            arrays.get("source_fast_lane_sha256", arrays.get("source_track_sha256"))
        )
        if candidate_hash and candidate_hash != source_sha256:
            return None
        distance = np.asarray(arrays["track_distance_m"], dtype=np.float64)
        speed = np.asarray(arrays["speed_mps"], dtype=np.float64)
        candidate_position = np.asarray(
            arrays.get("world_position_xz_m", arrays["world_position_xy_m"]),
            dtype=np.float64,
        )
        candidate_length = float(np.asarray(arrays["track_length_m"]).reshape(-1)[0])
        if (
            distance.ndim != 1
            or speed.shape != distance.shape
            or candidate_position.shape != (distance.size, 2)
            or distance.size < 2
            or not math.isclose(candidate_length, track_length_m, rel_tol=0.0, abs_tol=2.0)
        ):
            return None
        aligned_position = circular_interpolate(
            distance,
            candidate_position,
            target_distance_m,
            track_length_m,
        )
        rms_alignment_error = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (aligned_position - target_position_xy_m) ** 2,
                        axis=1,
                    )
                )
            )
        )
        if not math.isfinite(rms_alignment_error) or rms_alignment_error > 2.0:
            return None
        return circular_interpolate(
            distance,
            speed,
            target_distance_m,
            track_length_m,
        )
    except (KeyError, OSError, ValueError, TypeError):
        return None


def _normalised_track_distance(
    parsed: ParsedFastLane,
    position_xy: np.ndarray,
) -> np.ndarray:
    for candidate in (parsed.extra_length_m, parsed.point_length_m):
        values = np.asarray(candidate, dtype=np.float64)
        if (
            values.shape == (position_xy.shape[0],)
            and np.isfinite(values).all()
            and np.all(np.diff(values) >= -1e-3)
            and values[-1] > values[0]
        ):
            return values - values[0]
    delta = np.diff(position_xy, axis=0)
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(delta, axis=1))))


def _track_length(distance: np.ndarray, position_xy: np.ndarray) -> float:
    closing = float(np.linalg.norm(position_xy[0] - position_xy[-1]))
    length = float(distance[-1] + closing)
    if not math.isfinite(length) or length <= 0.0:
        raise TrackAssetBuildError("calculated track length is invalid")
    return length


def _normalised_forward_vectors(
    forward_xyz: np.ndarray,
    position_xy: np.ndarray,
) -> np.ndarray:
    forward = np.asarray(forward_xyz[:, [0, 2]], dtype=np.float64)
    norms = np.linalg.norm(forward, axis=1)
    valid = np.isfinite(forward).all(axis=1) & (norms > 1e-6)
    if not valid.all():
        previous = np.roll(position_xy, 1, axis=0)
        following = np.roll(position_xy, -1, axis=0)
        estimated = following - previous
        forward[~valid] = estimated[~valid]
        norms = np.linalg.norm(forward, axis=1)
    if np.any(norms <= 1e-9):
        raise TrackAssetBuildError("AI spline contains degenerate forward vectors")
    return forward / norms[:, None]


def _valid_side_widths(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.ndim == 1
        and right.shape == left.shape
        and np.isfinite(left).all()
        and np.isfinite(right).all()
        and np.count_nonzero(left > 0.05) >= max(2, int(0.95 * left.size))
        and np.count_nonzero(right > 0.05) >= max(2, int(0.95 * right.size))
    )


def _curvature(
    distance: np.ndarray,
    heading: np.ndarray,
    track_length: float,
) -> np.ndarray:
    unwrapped = np.unwrap(heading)
    extended_distance = np.concatenate(
        ([distance[-1] - track_length], distance, [track_length + distance[1]])
    )
    extended_heading = np.concatenate(
        ([unwrapped[-1] - 2.0 * math.pi], unwrapped, [unwrapped[1] + 2.0 * math.pi])
    )
    delta_distance = extended_distance[2:] - extended_distance[:-2]
    delta_heading = extended_heading[2:] - extended_heading[:-2]
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(
            delta_heading,
            delta_distance,
            out=np.zeros_like(delta_heading),
            where=np.abs(delta_distance) > 1e-9,
        )
    return result


def _npz_metadata_matches(
    path: Path,
    *,
    source_digest: str,
    asset_kind: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        arrays = load_npz_arrays(path)
        metadata_text = _string_scalar(arrays.get("metadata_json"))
        if not metadata_text:
            return False
        metadata = json.loads(metadata_text)
        return bool(
            metadata.get("asset_kind") == asset_kind
            and metadata.get("asset_schema") == _ASSET_SCHEMA
            and metadata.get("source_fast_lane_sha256") == source_digest
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _string_scalar(value: np.ndarray | None) -> str:
    if value is None:
        return ""
    array = np.asarray(value)
    if array.size != 1:
        return ""
    return str(array.reshape(-1)[0])


def _save_npz_atomically(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _layout_from_source(source: Path, track_id: str) -> str:
    try:
        track_index = [part.casefold() for part in source.parts].index(
            track_id.casefold()
        )
    except ValueError:
        return "default"
    between = source.parts[track_index + 1 : -2]
    return "_".join(between) if between else "default"


def _safe_component(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value.strip()
    )
    return cleaned or "default"
