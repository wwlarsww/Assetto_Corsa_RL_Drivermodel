"""Verify track, active reference and evaluation data use one world frame."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _text(arrays: Mapping[str, np.ndarray], name: str) -> str:
    value = arrays.get(name)
    if value is None:
        return ""
    array = np.asarray(value)
    return str(array.reshape(-1)[0]) if array.size == 1 else ""


def _points(arrays: Mapping[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in arrays:
            value = np.asarray(arrays[name], dtype=np.float64)
            if value.ndim == 2 and value.shape[1] >= 2 and value.shape[0] > 0:
                return value[:, :2]
    raise KeyError(f"none of the coordinate arrays exist: {names}")


def _downsample(points: np.ndarray, maximum: int = 1200) -> np.ndarray:
    if points.shape[0] <= maximum:
        return points
    index = np.linspace(0, points.shape[0] - 1, maximum).astype(int)
    return points[index]


def _nearest_distances(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    points = _downsample(np.asarray(points, dtype=np.float64))
    path = _downsample(np.asarray(path, dtype=np.float64))
    result = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], 128):
        current = points[start : start + 128]
        squared = np.sum((current[:, None, :] - path[None, :, :]) ** 2, axis=2)
        result[start : start + current.shape[0]] = np.sqrt(np.min(squared, axis=1))
    return result


def _symmetric_p95(first: np.ndarray, second: np.ndarray) -> float:
    return max(
        float(np.percentile(_nearest_distances(first, second), 95)),
        float(np.percentile(_nearest_distances(second, first), 95)),
    )


def main() -> int:
    """Verify static and dynamic coordinate frames."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path)
    parser.add_argument("--maximum-static-p95-m", type=float, default=20.0)
    parser.add_argument("--maximum-vehicle-distance-m", type=float, default=30.0)
    arguments = parser.parse_args()

    geometry = _load(arguments.geometry)
    reference = _load(arguments.reference)
    centerline = _points(
        geometry,
        ("centerline_world_xy_m", "centerline_world_xz_m"),
    )
    reference_line = _points(
        reference,
        ("world_position_xy_m", "world_position_xz_m"),
    )

    direct_static = _symmetric_p95(centerline, reference_line)
    swapped_static = _symmetric_p95(centerline[:, [1, 0]], reference_line)
    transform = _text(geometry, "coordinate_transform").replace(" ", "")
    report: dict[str, object] = {
        "geometry_coordinate_frame": _text(geometry, "coordinate_frame"),
        "geometry_coordinate_transform": transform,
        "static_centerline_reference_p95_m": direct_static,
        "static_p95_if_geometry_axes_swapped_m": swapped_static,
    }
    errors: list[str] = []

    if "[ac_x,ac_z]" in transform:
        errors.append("geometry uses native [AC x, AC z], not telemetry [AC z, AC x]")
    if swapped_static + 2.0 < direct_static:
        errors.append("swapping the geometry axes aligns it better with the reference")
    if direct_static > arguments.maximum_static_p95_m:
        errors.append(
            f"centerline/reference p95 {direct_static:.3f} m exceeds "
            f"{arguments.maximum_static_p95_m:.3f} m"
        )

    if arguments.trajectories is not None:
        trajectories = _load(arguments.trajectories)
        vehicle = _points(
            trajectories,
            ("vehicle_world_xy_m", "positions_xy_m", "vehicle_world_xz_m"),
        )
        vehicle_reference = _nearest_distances(vehicle, reference_line)
        vehicle_centerline = _nearest_distances(vehicle, centerline)
        vehicle_swapped_centerline = _nearest_distances(vehicle, centerline[:, [1, 0]])
        report.update(
            {
                "vehicle_reference_distance_median_m": float(np.median(vehicle_reference)),
                "vehicle_reference_distance_p95_m": float(np.percentile(vehicle_reference, 95)),
                "vehicle_centerline_distance_median_m": float(np.median(vehicle_centerline)),
                "vehicle_centerline_distance_if_geometry_swapped_median_m": float(
                    np.median(vehicle_swapped_centerline)
                ),
                "evaluation_artifact_schema": int(
                    np.asarray(
                        trajectories.get("evaluation_artifact_schema", np.asarray(1))
                    ).reshape(-1)[0]
                ),
            }
        )
        if float(np.median(vehicle_reference)) > arguments.maximum_vehicle_distance_m:
            errors.append("vehicle positions are not close to the active reference")
        if np.median(vehicle_swapped_centerline) + 5.0 < np.median(vehicle_centerline):
            errors.append("vehicle aligns better after swapping the geometry axes")

    report["status"] = "PASS" if not errors else "FAIL"
    report["errors"] = errors
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
