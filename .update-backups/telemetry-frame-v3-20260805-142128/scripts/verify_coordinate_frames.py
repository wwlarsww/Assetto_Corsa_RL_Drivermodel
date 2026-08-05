"""Verify static and dynamic coordinate frames before starting PPO training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

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


def _points(
    arrays: Mapping[str, np.ndarray],
    names: tuple[str, ...],
) -> np.ndarray:
    for name in names:
        if name in arrays:
            value = np.asarray(arrays[name], dtype=np.float64)
            if value.ndim == 2 and value.shape[1] >= 2:
                return value[:, :2]
    raise KeyError(f"none of the coordinate arrays exist: {names}")


def _sample_fraction(points: np.ndarray, count: int = 512) -> np.ndarray:
    query = np.linspace(0.0, points.shape[0], count, endpoint=False)
    lower = np.floor(query).astype(int) % points.shape[0]
    upper = (lower + 1) % points.shape[0]
    fraction = query - np.floor(query)
    return points[lower] * (1.0 - fraction[:, None]) + points[upper] * fraction[:, None]


def _nearest_distances(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    result = np.empty(points.shape[0], dtype=np.float64)
    chunk = 256
    for start in range(0, points.shape[0], chunk):
        current = points[start : start + chunk]
        squared = np.sum(
            (current[:, None, :] - path[None, :, :]) ** 2,
            axis=2,
        )
        result[start : start + current.shape[0]] = np.sqrt(np.min(squared, axis=1))
    return result


def _rms(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path)
    parser.add_argument("--maximum-static-rms-m", type=float, default=5.0)
    parser.add_argument("--maximum-vehicle-distance-m", type=float, default=30.0)
    arguments = parser.parse_args()

    geometry = _load(arguments.geometry)
    reference = _load(arguments.reference)
    centerline = _points(
        geometry,
        ("centerline_world_xz_m", "centerline_world_xy_m"),
    )
    reference_line = _points(
        reference,
        ("world_position_xz_m", "world_position_xy_m"),
    )
    center_sample = _sample_fraction(centerline)
    reference_sample = _sample_fraction(reference_line)
    static_rms = _rms(center_sample, reference_sample)
    swapped_rms = _rms(center_sample[:, [1, 0]], reference_sample)

    geometry_frame = _text(geometry, "coordinate_frame")
    geometry_transform = _text(geometry, "coordinate_transform")
    report: dict[str, object] = {
        "geometry_coordinate_frame": geometry_frame,
        "geometry_coordinate_transform": geometry_transform,
        "static_reference_rms_m": static_rms,
        "static_reference_rms_if_axes_swapped_m": swapped_rms,
    }
    errors: list[str] = []
    if "[ac_z,ac_x]" in geometry_transform.replace(" ", ""):
        errors.append("geometry still uses the broken [AC z, AC x] order")
    if swapped_rms + 1.0 < static_rms:
        errors.append("swapping geometry axes aligns it substantially better")
    if static_rms > arguments.maximum_static_rms_m:
        errors.append(
            f"static geometry/reference RMS {static_rms:.3f} m exceeds "
            f"{arguments.maximum_static_rms_m:.3f} m"
        )

    if arguments.trajectories is not None:
        trajectories = _load(arguments.trajectories)
        vehicle = _points(
            trajectories,
            ("vehicle_world_xz_m", "positions_xy_m"),
        )
        vehicle_distance = _nearest_distances(vehicle, reference_line)
        report.update(
            {
                "vehicle_reference_distance_median_m": float(
                    np.median(vehicle_distance)
                ),
                "vehicle_reference_distance_maximum_m": float(
                    np.max(vehicle_distance)
                ),
                "evaluation_artifact_schema": int(
                    np.asarray(
                        trajectories.get(
                            "evaluation_artifact_schema",
                            np.asarray(1),
                        )
                    ).reshape(-1)[0]
                ),
            }
        )
        if float(np.median(vehicle_distance)) > arguments.maximum_vehicle_distance_m:
            errors.append(
                "vehicle positions are not close to the selected reference line"
            )

    report["status"] = "PASS" if not errors else "FAIL"
    report["errors"] = errors
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
