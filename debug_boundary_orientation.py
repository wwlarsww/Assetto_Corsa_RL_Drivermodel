from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from assetto_corsa_rl.ac_integration.track_assets import (
    circular_interpolate,
)

COMPARISON_PATH = Path(r"runs\geometry-debug-drive-01\geometry_comparison.csv")

GEOMETRY_PATH = Path(
    r"data\generated\assetto_corsa_tracks" r"\monza\default\track_geometry.npz"
)


def signed_boundary_margin(
    positions: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Return a positive margin inside and a negative margin outside."""

    cross_section = right - left

    width_squared = np.sum(
        cross_section**2,
        axis=1,
    )

    valid = width_squared > 1e-12

    fraction = np.full(
        positions.shape[0],
        np.nan,
        dtype=np.float64,
    )

    fraction[valid] = (
        np.sum(
            (positions[valid] - left[valid]) * cross_section[valid],
            axis=1,
        )
        / width_squared[valid]
    )

    width = np.sqrt(width_squared)

    margin = np.full(
        positions.shape[0],
        np.nan,
        dtype=np.float64,
    )

    before_left = valid & (fraction < 0.0)
    after_right = valid & (fraction > 1.0)
    inside = valid & ~before_left & ~after_right

    margin[before_left] = fraction[before_left] * width[before_left]

    margin[after_right] = (1.0 - fraction[after_right]) * width[after_right]

    margin[inside] = (
        np.minimum(
            fraction[inside],
            1.0 - fraction[inside],
        )
        * width[inside]
    )

    return margin


def evaluate_boundaries(
    name: str,
    *,
    source_left: np.ndarray,
    source_right: np.ndarray,
    track_distance: np.ndarray,
    track_length: float,
    progress: np.ndarray,
    positions: np.ndarray,
    moving: np.ndarray,
    safe: np.ndarray,
) -> np.ndarray:
    interpolated_left = circular_interpolate(
        track_distance,
        source_left,
        progress,
        track_length,
    )

    interpolated_right = circular_interpolate(
        track_distance,
        source_right,
        progress,
        track_length,
    )

    margin = signed_boundary_margin(
        positions,
        interpolated_left,
        interpolated_right,
    )

    inside = margin >= 0.0

    moving_count = int(np.sum(moving))
    safe_count = int(np.sum(safe))

    print()
    print(name)
    print("-" * len(name))

    print(
        "Alle bewegten Samples innerhalb:",
        int(np.sum(moving & inside)),
        "von",
        moving_count,
        f"({100.0 * np.mean(inside[moving]):.2f} Prozent)",
    )

    print(
        "Samples mit 0 Reifen außerhalb und " "Geometrie innerhalb:",
        int(np.sum(safe & inside)),
        "von",
        safe_count,
        f"({100.0 * np.mean(inside[safe]):.2f} Prozent)",
    )

    print(
        "Samples mit 0 Reifen außerhalb, aber " "Geometrie außerhalb:",
        int(np.sum(safe & ~inside)),
    )

    print(
        "Median Randabstand bei 0 Reifen außerhalb:",
        float(np.median(margin[safe])),
        "m",
    )

    print(
        "5 Prozent Quantil bei 0 Reifen außerhalb:",
        float(np.percentile(margin[safe], 5)),
        "m",
    )

    print(
        "Minimum bei 0 Reifen außerhalb:",
        float(np.min(margin[safe])),
        "m",
    )

    return margin


if not COMPARISON_PATH.is_file():
    raise FileNotFoundError(
        f"Vergleichsdatei nicht gefunden: " f"{COMPARISON_PATH.resolve()}"
    )

if not GEOMETRY_PATH.is_file():
    raise FileNotFoundError(
        f"Geometriedatei nicht gefunden: " f"{GEOMETRY_PATH.resolve()}"
    )


with COMPARISON_PATH.open(
    "r",
    encoding="utf-8",
    newline="",
) as handle:
    rows = list(csv.DictReader(handle))


def float_column(name: str) -> np.ndarray:
    return np.asarray(
        [float(row[name]) for row in rows],
        dtype=np.float64,
    )


positions = np.column_stack(
    (
        float_column("world_xy_0_m"),
        float_column("world_xy_1_m"),
    )
)

progress = float_column("projected_progress_m")
speed_kmh = float_column("speed_kmh")

tyres_out = np.asarray(
    [int(float(row["number_of_tyres_out"])) for row in rows],
    dtype=np.int64,
)


with np.load(
    GEOMETRY_PATH,
    allow_pickle=False,
) as geometry:
    track_distance = np.asarray(
        geometry["track_distance_m"],
        dtype=np.float64,
    )

    center = np.asarray(
        geometry["centerline_world_xy_m"],
        dtype=np.float64,
    )

    stored_left = np.asarray(
        geometry["left_boundary_world_xy_m"],
        dtype=np.float64,
    )

    stored_right = np.asarray(
        geometry["right_boundary_world_xy_m"],
        dtype=np.float64,
    )

    forward = np.asarray(
        geometry["forward_vector_xy"],
        dtype=np.float64,
    )

    side_left = np.asarray(
        geometry["side_left_m"],
        dtype=np.float64,
    )

    side_right = np.asarray(
        geometry["side_right_m"],
        dtype=np.float64,
    )

    track_length = float(np.asarray(geometry["track_length_m"]).reshape(-1)[0])


moving = (speed_kmh >= 5.0) & np.isfinite(positions).all(axis=1)

safe = moving & (tyres_out == 0)


# Geometrische Tangente aus benachbarten Punkten.
previous_center = np.roll(
    center,
    1,
    axis=0,
)

next_center = np.roll(
    center,
    -1,
    axis=0,
)

tangent = next_center - previous_center

tangent_norm = np.linalg.norm(
    tangent,
    axis=1,
)

if np.any(tangent_norm <= 1e-12):
    raise RuntimeError("Die Centerline enthält ungültige Tangenten.")

tangent /= tangent_norm[:, None]


forward_norm = np.linalg.norm(
    forward,
    axis=1,
)

if np.any(forward_norm <= 1e-12):
    raise RuntimeError("Die Geometrie enthält ungültige Forward Vectoren.")

forward_normalized = forward / forward_norm[:, None]


alignment = np.sum(
    tangent * forward_normalized,
    axis=1,
)

alignment_angle_deg = np.degrees(
    np.arccos(
        np.clip(
            alignment,
            -1.0,
            1.0,
        )
    )
)


print("=" * 72)
print("1. Forward Vector gegen geometrische Tangente")
print("=" * 72)

print(
    "Alignment Minimum:",
    float(np.min(alignment)),
)

print(
    "Alignment 5 Prozent:",
    float(np.percentile(alignment, 5)),
)

print(
    "Alignment Median:",
    float(np.median(alignment)),
)

print(
    "Alignment 95 Prozent:",
    float(np.percentile(alignment, 95)),
)

print(
    "Alignment Maximum:",
    float(np.max(alignment)),
)

print()

print(
    "Fast gleiche Richtung, Dot > 0.9:",
    int(np.sum(alignment > 0.9)),
    "von",
    alignment.size,
)

print(
    "Fast entgegengesetzt, Dot < -0.9:",
    int(np.sum(alignment < -0.9)),
    "von",
    alignment.size,
)

print(
    "Median Winkelabweichung:",
    float(np.median(alignment_angle_deg)),
    "Grad",
)


# The canonical project plane is [AC z, AC x].
# This orientation matches the corrected track asset generator.
tangent_left_normal = np.column_stack(
    (
        -tangent[:, 1],
        tangent[:, 0],
    )
)

tangent_left_boundary = center + tangent_left_normal * side_left[:, None]

tangent_right_boundary = center - tangent_left_normal * side_right[:, None]


left_difference = np.linalg.norm(
    stored_left - tangent_left_boundary,
    axis=1,
)

right_difference = np.linalg.norm(
    stored_right - tangent_right_boundary,
    axis=1,
)


print()
print("=" * 72)
print("2. Gespeicherte gegen tangentenbasierte Grenzen")
print("=" * 72)

print(
    "Median Abweichung linker Rand:",
    float(np.median(left_difference)),
    "m",
)

print(
    "Maximum Abweichung linker Rand:",
    float(np.max(left_difference)),
    "m",
)

print(
    "Median Abweichung rechter Rand:",
    float(np.median(right_difference)),
    "m",
)

print(
    "Maximum Abweichung rechter Rand:",
    float(np.max(right_difference)),
    "m",
)


# Alternative Hypothese:
# SideLeft und SideRight wurden auf die jeweils falsche
# physische Streckenseite angewendet.
mirrored_left = 2.0 * center - stored_left

mirrored_right = 2.0 * center - stored_right


print()
print("=" * 72)
print("3. Vergleich der drei Randvarianten")
print("=" * 72)

evaluate_boundaries(
    "A: Aktuell gespeicherte Grenzen",
    source_left=stored_left,
    source_right=stored_right,
    track_distance=track_distance,
    track_length=track_length,
    progress=progress,
    positions=positions,
    moving=moving,
    safe=safe,
)

evaluate_boundaries(
    "B: Grenzen aus geometrischer Tangente",
    source_left=tangent_left_boundary,
    source_right=tangent_right_boundary,
    track_distance=track_distance,
    track_length=track_length,
    progress=progress,
    positions=positions,
    moving=moving,
    safe=safe,
)

evaluate_boundaries(
    "C: Gespiegelte SideLeft und SideRight Orientierung",
    source_left=mirrored_left,
    source_right=mirrored_right,
    track_distance=track_distance,
    track_length=track_length,
    progress=progress,
    positions=positions,
    moving=moving,
    safe=safe,
)
