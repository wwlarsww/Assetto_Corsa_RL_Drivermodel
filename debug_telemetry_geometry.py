from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from assetto_corsa_rl.ac_integration.track_assets import (
    circular_interpolate,
    project_world_position,
)

TELEMETRY_PATH = Path(r"runs\geometry-debug-drive-01\telemetry.csv")

SUMMARY_PATH = Path(r"runs\geometry-debug-drive-01\summary.json")

GEOMETRY_PATH = Path(
    r"data\generated\assetto_corsa_tracks" r"\monza\default\track_geometry.npz"
)

OUTPUT_PATH = Path(r"runs\geometry-debug-drive-01" r"\geometry_comparison.csv")


def circular_difference(
    first: np.ndarray,
    second: np.ndarray,
    period: float,
) -> np.ndarray:
    """Signed shortest difference on a periodic interval."""

    return (first - second + 0.5 * period) % period - 0.5 * period


def describe(
    name: str,
    values: np.ndarray,
    unit: str = "",
) -> None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        print(f"{name}: keine gültigen Werte")
        return

    suffix = f" {unit}" if unit else ""

    print(f"{name}:")
    print(f"  Minimum: {np.min(values): .6f}{suffix}")
    print(f"  5 Prozent: {np.percentile(values, 5): .6f}{suffix}")
    print(f"  Median: {np.median(values): .6f}{suffix}")
    print(f"  95 Prozent: {np.percentile(values, 95): .6f}{suffix}")
    print(f"  Maximum: {np.max(values): .6f}{suffix}")


def load_telemetry(
    path: Path,
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Telemetriedatei nicht gefunden: {path.resolve()}")

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError("Die Telemetriedatei ist leer.")

    def floats(name: str) -> np.ndarray:
        return np.asarray(
            [float(row[name]) for row in rows],
            dtype=np.float64,
        )

    def integers(name: str) -> np.ndarray:
        return np.asarray(
            [int(float(row[name])) for row in rows],
            dtype=np.int64,
        )

    native_x = floats("world_position_m_0")
    native_y = floats("world_position_m_1")
    native_z = floats("world_position_m_2")

    # Das Projekt verwendet:
    # world_xy = [AC z, AC x]
    position_xy = np.column_stack(
        (
            native_z,
            native_x,
        )
    )

    return {
        "time_s": floats("monotonic_time_s"),
        "position_xy_m": position_xy,
        "native_y_m": native_y,
        "normalized_position": floats("normalized_car_position"),
        "speed_kmh": floats("speed_kmh"),
        "status": integers("status"),
        "tyres_out": integers("number_of_tyres_out"),
    }


telemetry = load_telemetry(TELEMETRY_PATH)

if not GEOMETRY_PATH.is_file():
    raise FileNotFoundError(
        f"Geometriedatei nicht gefunden: " f"{GEOMETRY_PATH.resolve()}"
    )


with np.load(
    GEOMETRY_PATH,
    allow_pickle=False,
) as geometry:
    track_distance = np.asarray(
        geometry["track_distance_m"],
        dtype=np.float64,
    )

    centerline = np.asarray(
        geometry["centerline_world_xy_m"],
        dtype=np.float64,
    )

    left_boundary = np.asarray(
        geometry["left_boundary_world_xy_m"],
        dtype=np.float64,
    )

    right_boundary = np.asarray(
        geometry["right_boundary_world_xy_m"],
        dtype=np.float64,
    )

    forward_vectors = np.asarray(
        geometry["forward_vector_xy"],
        dtype=np.float64,
    )

    track_length = float(np.asarray(geometry["track_length_m"]).reshape(-1)[0])


summary = {}

if SUMMARY_PATH.is_file():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


positions = telemetry["position_xy_m"]
normalized_position = telemetry["normalized_position"]
speed_kmh = telemetry["speed_kmh"]
status = telemetry["status"]
tyres_out = telemetry["tyres_out"]


sample_count = positions.shape[0]

projected_progress = np.empty(
    sample_count,
    dtype=np.float64,
)

projection_distance = np.empty(
    sample_count,
    dtype=np.float64,
)

lateral_offset = np.empty(
    sample_count,
    dtype=np.float64,
)

longitudinal_residual = np.empty(
    sample_count,
    dtype=np.float64,
)

cross_section_fraction = np.empty(
    sample_count,
    dtype=np.float64,
)

signed_boundary_margin = np.empty(
    sample_count,
    dtype=np.float64,
)

left_margin = np.empty(
    sample_count,
    dtype=np.float64,
)

right_margin = np.empty(
    sample_count,
    dtype=np.float64,
)


for index, position in enumerate(positions):
    projection = project_world_position(
        centerline,
        track_distance,
        track_length,
        position,
    )

    progress = projection.progress_m

    center_point = circular_interpolate(
        track_distance,
        centerline,
        np.asarray([progress]),
        track_length,
    )[0]

    left_point = circular_interpolate(
        track_distance,
        left_boundary,
        np.asarray([progress]),
        track_length,
    )[0]

    right_point = circular_interpolate(
        track_distance,
        right_boundary,
        np.asarray([progress]),
        track_length,
    )[0]

    forward = circular_interpolate(
        track_distance,
        forward_vectors,
        np.asarray([progress]),
        track_length,
    )[0]

    forward_norm = float(np.linalg.norm(forward))

    if forward_norm <= 1e-12:
        raise RuntimeError(f"Ungültiger Forward Vector bei Sample {index}")

    forward = forward / forward_norm

    local_left = np.asarray(
        [
            -forward[1],
            forward[0],
        ],
        dtype=np.float64,
    )

    displacement = position - center_point

    lateral = float(
        np.dot(
            displacement,
            local_left,
        )
    )

    longitudinal = float(
        np.dot(
            displacement,
            forward,
        )
    )

    cross_section = right_point - left_point

    width_squared = float(
        np.dot(
            cross_section,
            cross_section,
        )
    )

    if width_squared <= 1e-12:
        raise RuntimeError(f"Ungültiger Querschnitt bei Sample {index}")

    width = float(np.sqrt(width_squared))

    fraction = float(
        np.dot(
            position - left_point,
            cross_section,
        )
        / width_squared
    )

    current_left_margin = fraction * width

    current_right_margin = (1.0 - fraction) * width

    if fraction < 0.0:
        signed_margin = fraction * width
    elif fraction > 1.0:
        signed_margin = (1.0 - fraction) * width
    else:
        signed_margin = min(
            current_left_margin,
            current_right_margin,
        )

    projected_progress[index] = progress
    projection_distance[index] = projection.distance_m
    lateral_offset[index] = lateral
    longitudinal_residual[index] = longitudinal
    cross_section_fraction[index] = fraction
    signed_boundary_margin[index] = signed_margin
    left_margin[index] = current_left_margin
    right_margin[index] = current_right_margin


ac_progress = normalized_position * track_length

progress_error_m = circular_difference(
    projected_progress,
    ac_progress,
    track_length,
)

inside_geometry = (cross_section_fraction >= 0.0) & (cross_section_fraction <= 1.0)

outside_left = cross_section_fraction < 0.0

outside_right = cross_section_fraction > 1.0

ac_reports_outside = tyres_out > 0

live = status == 2

moving = live & (speed_kmh >= 5.0) & np.isfinite(positions).all(axis=1)


movement_distance = float(
    np.sum(
        np.linalg.norm(
            np.diff(
                positions,
                axis=0,
            ),
            axis=1,
        )
    )
)


print("=" * 72)
print("1. Aufnahme")
print("=" * 72)

print("Samples:", sample_count)
print("Live Samples:", int(np.sum(live)))
print("Bewegte Live Samples:", int(np.sum(moving)))
print("Gefahrene Punktdistanz:", movement_distance, "m")
print(
    "Geschwindigkeit:",
    float(np.min(speed_kmh)),
    "bis",
    float(np.max(speed_kmh)),
    "km/h",
)

print()
print("Geometrie Streckenlänge:", track_length, "m")

if "track_spline_length_m" in summary:
    ac_track_length = float(summary["track_spline_length_m"])

    print(
        "AC Shared Memory Streckenlänge:",
        ac_track_length,
        "m",
    )

    print(
        "Differenz der Streckenlängen:",
        track_length - ac_track_length,
        "m",
    )


print()
print("=" * 72)
print("2. Fortschrittsvergleich")
print("=" * 72)

describe(
    "Signierter Fortschrittsfehler",
    progress_error_m[moving],
    "m",
)

describe(
    "Absoluter Fortschrittsfehler",
    np.abs(progress_error_m[moving]),
    "m",
)


print()
print("=" * 72)
print("3. Projektion auf die AI Linie")
print("=" * 72)

describe(
    "Abstand zur AI Linie",
    projection_distance[moving],
    "m",
)

describe(
    "Lateraler Abstand zur AI Linie",
    lateral_offset[moving],
    "m",
)

describe(
    "Tangentialer Restfehler",
    longitudinal_residual[moving],
    "m",
)


print()
print("=" * 72)
print("4. Vergleich mit den AI Grenzen")
print("=" * 72)

moving_count = max(
    int(np.sum(moving)),
    1,
)

print(
    "Innerhalb laut Geometrie:",
    int(np.sum(inside_geometry & moving)),
    f"({100.0 * np.sum(inside_geometry & moving) / moving_count:.2f} Prozent)",
)

print(
    "Außerhalb links:",
    int(np.sum(outside_left & moving)),
    f"({100.0 * np.sum(outside_left & moving) / moving_count:.2f} Prozent)",
)

print(
    "Außerhalb rechts:",
    int(np.sum(outside_right & moving)),
    f"({100.0 * np.sum(outside_right & moving) / moving_count:.2f} Prozent)",
)

describe(
    "Signierter Abstand zur nächsten Grenze",
    signed_boundary_margin[moving],
    "m",
)


print()
print("=" * 72)
print("5. Vergleich mit number_of_tyres_out")
print("=" * 72)

print(
    "AC meldet keine Reifen außerhalb, " "Geometrie innerhalb:",
    int(np.sum(moving & ~ac_reports_outside & inside_geometry)),
)

print(
    "AC meldet keine Reifen außerhalb, " "Geometrie außerhalb:",
    int(np.sum(moving & ~ac_reports_outside & ~inside_geometry)),
)

print(
    "AC meldet Reifen außerhalb, " "Geometrie innerhalb:",
    int(np.sum(moving & ac_reports_outside & inside_geometry)),
)

print(
    "AC meldet Reifen außerhalb, " "Geometrie außerhalb:",
    int(np.sum(moving & ac_reports_outside & ~inside_geometry)),
)

print()

for tyre_count in sorted(np.unique(tyres_out[moving])):
    selection = moving & (tyres_out == tyre_count)

    print(
        f"Reifen außerhalb = {tyre_count}:",
        int(np.sum(selection)),
        "Samples",
    )

    describe(
        "  Geometrischer Randabstand",
        signed_boundary_margin[selection],
        "m",
    )


print()
print("=" * 72)
print("6. Fortschrittsfehler nach Streckensektor")
print("=" * 72)

sector_count = 12

for sector in range(sector_count):
    lower = sector / sector_count
    upper = (sector + 1) / sector_count

    selection = moving & (normalized_position >= lower) & (normalized_position < upper)

    count = int(np.sum(selection))

    if count == 0:
        continue

    median_error = float(np.median(progress_error_m[selection]))

    median_margin = float(np.median(signed_boundary_margin[selection]))

    outside_percentage = float(100.0 * np.mean(~inside_geometry[selection]))

    print(
        f"Sektor {sector + 1:2d}: "
        f"Samples={count:4d}, "
        f"Fortschrittsfehler={median_error: .3f} m, "
        f"Randabstand={median_margin: .3f} m, "
        f"außerhalb={outside_percentage: .1f} Prozent"
    )


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    fieldnames = [
        "sample",
        "time_s",
        "speed_kmh",
        "normalized_car_position",
        "projected_progress_m",
        "progress_error_m",
        "world_xy_0_m",
        "world_xy_1_m",
        "projection_distance_m",
        "lateral_offset_m",
        "longitudinal_residual_m",
        "cross_section_fraction",
        "signed_boundary_margin_m",
        "inside_geometry",
        "number_of_tyres_out",
    ]

    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for index in range(sample_count):
        writer.writerow(
            {
                "sample": index,
                "time_s": telemetry["time_s"][index],
                "speed_kmh": speed_kmh[index],
                "normalized_car_position": normalized_position[index],
                "projected_progress_m": projected_progress[index],
                "progress_error_m": progress_error_m[index],
                "world_xy_0_m": positions[index, 0],
                "world_xy_1_m": positions[index, 1],
                "projection_distance_m": projection_distance[index],
                "lateral_offset_m": lateral_offset[index],
                "longitudinal_residual_m": longitudinal_residual[index],
                "cross_section_fraction": cross_section_fraction[index],
                "signed_boundary_margin_m": signed_boundary_margin[index],
                "inside_geometry": int(inside_geometry[index]),
                "number_of_tyres_out": tyres_out[index],
            }
        )


print()
print("=" * 72)
print("7. Ausgabedatei")
print("=" * 72)

print("Gespeichert unter:", OUTPUT_PATH)
