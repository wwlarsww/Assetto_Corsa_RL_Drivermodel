from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from assetto_corsa_rl.ac_integration.track_assets import (
    circular_interpolate,
)

COMPARISON_PATH = Path(r"runs\geometry-debug-drive-01\geometry_comparison.csv")

GEOMETRY_PATH = Path(
    r"data\generated\assetto_corsa_tracks" r"\monza\default\track_geometry.npz"
)

OUTPUT_DIRECTORY = Path(r"runs\geometry-debug-drive-01\mismatch_plots")

MINIMUM_SPEED_KMH = 5.0

# Sichtbarer Bereich vor und hinter dem jeweiligen Ereignis.
PLOT_PADDING_M = 120.0


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Datei nicht gefunden: {path.resolve()}")

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"Datei enthält keine Daten: {path.resolve()}")

    return rows


def float_column(
    rows: list[dict[str, str]],
    name: str,
) -> np.ndarray:
    return np.asarray(
        [float(row[name]) for row in rows],
        dtype=np.float64,
    )


def int_column(
    rows: list[dict[str, str]],
    name: str,
) -> np.ndarray:
    return np.asarray(
        [int(float(row[name])) for row in rows],
        dtype=np.int64,
    )


def circular_delta(
    values: np.ndarray,
    center: float,
    period: float,
) -> np.ndarray:
    """Signed shortest distance around a closed track."""

    return (
        np.asarray(values, dtype=np.float64) - float(center) + 0.5 * period
    ) % period - 0.5 * period


def contiguous_clusters(
    mask: np.ndarray,
) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)

    if indices.size == 0:
        return []

    split_locations = np.flatnonzero(np.diff(indices) > 1)

    return list(
        np.split(
            indices,
            split_locations + 1,
        )
    )


rows = load_rows(COMPARISON_PATH)

sample = int_column(rows, "sample")
speed_kmh = float_column(rows, "speed_kmh")
progress_m = float_column(rows, "projected_progress_m")
normalized_position = float_column(
    rows,
    "normalized_car_position",
)

position = np.column_stack(
    (
        float_column(rows, "world_xy_0_m"),
        float_column(rows, "world_xy_1_m"),
    )
)

lateral_offset_m = float_column(
    rows,
    "lateral_offset_m",
)

margin_m = float_column(
    rows,
    "signed_boundary_margin_m",
)

tyres_out = int_column(
    rows,
    "number_of_tyres_out",
)


with np.load(
    GEOMETRY_PATH,
    allow_pickle=False,
) as geometry:
    track_distance_m = np.asarray(
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

    side_left_m = np.asarray(
        geometry["side_left_m"],
        dtype=np.float64,
    )

    side_right_m = np.asarray(
        geometry["side_right_m"],
        dtype=np.float64,
    )

    track_length_m = float(np.asarray(geometry["track_length_m"]).reshape(-1)[0])


moving = np.isfinite(speed_kmh) & (speed_kmh >= MINIMUM_SPEED_KMH)

safe = tyres_out == 0

safe_but_outside = moving & safe & (margin_m < 0.0)

clusters = contiguous_clusters(safe_but_outside)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 78)
print("Gefundene Abweichungsereignisse")
print("=" * 78)

print("Anzahl:", len(clusters))
print("Samples:", int(np.sum(safe_but_outside)))


for event_number, cluster in enumerate(
    clusters,
    start=1,
):
    event_progress = progress_m[cluster]

    center_progress = float(np.median(event_progress))

    geometry_relative_m = circular_delta(
        track_distance_m,
        center_progress,
        track_length_m,
    )

    geometry_selection = np.abs(geometry_relative_m) <= PLOT_PADDING_M

    geometry_indices = np.flatnonzero(geometry_selection)

    geometry_order = np.argsort(geometry_relative_m[geometry_indices])

    geometry_indices = geometry_indices[geometry_order]

    telemetry_relative_m = circular_delta(
        progress_m,
        center_progress,
        track_length_m,
    )

    telemetry_selection = moving & (np.abs(telemetry_relative_m) <= PLOT_PADDING_M)

    telemetry_indices = np.flatnonzero(telemetry_selection)

    telemetry_order = np.argsort(telemetry_relative_m[telemetry_indices])

    telemetry_indices = telemetry_indices[telemetry_order]

    interpolated_side_left = circular_interpolate(
        track_distance_m,
        side_left_m,
        progress_m,
        track_length_m,
    )

    interpolated_side_right = circular_interpolate(
        track_distance_m,
        side_right_m,
        progress_m,
        track_length_m,
    )

    outside_left = lateral_offset_m[cluster] > interpolated_side_left[cluster]

    outside_right = lateral_offset_m[cluster] < -interpolated_side_right[cluster]

    required_extension_m = np.zeros(
        cluster.size,
        dtype=np.float64,
    )

    required_extension_m[outside_left] = (
        lateral_offset_m[cluster][outside_left]
        - interpolated_side_left[cluster][outside_left]
    )

    required_extension_m[outside_right] = (
        -interpolated_side_right[cluster][outside_right]
        - lateral_offset_m[cluster][outside_right]
    )

    worst_local_index = int(np.argmax(required_extension_m))

    worst_index = int(cluster[worst_local_index])

    event_start_relative = float(
        circular_delta(
            np.asarray([progress_m[cluster[0]]]),
            center_progress,
            track_length_m,
        )[0]
    )

    event_end_relative = float(
        circular_delta(
            np.asarray([progress_m[cluster[-1]]]),
            center_progress,
            track_length_m,
        )[0]
    )

    print()
    print(f"Ereignis {event_number}")
    print("-" * 50)

    print(
        "Samples:",
        int(cluster[0]),
        "bis",
        int(cluster[-1]),
    )

    print(
        "Rundenbereich:",
        f"{100.0 * normalized_position[cluster[0]]:.3f}",
        "bis",
        f"{100.0 * normalized_position[cluster[-1]]:.3f}",
        "Prozent",
    )

    print(
        "Fortschritt:",
        f"{progress_m[cluster[0]]:.3f}",
        "bis",
        f"{progress_m[cluster[-1]]:.3f}",
        "m",
    )

    print(
        "AI Breite links Minimum, Median, Maximum:",
        f"{np.min(interpolated_side_left[cluster]):.3f},",
        f"{np.median(interpolated_side_left[cluster]):.3f},",
        f"{np.max(interpolated_side_left[cluster]):.3f}",
        "m",
    )

    print(
        "Fahrzeugposition lateral Minimum, Median, Maximum:",
        f"{np.min(lateral_offset_m[cluster]):.3f},",
        f"{np.median(lateral_offset_m[cluster]):.3f},",
        f"{np.max(lateral_offset_m[cluster]):.3f}",
        "m",
    )

    print(
        "Erforderliche Erweiterung Minimum, Median, Maximum:",
        f"{np.min(required_extension_m):.3f},",
        f"{np.median(required_extension_m):.3f},",
        f"{np.max(required_extension_m):.3f}",
        "m",
    )

    print(
        "Schlimmster Sample:",
        int(sample[worst_index]),
    )

    print(
        "Schlimmste Weltposition:",
        position[worst_index],
    )

    print(
        "Schlimmster Fortschritt:",
        f"{progress_m[worst_index]:.3f} m",
    )

    print(
        "Schlimmste erforderliche Erweiterung:",
        f"{required_extension_m[worst_local_index]:.3f} m",
    )

    print(
        "Außerhalb links:",
        int(np.sum(outside_left)),
        "Samples",
    )

    print(
        "Außerhalb rechts:",
        int(np.sum(outside_right)),
        "Samples",
    )

    # ---------------------------------------------------------------
    # Karte der lokalen Geometrie
    # ---------------------------------------------------------------

    figure, axis = plt.subplots(figsize=(10, 8))

    axis.plot(
        centerline[geometry_indices, 0],
        centerline[geometry_indices, 1],
        linestyle="--",
        linewidth=1.5,
        label="AI Linie",
    )

    axis.plot(
        left_boundary[geometry_indices, 0],
        left_boundary[geometry_indices, 1],
        linewidth=2.0,
        label="Linke AI Grenze",
    )

    axis.plot(
        right_boundary[geometry_indices, 0],
        right_boundary[geometry_indices, 1],
        linewidth=2.0,
        label="Rechte AI Grenze",
    )

    valid_inside = telemetry_selection & (tyres_out == 0) & (margin_m >= 0.0)

    valid_outside = telemetry_selection & (tyres_out == 0) & (margin_m < 0.0)

    reported_outside = telemetry_selection & (tyres_out > 0)

    axis.scatter(
        position[valid_inside, 0],
        position[valid_inside, 1],
        s=10,
        label="0 Reifen außerhalb, Geometrie innerhalb",
    )

    axis.scatter(
        position[reported_outside, 0],
        position[reported_outside, 1],
        s=16,
        marker="^",
        label="Assetto Corsa meldet Reifen außerhalb",
    )

    axis.scatter(
        position[valid_outside, 0],
        position[valid_outside, 1],
        s=35,
        marker="x",
        label="0 Reifen außerhalb, Geometrie außerhalb",
    )

    axis.scatter(
        position[worst_index, 0],
        position[worst_index, 1],
        s=90,
        marker="X",
        label="Schlimmster Punkt",
    )

    axis.annotate(
        (
            f"Sample {sample[worst_index]}\n"
            f"{required_extension_m[worst_local_index]:.2f} m"
        ),
        (
            position[worst_index, 0],
            position[worst_index, 1],
        ),
        xytext=(10, 10),
        textcoords="offset points",
    )

    axis.set_title(f"Geometrieabweichung Ereignis {event_number}")

    axis.set_xlabel("Weltkoordinate 1 [m]")

    axis.set_ylabel("Weltkoordinate 2 [m]")

    axis.axis("equal")
    axis.grid(True)
    axis.legend()

    map_path = OUTPUT_DIRECTORY / (f"event_{event_number:02d}_map.png")

    figure.tight_layout()
    figure.savefig(
        map_path,
        dpi=180,
    )

    plt.close(figure)

    # ---------------------------------------------------------------
    # Laterale Breite und Fahrzeugposition
    # ---------------------------------------------------------------

    figure, axis = plt.subplots(figsize=(11, 6))

    axis.plot(
        geometry_relative_m[geometry_indices],
        side_left_m[geometry_indices],
        linewidth=2.0,
        label="Linke AI Breite",
    )

    axis.plot(
        geometry_relative_m[geometry_indices],
        -side_right_m[geometry_indices],
        linewidth=2.0,
        label="Rechte AI Breite",
    )

    axis.plot(
        telemetry_relative_m[telemetry_indices],
        lateral_offset_m[telemetry_indices],
        linewidth=1.2,
        label="Fahrzeugposition lateral",
    )

    cluster_relative_m = circular_delta(
        progress_m[cluster],
        center_progress,
        track_length_m,
    )

    axis.scatter(
        cluster_relative_m,
        lateral_offset_m[cluster],
        s=25,
        marker="x",
        label="Abweichung trotz 0 Reifen außerhalb",
    )

    axis.axvspan(
        min(
            event_start_relative,
            event_end_relative,
        ),
        max(
            event_start_relative,
            event_end_relative,
        ),
        alpha=0.15,
        label="Ereignisbereich",
    )

    axis.axhline(
        0.0,
        linewidth=0.8,
    )

    axis.set_title(
        (
            f"Laterales Profil Ereignis {event_number}\n"
            "Positive Werte liegen links der AI Linie"
        )
    )

    axis.set_xlabel("Relative Streckenposition [m]")

    axis.set_ylabel("Lateraler Abstand zur AI Linie [m]")

    axis.grid(True)
    axis.legend()

    profile_path = OUTPUT_DIRECTORY / (f"event_{event_number:02d}_lateral_profile.png")

    figure.tight_layout()
    figure.savefig(
        profile_path,
        dpi=180,
    )

    plt.close(figure)

    print("Kartenansicht:", map_path)
    print("Laterales Profil:", profile_path)


print()
print("=" * 78)
print("Fertig")
print("=" * 78)

print(
    "Ausgaben gespeichert unter:",
    OUTPUT_DIRECTORY.resolve(),
)
