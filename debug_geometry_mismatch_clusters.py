from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

COMPARISON_PATH = Path(r"runs\geometry-debug-drive-01\geometry_comparison.csv")

TELEMETRY_PATH = Path(r"runs\geometry-debug-drive-01\telemetry.csv")

MINIMUM_SPEED_KMH = 5.0

# Kurze Unterbrechungen bis 0,05 Sekunden werden zu einem
# gemeinsamen Ereignis zusammengefasst.
MAXIMUM_CLUSTER_GAP_SAMPLES = 5


def load_csv(path: Path) -> list[dict[str, str]]:
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


def find_clusters(
    mask: np.ndarray,
) -> list[np.ndarray]:
    selected = np.flatnonzero(mask)

    if selected.size == 0:
        return []

    split_locations = np.flatnonzero(
        np.diff(selected) > MAXIMUM_CLUSTER_GAP_SAMPLES + 1
    )

    return list(
        np.split(
            selected,
            split_locations + 1,
        )
    )


comparison_rows = load_csv(COMPARISON_PATH)
telemetry_rows = load_csv(TELEMETRY_PATH)

if len(comparison_rows) != len(telemetry_rows):
    raise RuntimeError(
        "geometry_comparison.csv und telemetry.csv "
        "haben unterschiedlich viele Zeilen: "
        f"{len(comparison_rows)} != {len(telemetry_rows)}"
    )


time_s = float_column(
    comparison_rows,
    "time_s",
)

speed_kmh = float_column(
    comparison_rows,
    "speed_kmh",
)

normalized_position = float_column(
    comparison_rows,
    "normalized_car_position",
)

projected_progress_m = float_column(
    comparison_rows,
    "projected_progress_m",
)

margin_m = float_column(
    comparison_rows,
    "signed_boundary_margin_m",
)

cross_fraction = float_column(
    comparison_rows,
    "cross_section_fraction",
)

world_x = float_column(
    comparison_rows,
    "world_xy_0_m",
)

world_y = float_column(
    comparison_rows,
    "world_xy_1_m",
)

inside_geometry = (
    int_column(
        comparison_rows,
        "inside_geometry",
    )
    != 0
)

tyres_out = int_column(
    comparison_rows,
    "number_of_tyres_out",
)

completed_laps = int_column(
    telemetry_rows,
    "completed_laps",
)

current_lap_time_ms = int_column(
    telemetry_rows,
    "current_lap_time_ms",
)


moving = np.isfinite(speed_kmh) & (speed_kmh >= MINIMUM_SPEED_KMH)

safe_but_outside = moving & (tyres_out == 0) & ~inside_geometry

tyres_out_but_center_inside = moving & (tyres_out > 0) & inside_geometry


def side_name(fraction: float) -> str:
    if fraction < 0.0:
        return "links"

    if fraction > 1.0:
        return "rechts"

    return "innerhalb"


def print_clusters(
    title: str,
    mask: np.ndarray,
) -> None:
    clusters = find_clusters(mask)

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

    print("Gesamte Treffer:", int(np.sum(mask)))
    print("Anzahl Ereignisse:", len(clusters))

    if not clusters:
        return

    for cluster_number, cluster in enumerate(
        clusters,
        start=1,
    ):
        start_index = int(cluster[0])
        end_index = int(cluster[-1])

        worst_index = int(cluster[np.argmin(margin_m[cluster])])

        start_time = time_s[start_index] - time_s[0]

        end_time = time_s[end_index] - time_s[0]

        duration = time_s[end_index] - time_s[start_index]

        print()
        print(f"Ereignis {cluster_number}")
        print("-" * 40)

        print(
            "Samplebereich:",
            start_index,
            "bis",
            end_index,
        )

        print(
            "Treffer im Ereignis:",
            cluster.size,
        )

        print(
            "Zeitbereich:",
            f"{start_time:.3f}",
            "bis",
            f"{end_time:.3f}",
            "s",
        )

        print(
            "Dauer:",
            f"{duration:.3f}",
            "s",
        )

        print(
            "Normalisierter Fortschritt:",
            f"{normalized_position[start_index]:.6f}",
            "bis",
            f"{normalized_position[end_index]:.6f}",
        )

        print(
            "Fortschritt in Metern:",
            f"{projected_progress_m[start_index]:.3f}",
            "bis",
            f"{projected_progress_m[end_index]:.3f}",
            "m",
        )

        print(
            "Runde:",
            completed_laps[start_index],
            "bis",
            completed_laps[end_index],
        )

        print(
            "Rundenzeit:",
            f"{current_lap_time_ms[start_index] / 1000.0:.3f}",
            "bis",
            f"{current_lap_time_ms[end_index] / 1000.0:.3f}",
            "s",
        )

        print(
            "Geschwindigkeit Minimum, Median, Maximum:",
            f"{np.min(speed_kmh[cluster]):.2f},",
            f"{np.median(speed_kmh[cluster]):.2f},",
            f"{np.max(speed_kmh[cluster]):.2f}",
            "km/h",
        )

        print(
            "Randabstand Minimum, Median, Maximum:",
            f"{np.min(margin_m[cluster]):.3f},",
            f"{np.median(margin_m[cluster]):.3f},",
            f"{np.max(margin_m[cluster]):.3f}",
            "m",
        )

        print(
            "Schlimmster Punkt:",
            f"Sample {worst_index},",
            f"Seite {side_name(cross_fraction[worst_index])},",
            f"Randabstand {margin_m[worst_index]:.3f} m",
        )

        print(
            "Weltposition am schlimmsten Punkt:",
            np.asarray(
                [
                    world_x[worst_index],
                    world_y[worst_index],
                ]
            ),
        )

        print(
            "Reifen außerhalb am schlimmsten Punkt:",
            tyres_out[worst_index],
        )


def print_progress_bins(
    mask: np.ndarray,
) -> None:
    print()
    print("=" * 78)
    print("Verteilung der sicheren Überschreitungen über die Runde")
    print("=" * 78)

    bin_count = 50
    bin_width = 1.0 / bin_count

    selected_positions = normalized_position[mask]

    if selected_positions.size == 0:
        print("Keine Treffer vorhanden.")
        return

    bin_indices = np.floor(np.mod(selected_positions, 1.0) * bin_count).astype(np.int64)

    bin_indices = np.clip(
        bin_indices,
        0,
        bin_count - 1,
    )

    counts = np.bincount(
        bin_indices,
        minlength=bin_count,
    )

    for index, count in enumerate(counts):
        if count == 0:
            continue

        lower = index * bin_width
        upper = (index + 1) * bin_width

        print(
            f"{100.0 * lower:6.2f} bis "
            f"{100.0 * upper:6.2f} Prozent: "
            f"{count:4d} Samples"
        )


print("=" * 78)
print("Grunddaten")
print("=" * 78)

print("Gesamte Samples:", len(comparison_rows))
print("Bewegte Samples:", int(np.sum(moving)))
print(
    "Sicher, aber außerhalb:",
    int(np.sum(safe_but_outside)),
)
print(
    "Reifen außerhalb, aber Fahrzeugpunkt innerhalb:",
    int(np.sum(tyres_out_but_center_inside)),
)


print_clusters(
    "A: Keine Reifen außerhalb, Fahrzeugpunkt aber außerhalb",
    safe_but_outside,
)

print_clusters(
    "B: Reifen außerhalb, Fahrzeugpunkt aber innerhalb",
    tyres_out_but_center_inside,
)

print_progress_bins(
    safe_but_outside,
)
