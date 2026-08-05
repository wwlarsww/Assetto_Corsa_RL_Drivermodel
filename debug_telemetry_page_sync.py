from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

TELEMETRY_PATH = Path(r"runs\geometry-debug-drive-01\telemetry.csv")

COMPARISON_PATH = Path(r"runs\geometry-debug-drive-01\geometry_comparison.csv")

MINIMUM_SPEED_KMH = 5.0

# Eine unveränderte Grafikseite über mindestens diesen Zeitraum
# gilt für diese Diagnose als verdächtig.
STALE_DURATION_S = 0.10


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
        raise RuntimeError(f"Datei ist leer: {path.resolve()}")

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


def equal_value_runs(
    values: np.ndarray,
) -> list[tuple[int, int]]:
    if values.size == 0:
        return []

    changes = np.flatnonzero(values[1:] != values[:-1]) + 1

    starts = np.concatenate(
        (
            np.asarray([0]),
            changes,
        )
    )

    ends = np.concatenate(
        (
            changes - 1,
            np.asarray([values.size - 1]),
        )
    )

    return [
        (int(start), int(end))
        for start, end in zip(
            starts,
            ends,
            strict=True,
        )
    ]


def true_clusters(
    mask: np.ndarray,
) -> list[np.ndarray]:
    selected = np.flatnonzero(mask)

    if selected.size == 0:
        return []

    split_points = np.flatnonzero(np.diff(selected) > 1)

    return list(
        np.split(
            selected,
            split_points + 1,
        )
    )


telemetry_rows = load_rows(TELEMETRY_PATH)
comparison_rows = load_rows(COMPARISON_PATH)

if len(telemetry_rows) != len(comparison_rows):
    raise RuntimeError("Die beiden CSV Dateien haben " "unterschiedlich viele Zeilen.")


time_s = float_column(
    telemetry_rows,
    "monotonic_time_s",
)

physics_packet_id = int_column(
    telemetry_rows,
    "physics_packet_id",
)

graphics_packet_id = int_column(
    telemetry_rows,
    "graphics_packet_id",
)

speed_kmh = float_column(
    telemetry_rows,
    "speed_kmh",
)

normalized_position = float_column(
    telemetry_rows,
    "normalized_car_position",
)

lap_time_ms = int_column(
    telemetry_rows,
    "current_lap_time_ms",
)

tyres_out = int_column(
    telemetry_rows,
    "number_of_tyres_out",
)

position = np.column_stack(
    (
        float_column(
            telemetry_rows,
            "world_position_m_2",
        ),
        float_column(
            telemetry_rows,
            "world_position_m_0",
        ),
    )
)

inside_geometry = (
    int_column(
        comparison_rows,
        "inside_geometry",
    )
    != 0
)

margin_m = float_column(
    comparison_rows,
    "signed_boundary_margin_m",
)

progress = float_column(
    comparison_rows,
    "normalized_car_position",
)


moving = speed_kmh >= MINIMUM_SPEED_KMH

safe_but_outside = moving & (tyres_out == 0) & ~inside_geometry


# ---------------------------------------------------------------------
# 1. Lange Wiederholungen derselben Grafikpaketnummer
# ---------------------------------------------------------------------

stale_graphics_packet = np.zeros(
    len(telemetry_rows),
    dtype=bool,
)

long_graphics_runs: list[tuple[int, int, float, int]] = []

for start, end in equal_value_runs(graphics_packet_id):
    duration_s = float(time_s[end] - time_s[start])

    if duration_s >= STALE_DURATION_S:
        stale_graphics_packet[start : end + 1] = True

        long_graphics_runs.append(
            (
                start,
                end,
                duration_s,
                int(graphics_packet_id[start]),
            )
        )


# ---------------------------------------------------------------------
# 2. Unveränderte Position trotz fahrendem Fahrzeug
# ---------------------------------------------------------------------

position_change_m = np.zeros(
    len(telemetry_rows),
    dtype=np.float64,
)

position_change_m[1:] = np.linalg.norm(
    np.diff(
        position,
        axis=0,
    ),
    axis=1,
)

same_position_as_previous = position_change_m <= 1e-6

same_progress_as_previous = np.zeros(
    len(telemetry_rows),
    dtype=bool,
)

same_progress_as_previous[1:] = np.abs(np.diff(normalized_position)) <= 1e-9

same_lap_time_as_previous = np.zeros(
    len(telemetry_rows),
    dtype=bool,
)

same_lap_time_as_previous[1:] = np.diff(lap_time_ms) == 0

unchanged_graphics_state = (
    moving
    & same_position_as_previous
    & same_progress_as_previous
    & same_lap_time_as_previous
)


stalled_graphics_state = np.zeros(
    len(telemetry_rows),
    dtype=bool,
)

long_pose_stalls: list[tuple[int, int, float]] = []

for cluster in true_clusters(unchanged_graphics_state):
    start = max(
        int(cluster[0]) - 1,
        0,
    )

    end = int(cluster[-1])

    duration_s = float(time_s[end] - time_s[start])

    if duration_s >= STALE_DURATION_S:
        stalled_graphics_state[start : end + 1] = True

        long_pose_stalls.append(
            (
                start,
                end,
                duration_s,
            )
        )


suspect_graphics_state = stale_graphics_packet | stalled_graphics_state


print("=" * 78)
print("1. Paketstatistik")
print("=" * 78)

print(
    "Samples:",
    len(telemetry_rows),
)

print(
    "Eindeutige Physikpakete:",
    np.unique(physics_packet_id).size,
)

print(
    "Eindeutige Grafikpakete:",
    np.unique(graphics_packet_id).size,
)

print(
    "Lange Wiederholungen einer Grafikpaketnummer:",
    len(long_graphics_runs),
)

for number, (
    start,
    end,
    duration_s,
    packet_id,
) in enumerate(
    long_graphics_runs,
    start=1,
):
    print(
        f"  Lauf {number}: "
        f"Samples {start} bis {end}, "
        f"Dauer {duration_s:.3f} s, "
        f"Grafikpaket {packet_id}"
    )


print()
print("=" * 78)
print("2. Eingefrorene Grafikzustände")
print("=" * 78)

print(
    "Lange Stillstände der Grafikdaten:",
    len(long_pose_stalls),
)

for number, (
    start,
    end,
    duration_s,
) in enumerate(
    long_pose_stalls,
    start=1,
):
    print(
        f"  Stillstand {number}: "
        f"Samples {start} bis {end}, "
        f"Dauer {duration_s:.3f} s, "
        f"Geschwindigkeit "
        f"{np.median(speed_kmh[start:end + 1]):.2f} km/h, "
        f"Fortschritt "
        f"{normalized_position[start]:.6f}"
    )


print()
print("=" * 78)
print("3. Einfluss auf die 122 sicheren Überschreitungen")
print("=" * 78)

print(
    "Sicher, aber außerhalb insgesamt:",
    int(np.sum(safe_but_outside)),
)

print(
    "Davon mit verdächtigem Grafikzustand:",
    int(np.sum(safe_but_outside & suspect_graphics_state)),
)

print(
    "Davon ohne verdächtigen Grafikzustand:",
    int(np.sum(safe_but_outside & ~suspect_graphics_state)),
)


print()
print("=" * 78)
print("4. Einzelne Abweichungsereignisse")
print("=" * 78)

for number, cluster in enumerate(
    true_clusters(safe_but_outside),
    start=1,
):
    start = int(cluster[0])
    end = int(cluster[-1])

    suspect_count = int(np.sum(suspect_graphics_state[cluster]))

    print()
    print(f"Ereignis {number}")
    print("-" * 40)

    print(
        "Samples:",
        start,
        "bis",
        end,
    )

    print(
        "Fortschritt:",
        f"{progress[start]:.6f}",
        "bis",
        f"{progress[end]:.6f}",
    )

    print(
        "Grafikpakete im Ereignis:",
        np.unique(graphics_packet_id[cluster]).size,
    )

    print(
        "Physikpakete im Ereignis:",
        np.unique(physics_packet_id[cluster]).size,
    )

    print(
        "Verdächtige Samples:",
        suspect_count,
        "von",
        cluster.size,
    )

    print(
        "Schlechtester Randabstand:",
        f"{np.min(margin_m[cluster]):.3f} m",
    )

    print(
        "Positionsänderung im Ereignis:",
        f"{np.sum(position_change_m[cluster]):.3f} m",
    )

