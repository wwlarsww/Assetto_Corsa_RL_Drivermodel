from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BC_DATASET = Path(r"data\processed\bmw_z4_gt3_monza\bc_dataset.npz")
EPSILON = 1e-6
SMOOTHING_WINDOW = 5


def resolve_trajectory_path() -> Path:
    configured = os.environ.get("TRAJECTORY_PATH")
    if configured and Path(configured).is_file():
        return Path(configured)
    candidates = sorted(
        Path("runs").rglob("trajectories.npz"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "Keine trajectories.npz gefunden und TRAJECTORY_PATH ist nicht gesetzt."
        )
    return candidates[0]


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    mask = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(mask) < 10:
        return float("nan")
    a = first[mask]
    b = second[mask]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.shape[0] < window:
        return values.copy()
    padding = window // 2
    padded = np.pad(values, ((padding, padding), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.column_stack(
        [
            np.convolve(padded[:, column], kernel, mode="valid")
            for column in range(values.shape[1])
        ]
    )


def world_to_local_xz(
    world_points: np.ndarray,
    vehicle_position: np.ndarray,
    heading_rad: float,
) -> np.ndarray:
    delta = np.asarray(world_points, dtype=np.float64) - np.asarray(
        vehicle_position, dtype=np.float64
    )[None, :]
    forward = np.asarray([np.cos(heading_rad), np.sin(heading_rad)])
    left = np.asarray([-np.sin(heading_rad), np.cos(heading_rad)])
    return np.column_stack((delta @ left, delta @ forward))


def actual_future_local(
    positions: np.ndarray,
    times: np.ndarray,
    headings: np.ndarray,
    episode_ids: np.ndarray,
    offsets_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.full((positions.shape[0], offsets_s.size, 2), np.nan)
    valid = np.zeros(positions.shape[0], dtype=bool)
    for episode_id in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode_id)
        if indices.size < 2:
            continue
        order = np.argsort(times[indices])
        indices = indices[order]
        episode_times = times[indices]
        episode_positions = positions[indices]
        unique_times, locations = np.unique(episode_times, return_index=True)
        unique_positions = episode_positions[locations]
        if unique_times.size < 2:
            continue
        for index in indices:
            query = times[index] + offsets_s
            if query[-1] > unique_times[-1]:
                continue
            future_world = np.column_stack(
                (
                    np.interp(query, unique_times, unique_positions[:, 0]),
                    np.interp(query, unique_times, unique_positions[:, 1]),
                )
            )
            result[index] = world_to_local_xz(
                future_world,
                positions[index],
                headings[index],
            )
            valid[index] = True
    return result, valid


def kinematic_speed(
    positions: np.ndarray,
    times: np.ndarray,
    episode_ids: np.ndarray,
) -> np.ndarray:
    result = np.full(positions.shape[0], np.nan)
    for episode_id in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode_id)
        if indices.size < 2:
            continue
        dt = np.diff(times[indices])
        distance = np.linalg.norm(np.diff(positions[indices], axis=0), axis=1)
        values = np.full(indices.size - 1, np.nan)
        valid = dt > 1e-4
        values[valid] = distance[valid] / dt[valid]
        result[indices[:-1]] = values
    return result


def kinematic_acceleration(
    positions: np.ndarray,
    times: np.ndarray,
    headings: np.ndarray,
    episode_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lateral = np.full(positions.shape[0], np.nan)
    forward_result = np.full(positions.shape[0], np.nan)
    valid_result = np.zeros(positions.shape[0], dtype=bool)
    for episode_id in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode_id)
        if indices.size < 9 or np.any(np.diff(times[indices]) <= 1e-4):
            continue
        smoothed = moving_average(positions[indices], SMOOTHING_WINDOW)
        velocity_world = np.gradient(smoothed, times[indices], axis=0, edge_order=2)
        acceleration_world = np.gradient(
            velocity_world, times[indices], axis=0, edge_order=2
        )
        heading = headings[indices]
        forward = np.column_stack((np.cos(heading), np.sin(heading)))
        left = np.column_stack((-np.sin(heading), np.cos(heading)))
        lateral[indices] = np.sum(acceleration_world * left, axis=1)
        forward_result[indices] = np.sum(acceleration_world * forward, axis=1)
        valid_result[indices[3:-3]] = True
    return lateral, forward_result, valid_result


def error_statistics(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
    difference = np.abs(np.asarray(first) - np.asarray(second))
    finite = difference[np.isfinite(difference)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(finite)),
        float(np.quantile(finite, 0.95)),
        float(np.max(finite)),
    )


trajectory_path = resolve_trajectory_path()
bc_dataset_path = Path(os.environ.get("BC_DATASET_PATH", str(DEFAULT_BC_DATASET)))
if not bc_dataset_path.is_file():
    raise FileNotFoundError(f"BC Datensatz nicht gefunden: {bc_dataset_path.resolve()}")

output_directory = trajectory_path.parent / "feature_geometry_diagnostics"
output_directory.mkdir(parents=True, exist_ok=True)

with np.load(trajectory_path, allow_pickle=False) as archive:
    trajectory = {name: np.asarray(archive[name]) for name in archive.files}
with np.load(bc_dataset_path, allow_pickle=False) as archive:
    feature_names = tuple(str(name) for name in archive["feature_names"].tolist())
    normalizer_mean = np.asarray(archive["normalizer_mean"], dtype=np.float64)
    normalizer_std = np.asarray(archive["normalizer_std"], dtype=np.float64)
    bc_train_normalized = np.asarray(archive["train_observations"], dtype=np.float64)

observations = np.asarray(trajectory["observations"], dtype=np.float64)
observations_normalized = np.asarray(
    trajectory["observations_normalized"], dtype=np.float64
)
if observations.shape != observations_normalized.shape:
    raise ValueError("observations und observations_normalized haben verschiedene Formen.")
if observations.shape[1] != len(feature_names):
    raise ValueError("Trajektorie und BC Datensatz verwenden verschiedene Featurezahlen.")

scale = np.maximum(normalizer_std, EPSILON)
raw = observations_normalized * scale[None, :] + normalizer_mean[None, :]
renormalized = (raw - normalizer_mean[None, :]) / scale[None, :]
normalization_error = float(np.max(np.abs(renormalized - observations_normalized)))
alias_error = float(np.max(np.abs(observations - observations_normalized)))

positions = np.asarray(trajectory["vehicle_world_xy_m"], dtype=np.float64)
headings = np.asarray(trajectory["vehicle_heading_rad"], dtype=np.float64)
times = np.asarray(trajectory["simulation_time_s"], dtype=np.float64)
episode_ids = np.asarray(trajectory["episode_ids"], dtype=np.int64)
offsets_s = np.asarray(trajectory["reference_time_offsets_s"], dtype=np.float64)

bezier_local = np.asarray(
    trajectory["bezier_prediction_local_xz_m"], dtype=np.float64
)
reference_local = np.asarray(
    trajectory["reference_preview_local_xz_m"], dtype=np.float64
)
left_local = np.asarray(
    trajectory["left_boundary_preview_local_xz_m"], dtype=np.float64
)
right_local = np.asarray(
    trajectory["right_boundary_preview_local_xz_m"], dtype=np.float64
)

raw_left = raw[:, 7:21].reshape(-1, 7, 2)
raw_right = raw[:, 21:35].reshape(-1, 7, 2)
raw_reference = raw[:, 35:75].reshape(-1, 20, 2)
left_error = error_statistics(raw_left, left_local)
right_error = error_statistics(raw_right, right_local)
reference_error = error_statistics(raw_reference, reference_local)

speed_live = raw[:, 0]
acceleration_x_live = raw[:, 1]
acceleration_z_live = raw[:, 2]

speed_from_position = kinematic_speed(positions, times, episode_ids)
speed_mask = (
    np.isfinite(speed_from_position)
    & np.isfinite(speed_live)
    & (speed_live > 2.0)
    & (speed_from_position < 120.0)
)
speed_mae = (
    float(np.mean(np.abs(speed_live[speed_mask] - speed_from_position[speed_mask])))
    if np.any(speed_mask)
    else float("nan")
)
speed_correlation = safe_correlation(
    speed_live[speed_mask], speed_from_position[speed_mask]
)

acceleration_left, acceleration_forward, acceleration_valid = kinematic_acceleration(
    positions, times, headings, episode_ids
)
acceleration_mask = (
    acceleration_valid
    & np.isfinite(acceleration_x_live)
    & np.isfinite(acceleration_z_live)
    & np.isfinite(acceleration_left)
    & np.isfinite(acceleration_forward)
    & (speed_live > 5.0)
    & (np.abs(acceleration_x_live) < 50.0)
    & (np.abs(acceleration_z_live) < 50.0)
    & (np.abs(acceleration_left) < 50.0)
    & (np.abs(acceleration_forward) < 50.0)
)
acceleration_correlations = {
    "live_x_vs_kinematic_left": safe_correlation(
        acceleration_x_live[acceleration_mask], acceleration_left[acceleration_mask]
    ),
    "live_x_vs_kinematic_forward": safe_correlation(
        acceleration_x_live[acceleration_mask], acceleration_forward[acceleration_mask]
    ),
    "live_z_vs_kinematic_left": safe_correlation(
        acceleration_z_live[acceleration_mask], acceleration_left[acceleration_mask]
    ),
    "live_z_vs_kinematic_forward": safe_correlation(
        acceleration_z_live[acceleration_mask], acceleration_forward[acceleration_mask]
    ),
}

actual_future, future_valid = actual_future_local(
    positions, times, headings, episode_ids, offsets_s
)
future_error = np.linalg.norm(bezier_local - actual_future, axis=2)
ade = np.full(positions.shape[0], np.nan)
fde = np.full(positions.shape[0], np.nan)
if np.any(future_valid):
    ade[future_valid] = np.mean(future_error[future_valid], axis=1)
    fde[future_valid] = future_error[future_valid, -1]

bezier_first_distance = np.linalg.norm(bezier_local[:, 0], axis=1)
reference_first_distance = np.linalg.norm(reference_local[:, 0], axis=1)

live_abs_z_p95 = np.quantile(np.abs(observations_normalized), 0.95, axis=0)
live_abs_z_max = np.max(np.abs(observations_normalized), axis=0)
live_fraction_above_5 = np.mean(np.abs(observations_normalized) > 5.0, axis=0)
bc_abs_z_p95 = np.quantile(np.abs(bc_train_normalized), 0.95, axis=0)

summary_path = output_directory / "feature_summary.csv"
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "index",
        "feature",
        "normalizer_mean",
        "normalizer_std",
        "live_raw_min",
        "live_raw_median",
        "live_raw_max",
        "live_abs_z_p95",
        "live_abs_z_max",
        "live_fraction_abs_z_above_5",
        "bc_train_abs_z_p95",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for index, name in enumerate(feature_names):
        writer.writerow(
            {
                "index": index,
                "feature": name,
                "normalizer_mean": normalizer_mean[index],
                "normalizer_std": normalizer_std[index],
                "live_raw_min": np.min(raw[:, index]),
                "live_raw_median": np.median(raw[:, index]),
                "live_raw_max": np.max(raw[:, index]),
                "live_abs_z_p95": live_abs_z_p95[index],
                "live_abs_z_max": live_abs_z_max[index],
                "live_fraction_abs_z_above_5": live_fraction_above_5[index],
                "bc_train_abs_z_p95": bc_abs_z_p95[index],
            }
        )

# Auffällige Normalisierung
feature_order = np.argsort(live_abs_z_p95)[-20:][::-1]
figure, axis = plt.subplots(figsize=(12, 8))
y = np.arange(feature_order.size)
axis.barh(y, live_abs_z_p95[feature_order])
axis.set_yticks(y, [f"{i}: {feature_names[i]}" for i in feature_order])
axis.invert_yaxis()
axis.set_xlabel("95 Prozent Quantil von |Z Score|")
axis.set_title("Live Features mit den größten normalisierten Beträgen")
axis.grid(True, axis="x")
figure.tight_layout()
normalization_plot = output_directory / "normalization_top_features.png"
figure.savefig(normalization_plot, dpi=180)
plt.close(figure)

# Geschwindigkeit
figure, axis = plt.subplots(figsize=(8, 8))
axis.scatter(speed_from_position[speed_mask], speed_live[speed_mask], s=10, alpha=0.5)
if np.any(speed_mask):
    upper = float(max(np.max(speed_from_position[speed_mask]), np.max(speed_live[speed_mask])))
    axis.plot([0.0, upper], [0.0, upper], linestyle="--")
axis.set_xlabel("Aus Weltposition abgeleitete Geschwindigkeit [m/s]")
axis.set_ylabel("Live Feature vehicle.speed_mps [m/s]")
axis.set_title("Plausibilitätsprüfung der absoluten Geschwindigkeit")
axis.grid(True)
figure.tight_layout()
speed_plot = output_directory / "speed_consistency.png"
figure.savefig(speed_plot, dpi=180)
plt.close(figure)

# Beschleunigungsachsen
figure, axis = plt.subplots(figsize=(9, 8))
axis.scatter(
    acceleration_left[acceleration_mask],
    acceleration_x_live[acceleration_mask],
    s=10,
    alpha=0.5,
    label="Live x gegen kinematisch links",
)
axis.scatter(
    acceleration_forward[acceleration_mask],
    acceleration_z_live[acceleration_mask],
    s=10,
    alpha=0.5,
    label="Live z gegen kinematisch vorwärts",
)
axis.set_xlabel("Kinematisch abgeleitete Beschleunigung [m/s²]")
axis.set_ylabel("Live Beschleunigungsfeature [m/s²]")
axis.set_title("Prüfung der Fahrzeugachsen für Beschleunigungen")
axis.grid(True)
axis.legend()
figure.tight_layout()
acceleration_plot = output_directory / "acceleration_axis_consistency.png"
figure.savefig(acceleration_plot, dpi=180)
plt.close(figure)

# Lokale Geometrie
valid_indices = np.flatnonzero(future_valid)
if valid_indices.size:
    representative_index = int(
        valid_indices[
            np.nanargmin(
                np.abs(ade[valid_indices] - np.nanmedian(ade[valid_indices]))
            )
        ]
    )
else:
    representative_index = int(np.argmax(speed_live))

figure, axis = plt.subplots(figsize=(9, 10))
prediction_with_origin = np.vstack(([[0.0, 0.0]], bezier_local[representative_index]))
axis.plot(
    prediction_with_origin[:, 0],
    prediction_with_origin[:, 1],
    marker="o",
    markersize=3,
    label="Bézier Vorhersage plus Fahrzeugursprung",
)
axis.plot(
    reference_local[representative_index, :, 0],
    reference_local[representative_index, :, 1],
    marker=".",
    label="AI Referenzvorschau",
)
axis.plot(
    left_local[representative_index, :, 0],
    left_local[representative_index, :, 1],
    marker=".",
    label="Linke Grenze",
)
axis.plot(
    right_local[representative_index, :, 0],
    right_local[representative_index, :, 1],
    marker=".",
    label="Rechte Grenze",
)
if future_valid[representative_index]:
    axis.plot(
        actual_future[representative_index, :, 0],
        actual_future[representative_index, :, 1],
        marker=".",
        label="Tatsächlich gefahrene Zukunft",
    )
axis.scatter([0.0], [0.0], s=100, marker="X", label="Fahrzeugursprung")
axis.set_xlabel("Lokales x: links positiv [m]")
axis.set_ylabel("Lokales z: vorwärts positiv [m]")
axis.set_title(f"Lokale Geometrie bei Sample {representative_index}")
axis.axis("equal")
axis.grid(True)
axis.legend()
figure.tight_layout()
geometry_plot = output_directory / "local_geometry_and_bezier.png"
figure.savefig(geometry_plot, dpi=180)
plt.close(figure)

print("=" * 78)
print("1. Dateien und Formen")
print("=" * 78)
print("Trajektorie:", trajectory_path.resolve())
print("BC Datensatz:", bc_dataset_path.resolve())
print("Samples:", observations.shape[0])
print("Features:", observations.shape[1])
print("Episoden:", np.unique(episode_ids).size)
print("Lokale Achsen:", str(trajectory["local_trajectory_axes"].reshape(-1)[0]))

print("\n" + "=" * 78)
print("2. Normalisierung")
print("=" * 78)
print("Maximaler Roundtrip Fehler:", f"{normalization_error:.9e}")
print("observations gegen observations_normalized:", f"{alias_error:.9e}")
print("Features mit mindestens einem |Z| > 5:", int(np.count_nonzero(live_abs_z_max > 5.0)))
print("Die 10 auffälligsten Features:")
for index in feature_order[:10]:
    print(
        f"  {index:2d} {feature_names[index]}: "
        f"Live p95={live_abs_z_p95[index]:.3f}, "
        f"BC p95={bc_abs_z_p95[index]:.3f}, "
        f"Live max={live_abs_z_max[index]:.3f}"
    )

print("\n" + "=" * 78)
print("3. Beobachtungsgeometrie gegen separat gespeicherte Geometrie")
print("=" * 78)
print("Linke Grenze Median, p95, Max:", f"{left_error[0]:.6f}, {left_error[1]:.6f}, {left_error[2]:.6f} m")
print("Rechte Grenze Median, p95, Max:", f"{right_error[0]:.6f}, {right_error[1]:.6f}, {right_error[2]:.6f} m")
print("Referenzlinie Median, p95, Max:", f"{reference_error[0]:.6f}, {reference_error[1]:.6f}, {reference_error[2]:.6f} m")
print("Frame Roundtrip Maximum:", f"{np.nanmax(trajectory['frame_roundtrip_error_m']):.9e} m")

print("\n" + "=" * 78)
print("4. Ursprung und Bézier Linie")
print("=" * 78)
print("Erster Referenzpunkt Abstand Median:", f"{np.median(reference_first_distance):.3f} m")
print("Erster Bézier Punkt Abstand Median:", f"{np.median(bezier_first_distance):.3f} m")
print("Erster Bézier Punkt ist immer exakt Ursprung:", bool(np.all(bezier_first_distance <= 1e-6)))
print("Der gespeicherte erste Punkt gehört zu t=0.25 s; der Plot ergänzt [0, 0] nur für die Darstellung.")
if np.any(future_valid):
    print("Bézier ADE Median, Mittel, p95:", f"{np.nanmedian(ade):.3f}, {np.nanmean(ade):.3f}, {np.nanquantile(ade, 0.95):.3f} m")
    print("Bézier FDE Median, Mittel, p95:", f"{np.nanmedian(fde):.3f}, {np.nanmean(fde):.3f}, {np.nanquantile(fde, 0.95):.3f} m")
    print("Samples mit vollständiger 5 Sekunden Zukunft:", int(np.count_nonzero(future_valid)))

print("\n" + "=" * 78)
print("5. Geschwindigkeit")
print("=" * 78)
print("Live Minimum, Median, Maximum:", f"{np.min(speed_live):.3f}, {np.median(speed_live):.3f}, {np.max(speed_live):.3f} m/s")
print("Negative Live Geschwindigkeiten:", int(np.count_nonzero(speed_live < -1e-6)))
print("Korrelation gegen Weltpositionsableitung:", f"{speed_correlation:.3f}")
print("Mittlerer absoluter Unterschied:", f"{speed_mae:.3f} m/s")

print("\n" + "=" * 78)
print("6. Beschleunigungsachsen")
print("=" * 78)
print("Vergleichssamples:", int(np.count_nonzero(acceleration_mask)))
for name, value in acceleration_correlations.items():
    print(f"{name}: {value:.3f}")
print("Erwartung: x gegen links und z gegen vorwärts sind positiv und größer als die Kreuzkorrelationen.")
print("Die Ableitung aus Grafikpositionen ist nur ein Plausibilitätstest und kein exakter Sensorvergleich.")

print("\n" + "=" * 78)
print("7. Ausgaben")
print("=" * 78)
print("Feature Tabelle:", summary_path)
print("Normalisierung:", normalization_plot)
print("Geschwindigkeit:", speed_plot)
print("Beschleunigungen:", acceleration_plot)
print("Geometrie und Bézier:", geometry_plot)
print("Ausgabeordner:", output_directory.resolve())
