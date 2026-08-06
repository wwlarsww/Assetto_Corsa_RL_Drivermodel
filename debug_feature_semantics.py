from __future__ import annotations

import os
from pathlib import Path

import numpy as np


DEFAULT_BC_DATASET = Path(
    r"data\processed\bmw_z4_gt3_monza\bc_dataset.npz"
)
EPSILON = 1e-6
SMOOTHING_WINDOW = 7


def resolve_trajectory_path() -> Path:
    configured = os.environ.get("TRAJECTORY_PATH")
    if configured and Path(configured).is_file():
        return Path(configured)

    candidates = sorted(
        Path("runs").rglob("trajectories.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Keine trajectories.npz gefunden.")
    return candidates[0]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values.copy()

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    mask = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(mask) < 10:
        return float("nan")

    first = first[mask]
    second = second[mask]
    if np.std(first) < 1e-9 or np.std(second) < 1e-9:
        return float("nan")

    return float(np.corrcoef(first, second)[0, 1])


trajectory_path = resolve_trajectory_path()
bc_dataset_path = Path(
    os.environ.get("BC_DATASET_PATH", str(DEFAULT_BC_DATASET))
)

with np.load(trajectory_path, allow_pickle=False) as archive:
    observations_normalized = np.asarray(
        archive["observations_normalized"],
        dtype=np.float64,
    )
    times = np.asarray(archive["simulation_time_s"], dtype=np.float64)
    episode_ids = np.asarray(archive["episode_ids"], dtype=np.int64)
    bezier = np.asarray(
        archive["bezier_prediction_local_xz_m"],
        dtype=np.float64,
    )
    reference = np.asarray(
        archive["reference_preview_local_xz_m"],
        dtype=np.float64,
    )

with np.load(bc_dataset_path, allow_pickle=False) as archive:
    feature_names = tuple(str(name) for name in archive["feature_names"].tolist())
    mean = np.asarray(archive["normalizer_mean"], dtype=np.float64)
    std = np.asarray(archive["normalizer_std"], dtype=np.float64)

raw = observations_normalized * np.maximum(std, EPSILON)[None, :] + mean[None, :]

speed = raw[:, 0]
acceleration_x = raw[:, 1]
acceleration_z = raw[:, 2]

speed_derivative = np.full(speed.shape, np.nan, dtype=np.float64)
valid_derivative = np.zeros(speed.shape, dtype=bool)

for episode_id in np.unique(episode_ids):
    indices = np.flatnonzero(episode_ids == episode_id)
    if indices.size < 9:
        continue

    episode_times = times[indices]
    if np.any(np.diff(episode_times) <= 1e-5):
        continue

    smoothed_speed = moving_average(speed[indices], SMOOTHING_WINDOW)
    derivative = np.gradient(
        smoothed_speed,
        episode_times,
        edge_order=2,
    )

    speed_derivative[indices] = derivative
    valid_derivative[indices[3:-3]] = True

mask = (
    valid_derivative
    & np.isfinite(acceleration_z)
    & np.isfinite(acceleration_x)
    & (speed > 2.0)
    & (np.abs(speed_derivative) < 30.0)
    & (np.abs(acceleration_z) < 30.0)
    & (np.abs(acceleration_x) < 30.0)
)

print("=" * 78)
print("1. Longitudinale Beschleunigung gegen d|v|/dt")
print("=" * 78)
print("Vergleichssamples:", int(np.count_nonzero(mask)))
print(
    "corr(acceleration_z, d speed / dt):",
    f"{correlation(acceleration_z[mask], speed_derivative[mask]):.3f}",
)
print(
    "corr(-acceleration_z, d speed / dt):",
    f"{correlation(-acceleration_z[mask], speed_derivative[mask]):.3f}",
)
print(
    "corr(acceleration_x, d speed / dt):",
    f"{correlation(acceleration_x[mask], speed_derivative[mask]):.3f}",
)
if np.any(mask):
    print(
        "Median |acceleration_z - d speed/dt|:",
        f"{np.median(np.abs(acceleration_z[mask] - speed_derivative[mask])):.3f} m/s²",
    )

print()
print("=" * 78)
print("2. Kritische Rohfeatures und Normalisierung")
print("=" * 78)

critical_indices = [0, 1, 2, 3, 4, 5, 6, 75, 76]
for index in critical_indices:
    if index >= raw.shape[1]:
        continue

    values = raw[:, index]
    z_values = observations_normalized[:, index]
    print()
    print(f"{index:2d} {feature_names[index]}")
    print(
        "  BC mean/std:",
        f"{mean[index]:.6g}",
        "/",
        f"{std[index]:.6g}",
    )
    print(
        "  Live raw min/median/max:",
        f"{np.min(values):.6g}",
        "/",
        f"{np.median(values):.6g}",
        "/",
        f"{np.max(values):.6g}",
    )
    print(
        "  Live Z p95/max:",
        f"{np.quantile(np.abs(z_values), 0.95):.3f}",
        "/",
        f"{np.max(np.abs(z_values)):.3f}",
    )

print()
print("=" * 78)
print("3. Erster Vorschaupunkt")
print("=" * 78)

bezier_first = bezier[:, 0, :]
reference_first = reference[:, 0, :]
expected_distance = np.maximum(speed, 0.0) * 0.25

bezier_distance = np.linalg.norm(bezier_first, axis=1)
reference_distance = np.linalg.norm(reference_first, axis=1)

print(
    "Erwartete Strecke bei 0,25 s aus Live Geschwindigkeit, Median:",
    f"{np.median(expected_distance):.3f} m",
)
print(
    "Erster Referenzpunkt Abstand, Median:",
    f"{np.median(reference_distance):.3f} m",
)
print(
    "Erster Bézier Punkt Abstand, Median:",
    f"{np.median(bezier_distance):.3f} m",
)
print(
    "Erster Bézier Punkt x Median:",
    f"{np.median(bezier_first[:, 0]):.3f} m",
)
print(
    "Erster Bézier Punkt z Median:",
    f"{np.median(bezier_first[:, 1]):.3f} m",
)
print(
    "Anteil erster Bézier Punkte mit z < 0:",
    f"{100.0 * np.mean(bezier_first[:, 1] < 0.0):.2f} Prozent",
)
print(
    "Median Verhältnis Bézier Abstand / erwartete Strecke:",
    f"{np.median(bezier_distance / np.maximum(expected_distance, 0.1)):.3f}",
)

print()
print("Trajektorie:", trajectory_path.resolve())
print("BC Datensatz:", bc_dataset_path.resolve())
