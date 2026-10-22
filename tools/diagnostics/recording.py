"""Portable, pickle-free lap storage and offline trajectory analysis."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.diagnostics.frame import Array, DiagnosticsFrame, MAX_FRAME_BYTES, SCHEMA_VERSION
from assetto_corsa_ai.predictor import horizontal_vehicle_axes

MAX_LAP_FRAMES = 36_000


def save_lap(path: Path, frames: tuple[DiagnosticsFrame, ...]) -> None:
    if not frames or len(frames) > MAX_LAP_FRAMES:
        raise ValueError("Lap frame count is outside the supported limits.")
    encoded = [frame.to_bytes() for frame in frames]
    offsets = np.cumsum([0] + [len(item) for item in encoded], dtype=np.int64)
    # Exclusive creation protects existing laps; no object arrays or pickle.
    with path.open("xb") as stream:
        np.savez_compressed(stream, payload=np.frombuffer(b"".join(encoded), dtype=np.uint8), offsets=offsets)


def load_lap(path: Path) -> tuple[DiagnosticsFrame, ...]:
    with np.load(path, allow_pickle=False) as archive:
        payload = np.asarray(archive["payload"], dtype=np.uint8)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
    if (payload.ndim != 1 or offsets.ndim != 1 or not 2 <= len(offsets) <= MAX_LAP_FRAMES + 1
            or offsets[0] != 0 or offsets[-1] != len(payload)
            or np.any(np.diff(offsets) <= 0) or np.any(np.diff(offsets) > MAX_FRAME_BYTES)):
        raise ValueError("Invalid lap archive layout.")
    frames = tuple(DiagnosticsFrame.from_bytes(payload[int(start):int(end)].tobytes())
                   for start, end in zip(offsets[:-1], offsets[1:]))
    if any(b.timestamp <= a.timestamp for a, b in zip(frames, frames[1:])):
        raise ValueError("Lap timestamps must be strictly increasing.")
    return frames


@dataclass(frozen=True)
class Recording:
    directory: Path
    metadata: dict[str, Any]
    laps: tuple[Path, ...]

    @classmethod
    def open(cls, directory: Path) -> "Recording":
        metadata: dict[str, Any] = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported recording schema.")
        return cls(directory, metadata, tuple(sorted(directory.glob("lap_*.npz"))))


def future_trajectory(frames: tuple[DiagnosticsFrame, ...], index: int, times: Array | None = None) -> Array | None:
    """Interpolate future world poses into this prediction's original vehicle frame.

    Missing horizons remain NaN. Never interpolate over a reset or a data gap.
    """
    origin = frames[index]
    if origin.position is None or origin.forward is None or not origin.horizons_seconds:
        return None
    try:
        forward, left = horizontal_vehicle_axes(np.asarray(origin.forward, dtype=np.float64))
    except ValueError:
        return None
    position = np.asarray(origin.position, dtype=np.float64)[[0, 2]]
    times = np.asarray([frame.timestamp for frame in frames]) if times is None else times
    result = np.full((len(origin.horizons_seconds), 2), np.nan)
    for horizon_index, horizon in enumerate(origin.horizons_seconds):
        end = int(np.searchsorted(times, origin.timestamp + horizon))
        if end <= index or end >= len(frames):
            continue
        a, b = frames[end - 1], frames[end]
        if (a.position is None or b.position is None or b.timestamp - a.timestamp > 0.25
                or any(f.reset or f.terminated or f.truncated or f.episode_id != origin.episode_id
                       or f.run_id != origin.run_id for f in frames[index:end + 1])):
            continue
        fraction = (origin.timestamp + horizon - a.timestamp) / (b.timestamp - a.timestamp)
        target = ((1 - fraction) * np.asarray(a.position)[[0, 2]]
                  + fraction * np.asarray(b.position)[[0, 2]])
        displacement = target - position
        result[horizon_index] = (float(displacement @ forward), float(displacement @ left))
    return result


def with_actual_future(frames: tuple[DiagnosticsFrame, ...], index: int, times: Array | None = None) -> DiagnosticsFrame:
    frame = frames[index]
    return replace(frame, actual_future_trajectory=future_trajectory(frames, index, times))


def prediction_errors(frame: DiagnosticsFrame) -> Array | None:
    predicted, actual = frame.predicted_trajectory, frame.actual_future_trajectory
    if predicted is None or actual is None or predicted.shape != actual.shape:
        return None
    return np.abs(predicted - actual)


def lap_statistics(frames: tuple[DiagnosticsFrame, ...]) -> dict[str, float | int | None]:
    times = np.asarray([frame.timestamp for frame in frames])
    replay_errors = [prediction_errors(with_actual_future(frames, index, times)) for index in range(len(frames))]
    def values(name: str) -> Array:
        items: list[float] = []
        for frame in frames:
            if name == "speed":
                value = frame.telemetry.speed_kmh if frame.telemetry else None
            elif name == "reward":
                value = frame.reward.total if frame.reward else None
            else:
                value = getattr(frame, name)
            if value is not None and np.isfinite(value):
                items.append(float(value))
        return np.asarray(items, dtype=np.float64)

    result: dict[str, float | int | None] = {"lap_time_s": frames[-1].timestamp - frames[0].timestamp if frames else 0.0}
    for field, label in (("speed", "speed_kmh"), ("lateral_offset", "absolute_lateral_offset_m"), ("reward", "reward")):
        array = values(field)
        if field == "lateral_offset":
            array = np.abs(array)
        result[f"mean_{label}"] = float(np.mean(array)) if array.size else None
        result[f"max_{label}"] = float(np.max(array)) if array.size else None
    rewards = values("reward")
    result["total_reward"] = float(np.sum(rewards)) if rewards.size else None
    for axis, column in (("forward", 0), ("lateral", 1)):
        errors: list[float] = []
        for error in replay_errors:
            if error is not None:
                errors.extend(error[:, column][np.isfinite(error[:, column])].tolist())
        result[f"mean_predictor_{axis}_mae_m"] = float(np.mean(errors)) if errors else None
    deltas = [abs(float(f.observed_action_delta[0])) for f in frames if f.observed_action_delta is not None]
    result["mean_absolute_steering_delta"] = float(np.mean(deltas)) if deltas else None
    controls = [float(f.observed_action[1]) for f in frames if f.observed_action is not None and abs(f.observed_action[1]) > 0.05]
    result["throttle_brake_switches"] = sum(a * b < 0 for a, b in zip(controls, controls[1:]))
    result["offtrack_events"] = sum(f.offtrack and (i == 0 or not frames[i - 1].offtrack) for i, f in enumerate(frames))
    return result
