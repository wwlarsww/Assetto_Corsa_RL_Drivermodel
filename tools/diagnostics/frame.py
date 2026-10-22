"""Versioned observation frames shared by live monitoring and recordings."""

from dataclasses import MISSING, asdict, dataclass, fields
import json
from math import isfinite
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from assetto_corsa_ai.telemetry import TelemetryFrame
from assetto_corsa_ai.static_info import SimulatorStaticInfo
from assetto_corsa_ai.reward import Reward

Array = NDArray[np.float64]
SCHEMA_VERSION = 1
MAX_FRAME_BYTES = 2_000_000
ARRAY_FIELDS = {
    "policy_mean": 1,
    "observed_action_delta": 1, "previous_observed_action": 1, "observed_action": 1,
    "raw_observation": 1, "normalized_observation": 1,
    "left_boundary": 2, "right_boundary": 2, "reference_trajectory": 2,
    "predicted_trajectory": 2, "actual_future_trajectory": 2,
}


def _numpy_json(value: object) -> object:
    if isinstance(value, np.ndarray):
        return cast(Array, value).tolist()
    if isinstance(value, np.generic):
        return cast(object, value.item())
    raise TypeError(f"Unsupported diagnostics value: {type(value).__name__}")


@dataclass(frozen=True)
class DiagnosticsFrame:
    timestamp: float
    sequence: int = 0
    run_id: str | None = None
    episode_id: int | None = None
    step: int | None = None
    telemetry_timestamp: float | None = None
    telemetry: TelemetryFrame | None = None
    static_info: SimulatorStaticInfo | None = None
    position: tuple[float, float, float] | None = None
    forward: tuple[float, float, float] | None = None
    progress: float | None = None
    normalized_progress: float | None = None
    track_length: float | None = None
    lateral_offset: float | None = None
    left_clearance: float | None = None
    right_clearance: float | None = None
    policy_mean: Array | None = None
    observed_action_delta: Array | None = None
    previous_observed_action: Array | None = None
    observed_action: Array | None = None
    raw_observation: Array | None = None
    normalized_observation: Array | None = None
    observation_names: tuple[str, ...] = ()
    clip_observations: float | None = None
    left_boundary: Array | None = None
    right_boundary: Array | None = None
    reference_trajectory: Array | None = None
    predicted_trajectory: Array | None = None
    actual_future_trajectory: Array | None = None
    horizons_seconds: tuple[float, ...] = ()
    predictor_forward_mae: float | None = None
    predictor_lateral_mae: float | None = None
    mean_predicted_progress: float | None = None
    mean_predicted_lateral_offset: float | None = None
    reward: Reward | None = None
    terminated: bool = False
    truncated: bool = False
    offtrack: bool = False
    reset: bool = False
    status: str = ""
    model_version: str | None = None
    model_timestamp: float | None = None
    policy_updates: int | None = None
    reward_style_weight: float = 1.0
    reward_source: str = ""
    termination_reason: str = ""
    reset_reason: str = ""

    def to_bytes(self) -> bytes:
        data = asdict(self)
        for name in ARRAY_FIELDS:
            value = getattr(self, name)
            data[name] = None if value is None else value.tolist()
        data["schema_version"] = SCHEMA_VERSION
        encoded = json.dumps(data, separators=(",", ":"), allow_nan=True, default=_numpy_json).encode("utf-8")
        if len(encoded) > MAX_FRAME_BYTES:
            raise ValueError("Diagnostics frame exceeds the size limit.")
        return encoded

    @classmethod
    def from_bytes(cls, payload: bytes) -> "DiagnosticsFrame":
        if len(payload) > MAX_FRAME_BYTES:
            raise ValueError("Diagnostics frame exceeds the size limit.")
        decoded: object = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("Invalid diagnostics frame.")
        data = cast(dict[str, Any], decoded)
        if data.pop("schema_version", None) != SCHEMA_VERSION:
            raise ValueError("Unsupported diagnostics schema.")
        # Read earlier recordings without exposing unused training internals.
        for old, new in (("final_action", "observed_action"), ("previous_action", "previous_observed_action"),
                         ("limited_delta_action", "observed_action_delta")):
            if old in data:
                data.setdefault(new, data.pop(old))
        for obsolete in ("sampled_action", "sampled_delta_action", "rl_timestamp", "predictor_loss"):
            data.pop(obsolete, None)
        if set(data) - {f.name for f in fields(cls)}:
            raise ValueError("Unexpected diagnostics fields.")
        for field in fields(cls):
            if field.name not in data:
                if field.default is MISSING:
                    raise ValueError(f"Missing diagnostics field: {field.name}")
                default: Any = field.default
                data[field.name] = list(cast(tuple[Any, ...], default)) if isinstance(default, tuple) else default
        if not isinstance(data["timestamp"], (float, int)) or not isfinite(data["timestamp"]):
            raise ValueError("Invalid frame timestamp.")
        for name in ("sequence", "episode_id", "step", "policy_updates"):
            if data[name] is not None and (type(data[name]) is not int or data[name] < 0):
                raise ValueError(f"Invalid {name}.")
        for name in ("terminated", "truncated", "offtrack", "reset"):
            if type(data[name]) is not bool:
                raise ValueError(f"Invalid {name} flag.")
        if not isinstance(data["status"], str) or (data["run_id"] is not None and not isinstance(data["run_id"], str)):
            raise ValueError("Invalid diagnostics identifiers/status.")
        for name in ("reward_source", "termination_reason", "reset_reason"):
            if not isinstance(data[name], str):
                raise ValueError(f"Invalid {name}.")
        if data["model_version"] is not None and not isinstance(data["model_version"], str):
            raise ValueError("Invalid model version.")
        for name in ("telemetry_timestamp", "progress", "normalized_progress", "track_length",
                     "lateral_offset", "left_clearance", "right_clearance", "clip_observations",
                     "predictor_forward_mae", "predictor_lateral_mae", "mean_predicted_progress", "mean_predicted_lateral_offset",
                     "model_timestamp", "reward_style_weight"):
            if data[name] is not None and (not isinstance(data[name], (int, float)) or isinstance(data[name], bool)):
                raise ValueError(f"Invalid numeric field {name}.")
        for name in ("telemetry", "static_info", "reward"):
            if data[name] is not None and not isinstance(data[name], dict):
                raise ValueError(f"Invalid {name} model.")
        for name in ("position", "forward"):
            if data[name] is not None:
                pose = np.asarray(data[name], dtype=np.float64)
                if pose.shape != (3,):
                    raise ValueError(f"Invalid {name} vector.")
                data[name] = pose.tolist()
        if (not isinstance(data["observation_names"], list)
                or not all(isinstance(name, str) for name in cast(list[object], data["observation_names"]))):
            raise ValueError("Invalid observation feature names.")
        horizons = np.asarray(data["horizons_seconds"], dtype=np.float64)
        if horizons.ndim != 1 or not np.all(np.isfinite(horizons)) or np.any(horizons <= 0) or np.any(np.diff(horizons) <= 0):
            raise ValueError("Invalid prediction horizons.")
        for name, dimensions in ARRAY_FIELDS.items():
            if data[name] is not None:
                array = np.asarray(data[name], dtype=np.float64)
                if dimensions == 2 and array.size == 0:
                    array = array.reshape(0, 2)
                if array.size > 50_000 or array.ndim != dimensions:
                    raise ValueError(f"Invalid dimensions for {name}.")
                if dimensions == 2 and array.shape[1] != 2:
                    raise ValueError(f"{name} requires (forward, left) pairs.")
                if name in ("observed_action", "observed_action_delta", "previous_observed_action", "policy_mean"):
                    if array.shape != (2,):
                        raise ValueError(f"{name} requires two controls.")
                data[name] = array
        if data["telemetry"] is not None:
            for key, value in data["telemetry"].items():
                if isinstance(value, list):
                    data["telemetry"][key] = tuple(tuple(cast(list[Any], item)) if isinstance(item, list) else item for item in cast(list[Any], value))
            data["telemetry"] = TelemetryFrame(**data["telemetry"])
        if data["static_info"] is not None:
            data["static_info"]["tyre_radius"] = tuple(data["static_info"]["tyre_radius"])
            data["static_info"] = SimulatorStaticInfo(**data["static_info"])
        if data["reward"] is not None:
            data["reward"] = Reward(**data["reward"])
        for name in ("position", "forward", "observation_names", "horizons_seconds"):
            if data[name] is not None:
                data[name] = tuple(data[name])
        return cls(**data)


def observation_names(boundary_count: int, reference_count: int) -> tuple[str, ...]:
    names = ["speed_mps", "longitudinal_acceleration_g", "lateral_acceleration_g", "yaw_rate",
             "front_slip_ratio", "rear_slip_ratio", "front_slip_angle", "rear_slip_angle"]
    for side, count in (("left_boundary", boundary_count), ("right_boundary", boundary_count),
                        ("reference", reference_count)):
        names.extend(f"{side}_{index}_{axis}" for index in range(count) for axis in ("forward", "left"))
    return tuple(names + ["steering", "longitudinal_control"])


def observation_geometry(raw: Array, boundary_count: int, reference_count: int) -> dict[str, Array]:
    if raw.shape != (10 + 4 * boundary_count + 2 * reference_count,):
        raise ValueError("Observation layout does not match configured track features.")
    end = 8 + 2 * boundary_count
    reference = end + 2 * boundary_count
    return {"left_boundary": raw[8:end].reshape(-1, 2).copy(),
            "right_boundary": raw[end:reference].reshape(-1, 2).copy(),
            "reference_trajectory": raw[reference:-2].reshape(-1, 2).copy()}
