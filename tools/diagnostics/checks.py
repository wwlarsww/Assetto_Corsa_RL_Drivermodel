"""Passive plausibility checks; no simulator or GUI dependencies."""

from dataclasses import dataclass

import numpy as np

from tools.diagnostics.buffer import FrameBuffer
from tools.diagnostics.frame import DiagnosticsFrame


@dataclass(frozen=True)
class HealthCheck:
    name: str
    level: str
    message: str


def reward_is_consistent(frame: DiagnosticsFrame, tolerance: float = 1e-5) -> bool | None:
    r = frame.reward
    if r is None:
        return None
    expected = r.performance_step + r.performance_prediction + frame.reward_style_weight * (r.style_step + r.style_prediction) + r.termination_penalty
    return bool(np.isfinite(r.total) and np.isclose(expected, r.total, atol=tolerance, rtol=tolerance))


class DiagnosticsChecks:
    def __init__(self) -> None:
        self.history = FrameBuffer(capacity=100, seconds=5.0)

    def evaluate(self, frame: DiagnosticsFrame, now: float | None = None, replay: bool = False) -> tuple[HealthCheck, ...]:
        self.history.append(frame)
        now = frame.timestamp if now is None else now
        results: list[HealthCheck] = []

        def add(name: str, valid: bool, success: str, warning: str) -> None:
            results.append(HealthCheck(name, "ok" if valid else "warning", success if valid else warning))

        if not replay:
            add("telemetry", frame.telemetry_timestamp is not None and now - frame.telemetry_timestamp < 1.0,
                "Telemetry live", "No Assetto Corsa telemetry available / telemetry stale")
            results.append(HealthCheck("model", "info", "No model checkpoint loaded" if frame.model_version is None else
                                       f"Checkpoint inference: update {frame.policy_updates}, step {frame.step}"))
            if frame.reset:
                results.append(HealthCheck("reset", "info", f"Reset inferred: {frame.reset_reason}"))
        for name, array in (("raw observation", frame.raw_observation), ("normalized observation", frame.normalized_observation)):
            if array is not None:
                add(name, bool(np.all(np.isfinite(array))), f"{name.capitalize()} finite", f"{name} contains NaN/Inf")
                if np.any(np.abs(array) > 10_000):
                    results.append(HealthCheck(name, "warning", f"Unusually large {name} values"))
        if frame.normalized_observation is not None and frame.clip_observations is not None:
            add("clipping", not bool(np.any(np.abs(frame.normalized_observation[:-2]) >= 0.98 * frame.clip_observations)),
                "Observation below clipping limit", "Normalized observation near VecNormalize clipping limit")
        if frame.progress is not None:
            fields = (frame.progress, frame.lateral_offset, frame.left_clearance, frame.right_clearance)
            add("projection", all(value is not None and np.isfinite(value) for value in fields)
                and frame.normalized_progress is not None and 0 <= frame.normalized_progress <= 1,
                "Track projection valid", "Track projection invalid")
        if frame.offtrack:
            results.append(HealthCheck("offtrack", "warning", "Vehicle offtrack"))
        if frame.observed_action_delta is not None:
            add("action delta", bool(np.all(np.abs(frame.observed_action_delta) <= 0.25)),
                "Action deltas bounded", "Large action deltas")
        history = self.history.frames()
        longitudinal = [float(f.observed_action[1]) for f in history if f.observed_action is not None and abs(f.observed_action[1]) > 0.05]
        switches = sum(a * b < 0 for a, b in zip(longitudinal, longitudinal[1:]))
        if switches >= 8:
            results.append(HealthCheck("pedal switching", "warning", f"{switches} throttle/brake switches in 5 s"))
        steering = [float(f.observed_action_delta[0]) for f in history if f.observed_action_delta is not None and abs(f.observed_action_delta[0]) >= 0.03]
        if sum(a * b < 0 for a, b in zip(steering, steering[1:])) >= 8:
            results.append(HealthCheck("steering", "warning", "High steering oscillation"))
        for axis in ("forward", "lateral"):
            errors = [getattr(f, f"predictor_{axis}_mae") for f in history]
            valid_errors = [float(v) for v in errors if v is not None and np.isfinite(v)]
            if len(valid_errors) >= 3 and valid_errors[-1] > max(0.5, valid_errors[0] * 1.5):
                results.append(HealthCheck("predictor", "warning", f"Predictor {axis} error rising: {valid_errors[-1]:.3f} m"))
        consistent = reward_is_consistent(frame)
        if consistent is not None:
            add("reward", consistent, "Reward decomposition consistent", "Reward components do not match total")
        return tuple(results)
