"""PPO-free staged validation of the write-capable live simulator boundary."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from assetto_corsa_rl.envs.environment import AssettoCorsaEnv

REFERENCE_SPEED_RESET_METHOD_VARIANT = "csp_hotlap_reference_speed_v1"


@dataclass(frozen=True, slots=True)
class LiveValidationIssue:
    """One failed invariant tied to a concrete validation stage."""

    stage: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ResetValidationSample:
    """Observed simulator state after one acknowledged reset."""

    index: int
    latency_s: float
    progress_m: float
    lateral_offset_m: float
    heading_error_rad: float
    speed_mps: float
    position_x_m: float
    position_y_m: float
    vehicle_heading_rad: float


@dataclass(frozen=True, slots=True)
class InputEchoSample:
    """Commanded action and corresponding raw AC input channels."""

    name: str
    commanded_pedal: float
    commanded_steering: float
    simulator_gas: float
    simulator_brake: float
    simulator_steer_angle: float


@dataclass(frozen=True, slots=True)
class TransportValidationSample:
    """One frame-synchronised transition used for the soak check."""

    index: int
    latency_s: float
    physics_packet_id: int
    graphics_packet_id: int
    slip_sequence: int
    collision_sequence: int
    speed_mps: float
    progress_m: float
    termination_reason: str | None


@dataclass(frozen=True, slots=True)
class DriveOffValidation:
    """Observed forward-gear engagement and motion after a reset."""

    steps_completed: int
    required_forward_gear: int
    first_observed_gear_raw: int | None
    maximum_observed_gear_raw: int | None
    maximum_observed_forward_gear: int | None
    maximum_rpms: int
    maximum_speed_mps: float
    forward_progress_m: float
    maximum_abs_slip_ratio: float
    termination_reason: str | None


@dataclass(frozen=True, slots=True)
class LiveValidationSummary:
    """Complete acceptance report for a single attached or launched session."""

    session_origin: str
    controller_index: int | None
    game_stop_requested: bool
    game_stopped: bool | None
    reset_method_variant: str
    reset_count: int
    reset_position_spread_m: float | None
    reset_progress_spread_m: float | None
    reset_heading_spread_rad: float | None
    maximum_abs_reset_lateral_offset_m: float | None
    maximum_abs_reset_heading_error_rad: float | None
    maximum_reset_speed_mps: float | None
    maximum_reset_latency_s: float | None
    soak_steps_requested: int
    soak_steps_completed: int
    maximum_step_latency_s: float | None
    mean_step_latency_s: float | None
    p95_step_latency_s: float | None
    unique_physics_packets: int
    unique_slip_packets: int
    termination_counts: dict[str, int]
    input_echo: tuple[InputEchoSample, ...]
    drive_off: DriveOffValidation | None
    issues: tuple[LiveValidationIssue, ...]

    @property
    def passed(self) -> bool:
        """Whether every required live invariant passed."""

        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable report."""

        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def _wrapped_difference(first: float, second: float, period: float) -> float:
    return abs((second - first + 0.5 * period) % period - 0.5 * period)


def _wrapped_delta(first: float, second: float, period: float) -> float:
    return (second - first + 0.5 * period) % period - 0.5 * period


def _reset_sample(
    index: int,
    latency_s: float,
    info: dict[str, Any],
) -> ResetValidationSample:
    telemetry = info["telemetry"]
    position = info["position_xy_m"]
    return ResetValidationSample(
        index=index,
        latency_s=latency_s,
        progress_m=float(telemetry["progress_m"]),
        lateral_offset_m=float(telemetry["lateral_offset_m"]),
        heading_error_rad=float(telemetry["heading_error_rad"]),
        speed_mps=float(telemetry["speed_mps"]),
        position_x_m=float(position[0]),
        position_y_m=float(position[1]),
        vehicle_heading_rad=float(info["vehicle_heading_rad"]),
    )


def _hold_action(
    environment: AssettoCorsaEnv,
    *,
    name: str,
    pedal: float,
    steering: float,
    hold_steps: int,
) -> InputEchoSample:
    environment.reset()
    latest_backend: dict[str, Any] = {}
    for _ in range(hold_steps):
        _, _, terminated, truncated, info = environment.step(
            np.asarray([pedal, steering], dtype=np.float32)
        )
        latest_backend = info["backend"]
        if terminated or truncated:
            break
    return InputEchoSample(
        name=name,
        commanded_pedal=pedal,
        commanded_steering=steering,
        simulator_gas=float(latest_backend["simulator_gas"]),
        simulator_brake=float(latest_backend["simulator_brake"]),
        simulator_steer_angle=float(latest_backend["simulator_steer_angle"]),
    )


def validate_live_environment(  # noqa: PLR0912, PLR0915 - staged acceptance report
    environment: AssettoCorsaEnv,
    *,
    session_origin: str,
    reset_method_variant: str,
    controller_index: int | None = None,
    reset_count: int = 8,
    soak_steps: int = 250,
    input_hold_steps: int = 4,
    reset_position_tolerance_m: float = 0.5,
    reset_progress_tolerance_m: float = 0.5,
    reset_heading_tolerance_rad: float = 0.05,
    maximum_reset_lateral_offset_m: float = 5.0,
    maximum_reset_heading_error_rad: float = 0.15,
    maximum_reset_speed_mps: float = 0.5,
    reference_start_speed_tolerance_mps: float = 2.0,
    control_frequency_hz: float = 20.0,
    expected_simulator_steering_sign: float = 1.0,
    drive_off_steps: int = 40,
    drive_off_throttle: float = 0.35,
    minimum_drive_off_speed_mps: float = 1.0,
    minimum_drive_off_progress_m: float = 1.0,
    minimum_forward_gear: int = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[
    LiveValidationSummary,
    tuple[ResetValidationSample, ...],
    tuple[TransportValidationSample, ...],
]:
    """Validate resets, real input echo and sustained transport without PPO."""

    if reset_count < 2:
        raise ValueError("reset_count must be at least two")
    if soak_steps <= 0 or input_hold_steps <= 0 or drive_off_steps <= 0:
        raise ValueError("soak_steps, input_hold_steps and drive_off_steps must be positive")
    if not 0 < drive_off_throttle <= 1:
        raise ValueError("drive_off_throttle must lie in (0, 1]")
    if minimum_drive_off_speed_mps <= 0 or minimum_drive_off_progress_m <= 0:
        raise ValueError("drive-off acceptance thresholds must be positive")
    if minimum_forward_gear < 1:
        raise ValueError("minimum_forward_gear must be at least one")
    if not math.isfinite(control_frequency_hz) or control_frequency_hz <= 0:
        raise ValueError("control_frequency_hz must be finite and positive")
    if (
        min(
            reset_position_tolerance_m,
            reset_progress_tolerance_m,
            reset_heading_tolerance_rad,
        )
        <= 0
    ):
        raise ValueError("reset repeatability tolerances must be positive")
    if (
        min(
            maximum_reset_lateral_offset_m,
            maximum_reset_heading_error_rad,
            maximum_reset_speed_mps,
            reference_start_speed_tolerance_mps,
        )
        <= 0
    ):
        raise ValueError("reset-state acceptance thresholds must be positive")
    if expected_simulator_steering_sign not in {-1.0, 1.0}:
        raise ValueError("expected_simulator_steering_sign must be -1.0 or 1.0")

    issues: list[LiveValidationIssue] = []
    reset_samples: list[ResetValidationSample] = []
    input_echo: list[InputEchoSample] = []
    transport_samples: list[TransportValidationSample] = []
    drive_off: DriveOffValidation | None = None
    track_length_m: float | None = None

    try:
        for index in range(reset_count):
            started = clock()
            _, info = environment.reset()
            reset_samples.append(_reset_sample(index + 1, clock() - started, info))
            track_length_m = environment.active_reference.track_length_m
    except Exception as error:
        issues.append(LiveValidationIssue("reset", "reset_failed", str(error)))

    position_spread: float | None = None
    progress_spread: float | None = None
    heading_spread: float | None = None
    maximum_lateral: float | None = None
    maximum_heading_error: float | None = None
    maximum_speed: float | None = None
    if reset_samples and track_length_m is not None:
        origin = reset_samples[0]
        position_spread = max(
            math.hypot(
                sample.position_x_m - origin.position_x_m, sample.position_y_m - origin.position_y_m
            )
            for sample in reset_samples
        )
        progress_spread = max(
            _wrapped_difference(origin.progress_m, sample.progress_m, track_length_m)
            for sample in reset_samples
        )
        heading_spread = max(
            _wrapped_difference(
                origin.vehicle_heading_rad,
                sample.vehicle_heading_rad,
                2.0 * math.pi,
            )
            for sample in reset_samples
        )
        maximum_lateral = max(abs(sample.lateral_offset_m) for sample in reset_samples)
        maximum_heading_error = max(abs(sample.heading_error_rad) for sample in reset_samples)
        maximum_speed = max(sample.speed_mps for sample in reset_samples)
        effective_position_tolerance_m = reset_position_tolerance_m
        effective_progress_tolerance_m = reset_progress_tolerance_m
        if reset_method_variant == REFERENCE_SPEED_RESET_METHOD_VARIANT:
            one_control_frame_m = maximum_speed / control_frequency_hz
            effective_position_tolerance_m = max(
                effective_position_tolerance_m,
                one_control_frame_m,
            )
            effective_progress_tolerance_m = max(
                effective_progress_tolerance_m,
                one_control_frame_m,
            )
        if position_spread > effective_position_tolerance_m:
            issues.append(
                LiveValidationIssue(
                    "reset",
                    "position_not_repeatable",
                    f"reset position spread {position_spread:.3f} m exceeds "
                    f"{effective_position_tolerance_m:.3f} m",
                )
            )
        if progress_spread > effective_progress_tolerance_m:
            issues.append(
                LiveValidationIssue(
                    "reset",
                    "progress_not_repeatable",
                    f"reset progress spread {progress_spread:.3f} m exceeds "
                    f"{effective_progress_tolerance_m:.3f} m",
                )
            )
        if heading_spread > reset_heading_tolerance_rad:
            issues.append(
                LiveValidationIssue(
                    "reset",
                    "heading_not_repeatable",
                    f"reset heading spread {heading_spread:.4f} rad exceeds "
                    f"{reset_heading_tolerance_rad:.4f} rad",
                )
            )
        if maximum_lateral > maximum_reset_lateral_offset_m:
            issues.append(
                LiveValidationIssue(
                    "reset",
                    "spawn_lateral_offset_too_large",
                    f"maximum reset lateral offset {maximum_lateral:.3f} m exceeds "
                    f"{maximum_reset_lateral_offset_m:.3f} m",
                )
            )
        if maximum_heading_error > maximum_reset_heading_error_rad:
            issues.append(
                LiveValidationIssue(
                    "reset",
                    "spawn_heading_error_too_large",
                    f"maximum reset heading error {maximum_heading_error:.4f} rad exceeds "
                    f"{maximum_reset_heading_error_rad:.4f} rad",
                )
            )
        if reset_method_variant == REFERENCE_SPEED_RESET_METHOD_VARIANT:
            try:
                maximum_speed_error = max(
                    abs(
                        sample.speed_mps
                        - environment.active_reference.human_mean_speed_at(sample.progress_m)
                    )
                    for sample in reset_samples
                )
            except (AttributeError, ValueError) as error:
                issues.append(
                    LiveValidationIssue(
                        "reset",
                        "human_mean_start_speed_unavailable",
                        str(error),
                    )
                )
            else:
                if maximum_speed_error > reference_start_speed_tolerance_mps:
                    issues.append(
                        LiveValidationIssue(
                            "reset",
                            "reference_start_speed_mismatch",
                            f"maximum reset-speed error {maximum_speed_error:.3f} m/s "
                            f"exceeds {reference_start_speed_tolerance_mps:.3f} m/s",
                        )
                    )
        elif maximum_speed > maximum_reset_speed_mps:
            issues.append(
                LiveValidationIssue(
                    "reset",
                    "spawn_not_stationary",
                    f"maximum reset speed {maximum_speed:.3f} m/s exceeds "
                    f"{maximum_reset_speed_mps:.3f} m/s",
                )
            )

    if len(reset_samples) == reset_count:
        try:
            input_echo.extend(
                (
                    _hold_action(
                        environment,
                        name="throttle",
                        pedal=0.35,
                        steering=0.0,
                        hold_steps=input_hold_steps,
                    ),
                    _hold_action(
                        environment,
                        name="brake",
                        pedal=-0.35,
                        steering=0.0,
                        hold_steps=input_hold_steps,
                    ),
                    _hold_action(
                        environment,
                        name="steer_left",
                        pedal=0.0,
                        steering=0.35,
                        hold_steps=input_hold_steps,
                    ),
                    _hold_action(
                        environment,
                        name="steer_right",
                        pedal=0.0,
                        steering=-0.35,
                        hold_steps=input_hold_steps,
                    ),
                    _hold_action(
                        environment,
                        name="neutral",
                        pedal=0.0,
                        steering=0.0,
                        hold_steps=input_hold_steps,
                    ),
                )
            )
        except Exception as error:
            issues.append(LiveValidationIssue("input", "input_probe_failed", str(error)))

    echo_by_name = {sample.name: sample for sample in input_echo}
    throttle = echo_by_name.get("throttle")
    brake = echo_by_name.get("brake")
    steer_left = echo_by_name.get("steer_left")
    steer_right = echo_by_name.get("steer_right")
    neutral = echo_by_name.get("neutral")
    if throttle is not None and (throttle.simulator_gas < 0.2 or throttle.simulator_brake > 0.05):
        issues.append(
            LiveValidationIssue(
                "input",
                "throttle_echo_mismatch",
                "AC reported gas/brake="
                f"{throttle.simulator_gas:.3f}/{throttle.simulator_brake:.3f}",
            )
        )
    if brake is not None and (brake.simulator_brake < 0.2 or brake.simulator_gas > 0.05):
        issues.append(
            LiveValidationIssue(
                "input",
                "brake_echo_mismatch",
                f"AC reported gas/brake={brake.simulator_gas:.3f}/{brake.simulator_brake:.3f}",
            )
        )
    if (
        steer_left is not None
        and steer_right is not None
        and (
            abs(steer_left.simulator_steer_angle - steer_right.simulator_steer_angle) < 0.05
            or expected_simulator_steering_sign * steer_left.simulator_steer_angle < 0.05
            or expected_simulator_steering_sign * steer_right.simulator_steer_angle > -0.05
        )
    ):
        issues.append(
            LiveValidationIssue(
                "input",
                "steering_echo_mismatch",
                "left/right commands did not produce opposing AC steering angles",
            )
        )
    if neutral is not None and (neutral.simulator_gas > 0.05 or neutral.simulator_brake > 0.05):
        issues.append(
            LiveValidationIssue(
                "input",
                "neutral_echo_mismatch",
                f"neutral AC gas/brake={neutral.simulator_gas:.3f}/{neutral.simulator_brake:.3f}",
            )
        )

    if len(reset_samples) == reset_count and track_length_m is not None:
        try:
            _, reset_info = environment.reset()
            initial_progress_m = float(reset_info["telemetry"]["progress_m"])
            observed_gears: list[int] = []
            maximum_drive_speed_mps = 0.0
            maximum_forward_progress_m = 0.0
            maximum_abs_slip_ratio = 0.0
            maximum_rpms = 0
            drive_termination: str | None = None
            completed_drive_steps = 0
            required_raw_gear = minimum_forward_gear + 1
            for index in range(drive_off_steps):
                pedal = 0.0 if index == 0 else drive_off_throttle
                _, _, terminated, truncated, info = environment.step(
                    np.asarray([pedal, 0.0], dtype=np.float32)
                )
                completed_drive_steps += 1
                backend = info["backend"]
                observed_gears.append(int(backend["simulator_gear"]))
                maximum_rpms = max(maximum_rpms, int(backend["simulator_rpms"]))
                maximum_abs_slip_ratio = max(
                    maximum_abs_slip_ratio,
                    abs(float(backend["front_slip_ratio"])),
                    abs(float(backend["rear_slip_ratio"])),
                )
                maximum_drive_speed_mps = max(
                    maximum_drive_speed_mps,
                    float(info["telemetry"]["speed_mps"]),
                )
                maximum_forward_progress_m = max(
                    maximum_forward_progress_m,
                    _wrapped_delta(
                        initial_progress_m,
                        float(info["telemetry"]["progress_m"]),
                        track_length_m,
                    ),
                )
                if terminated or truncated:
                    drive_termination = info["termination_reason"] or "truncated"
                    break
                if (
                    observed_gears[-1] >= required_raw_gear
                    and maximum_drive_speed_mps >= minimum_drive_off_speed_mps
                    and maximum_forward_progress_m >= minimum_drive_off_progress_m
                ):
                    break
            drive_off = DriveOffValidation(
                steps_completed=completed_drive_steps,
                required_forward_gear=minimum_forward_gear,
                first_observed_gear_raw=observed_gears[0] if observed_gears else None,
                maximum_observed_gear_raw=max(observed_gears) if observed_gears else None,
                maximum_observed_forward_gear=(
                    max(observed_gears) - 1 if observed_gears and max(observed_gears) >= 2 else None
                ),
                maximum_rpms=maximum_rpms,
                maximum_speed_mps=maximum_drive_speed_mps,
                forward_progress_m=maximum_forward_progress_m,
                maximum_abs_slip_ratio=maximum_abs_slip_ratio,
                termination_reason=drive_termination,
            )
        except Exception as error:
            issues.append(LiveValidationIssue("drive_off", "drive_off_probe_failed", str(error)))

    if drive_off is not None:
        # AC shared memory encodes reverse=0, neutral=1, first=2, second=3, etc.
        required_raw_gear = minimum_forward_gear + 1
        if (
            drive_off.maximum_observed_gear_raw is None
            or drive_off.maximum_observed_gear_raw < required_raw_gear
        ):
            issues.append(
                LiveValidationIssue(
                    "drive_off",
                    (
                        "automatic_upshift_missing"
                        if minimum_forward_gear > 1
                        else "forward_gear_missing"
                    ),
                    f"AC never reported forward gear {minimum_forward_gear} "
                    f"(maximum raw gear={drive_off.maximum_observed_gear_raw!r}, "
                    f"maximum RPM={drive_off.maximum_rpms})",
                )
            )
        if drive_off.maximum_speed_mps < minimum_drive_off_speed_mps:
            issues.append(
                LiveValidationIssue(
                    "drive_off",
                    "vehicle_did_not_accelerate",
                    f"maximum speed {drive_off.maximum_speed_mps:.3f} m/s is below "
                    f"{minimum_drive_off_speed_mps:.3f} m/s",
                )
            )
        if drive_off.forward_progress_m < minimum_drive_off_progress_m:
            issues.append(
                LiveValidationIssue(
                    "drive_off",
                    "forward_progress_missing",
                    f"forward progress {drive_off.forward_progress_m:.3f} m is below "
                    f"{minimum_drive_off_progress_m:.3f} m",
                )
            )
        if drive_off.termination_reason is not None:
            issues.append(
                LiveValidationIssue(
                    "drive_off",
                    "drive_off_terminated",
                    f"drive-off probe terminated with {drive_off.termination_reason}",
                )
            )

    termination_counts: Counter[str] = Counter()
    if len(reset_samples) == reset_count:
        try:
            environment.reset()
            for index in range(soak_steps):
                started = clock()
                _, _, terminated, truncated, info = environment.step(np.zeros(2, dtype=np.float32))
                latency_s = clock() - started
                backend = info["backend"]
                reason = info["termination_reason"]
                transport_samples.append(
                    TransportValidationSample(
                        index=index + 1,
                        latency_s=latency_s,
                        physics_packet_id=int(backend["physics_packet_id"]),
                        graphics_packet_id=int(backend["graphics_packet_id"]),
                        slip_sequence=int(backend["slip_sequence"]),
                        collision_sequence=int(backend["collision_sequence"]),
                        speed_mps=float(info["telemetry"]["speed_mps"]),
                        progress_m=float(info["telemetry"]["progress_m"]),
                        termination_reason=reason,
                    )
                )
                if reason is not None:
                    termination_counts[str(reason)] += 1
                if terminated or truncated:
                    environment.reset()
        except Exception as error:
            issues.append(LiveValidationIssue("soak", "connection_interrupted", str(error)))

    unique_physics = len({sample.physics_packet_id for sample in transport_samples})
    unique_slip = len({sample.slip_sequence for sample in transport_samples})
    if transport_samples and unique_physics != len(transport_samples):
        issues.append(
            LiveValidationIssue(
                "soak",
                "duplicate_physics_packet",
                f"{unique_physics}/{len(transport_samples)} physics packets were unique",
            )
        )
    if transport_samples and unique_slip != len(transport_samples):
        issues.append(
            LiveValidationIssue(
                "soak",
                "duplicate_slip_packet",
                f"{unique_slip}/{len(transport_samples)} slip packets were unique",
            )
        )

    summary = LiveValidationSummary(
        session_origin=session_origin,
        controller_index=controller_index,
        game_stop_requested=False,
        game_stopped=None,
        reset_method_variant=reset_method_variant,
        reset_count=len(reset_samples),
        reset_position_spread_m=position_spread,
        reset_progress_spread_m=progress_spread,
        reset_heading_spread_rad=heading_spread,
        maximum_abs_reset_lateral_offset_m=maximum_lateral,
        maximum_abs_reset_heading_error_rad=maximum_heading_error,
        maximum_reset_speed_mps=maximum_speed,
        maximum_reset_latency_s=(
            max(sample.latency_s for sample in reset_samples) if reset_samples else None
        ),
        soak_steps_requested=soak_steps,
        soak_steps_completed=len(transport_samples),
        maximum_step_latency_s=(
            max(sample.latency_s for sample in transport_samples) if transport_samples else None
        ),
        mean_step_latency_s=(
            float(np.mean([sample.latency_s for sample in transport_samples]))
            if transport_samples
            else None
        ),
        p95_step_latency_s=(
            float(np.quantile([sample.latency_s for sample in transport_samples], 0.95))
            if transport_samples
            else None
        ),
        unique_physics_packets=unique_physics,
        unique_slip_packets=unique_slip,
        termination_counts=dict(sorted(termination_counts.items())),
        input_echo=tuple(input_echo),
        drive_off=drive_off,
        issues=tuple(issues),
    )
    return summary, tuple(reset_samples), tuple(transport_samples)


def write_live_validation_artifacts(
    destination: Path,
    summary: LiveValidationSummary,
    reset_samples: tuple[ResetValidationSample, ...],
    transport_samples: tuple[TransportValidationSample, ...],
) -> None:
    """Persist a compact JSON report and raw reset/transport observations."""

    destination.mkdir(parents=True, exist_ok=False)
    (destination / "summary.json").write_text(
        json.dumps(summary.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for filename, samples in (
        ("resets.csv", reset_samples),
        ("transport.csv", transport_samples),
    ):
        if not samples:
            continue
        with (destination / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(asdict(samples[0])))
            writer.writeheader()
            writer.writerows(asdict(sample) for sample in samples)
