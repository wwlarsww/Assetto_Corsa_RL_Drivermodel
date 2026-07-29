"""Capture and validate read-only Assetto Corsa telemetry."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from assetto_corsa_rl.ac_integration.shared_memory import AssettoCorsaTelemetry

_STATUS_NAMES = {
    0: "off",
    1: "replay",
    2: "live",
    3: "pause",
}


class TelemetryReader(Protocol):
    """Small reader contract used by real shared memory and deterministic tests."""

    def read(self) -> AssettoCorsaTelemetry:
        """Return one copied snapshot."""


@dataclass(frozen=True, slots=True)
class DiagnosticCapture:
    """Unique physics frames and polling statistics from one timed capture."""

    snapshots: tuple[AssettoCorsaTelemetry, ...]
    poll_count: int
    elapsed_s: float
    requested_duration_s: float
    poll_frequency_hz: float


@dataclass(frozen=True, slots=True)
class DiagnosticIssue:
    """Machine-readable validation finding."""

    severity: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class FeatureReadiness:
    """Whether one thesis feature group is directly ready from shared memory."""

    feature: str
    status: str
    source: str


@dataclass(frozen=True, slots=True)
class TelemetryDiagnosticSummary:
    """Compact report used to accept or reject the live telemetry transport."""

    car_model: str
    track: str
    track_configuration: str
    assetto_corsa_version: str
    shared_memory_version: str
    track_spline_length_m: float
    requested_duration_s: float
    elapsed_s: float
    poll_count: int
    unique_physics_frames: int
    fresh_poll_fraction: float
    observed_physics_frequency_hz: float | None
    status_counts: dict[str, int]
    speed_min_kmh: float
    speed_max_kmh: float
    inferred_forward_local_velocity_axis: int | None
    inferred_forward_local_velocity_sign: int | None
    forward_axis_mean_error_kmh: float | None
    issues: tuple[DiagnosticIssue, ...]
    feature_readiness: tuple[FeatureReadiness, ...]

    @property
    def is_usable_transport(self) -> bool:
        """True when no transport-level error was detected."""

        return not any(issue.severity == "error" for issue in self.issues)


def capture_telemetry(
    reader: TelemetryReader,
    *,
    duration_s: float,
    poll_frequency_hz: float,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> DiagnosticCapture:
    """Poll for a fixed wall-clock duration and retain each physics packet once."""

    if duration_s <= 0 or poll_frequency_hz <= 0:
        raise ValueError("duration_s and poll_frequency_hz must be positive")
    interval = 1.0 / poll_frequency_hz
    start = clock()
    deadline = start + duration_s
    next_poll = start
    poll_count = 0
    last_packet_id: int | None = None
    snapshots: list[AssettoCorsaTelemetry] = []

    while clock() < deadline:
        snapshot = reader.read()
        poll_count += 1
        if snapshot.physics.packet_id != last_packet_id:
            snapshots.append(snapshot)
            last_packet_id = snapshot.physics.packet_id
        next_poll += interval
        delay = next_poll - clock()
        if delay > 0:
            sleep(delay)

    return DiagnosticCapture(
        snapshots=tuple(snapshots),
        poll_count=poll_count,
        elapsed_s=max(clock() - start, 0.0),
        requested_duration_s=duration_s,
        poll_frequency_hz=poll_frequency_hz,
    )


def _infer_forward_axis(
    snapshots: Sequence[AssettoCorsaTelemetry],
) -> tuple[int | None, int | None, float | None]:
    moving = [snapshot for snapshot in snapshots if snapshot.physics.speed_kmh >= 5.0]
    if len(moving) < 3:
        return None, None, None
    speed = np.asarray([snapshot.physics.speed_kmh for snapshot in moving], dtype=np.float64)
    local_velocity = np.asarray(
        [snapshot.physics.local_velocity_mps for snapshot in moving],
        dtype=np.float64,
    )
    candidates: list[tuple[float, int, int]] = []
    for axis in range(3):
        for sign in (-1, 1):
            error = float(np.mean(np.abs(sign * local_velocity[:, axis] * 3.6 - speed)))
            candidates.append((error, axis, sign))
    error, axis, sign = min(candidates)
    return axis, sign, error


def _all_finite(snapshot: AssettoCorsaTelemetry) -> bool:
    physics = snapshot.physics
    graphics = snapshot.graphics
    values = (
        physics.gas,
        physics.brake,
        physics.clutch,
        physics.steer_angle,
        physics.speed_kmh,
        physics.heading_rad,
        physics.brake_bias,
        *physics.velocity,
        *physics.acceleration_g,
        *physics.wheel_slip,
        *physics.wheel_load_n,
        *physics.wheel_angular_speed_rad_s,
        *physics.local_angular_velocity_rad_s,
        *physics.local_velocity_mps,
        graphics.distance_traveled_m,
        graphics.normalized_car_position,
        *graphics.car_coordinates_m,
        graphics.surface_grip,
    )
    return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float64))))


def _basic_range_issues(
    snapshots: Sequence[AssettoCorsaTelemetry],
) -> list[DiagnosticIssue]:
    static = snapshots[0].static
    issues: list[DiagnosticIssue] = []
    if any(not _all_finite(snapshot) for snapshot in snapshots):
        issues.append(
            DiagnosticIssue("error", "non_finite_signal", "At least one raw signal is non-finite.")
        )
    if any(
        snapshot.physics.gas < -0.01
        or snapshot.physics.gas > 1.01
        or snapshot.physics.brake < -0.01
        or snapshot.physics.brake > 1.01
        or snapshot.physics.clutch < -0.01
        or snapshot.physics.clutch > 1.01
        for snapshot in snapshots
    ):
        issues.append(
            DiagnosticIssue(
                "error",
                "control_range",
                "Gas, brake, or clutch left its documented [0, 1] range.",
            )
        )
    if any(
        snapshot.graphics.normalized_car_position < -0.01
        or snapshot.graphics.normalized_car_position > 1.01
        for snapshot in snapshots
    ):
        issues.append(
            DiagnosticIssue(
                "error",
                "normalised_position_range",
                "The normalised spline position left its documented [0, 1] range.",
            )
        )
    if static.track_spline_length_m <= 0:
        issues.append(
            DiagnosticIssue(
                "error",
                "missing_track_length",
                "The static page does not contain a positive track spline length.",
            )
        )
    if not static.car_model or not static.track:
        issues.append(
            DiagnosticIssue(
                "error",
                "missing_session_identity",
                "Car or track identifier is empty; wait until an on-track session is loaded.",
            )
        )
    return issues


def summarise_capture(capture: DiagnosticCapture) -> TelemetryDiagnosticSummary:
    """Run transport, range, state, and coordinate-axis checks."""

    snapshots = capture.snapshots
    if not snapshots:
        raise ValueError("capture does not contain a telemetry frame")
    first = snapshots[0]
    static = first.static
    issues = _basic_range_issues(snapshots)

    status_counter = Counter(
        _STATUS_NAMES.get(snapshot.graphics.status, f"unknown_{snapshot.graphics.status}")
        for snapshot in snapshots
    )
    if status_counter.get("live", 0) == 0:
        issues.append(
            DiagnosticIssue(
                "warning",
                "session_not_live",
                "No AC_LIVE frame was observed; menus, pause, and replay are not RL transitions.",
            )
        )

    observed_frequency: float | None = None
    if len(snapshots) >= 2:
        frame_duration = snapshots[-1].monotonic_time_s - snapshots[0].monotonic_time_s
        if frame_duration > 0:
            observed_frequency = (len(snapshots) - 1) / frame_duration
    else:
        issues.append(
            DiagnosticIssue(
                "error",
                "stale_physics_page",
                "Only one physics packet was observed; the page is stale or the game is paused.",
            )
        )

    fresh_fraction = len(snapshots) / max(capture.poll_count, 1)
    if fresh_fraction < 0.25:
        issues.append(
            DiagnosticIssue(
                "warning",
                "low_fresh_frame_fraction",
                "Fewer than 25% of polls returned a new physics packet.",
            )
        )

    axis, sign, axis_error = _infer_forward_axis(snapshots)
    if axis is None:
        issues.append(
            DiagnosticIssue(
                "warning",
                "forward_axis_unresolved",
                "Drive above 5 km/h for several frames to calibrate the local forward axis.",
            )
        )
    elif axis_error is not None and axis_error > 2.0:
        issues.append(
            DiagnosticIssue(
                "warning",
                "forward_axis_mismatch",
                f"Best local velocity axis still differs from speed by {axis_error:.2f} km/h.",
            )
        )

    issues.extend(
        (
            DiagnosticIssue(
                "warning",
                "slip_ratio_bridge_required",
                "Shared-memory wheelSlip semantics are not accepted as the thesis slip ratio; "
                "use the in-game acsys.CS.SlipRatio bridge.",
            ),
            DiagnosticIssue(
                "warning",
                "side_slip_bridge_required",
                "Front/rear side-slip angle is not published in shared memory; "
                "use the in-game acsys.CS.SlipAngle bridge.",
            ),
        )
    )

    speeds = [snapshot.physics.speed_kmh for snapshot in snapshots]
    return TelemetryDiagnosticSummary(
        car_model=static.car_model,
        track=static.track,
        track_configuration=static.track_configuration,
        assetto_corsa_version=static.assetto_corsa_version,
        shared_memory_version=static.shared_memory_version,
        track_spline_length_m=static.track_spline_length_m,
        requested_duration_s=capture.requested_duration_s,
        elapsed_s=capture.elapsed_s,
        poll_count=capture.poll_count,
        unique_physics_frames=len(snapshots),
        fresh_poll_fraction=fresh_fraction,
        observed_physics_frequency_hz=observed_frequency,
        status_counts=dict(sorted(status_counter.items())),
        speed_min_kmh=float(min(speeds)),
        speed_max_kmh=float(max(speeds)),
        inferred_forward_local_velocity_axis=axis,
        inferred_forward_local_velocity_sign=sign,
        forward_axis_mean_error_kmh=axis_error,
        issues=tuple(issues),
        feature_readiness=(
            FeatureReadiness("speed", "direct", "physics.speedKmh"),
            FeatureReadiness(
                "longitudinal/lateral acceleration",
                "calibration_required",
                "physics.accG raw axes",
            ),
            FeatureReadiness(
                "front/rear slip ratio",
                "bridge_required",
                "acsys.CS.SlipRatio",
            ),
            FeatureReadiness(
                "front/rear side-slip angle",
                "bridge_required",
                "acsys.CS.SlipAngle",
            ),
            FeatureReadiness(
                "world pose and track phase",
                "direct",
                "graphics.carCoordinates + normalizedCarPosition",
            ),
        ),
    )


def _csv_row(snapshot: AssettoCorsaTelemetry) -> dict[str, float | int]:
    physics = snapshot.physics
    graphics = snapshot.graphics
    row: dict[str, float | int] = {
        "monotonic_time_s": snapshot.monotonic_time_s,
        "physics_packet_id": physics.packet_id,
        "graphics_packet_id": graphics.packet_id,
        "status": graphics.status,
        "session": graphics.session,
        "completed_laps": graphics.completed_laps,
        "current_lap_time_ms": graphics.current_lap_time_ms,
        "normalized_car_position": graphics.normalized_car_position,
        "distance_traveled_m": graphics.distance_traveled_m,
        "speed_kmh": physics.speed_kmh,
        "gas": physics.gas,
        "brake": physics.brake,
        "clutch": physics.clutch,
        "steer_angle": physics.steer_angle,
        "heading_rad": physics.heading_rad,
        "number_of_tyres_out": physics.number_of_tyres_out,
    }
    for prefix, values in (
        ("world_position_m", graphics.car_coordinates_m),
        ("world_velocity_mps", physics.velocity),
        ("local_velocity_mps", physics.local_velocity_mps),
        ("acceleration_g", physics.acceleration_g),
        ("local_angular_velocity_rad_s", physics.local_angular_velocity_rad_s),
        ("wheel_slip_raw", physics.wheel_slip),
        ("wheel_angular_speed_rad_s", physics.wheel_angular_speed_rad_s),
    ):
        for index, value in enumerate(values):
            row[f"{prefix}_{index}"] = value
    return row


def write_diagnostic_artifacts(
    destination: Path,
    capture: DiagnosticCapture,
    summary: TelemetryDiagnosticSummary,
) -> None:
    """Write portable JSON summary and flat CSV raw signals."""

    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = [_csv_row(snapshot) for snapshot in capture.snapshots]
    if not rows:
        raise ValueError("cannot write an empty telemetry capture")
    with (destination / "telemetry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
