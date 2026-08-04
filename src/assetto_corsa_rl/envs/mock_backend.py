"""Deterministic lightweight vehicle model for tests and examples."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.envs.backend import validate_backend_action
from assetto_corsa_rl.references.fixed_reference import ReferenceTrajectory
from assetto_corsa_rl.types import (
    TRACK_PREVIEW_DISTANCES_M,
    BackendTransition,
    ReferenceLinePreview,
    TelemetryFrame,
    TerminationReason,
    TrackPreview,
    VehicleState,
)


@dataclass(slots=True)
class MockAssettoCorsaBackend:
    """Small kinematic model with deterministic safety boundaries.

    It is intentionally not a fidelity model of Assetto Corsa. Its job is to
    exercise I/O, observations, cumulative controls, rewards and reproducible
    termination paths without the game.
    """

    control_frequency_hz: float = 20.0
    track_length_m: float = 1000.0
    track_half_width_m: float = 5.0
    initial_speed_mps: float = 20.0
    minimum_speed_mps: float = 2.0
    stall_startup_grace_steps: int = 25
    stall_debounce_steps: int = 5
    critical_body_sideslip_angle_deg: float = 20.0
    minimum_body_sideslip_termination_speed_mps: float = 5.0
    _reference: ReferenceTrajectory | None = field(init=False, default=None, repr=False)
    _progress_m: float = field(init=False, default=0.0, repr=False)
    _lateral_offset_m: float = field(init=False, default=0.0, repr=False)
    _heading_error_rad: float = field(init=False, default=0.0, repr=False)
    _speed_mps: float = field(init=False, default=0.0, repr=False)
    _step_count: int = field(init=False, default=0, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _last_longitudinal_acceleration: float = field(init=False, default=0.0, repr=False)
    _last_lateral_acceleration: float = field(init=False, default=0.0, repr=False)
    _last_front_slip: float = field(init=False, default=0.0, repr=False)
    _last_rear_slip: float = field(init=False, default=0.0, repr=False)
    _last_body_sideslip_angle_rad: float = field(init=False, default=0.0, repr=False)
    _last_steering: float = field(init=False, default=0.0, repr=False)
    _stall_detection_armed: bool = field(init=False, default=False, repr=False)
    _consecutive_low_speed_steps: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        positive_values = (
            self.control_frequency_hz,
            self.track_length_m,
            self.track_half_width_m,
            self.initial_speed_mps,
            self.minimum_speed_mps,
            self.critical_body_sideslip_angle_deg,
            self.minimum_body_sideslip_termination_speed_mps,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError("mock backend physical settings must be positive")
        if self.minimum_speed_mps >= self.initial_speed_mps:
            raise ValueError("minimum speed must be below initial speed")
        if self.stall_debounce_steps <= 0:
            raise ValueError("stall debounce steps must be positive")
        if self.stall_startup_grace_steps <= 0:
            raise ValueError("stall startup grace steps must be positive")
        if self.critical_body_sideslip_angle_deg > 90.0:
            raise ValueError("critical body sideslip angle must not exceed 90 degrees")

    def _update_stall_detection(self) -> None:
        """Arm after driving off and debounce subsequent low-speed frames."""

        if self._speed_mps >= self.minimum_speed_mps:
            self._stall_detection_armed = True
            self._consecutive_low_speed_steps = 0
        elif self._stall_detection_armed:
            self._consecutive_low_speed_steps += 1

    def reset(
        self,
        *,
        seed: int | None,
        reference: ReferenceTrajectory,
        initial_action: npt.NDArray[np.float32] | None = None,
    ) -> TelemetryFrame:
        """Reset the deterministic vehicle state for one synthetic episode."""

        del seed
        if self._closed:
            raise RuntimeError("mock backend is closed")
        if not np.isclose(reference.track_length_m, self.track_length_m):
            raise ValueError("reference and mock backend track lengths differ")
        action = validate_backend_action(
            np.zeros(2, dtype=np.float32) if initial_action is None else initial_action
        )
        self._reference = reference
        self._progress_m = 0.0
        self._lateral_offset_m = 0.0
        self._heading_error_rad = 0.0
        self._speed_mps = self.initial_speed_mps
        self._step_count = 0
        self._stall_detection_armed = False
        self._consecutive_low_speed_steps = 0
        self._last_longitudinal_acceleration = 0.0
        self._last_lateral_acceleration = 0.0
        self._last_front_slip = 0.0
        self._last_rear_slip = 0.0
        self._last_body_sideslip_angle_rad = 0.0
        self._last_steering = float(action[1])
        return self._telemetry()

    def step(self, absolute_action: npt.NDArray[np.float32]) -> BackendTransition:
        """Advance the lightweight kinematic model by one control period."""

        if self._reference is None:
            raise RuntimeError("reset must be called before step")
        if self._closed:
            raise RuntimeError("mock backend is closed")
        action = validate_backend_action(absolute_action)
        pedal, steering = (float(action[0]), float(action[1]))
        dt = 1.0 / self.control_frequency_hz

        throttle = max(pedal, 0.0)
        brake = max(-pedal, 0.0)
        aerodynamic_drag = 0.005 * self._speed_mps**2
        longitudinal_acceleration = 6.0 * throttle - 10.0 * brake - aerodynamic_drag
        self._speed_mps = max(0.0, self._speed_mps + longitudinal_acceleration * dt)

        self._heading_error_rad += (
            steering * (0.35 + self._speed_mps / 80.0) - 0.15 * self._heading_error_rad
        ) * dt
        forward_velocity = max(self._speed_mps * np.cos(self._heading_error_rad), 0.0)
        lateral_velocity = self._speed_mps * np.sin(self._heading_error_rad)
        self._progress_m += forward_velocity * dt
        self._lateral_offset_m += lateral_velocity * dt

        steering_rate = (steering - self._last_steering) / dt
        front_slip = 0.12 * steering * self._speed_mps / 20.0 + 0.015 * steering_rate
        rear_slip = 0.08 * steering * self._speed_mps / 20.0 + 0.03 * brake
        lateral_acceleration = self._speed_mps**2 * steering / 18.0

        self._step_count += 1
        self._last_longitudinal_acceleration = longitudinal_acceleration
        self._last_lateral_acceleration = lateral_acceleration
        self._last_front_slip = front_slip
        self._last_rear_slip = rear_slip
        # The mock has no tyre model. Half the heading error is a stable,
        # deterministic body-sideslip proxy used only to exercise safety paths.
        self._last_body_sideslip_angle_rad = 0.5 * self._heading_error_rad
        self._last_steering = steering

        self._update_stall_detection()

        termination_reason = self._termination_reason()
        telemetry = self._telemetry()
        return BackendTransition(
            telemetry=telemetry,
            termination_reason=termination_reason,
            diagnostics={
                "mock_backend": True,
                "absolute_pedal": pedal,
                "absolute_steering": steering,
                "body_sideslip_angle_rad": self._last_body_sideslip_angle_rad,
                "body_sideslip_angle_deg": math.degrees(self._last_body_sideslip_angle_rad),
                "number_of_tyres_out": int(abs(self._lateral_offset_m) > self.track_half_width_m),
                "stall_detection_armed": self._stall_detection_armed,
                "consecutive_low_speed_steps": self._consecutive_low_speed_steps,
            },
        )

    def close(self) -> None:
        """Mark the mock backend closed and release its active reference."""

        self._closed = True
        self._reference = None

    def _termination_reason(self) -> TerminationReason | None:
        if self._progress_m >= self.track_length_m:
            return TerminationReason.LAP_COMPLETE
        if abs(self._lateral_offset_m) > self.track_half_width_m:
            return TerminationReason.OFF_TRACK
        if (
            self._stall_detection_armed
            and self._consecutive_low_speed_steps >= self.stall_debounce_steps
        ):
            return TerminationReason.STALLED
        if not self._stall_detection_armed and self._step_count >= self.stall_startup_grace_steps:
            return TerminationReason.STALLED
        if self._speed_mps >= self.minimum_body_sideslip_termination_speed_mps and abs(
            self._last_body_sideslip_angle_rad
        ) > math.radians(self.critical_body_sideslip_angle_deg):
            return TerminationReason.EXCESSIVE_BODY_SIDESLIP
        return None

    def _telemetry(self) -> TelemetryFrame:
        if self._reference is None:
            raise RuntimeError("mock backend has no active reference")
        position_xy_m = np.asarray(
            [self._progress_m, self._lateral_offset_m],
            dtype=np.float32,
        )
        reference_line_preview = ReferenceLinePreview(
            self._reference.ego_path_preview(
                self._progress_m,
                position_xy_m,
                self._heading_error_rad,
            )
        )
        distances = TRACK_PREVIEW_DISTANCES_M
        left = np.column_stack(
            (
                np.full_like(
                    distances,
                    self.track_half_width_m - self._lateral_offset_m,
                ),
                distances,
            )
        ).astype(np.float32)
        right = np.column_stack(
            (
                np.full_like(
                    distances,
                    -self.track_half_width_m - self._lateral_offset_m,
                ),
                distances,
            )
        ).astype(np.float32)

        return TelemetryFrame(
            vehicle_state=VehicleState(
                speed_mps=self._speed_mps,
                acceleration_x_mps2=self._last_lateral_acceleration,
                acceleration_z_mps2=self._last_longitudinal_acceleration,
                front_slip_ratio=self._last_front_slip,
                rear_slip_ratio=self._last_rear_slip,
                front_sideslip_angle_rad=0.7 * self._heading_error_rad,
                rear_sideslip_angle_rad=0.5 * self._heading_error_rad,
            ),
            track_preview=TrackPreview(left_boundary_xz_m=left, right_boundary_xz_m=right),
            reference_line_preview=reference_line_preview,
            progress_m=self._progress_m,
            lateral_offset_m=self._lateral_offset_m,
            heading_error_rad=self._heading_error_rad,
            vehicle_heading_rad=self._heading_error_rad,
            position_xy_m=position_xy_m,
            simulation_time_s=self._step_count / self.control_frequency_hz,
        )
