"""Write-capable, frame-synchronised backend for the original Assetto Corsa."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from typing import Protocol

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.ac_integration.collision_bridge import (
    AssettoCorsaCollisionBridge,
    CollisionState,
)
from assetto_corsa_rl.ac_integration.reset_bridge import (
    HOTLAP_REFERENCE_POSE_SPEED_RESET_METHOD_VARIANT,
    HOTLAP_REFERENCE_SPEED_RESET_METHOD_VARIANT,
    AssettoCorsaHotlapReferenceSpeedResetBridge,
    AssettoCorsaHotlapResetBridge,
    AssettoCorsaReferencePoseSpeedResetBridge,
    AssettoCorsaResetBridge,
    ResetAcknowledgement,
)
from assetto_corsa_rl.ac_integration.shared_memory import (
    AssettoCorsaSharedMemoryReader,
    AssettoCorsaTelemetry,
)
from assetto_corsa_rl.ac_integration.slip_bridge import (
    AssettoCorsaSlipBridgeFileReader,
    SlipBridgePacket,
    SlipBridgeTimeoutError,
)
from assetto_corsa_rl.ac_integration.slip_features import (
    assetto_corsa_body_sideslip_angle_rad,
    vanilla_ac_slip_features,
)
from assetto_corsa_rl.config import (
    LEGACY_LIVE_STEP_TIMING_METHOD_VARIANT,
    PRE_ACTION_PROGRESS_BASELINE_METHOD_VARIANT,
    TERMINATION_METHOD_VARIANT,
    UNRECOVERABLE_PROGRESS_STALL_TERMINATION_METHOD_VARIANT,
)
from assetto_corsa_rl.control.vgamepad import VGamepadControlWriter
from assetto_corsa_rl.envs.backend import validate_backend_action
from assetto_corsa_rl.features.preprocessing import stabilise_low_speed_slip
from assetto_corsa_rl.features.track_geometry import TrackGeometry
from assetto_corsa_rl.references.fixed_reference import ReferenceTrajectory
from assetto_corsa_rl.references.projection import wrapped_progress_delta
from assetto_corsa_rl.types import (
    BackendTransition,
    ReferenceLinePreview,
    TelemetryFrame,
    TerminationReason,
    TrackPreview,
    VehicleState,
)

STANDARD_GRAVITY_MPS2 = 9.80665
ASSETTO_CORSA_LIVE_STATUS = 2
LIVE_TELEMETRY_CALIBRATION_NAME = "assetto_corsa_gym_world_v1"
LOGGER = logging.getLogger(__name__)


class LiveBackendError(RuntimeError):
    """Base error for invalid sessions, stale frames, and unsafe live state."""


class LiveBackendTimeoutError(LiveBackendError):
    """Raised when no suitable fresh simulator frame arrives before a deadline."""


class LiveSessionMismatchError(LiveBackendError):
    """Raised when the running car/track does not match the experiment config."""


class _ReferencePoseResetError(LiveBackendError):
    """Raised for a fresh but transiently misaligned CSP pose-reset frame."""


class _TelemetryReader(Protocol):
    def read(self) -> AssettoCorsaTelemetry: ...

    def close(self) -> None: ...


class _SlipReader(Protocol):
    def receive(self, *, timeout_s: float) -> SlipBridgePacket: ...

    def close(self) -> None: ...


class _CollisionReader(Protocol):
    def synchronise(self, *, expected_track_id: str, timeout_s: float) -> CollisionState: ...

    def collision_since_synchronisation(
        self,
        *,
        timeout_s: float,
    ) -> tuple[bool, CollisionState]: ...

    def close(self) -> None: ...


class _ControlWriter(Protocol):
    def apply(self, absolute_action: npt.ArrayLike) -> None: ...

    def neutral(self) -> None: ...

    def close(self) -> None: ...


def _is_positive_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


class _ResetController(Protocol):
    def request_reset(
        self,
        *,
        timeout_s: float,
        start_speed_mps: float | None = None,
        start_position_xy_m: tuple[float, float] | None = None,
        start_heading_rad: float | None = None,
    ) -> ResetAcknowledgement: ...

    def close(self) -> None: ...


def _wrapped_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


class AssettoCorsaLiveBackend:
    """Compose telemetry, CSP reset, track geometry, and virtual X360 control.

    The backend never launches the game or edits user configuration. Every
    action remains active for one complete control period, and a transition is
    returned only after a newer live physics packet and a fresh slip packet.
    """

    def __init__(  # noqa: PLR0915 - explicit validation and transport state setup
        self,
        *,
        telemetry_reader: _TelemetryReader,
        slip_reader: _SlipReader,
        collision_reader: _CollisionReader,
        control_writer: _ControlWriter,
        reset_controller: _ResetController,
        track_geometry: TrackGeometry,
        control_frequency_hz: float,
        expected_car_id: str,
        expected_track_id: str,
        minimum_speed_mps: float,
        stall_startup_grace_steps: int,
        stall_debounce_steps: int,
        critical_body_sideslip_angle_deg: float,
        minimum_body_sideslip_termination_speed_mps: float,
        termination_method_variant: str = TERMINATION_METHOD_VARIANT,
        progress_stall_window_s: float = 1.0,
        progress_stall_minimum_progress_m: float = 2.0,
        frame_timeout_s: float = 2.0,
        reset_timeout_s: float = 10.0,
        reset_position_tolerance: float = 0.05,
        reset_reference_lateral_offset_m: float = 0.0,
        reset_reference_heading_tolerance_rad: float = 0.05,
        reset_speed_threshold_mps: float = 5.0,
        reset_reference_speed_tolerance_mps: float = 2.0,
        reset_stable_frames: int = 3,
        reset_max_abs_slip_ratio: float = 0.2,
        reset_slip_stable_frames: int = 1,
        reset_reference_speed_max_attempts: int = 1,
        reset_collision_quiescence_s: float = 0.5,
        reset_method_variant: str = "csp_hotlap_spawn_v1",
        step_timing_method_variant: str = LEGACY_LIVE_STEP_TIMING_METHOD_VARIANT,
        poll_interval_s: float = 0.001,
        monotonic_clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        positive = (
            control_frequency_hz,
            minimum_speed_mps,
            critical_body_sideslip_angle_deg,
            minimum_body_sideslip_termination_speed_mps,
            progress_stall_window_s,
            progress_stall_minimum_progress_m,
            frame_timeout_s,
            reset_timeout_s,
            reset_position_tolerance,
            reset_reference_heading_tolerance_rad,
            reset_speed_threshold_mps,
            reset_reference_speed_tolerance_mps,
            reset_max_abs_slip_ratio,
            poll_interval_s,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("live backend timing and safety values must be positive")
        if stall_debounce_steps <= 0:
            raise ValueError("stall_debounce_steps must be positive")
        if stall_startup_grace_steps <= 0:
            raise ValueError("stall_startup_grace_steps must be positive")
        if critical_body_sideslip_angle_deg > 90.0:
            raise ValueError("critical body sideslip angle must not exceed 90 degrees")
        if termination_method_variant not in {
            TERMINATION_METHOD_VARIANT,
            UNRECOVERABLE_PROGRESS_STALL_TERMINATION_METHOD_VARIANT,
        }:
            raise ValueError(
                f"unsupported termination method variant: {termination_method_variant!r}"
            )
        if reset_stable_frames <= 0:
            raise ValueError("reset_stable_frames must be positive")
        if reset_slip_stable_frames <= 0:
            raise ValueError("reset_slip_stable_frames must be positive")
        if not _is_positive_integer(reset_reference_speed_max_attempts):
            raise ValueError("reset_reference_speed_max_attempts must be a positive integer")
        if reset_collision_quiescence_s < 0:
            raise ValueError("reset_collision_quiescence_s must be non-negative")
        if not expected_car_id or not expected_track_id:
            raise ValueError("expected car and track identifiers must not be empty")
        if reset_method_variant not in {
            "csp_hotlap_spawn_v1",
            HOTLAP_REFERENCE_SPEED_RESET_METHOD_VARIANT,
            HOTLAP_REFERENCE_POSE_SPEED_RESET_METHOD_VARIANT,
            "python_ext_reset_car_v1",
        }:
            raise ValueError(f"unsupported reset method variant: {reset_method_variant!r}")
        if step_timing_method_variant not in {
            LEGACY_LIVE_STEP_TIMING_METHOD_VARIANT,
            PRE_ACTION_PROGRESS_BASELINE_METHOD_VARIANT,
        }:
            raise ValueError(
                f"unsupported live step timing method variant: {step_timing_method_variant!r}"
            )
        self.telemetry_reader = telemetry_reader
        self.slip_reader = slip_reader
        self.collision_reader = collision_reader
        self.control_writer = control_writer
        self.reset_controller = reset_controller
        self.track_geometry = track_geometry
        self.control_frequency_hz = control_frequency_hz
        self.expected_car_id = expected_car_id
        self.expected_track_id = expected_track_id
        self.minimum_speed_mps = minimum_speed_mps
        self.stall_startup_grace_steps = stall_startup_grace_steps
        self.stall_debounce_steps = stall_debounce_steps
        self.critical_body_sideslip_angle_rad = math.radians(critical_body_sideslip_angle_deg)
        self.minimum_body_sideslip_termination_speed_mps = (
            minimum_body_sideslip_termination_speed_mps
        )
        self.termination_method_variant = termination_method_variant
        self.progress_stall_window_steps = max(
            1,
            math.ceil(progress_stall_window_s * control_frequency_hz),
        )
        self.progress_stall_minimum_progress_m = progress_stall_minimum_progress_m
        self.frame_timeout_s = frame_timeout_s
        self.reset_timeout_s = reset_timeout_s
        self.reset_position_tolerance = reset_position_tolerance
        self.reset_reference_lateral_offset_m = reset_reference_lateral_offset_m
        self.reset_reference_heading_tolerance_rad = reset_reference_heading_tolerance_rad
        self.reset_speed_threshold_mps = reset_speed_threshold_mps
        self.reset_reference_speed_tolerance_mps = reset_reference_speed_tolerance_mps
        self.reset_stable_frames = reset_stable_frames
        self.reset_max_abs_slip_ratio = reset_max_abs_slip_ratio
        self.reset_slip_stable_frames = reset_slip_stable_frames
        self.reset_reference_speed_max_attempts = reset_reference_speed_max_attempts
        self.reset_collision_quiescence_s = reset_collision_quiescence_s
        self.reset_method_variant = reset_method_variant
        self.step_timing_method_variant = step_timing_method_variant
        self.poll_interval_s = poll_interval_s
        self._monotonic_clock = monotonic_clock
        self._sleep = sleep
        self._reference: ReferenceTrajectory | None = None
        self._last_physics_packet_id: int | None = None
        self._last_graphics_packet_id: int | None = None
        self._episode_origin_monotonic_s = 0.0
        self._episode_start_completed_laps = 0
        self._step_count = 0
        self._stall_detection_armed = False
        self._consecutive_low_speed_steps = 0
        self._progress_stall_progress_m: deque[float] = deque(
            maxlen=self.progress_stall_window_steps + 1
        )
        self._progress_stall_window_progress_m: float | None = None
        self._last_reset_acknowledgement: ResetAcknowledgement | None = None
        self._last_collision_state: CollisionState | None = None
        self._hotlap_start_progress_m: float | None = None
        self._last_reset_target_speed_mps: float | None = None
        self._last_reset_reference_lateral_error_m: float | None = None
        self._last_reset_reference_heading_error_rad: float | None = None
        self._last_reset_max_abs_slip_ratio: float | None = None
        self._last_reset_attempts: int | None = None
        self._last_snapshot_monotonic_s: float | None = None
        self._closed = False

    @classmethod
    def connect(
        cls,
        *,
        track_geometry: TrackGeometry,
        control_frequency_hz: float,
        expected_car_id: str,
        expected_track_id: str,
        minimum_speed_mps: float,
        stall_startup_grace_steps: int,
        stall_debounce_steps: int,
        critical_body_sideslip_angle_deg: float,
        minimum_body_sideslip_termination_speed_mps: float,
        termination_method_variant: str,
        progress_stall_window_s: float,
        progress_stall_minimum_progress_m: float,
        frame_timeout_s: float,
        reset_timeout_s: float,
        reset_position_tolerance: float,
        reset_reference_lateral_offset_m: float,
        reset_reference_heading_tolerance_rad: float,
        reset_speed_threshold_mps: float,
        reset_reference_speed_tolerance_mps: float,
        reset_stable_frames: int,
        reset_max_abs_slip_ratio: float,
        reset_slip_stable_frames: int,
        reset_reference_speed_max_attempts: int,
        reset_collision_quiescence_s: float,
        poll_interval_s: float,
        steering_output_sign: float,
        steering_output_gain: float = 1.0,
        reset_backend: str = "csp_hotlap_spawn_v1",
        step_timing_method_variant: str = LEGACY_LIVE_STEP_TIMING_METHOD_VARIANT,
        control_writer: _ControlWriter | None = None,
    ) -> AssettoCorsaLiveBackend:
        """Connect all live transports and clean up partially opened resources."""

        telemetry: AssettoCorsaSharedMemoryReader | None = None
        slip: AssettoCorsaSlipBridgeFileReader | None = None
        collision: AssettoCorsaCollisionBridge | None = None
        control: _ControlWriter | None = control_writer
        reset: _ResetController | None = None
        try:
            telemetry = AssettoCorsaSharedMemoryReader.connect()
            slip = AssettoCorsaSlipBridgeFileReader.connect()
            collision = AssettoCorsaCollisionBridge.connect()
            if control is None:
                control = VGamepadControlWriter(
                    steering_output_sign=steering_output_sign,
                    steering_output_gain=steering_output_gain,
                )
            if reset_backend == "csp_hotlap_spawn_v1":
                reset = AssettoCorsaHotlapResetBridge.connect(
                    expected_track_id=expected_track_id,
                )
            elif reset_backend == HOTLAP_REFERENCE_SPEED_RESET_METHOD_VARIANT:
                reset = AssettoCorsaHotlapReferenceSpeedResetBridge.connect(
                    expected_track_id=expected_track_id,
                )
            elif reset_backend == HOTLAP_REFERENCE_POSE_SPEED_RESET_METHOD_VARIANT:
                reset = AssettoCorsaReferencePoseSpeedResetBridge.connect(
                    expected_track_id=expected_track_id,
                )
            elif reset_backend == "python_ext_reset_car_v1":
                reset = AssettoCorsaResetBridge.connect()
            else:
                raise ValueError(f"unsupported reset backend: {reset_backend!r}")
            return cls(
                telemetry_reader=telemetry,
                slip_reader=slip,
                collision_reader=collision,
                control_writer=control,
                reset_controller=reset,
                track_geometry=track_geometry,
                control_frequency_hz=control_frequency_hz,
                expected_car_id=expected_car_id,
                expected_track_id=expected_track_id,
                minimum_speed_mps=minimum_speed_mps,
                stall_startup_grace_steps=stall_startup_grace_steps,
                stall_debounce_steps=stall_debounce_steps,
                critical_body_sideslip_angle_deg=critical_body_sideslip_angle_deg,
                minimum_body_sideslip_termination_speed_mps=(
                    minimum_body_sideslip_termination_speed_mps
                ),
                termination_method_variant=termination_method_variant,
                progress_stall_window_s=progress_stall_window_s,
                progress_stall_minimum_progress_m=progress_stall_minimum_progress_m,
                frame_timeout_s=frame_timeout_s,
                reset_timeout_s=reset_timeout_s,
                reset_position_tolerance=reset_position_tolerance,
                reset_reference_lateral_offset_m=reset_reference_lateral_offset_m,
                reset_reference_heading_tolerance_rad=(reset_reference_heading_tolerance_rad),
                reset_speed_threshold_mps=reset_speed_threshold_mps,
                reset_reference_speed_tolerance_mps=(reset_reference_speed_tolerance_mps),
                reset_stable_frames=reset_stable_frames,
                reset_max_abs_slip_ratio=reset_max_abs_slip_ratio,
                reset_slip_stable_frames=reset_slip_stable_frames,
                reset_reference_speed_max_attempts=reset_reference_speed_max_attempts,
                reset_collision_quiescence_s=reset_collision_quiescence_s,
                reset_method_variant=reset_backend,
                step_timing_method_variant=step_timing_method_variant,
                poll_interval_s=poll_interval_s,
            )
        except BaseException:
            for resource in (reset, control, collision, slip, telemetry):
                if resource is not None:
                    with suppress(Exception):
                        resource.close()
            raise

    def reset(  # noqa: PLR0915 - explicit reset transport and safety sequence
        self,
        *,
        seed: int | None,
        reference: ReferenceTrajectory,
        initial_action: npt.NDArray[np.float32] | None = None,
    ) -> TelemetryFrame:
        """Reset neutrally, then prime controls and return a stable start frame."""

        del seed  # The original simulator exposes no deterministic session seed.
        self._require_open()
        action = None if initial_action is None else validate_backend_action(initial_action)
        self.control_writer.neutral()
        self._reference = None
        try:
            first = self.telemetry_reader.read()
            self._validate_session(first, reference)
            if self.reset_method_variant in {
                HOTLAP_REFERENCE_SPEED_RESET_METHOD_VARIANT,
                HOTLAP_REFERENCE_POSE_SPEED_RESET_METHOD_VARIANT,
            }:
                after_physics_packet_id = first.physics.packet_id
                after_graphics_packet_id = first.graphics.packet_id
                if self._hotlap_start_progress_m is None:
                    self.reset_controller.request_reset(
                        timeout_s=self.reset_timeout_s,
                        start_speed_mps=0.0,
                    )
                    probe = self._wait_for_reset_state(
                        after_physics_packet_id,
                        after_graphics_packet_id,
                    )
                    self._hotlap_start_progress_m = self._project_reference_progress(
                        probe,
                        reference,
                    )
                    after_physics_packet_id = probe.physics.packet_id
                    after_graphics_packet_id = probe.graphics.packet_id
                target_speed_mps = reference.human_mean_speed_at(self._hotlap_start_progress_m)
                target_position_xy_m: tuple[float, float] | None = None
                target_heading_rad: float | None = None
                if self.reset_method_variant == HOTLAP_REFERENCE_POSE_SPEED_RESET_METHOD_VARIANT:
                    target_position, target_heading = reference.world_pose_at(
                        self._hotlap_start_progress_m
                    )
                    target_position = (
                        target_position
                        + np.asarray(
                            [-math.sin(target_heading), math.cos(target_heading)],
                            dtype=np.float32,
                        )
                        * self.reset_reference_lateral_offset_m
                    )
                    target_position_xy_m = (
                        float(target_position[0]),
                        float(target_position[1]),
                    )
                    target_heading_rad = float(target_heading)
                (
                    acknowledgement,
                    snapshot,
                    slip_packet,
                    pose_lateral_error_m,
                    pose_heading_error_rad,
                ) = self._perform_reference_speed_reset(
                    after_physics_packet_id=after_physics_packet_id,
                    after_graphics_packet_id=after_graphics_packet_id,
                    reference=reference,
                    target_speed_mps=target_speed_mps,
                    target_position_xy_m=target_position_xy_m,
                    target_heading_rad=target_heading_rad,
                )
                self._last_reset_max_abs_slip_ratio = self._maximum_abs_slip_ratio(
                    snapshot,
                    slip_packet,
                )
                if self.reset_method_variant == HOTLAP_REFERENCE_POSE_SPEED_RESET_METHOD_VARIANT:
                    self._last_reset_reference_lateral_error_m = pose_lateral_error_m
                    self._last_reset_reference_heading_error_rad = pose_heading_error_rad
                else:
                    self._last_reset_reference_lateral_error_m = None
                    self._last_reset_reference_heading_error_rad = None
                self._last_reset_target_speed_mps = target_speed_mps
            else:
                acknowledgement = self.reset_controller.request_reset(
                    timeout_s=self.reset_timeout_s
                )
                snapshot = self._wait_for_reset_state(
                    first.physics.packet_id,
                    first.graphics.packet_id,
                )
                snapshot = self._wait_for_reset_collision_quiescence(snapshot)
                slip_packet = self.slip_reader.receive(timeout_s=self.frame_timeout_s)
                self._last_reset_target_speed_mps = None
                self._last_reset_max_abs_slip_ratio = None
                self._last_reset_reference_lateral_error_m = None
                self._last_reset_reference_heading_error_rad = None
                self._last_reset_attempts = 1
            collision_state = self.collision_reader.synchronise(
                expected_track_id=self.expected_track_id,
                timeout_s=self.frame_timeout_s,
            )
            self._reference = reference
            self._last_physics_packet_id = snapshot.physics.packet_id
            self._last_graphics_packet_id = snapshot.graphics.packet_id
            self._last_snapshot_monotonic_s = snapshot.monotonic_time_s
            self._episode_origin_monotonic_s = snapshot.monotonic_time_s
            self._episode_start_completed_laps = snapshot.graphics.completed_laps
            self._step_count = 0
            self._stall_detection_armed = False
            self._consecutive_low_speed_steps = 0
            self._progress_stall_progress_m.clear()
            self._progress_stall_window_progress_m = None
            self._last_reset_acknowledgement = acknowledgement
            self._last_collision_state = collision_state
            if action is not None:
                self.control_writer.apply(action)
            initial_frame = self._build_frame(snapshot, slip_packet)
            self._progress_stall_progress_m.append(initial_frame.progress_m)
            return initial_frame
        except BaseException:
            self._reference = None
            self._last_physics_packet_id = None
            self._last_graphics_packet_id = None
            self._last_snapshot_monotonic_s = None
            self._last_reset_max_abs_slip_ratio = None
            self._last_reset_reference_lateral_error_m = None
            self._last_reset_reference_heading_error_rad = None
            self._last_reset_attempts = None
            self._progress_stall_progress_m.clear()
            self._progress_stall_window_progress_m = None
            self.control_writer.neutral()
            raise

    def step(self, absolute_action: npt.NDArray[np.float32]) -> BackendTransition:
        """Apply one action for a complete period and return one fresh live frame."""

        self._require_open()
        if (
            self._reference is None
            or self._last_physics_packet_id is None
            or self._last_graphics_packet_id is None
        ):
            raise RuntimeError("reset must be called before step")
        action = validate_backend_action(absolute_action)
        reward_progress_start_m: float | None = None
        observation_to_action_delay_s: float | None = None
        after_physics_packet_id = self._last_physics_packet_id
        after_graphics_packet_id = self._last_graphics_packet_id
        if self.step_timing_method_variant == PRE_ACTION_PROGRESS_BASELINE_METHOD_VARIANT:
            baseline = self.telemetry_reader.read()
            self._validate_session(baseline, self._reference)
            if baseline.graphics.status != ASSETTO_CORSA_LIVE_STATUS:
                raise LiveBackendError("pre-action telemetry does not report a LIVE session")
            reward_progress_start_m = self._project_reference_progress(
                baseline,
                self._reference,
            )
            after_physics_packet_id = baseline.physics.packet_id
            after_graphics_packet_id = baseline.graphics.packet_id
            if self._last_snapshot_monotonic_s is not None:
                observation_to_action_delay_s = max(
                    baseline.monotonic_time_s - self._last_snapshot_monotonic_s,
                    0.0,
                )
        action_time = self._monotonic_clock()
        try:
            self.control_writer.apply(action)
            snapshot = self._wait_for_fresh_frame(
                after_physics_packet_id=after_physics_packet_id,
                after_graphics_packet_id=after_graphics_packet_id,
                not_before_s=action_time + 1.0 / self.control_frequency_hz,
                timeout_s=self.frame_timeout_s,
            )
            slip_packet = self.slip_reader.receive(timeout_s=self.frame_timeout_s)
            collision_detected, collision_state = (
                self.collision_reader.collision_since_synchronisation(
                    timeout_s=self.frame_timeout_s,
                )
            )
            frame = self._build_frame(snapshot, slip_packet)
        except BaseException:
            self.control_writer.neutral()
            raise
        self._last_physics_packet_id = snapshot.physics.packet_id
        self._last_graphics_packet_id = snapshot.graphics.packet_id
        self._last_snapshot_monotonic_s = snapshot.monotonic_time_s
        self._last_collision_state = collision_state
        self._step_count += 1
        body_sideslip_angle_rad = assetto_corsa_body_sideslip_angle_rad(
            snapshot.physics.local_velocity_mps
        )
        self._update_stall_detection(frame.vehicle_state.speed_mps)
        self._update_progress_stall_detection(frame.progress_m)
        termination = self._termination_reason(
            snapshot,
            frame,
            collision_detected,
            body_sideslip_angle_rad,
        )
        diagnostics = {
            "live_backend": True,
            "telemetry_calibration": LIVE_TELEMETRY_CALIBRATION_NAME,
            "step_timing_method_variant": self.step_timing_method_variant,
            "reward_progress_start_m": reward_progress_start_m,
            "observation_to_action_delay_s": observation_to_action_delay_s,
            "action_hold_duration_s": max(self._monotonic_clock() - action_time, 0.0),
            "physics_packet_id": snapshot.physics.packet_id,
            "graphics_packet_id": snapshot.graphics.packet_id,
            "slip_sequence": slip_packet.sequence,
            "collision_sequence": collision_state.collision_sequence,
            "collision_simulator_frame": collision_state.simulator_frame,
            "reset_sequence": (
                self._last_reset_acknowledgement.sequence
                if self._last_reset_acknowledgement is not None
                else None
            ),
            "reset_target_speed_mps": self._last_reset_target_speed_mps,
            "reset_start_reference_progress_m": self._hotlap_start_progress_m,
            "reset_max_abs_slip_ratio": self._last_reset_max_abs_slip_ratio,
            "reset_attempts": self._last_reset_attempts,
            "reset_reference_lateral_error_m": (self._last_reset_reference_lateral_error_m),
            "reset_reference_heading_error_rad": (self._last_reset_reference_heading_error_rad),
            "simulator_seed_supported": False,
            "absolute_pedal": float(action[0]),
            "absolute_steering": float(action[1]),
            "simulator_gas": snapshot.physics.gas,
            "simulator_brake": snapshot.physics.brake,
            "simulator_steer_angle": snapshot.physics.steer_angle,
            "simulator_gear": snapshot.physics.gear,
            "simulator_rpms": snapshot.physics.rpms,
            "front_slip_ratio": frame.vehicle_state.front_slip_ratio,
            "rear_slip_ratio": frame.vehicle_state.rear_slip_ratio,
            "body_sideslip_angle_rad": body_sideslip_angle_rad,
            "body_sideslip_angle_deg": math.degrees(body_sideslip_angle_rad),
            "number_of_tyres_out": snapshot.physics.number_of_tyres_out,
            "stall_detection_armed": self._stall_detection_armed,
            "consecutive_low_speed_steps": self._consecutive_low_speed_steps,
            "termination_method_variant": self.termination_method_variant,
            "progress_stall_window_steps": self.progress_stall_window_steps,
            "progress_stall_minimum_progress_m": self.progress_stall_minimum_progress_m,
            "progress_stall_window_progress_m": self._progress_stall_window_progress_m,
            "progress_stall_triggered": self._progress_stall_termination_due(),
            "reset_method_variant": (
                self._last_reset_acknowledgement.method_variant
                if self._last_reset_acknowledgement is not None
                else None
            ),
        }
        return BackendTransition(
            telemetry=frame,
            termination_reason=termination,
            diagnostics=diagnostics,
        )

    def _perform_reference_speed_reset(
        self,
        *,
        after_physics_packet_id: int,
        after_graphics_packet_id: int,
        reference: ReferenceTrajectory,
        target_speed_mps: float,
        target_position_xy_m: tuple[float, float] | None,
        target_heading_rad: float | None,
    ) -> tuple[
        ResetAcknowledgement,
        AssettoCorsaTelemetry,
        SlipBridgePacket,
        float | None,
        float | None,
    ]:
        """Retry a fully acknowledged moving reset after unsafe settling or pose."""

        last_failure: (
            LiveBackendTimeoutError | _ReferencePoseResetError | SlipBridgeTimeoutError | None
        ) = None
        for attempt in range(1, self.reset_reference_speed_max_attempts + 1):
            acknowledgement = self.reset_controller.request_reset(
                timeout_s=self.reset_timeout_s,
                start_speed_mps=target_speed_mps,
                start_position_xy_m=target_position_xy_m,
                start_heading_rad=target_heading_rad,
            )
            try:
                snapshot = self._wait_for_reference_speed_reset_state(
                    after_physics_packet_id,
                    after_graphics_packet_id,
                    target_speed_mps=target_speed_mps,
                )
                slip_packet = self.slip_reader.receive(timeout_s=self.frame_timeout_s)
                snapshot, slip_packet = self._wait_for_reference_speed_slip_stability(
                    snapshot,
                    slip_packet,
                    target_speed_mps=target_speed_mps,
                )
                pose_lateral_error_m: float | None = None
                pose_heading_error_rad: float | None = None
                if target_position_xy_m is not None:
                    (
                        pose_lateral_error_m,
                        pose_heading_error_rad,
                    ) = self._validate_reference_pose_reset(snapshot, reference)
            except (
                LiveBackendTimeoutError,
                _ReferencePoseResetError,
                SlipBridgeTimeoutError,
            ) as error:
                last_failure = error
                if attempt >= self.reset_reference_speed_max_attempts:
                    break
                LOGGER.warning(
                    "reference-speed reset attempt %d/%d was unsafe: %s; retrying",
                    attempt,
                    self.reset_reference_speed_max_attempts,
                    error,
                )
                self.control_writer.neutral()
                retry_baseline = self.telemetry_reader.read()
                self._validate_session(retry_baseline, reference)
                after_physics_packet_id = retry_baseline.physics.packet_id
                after_graphics_packet_id = retry_baseline.graphics.packet_id
                continue

            self._last_reset_attempts = attempt
            return (
                acknowledgement,
                snapshot,
                slip_packet,
                pose_lateral_error_m,
                pose_heading_error_rad,
            )

        if last_failure is None:
            raise RuntimeError("reference-speed reset retry loop ended without a result")
        self._last_reset_attempts = self.reset_reference_speed_max_attempts
        raise LiveBackendTimeoutError(
            "reference-speed reset remained unsafe after "
            f"{self.reset_reference_speed_max_attempts} attempt(s); "
            f"last failure: {last_failure}"
        ) from last_failure

    def _wait_for_reset_state(
        self,
        after_physics_packet_id: int,
        after_graphics_packet_id: int,
    ) -> AssettoCorsaTelemetry:
        deadline = self._monotonic_clock() + self.reset_timeout_s
        stable = 0
        physics_packet_id = after_physics_packet_id
        graphics_packet_id = after_graphics_packet_id
        latest: AssettoCorsaTelemetry | None = None
        previous_position_m: tuple[float, float, float] | None = None
        while self._monotonic_clock() < deadline:
            remaining = deadline - self._monotonic_clock()
            try:
                latest = self._wait_for_fresh_frame(
                    after_physics_packet_id=physics_packet_id,
                    after_graphics_packet_id=graphics_packet_id,
                    not_before_s=self._monotonic_clock(),
                    timeout_s=max(
                        min(remaining, self.frame_timeout_s),
                        self.poll_interval_s,
                    ),
                )
            except LiveBackendTimeoutError:
                # CSP can pause physics frames briefly while ext_resetCar() moves
                # the car. Keep polling within the larger reset timeout budget.
                continue
            physics_packet_id = latest.physics.packet_id
            graphics_packet_id = latest.graphics.packet_id
            position_m = latest.graphics.car_coordinates_m
            position_stable = (
                previous_position_m is None
                or math.dist(
                    position_m,
                    previous_position_m,
                )
                <= self.reset_position_tolerance
            )
            slow = latest.physics.speed_kmh / 3.6 <= self.reset_speed_threshold_mps
            stable = stable + 1 if position_stable and slow and not latest.graphics.is_in_pit else 0
            previous_position_m = position_m
            if stable >= self.reset_stable_frames:
                return latest
        raise LiveBackendTimeoutError(
            "reset was acknowledged but no stable non-pit frame arrived before timeout"
        )

    def _wait_for_reset_collision_quiescence(
        self,
        snapshot: AssettoCorsaTelemetry,
    ) -> AssettoCorsaTelemetry:
        """Keep controls neutral until delayed reset-only collisions have fired."""

        if self.reset_collision_quiescence_s == 0:
            return snapshot
        return self._wait_for_fresh_frame(
            after_physics_packet_id=snapshot.physics.packet_id,
            after_graphics_packet_id=snapshot.graphics.packet_id,
            not_before_s=self._monotonic_clock() + self.reset_collision_quiescence_s,
            timeout_s=self.reset_collision_quiescence_s + self.frame_timeout_s,
        )

    def _wait_for_reference_speed_reset_state(
        self,
        after_physics_packet_id: int,
        after_graphics_packet_id: int,
        *,
        target_speed_mps: float,
    ) -> AssettoCorsaTelemetry:
        """Return the first fresh non-pit frame at the requested CSP speed."""

        deadline = self._monotonic_clock() + self.reset_timeout_s
        physics_packet_id = after_physics_packet_id
        graphics_packet_id = after_graphics_packet_id
        latest_speed_mps: float | None = None
        while self._monotonic_clock() < deadline:
            remaining = deadline - self._monotonic_clock()
            try:
                latest = self._wait_for_fresh_frame(
                    after_physics_packet_id=physics_packet_id,
                    after_graphics_packet_id=graphics_packet_id,
                    not_before_s=self._monotonic_clock(),
                    timeout_s=max(
                        min(remaining, self.frame_timeout_s),
                        self.poll_interval_s,
                    ),
                )
            except LiveBackendTimeoutError:
                continue
            physics_packet_id = latest.physics.packet_id
            graphics_packet_id = latest.graphics.packet_id
            latest_speed_mps = latest.physics.speed_kmh / 3.6
            if (
                not latest.graphics.is_in_pit
                and abs(latest_speed_mps - target_speed_mps)
                <= self.reset_reference_speed_tolerance_mps
            ):
                return latest
        observed = "none" if latest_speed_mps is None else f"{latest_speed_mps:.3f} m/s"
        raise LiveBackendTimeoutError(
            "reference-speed reset was acknowledged but the requested "
            f"{target_speed_mps:.3f} m/s was not observed; latest={observed}"
        )

    def _wait_for_reference_speed_slip_stability(
        self,
        snapshot: AssettoCorsaTelemetry,
        slip_packet: SlipBridgePacket,
        *,
        target_speed_mps: float,
    ) -> tuple[AssettoCorsaTelemetry, SlipBridgePacket]:
        """Reject the CSP wheel-speed transient before exposing BC features."""

        deadline = self._monotonic_clock() + self.reset_timeout_s
        stable_frames = 0
        latest_max_abs_slip = self._maximum_abs_slip_ratio(snapshot, slip_packet)
        latest_speed_mps = snapshot.physics.speed_kmh / 3.6
        first_speed_mps = latest_speed_mps
        first_max_abs_slip = latest_max_abs_slip
        peak_max_abs_slip = latest_max_abs_slip
        while True:
            speed_is_valid = (
                abs(latest_speed_mps - target_speed_mps) <= self.reset_reference_speed_tolerance_mps
            )
            if speed_is_valid and latest_max_abs_slip <= self.reset_max_abs_slip_ratio + 1e-6:
                stable_frames += 1
            else:
                stable_frames = 0
            if stable_frames >= self.reset_slip_stable_frames:
                return snapshot, slip_packet

            remaining = deadline - self._monotonic_clock()
            if remaining <= 0.0:
                position_xy_m = (
                    snapshot.graphics.car_coordinates_m[2],
                    snapshot.graphics.car_coordinates_m[0],
                )
                heading_rad = _wrapped_angle(-snapshot.physics.heading_rad)
                raise LiveBackendTimeoutError(
                    "reference-speed reset did not reach stable wheel slip; "
                    f"first_speed={first_speed_mps:.3f} m/s, "
                    f"first_max_abs_slip_ratio={first_max_abs_slip:.3f}, "
                    f"latest_speed={latest_speed_mps:.3f} m/s, "
                    f"latest_max_abs_slip_ratio={latest_max_abs_slip:.3f}, "
                    f"peak_max_abs_slip_ratio={peak_max_abs_slip:.3f}, "
                    f"latest_position_xy=({position_xy_m[0]:.3f}, "
                    f"{position_xy_m[1]:.3f}) m, "
                    f"latest_heading={heading_rad:.4f} rad, "
                    f"required<={self.reset_max_abs_slip_ratio:.3f} for "
                    f"{self.reset_slip_stable_frames} frames"
                )
            try:
                snapshot = self._wait_for_fresh_frame(
                    after_physics_packet_id=snapshot.physics.packet_id,
                    after_graphics_packet_id=snapshot.graphics.packet_id,
                    not_before_s=self._monotonic_clock(),
                    timeout_s=max(
                        min(remaining, self.frame_timeout_s),
                        self.poll_interval_s,
                    ),
                )
            except LiveBackendTimeoutError:
                # CSP can briefly suspend both shared-memory pages while the
                # explicit pose is settled. Honour the larger reset timeout.
                continue
            try:
                slip_packet = self.slip_reader.receive(
                    timeout_s=max(
                        min(remaining, self.frame_timeout_s),
                        self.poll_interval_s,
                    )
                )
            except SlipBridgeTimeoutError:
                # Treat a transient file-bridge gap like a shared-memory gap;
                # the outer reset deadline remains the authoritative bound.
                continue
            latest_speed_mps = snapshot.physics.speed_kmh / 3.6
            latest_max_abs_slip = self._maximum_abs_slip_ratio(snapshot, slip_packet)
            peak_max_abs_slip = max(peak_max_abs_slip, latest_max_abs_slip)

    def _validate_reference_pose_reset(
        self,
        snapshot: AssettoCorsaTelemetry,
        reference: ReferenceTrajectory,
    ) -> tuple[float, float]:
        """Require the moving reset frame to remain aligned with the reference."""

        position_xy_m = np.asarray(
            [
                snapshot.graphics.car_coordinates_m[2],
                snapshot.graphics.car_coordinates_m[0],
            ],
            dtype=np.float32,
        )
        progress_m = float(reference.project_world_positions(position_xy_m[None, :])[0])
        reference_position, reference_heading = reference.world_pose_at(progress_m)
        displacement = position_xy_m - reference_position
        lateral_error_m = float(
            -displacement[0] * math.sin(reference_heading)
            + displacement[1] * math.cos(reference_heading)
        )
        vehicle_heading_rad = _wrapped_angle(-snapshot.physics.heading_rad)
        heading_error_rad = _wrapped_angle(vehicle_heading_rad - reference_heading)
        lateral_alignment_error_m = lateral_error_m - self.reset_reference_lateral_offset_m
        if abs(lateral_alignment_error_m) > self.reset_position_tolerance:
            raise _ReferencePoseResetError(
                "reference-pose reset was acknowledged but lateral alignment "
                f"is {lateral_error_m:.3f} m for target "
                f"{self.reset_reference_lateral_offset_m:.3f} m; tolerance="
                f"{self.reset_position_tolerance:.3f} m"
            )
        if abs(heading_error_rad) > self.reset_reference_heading_tolerance_rad:
            raise _ReferencePoseResetError(
                "reference-pose reset was acknowledged but heading alignment "
                f"is {heading_error_rad:.4f} rad; tolerance="
                f"{self.reset_reference_heading_tolerance_rad:.4f} rad"
            )
        return lateral_error_m, heading_error_rad

    @staticmethod
    def _maximum_abs_slip_ratio(
        snapshot: AssettoCorsaTelemetry,
        slip_packet: SlipBridgePacket,
    ) -> float:
        raw = vanilla_ac_slip_features(slip_packet)
        slip = stabilise_low_speed_slip(
            snapshot.physics.speed_kmh / 3.6,
            np.asarray(
                [
                    raw.front_slip_ratio,
                    raw.rear_slip_ratio,
                    raw.front_sideslip_angle_rad,
                    raw.rear_sideslip_angle_rad,
                ],
                dtype=np.float32,
            ),
        )
        return float(np.max(np.abs(slip[:2])))

    @staticmethod
    def _project_reference_progress(
        snapshot: AssettoCorsaTelemetry,
        reference: ReferenceTrajectory,
    ) -> float:
        position_xy_m = np.asarray(
            [
                snapshot.graphics.car_coordinates_m[2],
                snapshot.graphics.car_coordinates_m[0],
            ],
            dtype=np.float32,
        )
        return float(reference.project_world_positions(position_xy_m[None, :])[0])

    def _wait_for_fresh_frame(
        self,
        *,
        after_physics_packet_id: int,
        after_graphics_packet_id: int,
        not_before_s: float,
        timeout_s: float,
    ) -> AssettoCorsaTelemetry:
        """Wait for a live snapshot with fresh physics and graphics pages.

        Assetto Corsa publishes both dynamic pages independently.  Accepting a
        new physics packet together with an old graphics packet can repeat an
        obsolete world pose for many control steps.  Requiring both packet IDs
        to advance keeps position, lap phase, and tyre state temporally usable.
        """

        deadline = self._monotonic_clock() + timeout_s
        last_status: int | None = None
        last_physics_packet = after_physics_packet_id
        last_graphics_packet = after_graphics_packet_id
        while True:
            snapshot = self.telemetry_reader.read()
            last_status = snapshot.graphics.status
            last_physics_packet = snapshot.physics.packet_id
            last_graphics_packet = snapshot.graphics.packet_id
            now = self._monotonic_clock()
            if (
                now >= not_before_s
                and snapshot.physics.packet_id != after_physics_packet_id
                and snapshot.graphics.packet_id != after_graphics_packet_id
                and snapshot.graphics.status == ASSETTO_CORSA_LIVE_STATUS
            ):
                return snapshot
            remaining = deadline - now
            if remaining <= 0:
                raise LiveBackendTimeoutError(
                    "no fresh synchronised live telemetry frame before timeout "
                    f"(last_status={last_status}, "
                    f"last_physics_packet={last_physics_packet}, "
                    f"last_graphics_packet={last_graphics_packet})"
                )
            self._sleep(min(self.poll_interval_s, remaining))

    def _validate_session(
        self,
        snapshot: AssettoCorsaTelemetry,
        reference: ReferenceTrajectory,
    ) -> None:
        static = snapshot.static
        track_id = static.track
        if static.track_configuration:
            track_id += f"-{static.track_configuration}"
        mismatches: list[str] = []
        if static.car_model != self.expected_car_id:
            mismatches.append(f"car={static.car_model!r}, expected {self.expected_car_id!r}")
        if track_id != self.expected_track_id:
            mismatches.append(f"track={track_id!r}, expected {self.expected_track_id!r}")
        length_tolerance_m = max(1.0, 0.001 * reference.track_length_m)
        if abs(static.track_spline_length_m - reference.track_length_m) > length_tolerance_m:
            mismatches.append(
                "track spline length="
                f"{static.track_spline_length_m:.3f} m, expected {reference.track_length_m:.3f} m"
            )
        if snapshot.graphics.status != ASSETTO_CORSA_LIVE_STATUS:
            mismatches.append(f"graphics status={snapshot.graphics.status}, expected live status 2")
        if mismatches:
            raise LiveSessionMismatchError("; ".join(mismatches))

    def _build_frame(
        self,
        snapshot: AssettoCorsaTelemetry,
        slip_packet: SlipBridgePacket,
    ) -> TelemetryFrame:
        if self._reference is None:
            raise RuntimeError("live backend has no active reference")
        reference = self._reference
        physics = snapshot.physics
        graphics = snapshot.graphics
        # AssettoCorsaGym's public world convention is (AC z, AC x). Vertical
        # AC y is discarded. Its yaw convention is -physics.heading.
        position_xy_m = np.asarray(
            [graphics.car_coordinates_m[2], graphics.car_coordinates_m[0]],
            dtype=np.float32,
        )
        vehicle_heading_rad = _wrapped_angle(-physics.heading_rad)
        progress_m = float(reference.project_world_positions(position_xy_m[None, :])[0])
        reference_position, reference_heading = reference.world_pose_at(progress_m)
        displacement = position_xy_m - reference_position
        lateral_offset_m = float(
            -displacement[0] * math.sin(reference_heading)
            + displacement[1] * math.cos(reference_heading)
        )
        heading_error_rad = _wrapped_angle(vehicle_heading_rad - reference_heading)
        left, right = self.track_geometry.preview_ego(
            progress_m,
            position_xy_m,
            vehicle_heading_rad,
        )
        speed_mps = physics.speed_kmh / 3.6
        raw_slip = vanilla_ac_slip_features(slip_packet)
        slip = stabilise_low_speed_slip(
            speed_mps,
            np.asarray(
                [
                    raw_slip.front_slip_ratio,
                    raw_slip.rear_slip_ratio,
                    raw_slip.front_sideslip_angle_rad,
                    raw_slip.rear_sideslip_angle_rad,
                ],
                dtype=np.float32,
            ),
        )
        simulation_time_s = max(
            0.0,
            snapshot.monotonic_time_s - self._episode_origin_monotonic_s,
        )
        return TelemetryFrame(
            vehicle_state=VehicleState(
                speed_mps=speed_mps,
                acceleration_x_mps2=physics.acceleration_g[0] * STANDARD_GRAVITY_MPS2,
                acceleration_z_mps2=physics.acceleration_g[2] * STANDARD_GRAVITY_MPS2,
                front_slip_ratio=float(slip[0]),
                rear_slip_ratio=float(slip[1]),
                front_sideslip_angle_rad=float(slip[2]),
                rear_sideslip_angle_rad=float(slip[3]),
            ),
            track_preview=TrackPreview(left_boundary_xz_m=left, right_boundary_xz_m=right),
            reference_line_preview=ReferenceLinePreview(
                reference.ego_path_preview(
                    progress_m,
                    position_xy_m,
                    vehicle_heading_rad,
                )
            ),
            progress_m=progress_m,
            lateral_offset_m=lateral_offset_m,
            heading_error_rad=heading_error_rad,
            vehicle_heading_rad=vehicle_heading_rad,
            position_xy_m=position_xy_m,
            simulation_time_s=simulation_time_s,
        )

    def _termination_reason(
        self,
        snapshot: AssettoCorsaTelemetry,
        frame: TelemetryFrame,
        collision_detected: bool,
        body_sideslip_angle_rad: float,
    ) -> TerminationReason | None:
        if collision_detected:
            return TerminationReason.CRASH
        if snapshot.graphics.completed_laps > self._episode_start_completed_laps:
            return TerminationReason.LAP_COMPLETE
        if snapshot.physics.number_of_tyres_out > 0:
            return TerminationReason.OFF_TRACK
        if self._stall_termination_due() or self._progress_stall_termination_due():
            return TerminationReason.STALLED
        if (
            frame.vehicle_state.speed_mps >= self.minimum_body_sideslip_termination_speed_mps
            and abs(body_sideslip_angle_rad) > self.critical_body_sideslip_angle_rad
        ):
            return TerminationReason.EXCESSIVE_BODY_SIDESLIP
        return None

    def _stall_termination_due(self) -> bool:
        if self._stall_detection_armed:
            return self._consecutive_low_speed_steps >= self.stall_debounce_steps
        return self._step_count >= self.stall_startup_grace_steps

    def _update_stall_detection(self, speed_mps: float) -> None:
        """Arm after driving off and debounce subsequent low-speed frames."""

        if speed_mps >= self.minimum_speed_mps:
            self._stall_detection_armed = True
            self._consecutive_low_speed_steps = 0
        elif self._stall_detection_armed:
            self._consecutive_low_speed_steps += 1

    def _update_progress_stall_detection(self, progress_m: float) -> None:
        """Track net reference progress over the configured rolling window."""

        self._progress_stall_progress_m.append(progress_m)
        self._progress_stall_window_progress_m = None
        if len(self._progress_stall_progress_m) < self.progress_stall_window_steps + 1:
            return
        if self._reference is None:
            raise RuntimeError("progress-stall detection requires an active reference")
        self._progress_stall_window_progress_m = wrapped_progress_delta(
            self._progress_stall_progress_m[0],
            self._progress_stall_progress_m[-1],
            self._reference.track_length_m,
        )

    def _progress_stall_termination_due(self) -> bool:
        return (
            self.termination_method_variant
            == UNRECOVERABLE_PROGRESS_STALL_TERMINATION_METHOD_VARIANT
            and self._progress_stall_window_progress_m is not None
            and self._progress_stall_window_progress_m < self.progress_stall_minimum_progress_m
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("live backend is closed")

    def close(self) -> None:
        """Neutralise controls and close every transport idempotently."""

        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for resource in (
            self.control_writer,
            self.reset_controller,
            self.collision_reader,
            self.slip_reader,
            self.telemetry_reader,
        ):
            try:
                resource.close()
            except BaseException as error:  # pragma: no cover - best-effort teardown
                errors.append(error)
        self._reference = None
        if errors:
            raise LiveBackendError(f"failed to close {len(errors)} live backend resource(s)")
