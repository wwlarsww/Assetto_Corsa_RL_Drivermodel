from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pytest

from assetto_corsa_rl.ac_integration.collision_bridge import CollisionState
from assetto_corsa_rl.ac_integration.live_backend import (
    AssettoCorsaLiveBackend,
    LiveBackendError,
    LiveBackendTimeoutError,
    LiveSessionMismatchError,
)
from assetto_corsa_rl.ac_integration.reset_bridge import (
    ResetAcknowledgement,
    ResetBridgeStatus,
)
from assetto_corsa_rl.ac_integration.shared_memory import (
    AssettoCorsaTelemetry,
    GraphicsPage,
    PhysicsPage,
    StaticPage,
)
from assetto_corsa_rl.ac_integration.slip_bridge import (
    SlipBridgePacket,
    SlipBridgeTimeoutError,
)
from assetto_corsa_rl.config import (
    PRE_ACTION_PROGRESS_BASELINE_METHOD_VARIANT,
    TERMINATION_METHOD_VARIANT,
    UNRECOVERABLE_PROGRESS_STALL_TERMINATION_METHOD_VARIANT,
)
from assetto_corsa_rl.features.track_geometry import TrackGeometry
from assetto_corsa_rl.references.fixed_reference import ReferenceTrajectory
from assetto_corsa_rl.types import TerminationReason


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class _TelemetryReader:
    def __init__(self, frames: Sequence[AssettoCorsaTelemetry]) -> None:
        self.frames = list(frames)
        self.index = 0
        self.closed = False

    def read(self) -> AssettoCorsaTelemetry:
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame

    def close(self) -> None:
        self.closed = True


class _SlipReader:
    def __init__(
        self,
        slip_ratio: tuple[float, float, float, float] = (0.1, 0.3, -0.2, 0.0),
        slip_ratio_sequence: Sequence[tuple[float, float, float, float]] | None = None,
    ) -> None:
        self.sequence = 0
        self.closed = False
        self.slip_ratio = slip_ratio
        self.slip_ratio_sequence = list(slip_ratio_sequence or ())

    def receive(self, *, timeout_s: float) -> SlipBridgePacket:
        assert timeout_s > 0
        slip_ratio = (
            self.slip_ratio_sequence[min(self.sequence, len(self.slip_ratio_sequence) - 1)]
            if self.slip_ratio_sequence
            else self.slip_ratio
        )
        packet = SlipBridgePacket(
            sequence=self.sequence,
            source_time_s=10.0 + self.sequence,
            received_monotonic_s=20.0 + self.sequence,
            slip_ratio=slip_ratio,
            slip_angle_raw=(1.0, 3.0, -2.0, 0.0),
        )
        self.sequence += 1
        return packet

    def close(self) -> None:
        self.closed = True


class _TransientTimeoutSlipReader(_SlipReader):
    def __init__(self) -> None:
        super().__init__(
            slip_ratio_sequence=[
                (0.0, 0.0, -0.7, -0.7),
                (0.0, 0.0, 0.1, 0.1),
            ]
        )
        self.calls = 0

    def receive(self, *, timeout_s: float) -> SlipBridgePacket:
        self.calls += 1
        if self.calls == 2:
            raise SlipBridgeTimeoutError("transient test gap")
        return super().receive(timeout_s=timeout_s)


class _CollisionReader:
    def __init__(
        self,
        *,
        collision_on_step: bool = False,
        clock: _Clock | None = None,
    ) -> None:
        self.collision_on_step = collision_on_step
        self.clock = clock
        self.synchronised_at_s: float | None = None
        self.closed = False

    @staticmethod
    def _state(sequence: int) -> CollisionState:
        return CollisionState("monza", sequence, 100 + sequence, 1.0)

    def synchronise(self, *, expected_track_id: str, timeout_s: float) -> CollisionState:
        assert expected_track_id == "monza"
        assert timeout_s > 0
        self.synchronised_at_s = self.clock() if self.clock is not None else None
        return self._state(4)

    def collision_since_synchronisation(
        self,
        *,
        timeout_s: float,
    ) -> tuple[bool, CollisionState]:
        assert timeout_s > 0
        return self.collision_on_step, self._state(5 if self.collision_on_step else 4)

    def close(self) -> None:
        self.closed = True


class _ControlWriter:
    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []
        self.neutral_count = 0
        self.closed = False

    def apply(self, absolute_action: npt.ArrayLike) -> None:
        self.actions.append(np.asarray(absolute_action, dtype=np.float32).copy())

    def neutral(self) -> None:
        self.neutral_count += 1

    def close(self) -> None:
        self.closed = True


class _ResetController:
    def __init__(self) -> None:
        self.timeouts: list[float] = []
        self.start_speeds_mps: list[float | None] = []
        self.start_positions_xy_m: list[tuple[float, float] | None] = []
        self.start_headings_rad: list[float | None] = []
        self.closed = False

    def request_reset(
        self,
        *,
        timeout_s: float,
        start_speed_mps: float | None = None,
        start_position_xy_m: tuple[float, float] | None = None,
        start_heading_rad: float | None = None,
    ) -> ResetAcknowledgement:
        self.timeouts.append(timeout_s)
        self.start_speeds_mps.append(start_speed_mps)
        self.start_positions_xy_m.append(start_position_xy_m)
        self.start_headings_rad.append(start_heading_rad)
        return ResetAcknowledgement(9, ResetBridgeStatus.SUCCESS, 100.0)

    def close(self) -> None:
        self.closed = True


def _physics(
    packet_id: int,
    *,
    speed_kmh: float = 36.0,
    local_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 10.0),
    number_of_tyres_out: int = 0,
) -> PhysicsPage:
    return PhysicsPage(
        packet_id=packet_id,
        gas=0.0,
        brake=0.0,
        clutch=0.0,
        gear=2,
        rpms=5000,
        steer_angle=0.0,
        speed_kmh=speed_kmh,
        velocity=(0.0, 0.0, 10.0),
        acceleration_g=(0.5, 0.0, 1.0),
        wheel_slip=(0.0, 0.0, 0.0, 0.0),
        wheel_load_n=(1000.0, 1000.0, 1000.0, 1000.0),
        wheel_angular_speed_rad_s=(1.0, 1.0, 1.0, 1.0),
        heading_rad=0.0,
        local_angular_velocity_rad_s=(0.0, 0.0, 0.0),
        local_velocity_mps=local_velocity_mps,
        tyre_contact_heading=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        number_of_tyres_out=number_of_tyres_out,
        brake_bias=0.6,
    )


def _graphics(packet_id: int, *, phase: float, completed_laps: int = 0) -> GraphicsPage:
    return GraphicsPage(
        packet_id=packet_id,
        status=2,
        session=4,
        completed_laps=completed_laps,
        current_lap_time_ms=0,
        last_lap_time_ms=0,
        distance_traveled_m=10.0,
        is_in_pit=False,
        normalized_car_position=phase,
        car_coordinates_m=(0.0, 0.0, 10.0),
        is_in_pit_lane=False,
        surface_grip=1.0,
    )


def _static() -> StaticPage:
    return StaticPage(
        shared_memory_version="1.1",
        assetto_corsa_version="1.16.4",
        car_model="bmw_z4_gt3",
        track="monza",
        track_configuration="",
        track_spline_length_m=1000.0,
        tyre_radius_m=(0.3, 0.3, 0.3, 0.3),
    )


def _telemetry(
    packet_id: int,
    *,
    phase: float,
    laps: int = 0,
    speed_kmh: float = 36.0,
    local_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 10.0),
    number_of_tyres_out: int = 0,
) -> AssettoCorsaTelemetry:
    return AssettoCorsaTelemetry(
        monotonic_time_s=packet_id * 0.1,
        physics=_physics(
            packet_id,
            speed_kmh=speed_kmh,
            local_velocity_mps=local_velocity_mps,
            number_of_tyres_out=number_of_tyres_out,
        ),
        graphics=_graphics(packet_id, phase=phase, completed_laps=laps),
        static=_static(),
    )


def _reference() -> ReferenceTrajectory:
    return ReferenceTrajectory.constant(track_length_m=1000.0, speed_mps=25.0)


def _geometry() -> TrackGeometry:
    distance = np.asarray([0.0, 333.0, 666.0], dtype=np.float32)
    centre_x = distance.copy()
    return TrackGeometry(
        distance_m=distance,
        track_length_m=1000.0,
        left_xy_m=np.column_stack((centre_x, np.full(3, 5.0))),
        right_xy_m=np.column_stack((centre_x, np.full(3, -5.0))),
    )


def _backend(
    frames: Sequence[AssettoCorsaTelemetry],
    *,
    collision_on_step: bool = False,
    reset_collision_quiescence_s: float = 0.0,
    reset_method_variant: str = "csp_hotlap_spawn_v1",
    slip_ratio: tuple[float, float, float, float] = (0.1, 0.3, -0.2, 0.0),
    slip_ratio_sequence: Sequence[tuple[float, float, float, float]] | None = None,
    step_timing_method_variant: str = "observation_to_post_hold_v1",
    reset_max_abs_slip_ratio: float = 0.2,
    reset_slip_stable_frames: int = 1,
    reset_reference_speed_max_attempts: int = 1,
    reset_reference_lateral_offset_m: float = 0.0,
    termination_method_variant: str = TERMINATION_METHOD_VARIANT,
    progress_stall_window_s: float = 1.0,
    progress_stall_minimum_progress_m: float = 2.0,
) -> tuple[
    AssettoCorsaLiveBackend,
    _ControlWriter,
    _ResetController,
]:
    clock = _Clock()
    control = _ControlWriter()
    reset = _ResetController()
    backend = AssettoCorsaLiveBackend(
        telemetry_reader=_TelemetryReader(frames),
        slip_reader=_SlipReader(slip_ratio, slip_ratio_sequence),
        collision_reader=_CollisionReader(collision_on_step=collision_on_step, clock=clock),
        control_writer=control,
        reset_controller=reset,
        track_geometry=_geometry(),
        control_frequency_hz=25.0,
        expected_car_id="bmw_z4_gt3",
        expected_track_id="monza",
        minimum_speed_mps=2.0,
        stall_startup_grace_steps=25,
        stall_debounce_steps=3,
        termination_method_variant=termination_method_variant,
        progress_stall_window_s=progress_stall_window_s,
        progress_stall_minimum_progress_m=progress_stall_minimum_progress_m,
        critical_body_sideslip_angle_deg=20.0,
        minimum_body_sideslip_termination_speed_mps=5.0,
        frame_timeout_s=0.5,
        reset_timeout_s=1.0,
        reset_position_tolerance=0.05,
        reset_reference_lateral_offset_m=reset_reference_lateral_offset_m,
        reset_reference_heading_tolerance_rad=0.05,
        reset_speed_threshold_mps=15.0,
        reset_reference_speed_tolerance_mps=0.5,
        reset_stable_frames=3,
        reset_max_abs_slip_ratio=reset_max_abs_slip_ratio,
        reset_slip_stable_frames=reset_slip_stable_frames,
        reset_reference_speed_max_attempts=reset_reference_speed_max_attempts,
        reset_collision_quiescence_s=reset_collision_quiescence_s,
        reset_method_variant=reset_method_variant,
        step_timing_method_variant=step_timing_method_variant,
        poll_interval_s=0.001,
        monotonic_clock=clock,
        sleep=clock.sleep,
    )
    return backend, control, reset


def test_live_backend_records_collision_baseline_after_reset_quiescence() -> None:
    backend, _control, _reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.01),
            _telemetry(3, phase=0.01),
            _telemetry(4, phase=0.01),
            _telemetry(5, phase=0.01),
            _telemetry(6, phase=0.01),
        ],
        reset_collision_quiescence_s=0.2,
    )

    backend.reset(seed=None, reference=_reference())

    collision_reader = backend.collision_reader
    assert isinstance(collision_reader, _CollisionReader)
    assert collision_reader.synchronised_at_s == pytest.approx(0.2)


def test_live_backend_acknowledges_reset_maps_frames_and_lap_completion() -> None:
    backend, control, reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.01),
            _telemetry(3, phase=0.01),
            _telemetry(4, phase=0.01),
            _telemetry(5, phase=0.02, laps=1),
        ]
    )
    initial = backend.reset(seed=7, reference=_reference())
    assert reset.timeouts == [1.0]
    assert initial.position_xy_m == pytest.approx([10.0, 0.0])
    assert initial.vehicle_state.speed_mps == pytest.approx(10.0)
    assert initial.vehicle_state.front_slip_ratio == pytest.approx(0.2)
    assert initial.track_preview.left_boundary_xz_m[0, 0] == pytest.approx(5.0)

    transition = backend.step(np.asarray([0.5, -0.25], dtype=np.float32))
    assert control.actions[0] == pytest.approx([0.5, -0.25])
    assert transition.termination_reason is TerminationReason.LAP_COMPLETE
    assert transition.diagnostics is not None
    assert transition.diagnostics["physics_packet_id"] == 5
    assert transition.diagnostics["reset_sequence"] == 9
    assert transition.diagnostics["simulator_gear"] == 2
    assert transition.diagnostics["simulator_rpms"] == 5000
    backend.close()
    assert control.closed


def test_live_backend_samples_progress_immediately_before_applying_action() -> None:
    backend, control, _reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.01),
            _telemetry(3, phase=0.01),
            _telemetry(4, phase=0.01),
            _telemetry(5, phase=0.015),
            _telemetry(6, phase=0.02),
        ],
        step_timing_method_variant=PRE_ACTION_PROGRESS_BASELINE_METHOD_VARIANT,
    )
    backend.reset(seed=None, reference=_reference())

    transition = backend.step(np.asarray([0.5, 0.25], dtype=np.float32))

    assert control.actions[-1] == pytest.approx([0.5, 0.25])
    assert transition.diagnostics is not None
    assert transition.diagnostics["step_timing_method_variant"] == (
        PRE_ACTION_PROGRESS_BASELINE_METHOD_VARIANT
    )
    assert transition.diagnostics["reward_progress_start_m"] is not None
    assert transition.diagnostics["observation_to_action_delay_s"] == pytest.approx(0.1)
    assert transition.diagnostics["action_hold_duration_s"] >= 1.0 / 25.0


def test_live_backend_primes_configured_action_after_safe_reset() -> None:
    backend, control, _reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.01),
            _telemetry(3, phase=0.01),
            _telemetry(4, phase=0.01),
        ]
    )

    backend.reset(
        seed=None,
        reference=_reference(),
        initial_action=np.asarray([1.0, 0.0], dtype=np.float32),
    )

    assert control.neutral_count == 1
    assert len(control.actions) == 1
    np.testing.assert_allclose(control.actions[0], [1.0, 0.0])


def test_live_backend_probes_hotlap_start_then_uses_human_mean_speed() -> None:
    backend, _control, reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(
                2,
                phase=0.85,
                speed_kmh=0.0,
                local_velocity_mps=(0.0, 0.0, 0.0),
            ),
            _telemetry(
                3,
                phase=0.85,
                speed_kmh=0.0,
                local_velocity_mps=(0.0, 0.0, 0.0),
            ),
            _telemetry(
                4,
                phase=0.85,
                speed_kmh=0.0,
                local_velocity_mps=(0.0, 0.0, 0.0),
            ),
            _telemetry(
                5,
                phase=0.85,
                speed_kmh=90.0,
                local_velocity_mps=(0.0, 0.0, 25.0),
            ),
        ],
        reset_method_variant="csp_hotlap_reference_speed_v1",
    )
    reference = ReferenceTrajectory.constant(
        track_length_m=1000.0,
        speed_mps=20.0,
        human_mean_speed_mps=25.0,
    )

    initial = backend.reset(seed=None, reference=reference)

    assert reset.start_speeds_mps == [0.0, 25.0]
    assert initial.progress_m == pytest.approx(10.0)
    assert initial.vehicle_state.speed_mps == pytest.approx(25.0)


def test_live_backend_requests_reference_pose_after_hotlap_probe() -> None:
    backend, _control, reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.85, speed_kmh=0.0),
            _telemetry(3, phase=0.85, speed_kmh=0.0),
            _telemetry(4, phase=0.85, speed_kmh=0.0),
            _telemetry(5, phase=0.85, speed_kmh=90.0),
        ],
        reset_method_variant="csp_reference_pose_speed_v2",
    )
    reference = ReferenceTrajectory.constant(
        track_length_m=1000.0,
        speed_mps=20.0,
        human_mean_speed_mps=25.0,
    )

    backend.reset(seed=None, reference=reference)

    assert reset.start_speeds_mps == [0.0, 25.0]
    assert reset.start_positions_xy_m[0] is None
    assert reset.start_positions_xy_m[1] == pytest.approx((10.0, 0.0))
    assert reset.start_headings_rad[1] == pytest.approx(0.0)
    assert backend._last_reset_reference_lateral_error_m == pytest.approx(0.0)
    assert backend._last_reset_reference_heading_error_rad == pytest.approx(0.0)


def test_live_backend_applies_and_validates_reference_lateral_start_offset() -> None:
    frames = [
        replace(
            _telemetry(packet_id, phase=0.85, speed_kmh=speed_kmh),
            graphics=replace(
                _telemetry(packet_id, phase=0.85).graphics,
                car_coordinates_m=(-1.0, 0.0, 10.0),
            ),
        )
        for packet_id, speed_kmh in ((1, 36.0), (2, 0.0), (3, 0.0), (4, 0.0), (5, 90.0))
    ]
    backend, _control, reset = _backend(
        frames,
        reset_method_variant="csp_reference_pose_speed_v2",
        reset_reference_lateral_offset_m=-1.0,
    )
    reference = ReferenceTrajectory.constant(
        track_length_m=1000.0,
        speed_mps=20.0,
        human_mean_speed_mps=25.0,
    )

    backend.reset(seed=None, reference=reference)

    assert reset.start_positions_xy_m[1] == pytest.approx((10.0, -1.0))
    assert backend._last_reset_reference_lateral_error_m == pytest.approx(-1.0)


def test_reference_pose_reset_rejects_lateral_misalignment() -> None:
    snapshot = _telemetry(1, phase=0.01)
    snapshot = replace(
        snapshot,
        graphics=replace(snapshot.graphics, car_coordinates_m=(0.1, 0.0, 10.0)),
    )
    backend, _control, _reset = _backend([snapshot])

    with pytest.raises(LiveBackendError, match="lateral alignment"):
        backend._validate_reference_pose_reset(snapshot, _reference())


def test_reference_pose_reset_rejects_heading_misalignment() -> None:
    snapshot = _telemetry(1, phase=0.01)
    snapshot = replace(
        snapshot,
        physics=replace(snapshot.physics, heading_rad=-0.1),
    )
    backend, _control, _reset = _backend([snapshot])

    with pytest.raises(LiveBackendError, match="heading alignment"):
        backend._validate_reference_pose_reset(snapshot, _reference())


def test_reference_speed_reset_waits_for_stable_bc_compatible_wheel_slip() -> None:
    backend, _control, _reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.85, speed_kmh=0.0),
            _telemetry(3, phase=0.85, speed_kmh=0.0),
            _telemetry(4, phase=0.85, speed_kmh=0.0),
            _telemetry(5, phase=0.85, speed_kmh=90.0),
            _telemetry(6, phase=0.851, speed_kmh=90.0),
            _telemetry(7, phase=0.852, speed_kmh=90.0),
        ],
        reset_method_variant="csp_hotlap_reference_speed_v1",
        slip_ratio_sequence=[
            (0.0, 0.0, -0.7, -0.7),
            (0.0, 0.0, 0.1, 0.1),
            (0.0, 0.0, 0.05, 0.05),
        ],
        reset_max_abs_slip_ratio=0.2,
        reset_slip_stable_frames=2,
    )
    reference = ReferenceTrajectory.constant(
        track_length_m=1000.0,
        speed_mps=20.0,
        human_mean_speed_mps=25.0,
    )

    initial = backend.reset(seed=None, reference=reference)

    assert abs(initial.vehicle_state.rear_slip_ratio) <= 0.2
    slip_reader = backend.slip_reader
    assert isinstance(slip_reader, _SlipReader)
    assert slip_reader.sequence == 3


def test_reference_speed_slip_wait_survives_one_frame_timeout() -> None:
    repeated = _telemetry(5, phase=0.85, speed_kmh=90.0)
    backend, _control, _reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.85, speed_kmh=0.0),
            _telemetry(3, phase=0.85, speed_kmh=0.0),
            _telemetry(4, phase=0.85, speed_kmh=0.0),
            repeated,
            *([repeated] * 510),
            _telemetry(6, phase=0.851, speed_kmh=90.0),
        ],
        reset_method_variant="csp_hotlap_reference_speed_v1",
        slip_ratio_sequence=[
            (0.0, 0.0, -0.7, -0.7),
            (0.0, 0.0, 0.1, 0.1),
        ],
    )
    reference = ReferenceTrajectory.constant(
        track_length_m=1000.0,
        speed_mps=20.0,
        human_mean_speed_mps=25.0,
    )

    initial = backend.reset(seed=None, reference=reference)

    assert abs(initial.vehicle_state.rear_slip_ratio) <= 0.2


def test_reference_speed_slip_wait_survives_one_bridge_timeout() -> None:
    backend, _control, _reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.85, speed_kmh=0.0),
            _telemetry(3, phase=0.85, speed_kmh=0.0),
            _telemetry(4, phase=0.85, speed_kmh=0.0),
            _telemetry(5, phase=0.85, speed_kmh=90.0),
            _telemetry(6, phase=0.851, speed_kmh=90.0),
            _telemetry(7, phase=0.852, speed_kmh=90.0),
        ],
        reset_method_variant="csp_hotlap_reference_speed_v1",
    )
    slip_reader = _TransientTimeoutSlipReader()
    backend.slip_reader = slip_reader
    reference = ReferenceTrajectory.constant(
        track_length_m=1000.0,
        speed_mps=20.0,
        human_mean_speed_mps=25.0,
    )

    initial = backend.reset(seed=None, reference=reference)

    assert abs(initial.vehicle_state.rear_slip_ratio) <= 0.2
    assert slip_reader.calls == 3


def test_reference_speed_reset_retries_a_fully_acknowledged_unsafe_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, control, reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.85, speed_kmh=0.0),
            _telemetry(3, phase=0.85, speed_kmh=0.0),
            _telemetry(4, phase=0.85, speed_kmh=0.0),
            _telemetry(5, phase=0.85, speed_kmh=90.0),
            _telemetry(6, phase=0.85, speed_kmh=0.0),
            _telemetry(7, phase=0.85, speed_kmh=90.0),
        ],
        reset_method_variant="csp_hotlap_reference_speed_v1",
        reset_reference_speed_max_attempts=2,
    )
    original_wait = backend._wait_for_reference_speed_slip_stability
    calls = 0

    def fail_once(
        snapshot: AssettoCorsaTelemetry,
        slip_packet: SlipBridgePacket,
        *,
        target_speed_mps: float,
    ) -> tuple[AssettoCorsaTelemetry, SlipBridgePacket]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LiveBackendTimeoutError("transient unsafe tyre settling")
        return original_wait(
            snapshot,
            slip_packet,
            target_speed_mps=target_speed_mps,
        )

    monkeypatch.setattr(backend, "_wait_for_reference_speed_slip_stability", fail_once)
    reference = ReferenceTrajectory.constant(
        track_length_m=1000.0,
        speed_mps=20.0,
        human_mean_speed_mps=25.0,
    )

    initial = backend.reset(seed=None, reference=reference)

    assert initial.vehicle_state.speed_mps == pytest.approx(25.0)
    assert reset.start_speeds_mps == [0.0, 25.0, 25.0]
    assert backend._last_reset_attempts == 2
    assert control.neutral_count == 2


def test_reference_pose_reset_retries_a_transient_alignment_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, control, reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.85, speed_kmh=0.0),
            _telemetry(3, phase=0.85, speed_kmh=0.0),
            _telemetry(4, phase=0.85, speed_kmh=0.0),
            _telemetry(5, phase=0.85, speed_kmh=90.0),
            _telemetry(6, phase=0.85, speed_kmh=0.0),
            _telemetry(7, phase=0.85, speed_kmh=90.0),
        ],
        reset_method_variant="csp_reference_pose_speed_v2",
        reset_reference_speed_max_attempts=2,
    )
    original_validate = backend._validate_reference_pose_reset
    calls = 0

    def fail_once(
        snapshot: AssettoCorsaTelemetry,
        reference: ReferenceTrajectory,
    ) -> tuple[float, float]:
        nonlocal calls
        calls += 1
        if calls == 1:
            misaligned = replace(
                snapshot,
                graphics=replace(
                    snapshot.graphics,
                    car_coordinates_m=(0.1, 0.0, 10.0),
                ),
            )
            return original_validate(misaligned, reference)
        return original_validate(snapshot, reference)

    monkeypatch.setattr(backend, "_validate_reference_pose_reset", fail_once)
    reference = ReferenceTrajectory.constant(
        track_length_m=1000.0,
        speed_mps=20.0,
        human_mean_speed_mps=25.0,
    )

    initial = backend.reset(seed=None, reference=reference)

    assert initial.vehicle_state.speed_mps == pytest.approx(25.0)
    assert reset.start_speeds_mps == [0.0, 25.0, 25.0]
    assert reset.start_positions_xy_m == [None, (10.0, 0.0), (10.0, 0.0)]
    assert backend._last_reset_attempts == 2
    assert backend._last_reset_reference_lateral_error_m == pytest.approx(0.0)
    assert backend._last_reset_reference_heading_error_rad == pytest.approx(0.0)
    assert control.neutral_count == 2


def test_reference_speed_reset_reports_exhausted_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, _control, reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.85, speed_kmh=0.0),
            _telemetry(3, phase=0.85, speed_kmh=0.0),
            _telemetry(4, phase=0.85, speed_kmh=0.0),
            _telemetry(5, phase=0.85, speed_kmh=90.0),
            _telemetry(6, phase=0.85, speed_kmh=0.0),
            _telemetry(7, phase=0.85, speed_kmh=90.0),
        ],
        reset_method_variant="csp_hotlap_reference_speed_v1",
        reset_reference_speed_max_attempts=2,
    )

    def always_fail(
        _snapshot: AssettoCorsaTelemetry,
        _slip_packet: SlipBridgePacket,
        *,
        target_speed_mps: float,
    ) -> tuple[AssettoCorsaTelemetry, SlipBridgePacket]:
        del target_speed_mps
        raise LiveBackendTimeoutError("unsafe tyre settling")

    monkeypatch.setattr(backend, "_wait_for_reference_speed_slip_stability", always_fail)
    reference = ReferenceTrajectory.constant(
        track_length_m=1000.0,
        speed_mps=20.0,
        human_mean_speed_mps=25.0,
    )

    with pytest.raises(LiveBackendTimeoutError, match="unsafe after 2 attempt"):
        backend.reset(seed=None, reference=reference)

    assert reset.start_speeds_mps == [0.0, 25.0, 25.0]


def test_live_backend_terminates_when_one_tyre_is_out() -> None:
    backend, _control, _reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.01),
            _telemetry(3, phase=0.01),
            _telemetry(4, phase=0.01),
            _telemetry(5, phase=0.02, number_of_tyres_out=1),
        ]
    )
    backend.reset(seed=None, reference=_reference())

    transition = backend.step(np.zeros(2, dtype=np.float32))

    assert transition.termination_reason is TerminationReason.OFF_TRACK
    assert transition.diagnostics is not None
    assert transition.diagnostics["number_of_tyres_out"] == 1


def test_live_backend_debounces_stall_after_successful_motion() -> None:
    backend, _control, _reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.01),
            _telemetry(3, phase=0.01),
            _telemetry(4, phase=0.01),
        ]
    )
    frame = backend.reset(seed=None, reference=_reference())
    snapshot = _telemetry(5, phase=0.02, speed_kmh=0.0)
    backend._update_stall_detection(10.0)

    for _ in range(2):
        backend._update_stall_detection(0.0)
        assert backend._termination_reason(snapshot, frame, False, 0.0) is None
    backend._update_stall_detection(0.0)

    assert backend._termination_reason(snapshot, frame, False, 0.0) is TerminationReason.STALLED


def test_live_backend_progress_stall_terminates_a_rolling_vehicle_after_window() -> None:
    backend, _control, _reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.01),
            _telemetry(3, phase=0.01),
            _telemetry(4, phase=0.01),
            *([_telemetry(5, phase=0.02, speed_kmh=36.0)] * 50),
            *([_telemetry(6, phase=0.02, speed_kmh=36.0)] * 50),
            *([_telemetry(7, phase=0.02, speed_kmh=36.0)] * 50),
        ],
        termination_method_variant=UNRECOVERABLE_PROGRESS_STALL_TERMINATION_METHOD_VARIANT,
        progress_stall_window_s=0.12,
        progress_stall_minimum_progress_m=2.0,
    )
    backend.reset(seed=None, reference=_reference())

    first = backend.step(np.zeros(2, dtype=np.float32))
    second = backend.step(np.zeros(2, dtype=np.float32))
    third = backend.step(np.zeros(2, dtype=np.float32))

    assert first.termination_reason is None
    assert second.termination_reason is None
    assert third.termination_reason is TerminationReason.STALLED
    assert third.diagnostics is not None
    assert third.diagnostics["progress_stall_window_steps"] == 3
    assert third.diagnostics["progress_stall_window_progress_m"] == pytest.approx(0.0)
    assert third.diagnostics["progress_stall_triggered"] is True


def test_live_backend_v1_does_not_apply_progress_stall() -> None:
    backend, _control, _reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.01),
            _telemetry(3, phase=0.01),
            _telemetry(4, phase=0.01),
            _telemetry(5, phase=0.02, speed_kmh=36.0),
        ],
        progress_stall_window_s=0.04,
        progress_stall_minimum_progress_m=2.0,
    )
    backend.reset(seed=None, reference=_reference())

    transition = backend.step(np.zeros(2, dtype=np.float32))

    assert transition.termination_reason is None
    assert transition.diagnostics is not None
    assert transition.diagnostics["progress_stall_triggered"] is False


def test_live_backend_ignores_longitudinal_tyre_slip_for_termination() -> None:
    backend, _control, _reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.01),
            _telemetry(3, phase=0.01),
            _telemetry(4, phase=0.01),
            _telemetry(5, phase=0.02),
        ],
        slip_ratio=(0.0, 0.0, 7.0, 8.0),
    )
    backend.reset(seed=None, reference=_reference())

    transition = backend.step(np.zeros(2, dtype=np.float32))

    assert transition.telemetry.vehicle_state.rear_slip_ratio == pytest.approx(7.5)
    assert transition.termination_reason is None


def test_live_backend_terminates_above_twenty_degree_body_sideslip() -> None:
    backend, _control, _reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.01),
            _telemetry(3, phase=0.01),
            _telemetry(4, phase=0.01),
            _telemetry(
                5,
                phase=0.02,
                local_velocity_mps=(10.0, 0.0, 10.0),
            ),
        ]
    )
    backend.reset(seed=None, reference=_reference())

    transition = backend.step(np.zeros(2, dtype=np.float32))

    assert transition.termination_reason is TerminationReason.EXCESSIVE_BODY_SIDESLIP
    assert transition.diagnostics is not None
    assert transition.diagnostics["body_sideslip_angle_deg"] == pytest.approx(45.0)


def test_live_backend_tolerates_a_short_csp_reset_frame_pause() -> None:
    stalled = _telemetry(1, phase=0.5)
    backend, _control, reset = _backend(
        [
            *([stalled] * 602),
            _telemetry(2, phase=0.85),
            _telemetry(3, phase=0.85),
            _telemetry(4, phase=0.85),
        ]
    )

    initial = backend.reset(seed=None, reference=_reference())

    assert reset.timeouts == [1.0]
    assert initial.vehicle_state.speed_mps == pytest.approx(10.0)


def test_live_backend_rejects_wrong_session_before_reset_command() -> None:
    wrong = _telemetry(1, phase=0.0)
    wrong = replace(wrong, static=replace(wrong.static, car_model="wrong_car"))
    backend, control, reset = _backend([wrong])
    with pytest.raises(LiveSessionMismatchError, match="wrong_car"):
        backend.reset(seed=None, reference=_reference())
    assert reset.timeouts == []
    assert control.neutral_count >= 2


def test_live_backend_terminates_on_latched_csp_collision() -> None:
    backend, _control, _reset = _backend(
        [
            _telemetry(1, phase=0.5),
            _telemetry(2, phase=0.01),
            _telemetry(3, phase=0.01),
            _telemetry(4, phase=0.01),
            _telemetry(5, phase=0.02),
        ],
        collision_on_step=True,
    )
    backend.reset(seed=None, reference=_reference())
    transition = backend.step(np.zeros(2, dtype=np.float32))
    assert transition.termination_reason is TerminationReason.CRASH
    assert transition.diagnostics is not None
    assert transition.diagnostics["collision_sequence"] == 5
