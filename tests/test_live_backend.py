from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pytest

from assetto_corsa_rl.ac_integration.collision_bridge import CollisionState
from assetto_corsa_rl.ac_integration.live_backend import (
    AssettoCorsaLiveBackend,
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
from assetto_corsa_rl.ac_integration.slip_bridge import SlipBridgePacket
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
    ) -> None:
        self.sequence = 0
        self.closed = False
        self.slip_ratio = slip_ratio

    def receive(self, *, timeout_s: float) -> SlipBridgePacket:
        assert timeout_s > 0
        packet = SlipBridgePacket(
            sequence=self.sequence,
            source_time_s=10.0 + self.sequence,
            received_monotonic_s=20.0 + self.sequence,
            slip_ratio=self.slip_ratio,
            slip_angle_raw=(1.0, 3.0, -2.0, 0.0),
        )
        self.sequence += 1
        return packet

    def close(self) -> None:
        self.closed = True


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
        self.closed = False

    def request_reset(self, *, timeout_s: float) -> ResetAcknowledgement:
        self.timeouts.append(timeout_s)
        return ResetAcknowledgement(9, ResetBridgeStatus.SUCCESS, 100.0)

    def close(self) -> None:
        self.closed = True


def _physics(
    packet_id: int,
    *,
    speed_kmh: float = 36.0,
    local_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 10.0),
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
        number_of_tyres_out=0,
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
) -> AssettoCorsaTelemetry:
    return AssettoCorsaTelemetry(
        monotonic_time_s=packet_id * 0.1,
        physics=_physics(
            packet_id,
            speed_kmh=speed_kmh,
            local_velocity_mps=local_velocity_mps,
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
    slip_ratio: tuple[float, float, float, float] = (0.1, 0.3, -0.2, 0.0),
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
        slip_reader=_SlipReader(slip_ratio),
        collision_reader=_CollisionReader(collision_on_step=collision_on_step, clock=clock),
        control_writer=control,
        reset_controller=reset,
        track_geometry=_geometry(),
        control_frequency_hz=25.0,
        expected_car_id="bmw_z4_gt3",
        expected_track_id="monza",
        minimum_speed_mps=2.0,
        minimum_speed_grace_steps=10,
        critical_body_sideslip_angle_deg=20.0,
        minimum_body_sideslip_termination_speed_mps=5.0,
        frame_timeout_s=0.5,
        reset_timeout_s=1.0,
        reset_position_tolerance=0.05,
        reset_speed_threshold_mps=15.0,
        reset_stable_frames=3,
        reset_collision_quiescence_s=reset_collision_quiescence_s,
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
