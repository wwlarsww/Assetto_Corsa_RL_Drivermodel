from __future__ import annotations

from collections.abc import Sequence

from assetto_corsa_rl.ac_integration.diagnostics import capture_telemetry
from assetto_corsa_rl.ac_integration.live_backend import AssettoCorsaLiveBackend
from assetto_corsa_rl.ac_integration.shared_memory import (
    AssettoCorsaTelemetry,
    GraphicsPage,
    PhysicsPage,
    StaticPage,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class _Reader:
    def __init__(self, frames: Sequence[AssettoCorsaTelemetry]) -> None:
        self.frames = list(frames)
        self.index = 0

    def read(self) -> AssettoCorsaTelemetry:
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame


def _telemetry(
    physics_packet_id: int,
    graphics_packet_id: int,
) -> AssettoCorsaTelemetry:
    zero3 = (0.0, 0.0, 0.0)
    zero4 = (0.0, 0.0, 0.0, 0.0)
    return AssettoCorsaTelemetry(
        monotonic_time_s=physics_packet_id / 100.0,
        physics=PhysicsPage(
            packet_id=physics_packet_id,
            gas=0.0,
            brake=0.0,
            clutch=0.0,
            gear=1,
            rpms=1000,
            steer_angle=0.0,
            speed_kmh=36.0,
            velocity=(0.0, 0.0, 10.0),
            acceleration_g=zero3,
            wheel_slip=zero4,
            wheel_load_n=zero4,
            wheel_angular_speed_rad_s=zero4,
            heading_rad=0.0,
            local_angular_velocity_rad_s=zero3,
            local_velocity_mps=(0.0, 0.0, 10.0),
            tyre_contact_heading=(zero3, zero3, zero3, zero3),
            number_of_tyres_out=0,
            brake_bias=0.5,
        ),
        graphics=GraphicsPage(
            packet_id=graphics_packet_id,
            status=2,
            session=4,
            completed_laps=0,
            current_lap_time_ms=0,
            last_lap_time_ms=0,
            distance_traveled_m=0.0,
            is_in_pit=False,
            normalized_car_position=0.1,
            car_coordinates_m=(1.0, 0.0, 2.0),
            is_in_pit_lane=False,
            surface_grip=1.0,
        ),
        static=StaticPage(
            shared_memory_version="1.1",
            assetto_corsa_version="1.16.4",
            car_model="test_car",
            track="test_track",
            track_configuration="",
            track_spline_length_m=1000.0,
            tyre_radius_m=(0.3, 0.3, 0.3, 0.3),
        ),
    )


def test_live_backend_waits_for_fresh_physics_and_graphics() -> None:
    clock = _Clock()
    backend = object.__new__(AssettoCorsaLiveBackend)
    backend.telemetry_reader = _Reader(
        [
            _telemetry(2, 10),
            _telemetry(3, 10),
            _telemetry(4, 11),
        ]
    )
    backend._monotonic_clock = clock
    backend._sleep = clock.sleep
    backend.poll_interval_s = 0.001

    snapshot = backend._wait_for_fresh_frame(
        after_physics_packet_id=1,
        after_graphics_packet_id=10,
        not_before_s=0.0,
        timeout_s=0.1,
    )

    assert snapshot.physics.packet_id == 4
    assert snapshot.graphics.packet_id == 11


def test_diagnostic_capture_skips_stale_graphics_snapshots() -> None:
    clock = _Clock()
    capture = capture_telemetry(
        _Reader(
            [
                _telemetry(1, 10),
                _telemetry(2, 10),
                _telemetry(3, 11),
                _telemetry(4, 12),
            ]
        ),
        duration_s=0.04,
        poll_frequency_hz=100.0,
        clock=clock,
        sleep=clock.sleep,
    )

    packet_pairs = [
        (snapshot.physics.packet_id, snapshot.graphics.packet_id) for snapshot in capture.snapshots
    ]
    assert packet_pairs == [(1, 10), (3, 11), (4, 12)]
