from __future__ import annotations

import json
from pathlib import Path

import pytest

from assetto_corsa_rl.ac_integration.diagnostics import (
    DiagnosticCapture,
    capture_telemetry,
    summarise_capture,
    write_diagnostic_artifacts,
)
from assetto_corsa_rl.ac_integration.shared_memory import (
    AssettoCorsaTelemetry,
    GraphicsPage,
    PhysicsPage,
    StaticPage,
)


def _snapshot(
    packet_id: int,
    timestamp_s: float,
    *,
    speed_kmh: float = 36.0,
    status: int = 2,
    gas: float = 0.2,
) -> AssettoCorsaTelemetry:
    return AssettoCorsaTelemetry(
        monotonic_time_s=timestamp_s,
        physics=PhysicsPage(
            packet_id=packet_id,
            gas=gas,
            brake=0.0,
            clutch=0.0,
            gear=3,
            rpms=4000,
            steer_angle=0.0,
            speed_kmh=speed_kmh,
            velocity=(0.0, 0.0, speed_kmh / 3.6),
            acceleration_g=(0.0, 0.0, 0.0),
            wheel_slip=(0.0, 0.0, 0.0, 0.0),
            wheel_load_n=(1000.0, 1000.0, 1000.0, 1000.0),
            wheel_angular_speed_rad_s=(30.0, 30.0, 30.0, 30.0),
            heading_rad=0.0,
            local_angular_velocity_rad_s=(0.0, 0.0, 0.0),
            local_velocity_mps=(0.0, 0.0, speed_kmh / 3.6),
            tyre_contact_heading=((0.0, 0.0, 1.0),) * 4,
            number_of_tyres_out=0,
            brake_bias=0.6,
        ),
        graphics=GraphicsPage(
            packet_id=packet_id,
            status=status,
            session=3,
            completed_laps=0,
            current_lap_time_ms=packet_id * 10,
            last_lap_time_ms=0,
            distance_traveled_m=float(packet_id),
            is_in_pit=False,
            normalized_car_position=0.1,
            car_coordinates_m=(float(packet_id), 0.0, 0.0),
            is_in_pit_lane=False,
            surface_grip=1.0,
        ),
        static=StaticPage(
            shared_memory_version="1.1",
            assetto_corsa_version="1.16.4",
            car_model="ks_bmw_z4_gt3",
            track="monza",
            track_configuration="",
            track_spline_length_m=5793.0,
            tyre_radius_m=(0.31, 0.31, 0.33, 0.33),
        ),
    )


def test_summary_infers_forward_axis_and_marks_bridge_features() -> None:
    snapshots = tuple(_snapshot(index, index / 100.0) for index in range(1, 6))
    summary = summarise_capture(
        DiagnosticCapture(
            snapshots=snapshots,
            poll_count=5,
            elapsed_s=0.05,
            requested_duration_s=0.05,
            poll_frequency_hz=100.0,
        )
    )
    assert summary.is_usable_transport
    assert summary.inferred_forward_local_velocity_axis == 2
    assert summary.inferred_forward_local_velocity_sign == 1
    assert summary.forward_axis_mean_error_kmh == pytest.approx(0.0)
    assert {issue.code for issue in summary.issues} >= {
        "slip_ratio_bridge_required",
        "side_slip_bridge_required",
    }


def test_summary_reports_invalid_transport_and_writes_artifacts(tmp_path: Path) -> None:
    capture = DiagnosticCapture(
        snapshots=(_snapshot(1, 0.0, status=3, gas=2.0),),
        poll_count=10,
        elapsed_s=1.0,
        requested_duration_s=1.0,
        poll_frequency_hz=10.0,
    )
    summary = summarise_capture(capture)
    assert not summary.is_usable_transport
    codes = {issue.code for issue in summary.issues}
    assert {"control_range", "session_not_live", "stale_physics_page"} <= codes
    write_diagnostic_artifacts(tmp_path, capture, summary)
    assert (tmp_path / "telemetry.csv").read_text(encoding="utf-8").count("\n") == 2
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["car_model"] == "ks_bmw_z4_gt3"


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += duration


class _FakeReader:
    def __init__(self, clock: _FakeClock) -> None:
        self.clock = clock
        self.calls = 0

    def read(self) -> AssettoCorsaTelemetry:
        self.calls += 1
        packet_id = (self.calls + 1) // 2
        return _snapshot(packet_id, self.clock())


def test_capture_deduplicates_physics_packets() -> None:
    clock = _FakeClock()
    capture = capture_telemetry(
        _FakeReader(clock),
        duration_s=0.05,
        poll_frequency_hz=100.0,
        clock=clock,
        sleep=clock.sleep,
    )
    assert capture.poll_count == 5
    assert len(capture.snapshots) == 3
    with pytest.raises(ValueError, match="positive"):
        capture_telemetry(_FakeReader(clock), duration_s=0.0, poll_frequency_hz=1.0)
    with pytest.raises(ValueError, match="does not contain"):
        summarise_capture(
            DiagnosticCapture(
                snapshots=(),
                poll_count=0,
                elapsed_s=0.0,
                requested_duration_s=1.0,
                poll_frequency_hz=1.0,
            )
        )
