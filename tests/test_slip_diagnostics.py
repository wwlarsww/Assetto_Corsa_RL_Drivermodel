from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

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
from assetto_corsa_rl.ac_integration.slip_diagnostics import (
    SlipDiagnosticCapture,
    capture_slip_telemetry,
    summarise_slip_capture,
    write_slip_diagnostic_artifacts,
)


def _packet(
    sequence: int,
    time_s: float,
    *,
    ratio: tuple[float, float, float, float] = (0.1, 0.2, -0.1, -0.2),
    angle: tuple[float, float, float, float] = (0.01, 0.02, -0.01, -0.02),
) -> SlipBridgePacket:
    return SlipBridgePacket(
        sequence=sequence,
        source_time_s=1_000.0 + time_s,
        received_monotonic_s=time_s,
        slip_ratio=ratio,
        slip_angle_raw=angle,
    )


def _shared_snapshot(packet_id: int) -> AssettoCorsaTelemetry:
    return AssettoCorsaTelemetry(
        monotonic_time_s=float(packet_id),
        physics=PhysicsPage(
            packet_id=packet_id,
            gas=0.6,
            brake=0.1,
            clutch=0.0,
            gear=3,
            rpms=4000,
            steer_angle=0.2,
            speed_kmh=72.0,
            velocity=(1.0, 0.0, 20.0),
            acceleration_g=(0.4, 0.0, 0.2),
            wheel_slip=(1.0, 2.0, 3.0, 4.0),
            wheel_load_n=(1000.0, 1000.0, 1000.0, 1000.0),
            wheel_angular_speed_rad_s=(50.0, 50.0, 50.0, 50.0),
            heading_rad=0.1,
            local_angular_velocity_rad_s=(0.0, 0.1, 0.0),
            local_velocity_mps=(1.0, 0.0, 20.0),
            tyre_contact_heading=(
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 1.0),
            ),
            number_of_tyres_out=0,
            brake_bias=0.6,
        ),
        graphics=GraphicsPage(
            packet_id=packet_id,
            status=2,
            session=3,
            completed_laps=0,
            current_lap_time_ms=100,
            last_lap_time_ms=0,
            distance_traveled_m=10.0,
            is_in_pit=False,
            normalized_car_position=0.1,
            car_coordinates_m=(1.0, 2.0, 3.0),
            is_in_pit_lane=False,
            surface_grip=1.0,
        ),
        static=StaticPage(
            shared_memory_version="1.7",
            assetto_corsa_version="1.16.4",
            car_model="bmw_z4_gt3",
            track="monza",
            track_configuration="",
            track_spline_length_m=5758.0,
            tyre_radius_m=(0.3, 0.3, 0.3, 0.3),
        ),
    )


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class _FakeReader:
    def __init__(self, clock: _FakeClock, packets: list[SlipBridgePacket]) -> None:
        self.clock = clock
        self.packets = packets

    def receive(self, *, timeout_s: float) -> SlipBridgePacket:
        self.clock.value += min(timeout_s, 0.01)
        if not self.packets:
            raise SlipBridgeTimeoutError("empty")
        return self.packets.pop(0)


class _FakeSharedReader:
    def __init__(self) -> None:
        self.calls = 0

    def read(self) -> AssettoCorsaTelemetry:
        self.calls += 1
        return _shared_snapshot(self.calls)


def test_capture_is_bounded_and_deduplicates_consecutive_packets() -> None:
    clock = _FakeClock()
    shared_reader = _FakeSharedReader()
    capture = capture_slip_telemetry(
        _FakeReader(
            clock,
            [_packet(1, 0.0), _packet(1, 0.01), _packet(2, 0.02)],
        ),
        duration_s=0.05,
        shared_telemetry_reader=shared_reader,
        clock=clock,
        receive_slice_s=0.02,
    )
    assert [packet.sequence for packet in capture.packets] == [1, 2]
    assert len(capture.shared_telemetry) == 2
    assert shared_reader.calls == 2
    assert capture.elapsed_s == pytest.approx(0.05)
    with pytest.raises(ValueError, match="duration_s"):
        capture_slip_telemetry(_FakeReader(clock, []), duration_s=0.0)
    with pytest.raises(ValueError, match="receive_slice_s"):
        capture_slip_telemetry(
            _FakeReader(clock, []),
            duration_s=1.0,
            receive_slice_s=0.0,
        )


def test_summary_reports_no_or_stale_packets() -> None:
    empty = summarise_slip_capture(
        SlipDiagnosticCapture(packets=(), elapsed_s=1.0, requested_duration_s=1.0)
    )
    assert not empty.is_usable_transport
    assert empty.slip_ratio_min is None
    assert {issue.code for issue in empty.issues} == {"bridge_no_packets"}

    stale = summarise_slip_capture(
        SlipDiagnosticCapture(
            packets=(_packet(1, 0.0),),
            elapsed_s=1.0,
            requested_duration_s=1.0,
        )
    )
    assert not stale.is_usable_transport
    assert {issue.code for issue in stale.issues} >= {"bridge_stale"}


def test_summary_reports_ranges_frequency_gaps_and_suspicious_values() -> None:
    summary = summarise_slip_capture(
        SlipDiagnosticCapture(
            packets=(
                _packet(0xFFFFFFFF, 0.0),
                _packet(
                    1,
                    0.1,
                    ratio=(11.0, 0.3, -0.4, -0.5),
                    angle=(math.nan, 0.03, -0.04, -0.05),
                ),
            ),
            elapsed_s=0.1,
            requested_duration_s=0.1,
        )
    )
    codes = {issue.code for issue in summary.issues}
    assert {
        "bridge_low_frequency",
        "bridge_non_finite",
        "slip_ratio_scale_suspicious",
    } <= codes
    assert summary.estimated_missed_packets == 1
    assert summary.observed_frequency_hz == pytest.approx(10.0)
    assert summary.slip_ratio_max is not None
    assert summary.slip_ratio_max[0] == 11.0
    assert not summary.is_usable_transport


def test_summary_accepts_live_packets_and_writes_named_wheel_columns(tmp_path: Path) -> None:
    packets = tuple(_packet(index, index / 50.0) for index in range(5))
    capture = SlipDiagnosticCapture(
        packets=packets,
        elapsed_s=0.1,
        requested_duration_s=0.1,
        shared_telemetry=tuple(_shared_snapshot(index) for index in range(5)),
    )
    summary = summarise_slip_capture(capture)
    assert summary.is_usable_transport
    assert summary.observed_frequency_hz == pytest.approx(50.0)
    assert summary.estimated_missed_packets == 0
    assert summary.duplicate_packets == 0
    assert summary.shared_memory_pairs == 5
    assert summary.slip_calibration_name == "vanilla_ac_python_v1"
    assert summary.slip_angle_raw_unit == "degree"
    assert summary.slip_angle_scale_rad_per_raw == pytest.approx(math.pi / 180.0)
    assert summary.axle_aggregation == "arithmetic_left_right_mean"
    write_slip_diagnostic_artifacts(tmp_path, capture, summary)
    with (tmp_path / "slip_telemetry.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert rows[0]["slip_ratio_front_left"] == "0.1"
    assert rows[0]["slip_angle_raw_rear_right"] == "-0.02"
    assert rows[0]["speed_kmh"] == "72.0"
    assert rows[0]["wheel_slip_shared_raw_rear_right"] == "4.0"
    assert (tmp_path / "summary.json").is_file()


def test_summary_and_writer_reject_mismatched_shared_pairs(tmp_path: Path) -> None:
    capture = SlipDiagnosticCapture(
        packets=(_packet(1, 0.0), _packet(2, 0.01)),
        elapsed_s=0.02,
        requested_duration_s=0.02,
        shared_telemetry=(_shared_snapshot(1),),
    )
    summary = summarise_slip_capture(capture)
    assert not summary.is_usable_transport
    assert {issue.code for issue in summary.issues} >= {"shared_pair_count_mismatch"}
    with pytest.raises(ValueError, match="counts"):
        write_slip_diagnostic_artifacts(tmp_path, capture, summary)
