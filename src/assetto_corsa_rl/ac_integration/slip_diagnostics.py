"""Capture and validate the local Assetto Corsa tyre-signal bridge."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Protocol

import numpy as np

from assetto_corsa_rl.ac_integration.shared_memory import (
    AssettoCorsaTelemetry,
    WheelValues,
)
from assetto_corsa_rl.ac_integration.slip_bridge import (
    WHEEL_NAMES,
    SlipBridgePacket,
    SlipBridgeTimeoutError,
)
from assetto_corsa_rl.ac_integration.slip_features import (
    VANILLA_AC_SLIP_ANGLE_RAD_PER_RAW,
    VANILLA_AC_SLIP_ANGLE_RAW_UNIT,
    VANILLA_AC_SLIP_AXLE_AGGREGATION,
    VANILLA_AC_SLIP_CALIBRATION_NAME,
)


class SlipPacketReader(Protocol):
    """Small receive contract implemented by local-file, UDP, and test readers."""

    def receive(self, *, timeout_s: float) -> SlipBridgePacket:
        """Return one packet or raise after the bounded timeout."""

        ...


class SharedTelemetryReader(Protocol):
    """Read-only contract used to pair bridge packets with vehicle state."""

    def read(self) -> AssettoCorsaTelemetry:
        """Return the latest copied shared-memory snapshot."""

        ...


@dataclass(frozen=True, slots=True)
class SlipDiagnosticCapture:
    """Unique bridge packets collected during a fixed wall-clock window."""

    packets: tuple[SlipBridgePacket, ...]
    elapsed_s: float
    requested_duration_s: float
    shared_telemetry: tuple[AssettoCorsaTelemetry, ...] = ()


@dataclass(frozen=True, slots=True)
class SlipDiagnosticIssue:
    """Machine-readable bridge validation finding."""

    severity: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SlipDiagnosticSummary:
    """Protocol, timing, range, and unit-readiness report."""

    requested_duration_s: float
    elapsed_s: float
    unique_packets: int
    observed_frequency_hz: float | None
    first_sequence: int | None
    last_sequence: int | None
    estimated_missed_packets: int
    duplicate_packets: int
    slip_ratio_min: tuple[float, float, float, float] | None
    slip_ratio_max: tuple[float, float, float, float] | None
    slip_angle_raw_min: tuple[float, float, float, float] | None
    slip_angle_raw_max: tuple[float, float, float, float] | None
    issues: tuple[SlipDiagnosticIssue, ...]
    shared_memory_pairs: int = 0
    slip_calibration_name: str = VANILLA_AC_SLIP_CALIBRATION_NAME
    slip_angle_raw_unit: str = VANILLA_AC_SLIP_ANGLE_RAW_UNIT
    slip_angle_scale_rad_per_raw: float = VANILLA_AC_SLIP_ANGLE_RAD_PER_RAW
    axle_aggregation: str = VANILLA_AC_SLIP_AXLE_AGGREGATION

    @property
    def is_usable_transport(self) -> bool:
        """True when packets are live and structurally valid."""

        return not any(issue.severity == "error" for issue in self.issues)


def capture_slip_telemetry(
    reader: SlipPacketReader,
    *,
    duration_s: float,
    shared_telemetry_reader: SharedTelemetryReader | None = None,
    clock: Callable[[], float] = time.monotonic,
    receive_slice_s: float = 0.1,
) -> SlipDiagnosticCapture:
    """Receive unique packets for a bounded duration."""

    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if receive_slice_s <= 0:
        raise ValueError("receive_slice_s must be positive")
    start = clock()
    deadline = start + duration_s
    packets: list[SlipBridgePacket] = []
    shared_telemetry: list[AssettoCorsaTelemetry] = []
    last_sequence: int | None = None
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            break
        try:
            packet = reader.receive(timeout_s=min(receive_slice_s, remaining))
        except SlipBridgeTimeoutError:
            continue
        if packet.sequence != last_sequence:
            packets.append(packet)
            if shared_telemetry_reader is not None:
                shared_telemetry.append(shared_telemetry_reader.read())
            last_sequence = packet.sequence
    return SlipDiagnosticCapture(
        packets=tuple(packets),
        elapsed_s=max(clock() - start, 0.0),
        requested_duration_s=duration_s,
        shared_telemetry=tuple(shared_telemetry),
    )


def _sequence_findings(packets: Sequence[SlipBridgePacket]) -> tuple[int, int]:
    missed = 0
    duplicates = 0
    for previous, current in pairwise(packets):
        delta = (current.sequence - previous.sequence) & 0xFFFFFFFF
        if delta == 0:
            duplicates += 1
        elif 1 < delta < 0x80000000:
            missed += delta - 1
    return missed, duplicates


def _wheel_extrema(values: np.ndarray) -> tuple[WheelValues, WheelValues]:
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    return (
        float(minimum[0]),
        float(minimum[1]),
        float(minimum[2]),
        float(minimum[3]),
    ), (
        float(maximum[0]),
        float(maximum[1]),
        float(maximum[2]),
        float(maximum[3]),
    )


def summarise_slip_capture(capture: SlipDiagnosticCapture) -> SlipDiagnosticSummary:
    """Validate availability, timing, ranges, and unresolved semantics."""

    packets = list(capture.packets)
    issues: list[SlipDiagnosticIssue] = []
    if not packets:
        issues.append(
            SlipDiagnosticIssue(
                "error",
                "bridge_no_packets",
                "No packets arrived; install and enable the in-game Assetto Corsa RL Bridge app.",
            )
        )
        return SlipDiagnosticSummary(
            requested_duration_s=capture.requested_duration_s,
            elapsed_s=capture.elapsed_s,
            unique_packets=0,
            observed_frequency_hz=None,
            first_sequence=None,
            last_sequence=None,
            estimated_missed_packets=0,
            duplicate_packets=0,
            slip_ratio_min=None,
            slip_ratio_max=None,
            slip_angle_raw_min=None,
            slip_angle_raw_max=None,
            issues=tuple(issues),
            shared_memory_pairs=0,
        )

    observed_frequency: float | None = None
    if len(packets) >= 2:
        interval = packets[-1].received_monotonic_s - packets[0].received_monotonic_s
        if interval > 0:
            observed_frequency = (len(packets) - 1) / interval
    else:
        issues.append(
            SlipDiagnosticIssue(
                "error",
                "bridge_stale",
                "Only one bridge packet arrived during the capture.",
            )
        )

    ratio_values = np.asarray([packet.slip_ratio for packet in packets], dtype=np.float64)
    angle_values = np.asarray([packet.slip_angle_raw for packet in packets], dtype=np.float64)
    if not np.all(np.isfinite(ratio_values)) or not np.all(np.isfinite(angle_values)):
        issues.append(
            SlipDiagnosticIssue(
                "error",
                "bridge_non_finite",
                "The bridge produced a non-finite tyre value.",
            )
        )
    ratio_min, ratio_max = _wheel_extrema(ratio_values)
    angle_min, angle_max = _wheel_extrema(angle_values)
    missed, duplicates = _sequence_findings(packets)
    if capture.shared_telemetry and len(capture.shared_telemetry) != len(packets):
        issues.append(
            SlipDiagnosticIssue(
                "error",
                "shared_pair_count_mismatch",
                "The number of shared-memory snapshots does not match the bridge packets.",
            )
        )
    if observed_frequency is not None and observed_frequency < 20.0:
        issues.append(
            SlipDiagnosticIssue(
                "warning",
                "bridge_low_frequency",
                f"The observed bridge rate is only {observed_frequency:.1f} Hz.",
            )
        )
    if max(abs(value) for value in (*ratio_min, *ratio_max)) > 10.0:
        issues.append(
            SlipDiagnosticIssue(
                "warning",
                "slip_ratio_scale_suspicious",
                "An absolute slip-ratio value exceeded 10; inspect API semantics and wheel order.",
            )
        )
    return SlipDiagnosticSummary(
        requested_duration_s=capture.requested_duration_s,
        elapsed_s=capture.elapsed_s,
        unique_packets=len(packets),
        observed_frequency_hz=observed_frequency,
        first_sequence=packets[0].sequence,
        last_sequence=packets[-1].sequence,
        estimated_missed_packets=missed,
        duplicate_packets=duplicates,
        slip_ratio_min=(
            ratio_min[0],
            ratio_min[1],
            ratio_min[2],
            ratio_min[3],
        ),
        slip_ratio_max=(
            ratio_max[0],
            ratio_max[1],
            ratio_max[2],
            ratio_max[3],
        ),
        slip_angle_raw_min=(
            angle_min[0],
            angle_min[1],
            angle_min[2],
            angle_min[3],
        ),
        slip_angle_raw_max=(
            angle_max[0],
            angle_max[1],
            angle_max[2],
            angle_max[3],
        ),
        issues=tuple(issues),
        shared_memory_pairs=len(capture.shared_telemetry),
    )


def write_slip_diagnostic_artifacts(
    destination: Path,
    capture: SlipDiagnosticCapture,
    summary: SlipDiagnosticSummary,
) -> None:
    """Write a portable summary and tyre-level CSV."""

    if capture.shared_telemetry and len(capture.shared_telemetry) != len(capture.packets):
        raise ValueError("shared telemetry and bridge packet counts must match")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = ["sequence", "source_time_s", "received_monotonic_s"]
    fields.extend(f"slip_ratio_{wheel}" for wheel in WHEEL_NAMES)
    fields.extend(f"slip_angle_raw_{wheel}" for wheel in WHEEL_NAMES)
    if capture.shared_telemetry:
        fields.extend(
            (
                "speed_kmh",
                "gas",
                "brake",
                "steer_angle",
                "local_velocity_mps_0",
                "local_velocity_mps_1",
                "local_velocity_mps_2",
                "acceleration_g_0",
                "acceleration_g_1",
                "acceleration_g_2",
            )
        )
        fields.extend(f"wheel_slip_shared_raw_{wheel}" for wheel in WHEEL_NAMES)
    with (destination / "slip_telemetry.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, packet in enumerate(capture.packets):
            row: dict[str, float | int] = {
                "sequence": packet.sequence,
                "source_time_s": packet.source_time_s,
                "received_monotonic_s": packet.received_monotonic_s,
            }
            row.update(
                {
                    f"slip_ratio_{wheel}": value
                    for wheel, value in zip(WHEEL_NAMES, packet.slip_ratio, strict=True)
                }
            )
            row.update(
                {
                    f"slip_angle_raw_{wheel}": value
                    for wheel, value in zip(WHEEL_NAMES, packet.slip_angle_raw, strict=True)
                }
            )
            if capture.shared_telemetry:
                telemetry = capture.shared_telemetry[index]
                physics = telemetry.physics
                row.update(
                    {
                        "speed_kmh": physics.speed_kmh,
                        "gas": physics.gas,
                        "brake": physics.brake,
                        "steer_angle": physics.steer_angle,
                        **{
                            f"local_velocity_mps_{axis}": value
                            for axis, value in enumerate(physics.local_velocity_mps)
                        },
                        **{
                            f"acceleration_g_{axis}": value
                            for axis, value in enumerate(physics.acceleration_g)
                        },
                        **{
                            f"wheel_slip_shared_raw_{wheel}": value
                            for wheel, value in zip(
                                WHEEL_NAMES,
                                physics.wheel_slip,
                                strict=True,
                            )
                        },
                    }
                )
            writer.writerow(row)
