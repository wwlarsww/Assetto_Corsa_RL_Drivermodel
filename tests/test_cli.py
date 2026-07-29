from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from assetto_corsa_rl import cli
from assetto_corsa_rl.ac_integration.diagnostics import (
    FeatureReadiness,
    TelemetryDiagnosticSummary,
)
from assetto_corsa_rl.ac_integration.shared_memory import (
    AssettoCorsaTelemetryUnavailableError,
)
from assetto_corsa_rl.ac_integration.slip_diagnostics import (
    SlipDiagnosticIssue,
    SlipDiagnosticSummary,
)
from assetto_corsa_rl.cli import main


def test_mock_cli_without_recording() -> None:
    assert main(["mock-run", "--config", "configs/mock.toml", "--no-record"]) == 0


def test_mock_rl_readiness_cli() -> None:
    assert main(["rl-ready", "--config", "configs/mock.toml", "--backend", "mock"]) == 0


def test_telemetry_cli_reports_unavailable_game(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(**_kwargs: object) -> None:
        raise AssettoCorsaTelemetryUnavailableError("game is not running")

    monkeypatch.setattr(cli, "run_telemetry_diagnostic", unavailable)
    assert main(["telemetry-diagnose", "--duration", "1", "--no-record"]) == 2


def _diagnostic_summary() -> TelemetryDiagnosticSummary:
    return TelemetryDiagnosticSummary(
        car_model="ks_bmw_z4_gt3",
        track="monza",
        track_configuration="",
        assetto_corsa_version="1.16.4",
        shared_memory_version="1.1",
        track_spline_length_m=5793.0,
        requested_duration_s=1.0,
        elapsed_s=1.0,
        poll_count=10,
        unique_physics_frames=10,
        fresh_poll_fraction=1.0,
        observed_physics_frequency_hz=10.0,
        status_counts={"live": 10},
        speed_min_kmh=10.0,
        speed_max_kmh=20.0,
        inferred_forward_local_velocity_axis=2,
        inferred_forward_local_velocity_sign=1,
        forward_axis_mean_error_kmh=0.0,
        issues=(),
        feature_readiness=(FeatureReadiness("speed", "direct", "speedKmh"),),
    )


def test_telemetry_cli_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "run_telemetry_diagnostic",
        lambda **_kwargs: (_diagnostic_summary(), None),
    )
    assert main(["telemetry-diagnose", "--duration", "1", "--no-record"]) == 0


def test_run_telemetry_diagnostic_writes_requested_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ContextReader:
        def __enter__(self) -> _ContextReader:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    reader = _ContextReader()
    reader_type: Any = vars(cli)["AssettoCorsaSharedMemoryReader"]
    monkeypatch.setattr(reader_type, "connect", lambda: reader)
    capture = object()
    monkeypatch.setattr(cli, "capture_telemetry", lambda *_args, **_kwargs: capture)
    monkeypatch.setattr(cli, "summarise_capture", lambda _capture: _diagnostic_summary())
    written: list[object] = []
    monkeypatch.setattr(
        cli,
        "write_diagnostic_artifacts",
        lambda destination, _capture, _summary: written.append(destination),
    )
    summary, destination = cli.run_telemetry_diagnostic(
        duration_s=1.0,
        poll_frequency_hz=10.0,
        output_directory=tmp_path,
        record=True,
    )
    assert summary.car_model == "ks_bmw_z4_gt3"
    assert destination == tmp_path
    assert written == [tmp_path]


def _slip_summary(*, usable: bool = True) -> SlipDiagnosticSummary:
    issues = () if usable else (SlipDiagnosticIssue("error", "bridge_no_packets", "missing"),)
    return SlipDiagnosticSummary(
        requested_duration_s=1.0,
        elapsed_s=1.0,
        unique_packets=50 if usable else 0,
        observed_frequency_hz=50.0 if usable else None,
        first_sequence=1 if usable else None,
        last_sequence=50 if usable else None,
        estimated_missed_packets=0,
        duplicate_packets=0,
        slip_ratio_min=(-0.1, -0.2, -0.3, -0.4) if usable else None,
        slip_ratio_max=(0.1, 0.2, 0.3, 0.4) if usable else None,
        slip_angle_raw_min=(-0.01, -0.02, -0.03, -0.04) if usable else None,
        slip_angle_raw_max=(0.01, 0.02, 0.03, 0.04) if usable else None,
        issues=issues,
    )


def test_slip_cli_reports_success_and_missing_packets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "run_slip_diagnostic",
        lambda **_kwargs: (_slip_summary(), None),
    )
    assert main(["slip-diagnose", "--duration", "1", "--no-record"]) == 0
    monkeypatch.setattr(
        cli,
        "run_slip_diagnostic",
        lambda **_kwargs: (_slip_summary(usable=False), None),
    )
    assert main(["slip-diagnose", "--duration", "1", "--no-record"]) == 3


def test_slip_cli_reports_bind_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def occupied(**_kwargs: object) -> None:
        raise OSError("occupied")

    monkeypatch.setattr(cli, "run_slip_diagnostic", occupied)
    assert main(["slip-diagnose", "--duration", "1", "--no-record"]) == 2


def test_slip_cli_reports_unavailable_shared_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(**_kwargs: object) -> None:
        raise AssettoCorsaTelemetryUnavailableError("game is not on track")

    monkeypatch.setattr(cli, "run_slip_diagnostic", unavailable)
    assert (
        main(
            [
                "slip-diagnose",
                "--duration",
                "1",
                "--with-shared-memory",
                "--no-record",
            ]
        )
        == 2
    )


def test_run_slip_diagnostic_writes_requested_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ContextReceiver:
        def __enter__(self) -> _ContextReceiver:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    receiver = _ContextReceiver()
    reader_type: Any = vars(cli)["AssettoCorsaSlipBridgeFileReader"]
    monkeypatch.setattr(reader_type, "connect", lambda *_args: receiver)
    shared_reader_type: Any = vars(cli)["AssettoCorsaSharedMemoryReader"]
    monkeypatch.setattr(shared_reader_type, "connect", lambda: receiver)
    capture = object()
    monkeypatch.setattr(cli, "capture_slip_telemetry", lambda *_args, **_kwargs: capture)
    monkeypatch.setattr(cli, "summarise_slip_capture", lambda _capture: _slip_summary())
    written: list[object] = []
    monkeypatch.setattr(
        cli,
        "write_slip_diagnostic_artifacts",
        lambda destination, _capture, _summary: written.append(destination),
    )
    summary, destination = cli.run_slip_diagnostic(
        duration_s=1.0,
        transport="file",
        bridge_file=None,
        port=27942,
        pair_shared_memory=True,
        output_directory=tmp_path,
        record=True,
    )
    assert summary.unique_packets == 50
    assert destination == tmp_path
    assert written == [tmp_path]


def test_run_slip_diagnostic_supports_explicit_udp_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ContextReceiver:
        def __enter__(self) -> _ContextReceiver:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    receiver_type: Any = vars(cli)["AssettoCorsaSlipBridgeReceiver"]
    monkeypatch.setattr(receiver_type, "bind", lambda **_kwargs: _ContextReceiver())
    monkeypatch.setattr(cli, "capture_slip_telemetry", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "summarise_slip_capture", lambda _capture: _slip_summary())
    summary, destination = cli.run_slip_diagnostic(
        duration_s=1.0,
        transport="udp",
        bridge_file=None,
        port=27942,
        pair_shared_memory=False,
        output_directory=None,
        record=False,
    )
    assert summary.unique_packets == 50
    assert destination is None
    with pytest.raises(ValueError, match="unsupported"):
        cli.run_slip_diagnostic(
            duration_s=1.0,
            transport="invalid",
            bridge_file=None,
            port=27942,
            pair_shared_memory=False,
            output_directory=None,
            record=False,
        )
