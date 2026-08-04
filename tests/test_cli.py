from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
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
from assetto_corsa_rl.training.readiness import TrainingReadinessReport


def test_cli_help_explains_the_local_training_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli.build_parser()
    assert "Research training is local-only" in parser.format_help()

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["rl-train", "--help"])
    assert exit_info.value.code == 0
    command_help = capsys.readouterr().out
    assert "deterministic mock or live Assetto Corsa backend" in command_help
    assert "(default: mock)" in command_help
    arguments = parser.parse_args(["rl-train", "--backend", "assetto-corsa", "--launch-game"])
    assert arguments.launch_game is True
    assert arguments.monitor_ui is True
    assert parser.parse_args(["rl-train", "--no-monitor-ui"]).monitor_ui is False

    live_arguments = parser.parse_args(
        [
            "live-validate",
            "--drive-off-steps",
            "300",
            "--drive-off-throttle",
            "0.5",
            "--minimum-forward-gear",
            "2",
        ]
    )
    assert live_arguments.drive_off_steps == 300
    assert live_arguments.drive_off_throttle == pytest.approx(0.5)
    assert live_arguments.minimum_forward_gear == 2


def test_bc_warm_start_is_published_to_stable_paths(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoint.pt"
    summary = tmp_path / "run" / "summary.json"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    summary.write_text('{"metric": 1}', encoding="utf-8")
    config: Any = SimpleNamespace(
        ppo=SimpleNamespace(
            initial_checkpoint_path=str(tmp_path / "processed" / "bc_checkpoint.pt"),
            initial_checkpoint_metrics_path=str(tmp_path / "processed" / "bc_summary.json"),
        )
    )

    checkpoint_destination, metrics_destination = cli._publish_bc_warm_start(
        config,
        checkpoint_path=checkpoint,
        summary_path=summary,
    )

    assert checkpoint_destination is not None
    assert metrics_destination is not None
    assert checkpoint_destination.read_bytes() == b"checkpoint"
    assert metrics_destination.read_text("utf-8") == '{"metric": 1}'


def test_mock_cli_without_recording() -> None:
    assert main(["mock-run", "--config", "configs/mock.toml", "--no-record"]) == 0


def test_mock_rl_readiness_cli() -> None:
    assert main(["rl-ready", "--config", "configs/mock.toml", "--backend", "mock"]) == 0


def test_x360_setup_holds_only_a_neutral_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Writer:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    writer = _Writer()
    slept: list[float] = []
    monkeypatch.setattr(cli, "VGamepadControlWriter", lambda: writer)
    monkeypatch.setattr(time, "sleep", slept.append)
    assert main(["x360-setup", "--duration", "3"]) == 0
    assert slept == [3.0]
    assert writer.closed


def test_live_rl_training_selects_live_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Writer:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _Environment:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    writer = _Writer()
    environment = _Environment()
    launcher_events: list[str] = []
    device_events: list[str] = []

    class _Launcher:
        def __init__(self, _config: object, **_kwargs: object) -> None:
            device_events.append("launcher")
            launcher_events.append("created")

        def __enter__(self) -> _Launcher:
            launcher_events.append("entered")
            return self

        def __exit__(self, *_args: object) -> None:
            launcher_events.append("restored")

        @staticmethod
        def stop_owned_session() -> bool:
            launcher_events.append("stopped")
            return True

    monkeypatch.setattr(
        cli,
        "assess_training_readiness",
        lambda *_args, **_kwargs: TrainingReadinessReport("assetto-corsa", ()),
    )
    monkeypatch.setattr(
        cli,
        "create_assetto_corsa_environment",
        lambda _config, *, control_writer: (
            environment if control_writer is writer else pytest.fail("wrong control writer")
        ),
    )

    def create_writer(**_kwargs: object) -> _Writer:
        device_events.append("gamepad")
        return writer

    monkeypatch.setattr(cli, "VGamepadControlWriter", create_writer)
    monkeypatch.setattr(cli, "AssettoCorsaSessionLauncher", _Launcher)
    observed_backends: list[str] = []

    def train_stub(
        _config: object,
        selected_environment: object,
        *,
        command: list[str],
        backend_name: str,
        device: str,
        monitor_ui: bool,
        monitor_backend: str,
    ) -> SimpleNamespace:
        assert selected_environment is environment
        assert command[0] == "ac-rl-driver"
        assert device == "cpu"
        assert monitor_ui is True
        assert monitor_backend == "aim"
        observed_backends.append(backend_name)
        return SimpleNamespace(
            environment_steps=1,
            iterations=1,
            policy_version=1,
            run_directory=Path("runs/test"),
        )

    monkeypatch.setattr(cli, "train_ppo", train_stub)
    assert (
        main(
            [
                "rl-train",
                "--config",
                "configs/mock.toml",
                "--backend",
                "assetto-corsa",
                "--launch-game",
            ]
        )
        == 0
    )
    assert observed_backends == ["assetto-corsa"]
    assert launcher_events == ["created", "entered", "stopped", "restored"]
    assert device_events == ["gamepad", "launcher"]
    assert environment.closed
    assert writer.closed


def test_standalone_training_monitor_cli_selects_and_opens_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "ppo-mock" / "run"
    opened: list[tuple[Path, float, int | None]] = []
    monkeypatch.setattr(
        cli,
        "resolve_training_run_directory",
        lambda run, *, runs_root: (
            run_directory
            if run is None and runs_root == Path("custom-runs")
            else pytest.fail("wrong monitor selection")
        ),
    )
    monkeypatch.setattr(
        cli,
        "monitor_training_run",
        lambda run, *, poll_interval_s, parent_pid, start_ui: (
            opened.append((run, poll_interval_s, parent_pid))
            if start_ui
            else pytest.fail("Aim UI should start by default")
        ),
    )

    assert (
        main(
            [
                "training-monitor",
                "--runs-root",
                "custom-runs",
                "--poll-interval",
                "0.5",
            ]
        )
        == 0
    )
    assert opened == [(run_directory, 0.5, None)]


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
