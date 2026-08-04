from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from assetto_corsa_rl.training.aim_integration import AimMonitor

from assetto_corsa_rl.training import monitor as monitor_module
from assetto_corsa_rl.training.monitor import (
    launch_training_monitor,
    monitor_training_run,
    resolve_training_run_directory,
)


class _FakeAimRun:
    def __init__(self) -> None:
        self.tracked: list[tuple[str, float, int]] = []
        self.metadata: dict[str, object] = {}
        self.closed = False

    def track(self, value: float, *, name: str, step: int) -> None:
        self.tracked.append((name, value, step))

    def __setitem__(self, key: str, value: object) -> None:
        self.metadata[key] = value

    def close(self) -> None:
        self.closed = True


def test_aim_monitor_tracks_scalars_and_starts_local_ui(tmp_path: Path) -> None:
    run = _FakeAimRun()
    processes: list[tuple[list[str], dict[str, Any]]] = []

    def process_factory(command: list[str], **options: Any) -> SimpleNamespace:
        processes.append((command, options))
        return SimpleNamespace(pid=4321)

    monitor = AimMonitor(
        run_directory=tmp_path,
        run_label="MOCK",
        total_environment_steps=100,
        start_ui=True,
        run_factory=lambda **_kwargs: run,
        process_factory=process_factory,
        aim_executable="aim",
    )
    monitor.update(
        {
            "environment_steps": "25",
            "speed_mean_mps": "12.5",
            "approximate_kl": float("nan"),
            "kl_status": "in_band",
        }
    )
    monitor.finish(completed=True)
    monitor.close()

    assert ("speed_mean_mps", 12.5, 25) in run.tracked
    assert not any(name == "approximate_kl" for name, _value, _step in run.tracked)
    assert run.metadata["kl_status"] == "in_band"
    assert run.metadata["status"] == "complete"
    assert run.closed
    command, options = processes[0]
    assert command[:2] == ["aim", "up"]
    assert command[command.index("--repo") + 1] == str(tmp_path.resolve())
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "43800"
    assert options["close_fds"] is True


class _FakeMonitor:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.is_enabled = True
        self.updates: list[dict[str, str]] = []
        self.finished: list[bool] = []
        self.waited = False
        self.closed = False

    def update(self, metrics: dict[str, str]) -> None:
        self.updates.append(metrics)

    def finish(self, *, completed: bool) -> None:
        self.finished.append(completed)

    def wait_for_close(self) -> None:
        self.waited = True

    def process_events(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True
        self.is_enabled = False


def _write_run_metadata(run_directory: Path, *, total_steps: int = 100) -> None:
    run_directory.mkdir(parents=True)
    (run_directory / "run.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "ppo": {
                        "total_environment_steps": total_steps,
                        "method_variant": "test_variant",
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_standalone_monitor_streams_completed_metrics(tmp_path: Path) -> None:
    run_directory = tmp_path / "ppo-mock" / "run"
    _write_run_metadata(run_directory)
    with (run_directory / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("iteration", "environment_steps"))
        writer.writeheader()
        writer.writerow({"iteration": 1, "environment_steps": 25})
    (run_directory / "summary.json").write_text("{}\n", encoding="utf-8")
    created: list[_FakeMonitor] = []

    def factory(**kwargs: Any) -> _FakeMonitor:
        monitor = _FakeMonitor(**kwargs)
        created.append(monitor)
        return monitor

    monitor_training_run(run_directory, monitor_factory=factory, start_ui=True)

    monitor = created[0]
    assert monitor.kwargs["run_directory"] == run_directory
    assert monitor.kwargs["total_environment_steps"] == 100
    assert monitor.kwargs["start_ui"] is True
    assert monitor.updates == [{"iteration": "1", "environment_steps": "25"}]
    assert monitor.finished == [True]
    assert monitor.waited
    assert monitor.closed


def test_monitor_selects_newest_ppo_run(tmp_path: Path) -> None:
    older = tmp_path / "group" / "ppo-mock" / "older"
    newer = tmp_path / "group" / "ppo-assetto-corsa" / "newer"
    _write_run_metadata(older)
    _write_run_metadata(newer)
    older_metadata = older / "run.json"
    newer_metadata = newer / "run.json"
    older_metadata_time = newer_metadata.stat().st_mtime - 10
    os.utime(older_metadata, (older_metadata_time, older_metadata_time))

    assert resolve_training_run_directory(None, runs_root=tmp_path) == newer.resolve()


def test_monitor_launcher_starts_independent_aim_follower(tmp_path: Path) -> None:
    run_directory = tmp_path / "ppo-mock" / "run"
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def process_factory(command: list[str], **options: Any) -> SimpleNamespace:
        calls.append((command, options))
        return SimpleNamespace(pid=4321)

    assert launch_training_monitor(run_directory, process_factory=process_factory) == 4321
    command, options = calls[0]
    assert command[1:3] == ["-m", "assetto_corsa_rl"]
    assert "training-monitor" in command
    assert "--monitor-start-ui" in command
    assert str(run_directory.resolve()) in command
    assert options["close_fds"] is True


def test_monitor_launcher_does_not_spawn_when_aim_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(monitor_module, "aim_is_available", lambda: False)

    assert launch_training_monitor(tmp_path) is None


def test_aim_monitor_rejects_invalid_total_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="total_environment_steps"):
        AimMonitor(run_directory=tmp_path, total_environment_steps=0)
