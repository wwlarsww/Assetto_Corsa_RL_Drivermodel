from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from assetto_corsa_rl.training.monitor import (  # noqa: E402
    LiveMonitor,
    launch_training_monitor,
    monitor_training_run,
    resolve_training_run_directory,
)


def _metrics(iteration: int) -> dict[str, float | int]:
    return {
        "iteration": iteration,
        "policy_version": iteration,
        "environment_steps": iteration * 10,
        "episodes_completed": iteration // 2,
        "environment_reward_mean": float(iteration),
        "shaped_reward_mean": float(iteration) + 0.25,
        "policy_loss": -0.01 * iteration,
        "value_loss": 0.5 / iteration,
        "predictor_loss": 0.25 / iteration,
        "entropy": 1.5 - 0.1 * iteration,
        "approximate_kl": 0.001 * iteration,
        "clip_fraction": 0.02 * iteration,
    }


def test_dashboard_separates_diagnostics_and_bounds_history() -> None:
    monitor = LiveMonitor(
        total_environment_steps=40,
        run_label="MOCK · test",
        max_history_points=3,
        refresh_interval_s=0.0,
    )
    assert monitor.is_enabled
    try:
        for iteration in range(1, 5):
            monitor.update(_metrics(iteration))

        assert tuple(monitor._history["environment_reward_mean"]) == (2.0, 3.0, 4.0)
        assert tuple(monitor._lines["policy_loss"].get_xdata()) == (20.0, 30.0, 40.0)
        assert (
            monitor._axes_by_metric["environment_reward_mean"]
            is not monitor._axes_by_metric["policy_loss"]
        )
        assert monitor._axes_by_metric["policy_loss"] is not monitor._axes_by_metric["value_loss"]
        assert "100.0%" in monitor._kpi_text["progress"].get_text()
        assert "policy v4" in monitor._meta_text.get_text()
        assert monitor._axes_by_metric["episodes_completed"].get_ylim()[1] > 2.0
        step_formatter = monitor._axes_by_metric["policy_loss"].xaxis.get_major_formatter()
        assert "k" in step_formatter(10_000, 0)

        monitor.finish(completed=True)
        assert "COMPLETE" in monitor._status_text.get_text()
    finally:
        monitor.close()
    assert not monitor.is_enabled


def test_dashboard_ignores_invalid_metric_values() -> None:
    monitor = LiveMonitor(refresh_interval_s=0.0)
    assert monitor.is_enabled
    try:
        monitor.update(
            {
                "environment_steps": 10,
                "environment_reward_mean": float("nan"),
                "policy_loss": "not-a-number",
                "entropy": float("inf"),
                "episodes_completed": False,
            }
        )
        assert not monitor._history["environment_reward_mean"]
        assert not monitor._history["policy_loss"]
        assert not monitor._history["entropy"]
        assert not monitor._history["episodes_completed"]
    finally:
        monitor.close()


def test_dashboard_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="total_environment_steps"):
        LiveMonitor(total_environment_steps=0)
    with pytest.raises(ValueError, match="max_history_points"):
        LiveMonitor(max_history_points=1)
    with pytest.raises(ValueError, match="refresh_interval_s"):
        LiveMonitor(refresh_interval_s=-0.1)


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
        self.is_enabled = False

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


def test_standalone_monitor_reads_completed_metrics_and_waits_for_close(tmp_path: Path) -> None:
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

    monitor_training_run(run_directory, monitor_factory=factory)

    monitor = created[0]
    assert monitor.kwargs["total_environment_steps"] == 100
    assert "test_variant" in monitor.kwargs["run_label"]
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
    older_metadata.touch()
    newer_metadata.touch()
    older_metadata_time = newer_metadata.stat().st_mtime - 10
    os.utime(older_metadata, (older_metadata_time, older_metadata_time))

    assert resolve_training_run_directory(None, runs_root=tmp_path) == newer.resolve()


def test_monitor_launcher_starts_independent_cli_process(tmp_path: Path) -> None:
    run_directory = tmp_path / "ppo-mock" / "run"
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def process_factory(command: list[str], **options: Any) -> SimpleNamespace:
        calls.append((command, options))
        return SimpleNamespace(pid=4321)

    assert launch_training_monitor(run_directory, process_factory=process_factory) == 4321
    command, options = calls[0]
    assert command[1:3] == ["-m", "assetto_corsa_rl"]
    assert "training-monitor" in command
    assert str(run_directory.resolve()) in command
    assert options["close_fds"] is True
