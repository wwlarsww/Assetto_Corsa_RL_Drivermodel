from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from assetto_corsa_rl.cli import run_mock
from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.experiment import ExperimentRecorder


def test_recorded_mock_run_contains_traceable_artefacts(tmp_path: Path) -> None:
    baseline = ProjectConfig.from_toml("configs/mock.toml")
    config = replace(
        baseline,
        experiment=replace(
            baseline.experiment,
            output_directory=str(tmp_path),
        ),
    )
    summary = run_mock(config, record=True, command=["test", "mock-run"])
    run_directory = Path(str(summary["run_directory"]))

    with (run_directory / "run.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (run_directory / "summary.json").open(encoding="utf-8") as handle:
        persisted_summary = json.load(handle)
    with (run_directory / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert metadata["configuration"]["experiment"]["seed"] == 769913
    assert metadata["command"] == ["test", "mock-run"]
    assert persisted_summary["termination_reason"] == "lap_complete"
    assert len(rows) == summary["steps"]
    assert rows[-1]["termination_reason"] == "lap_complete"


def test_recorder_flushes_each_metric_for_the_external_monitor(tmp_path: Path) -> None:
    recorder = ExperimentRecorder.create(
        output_directory=tmp_path,
        config=ProjectConfig(),
        command=["test"],
    )
    try:
        recorder.record_step({"iteration": 1, "environment_steps": 25})
        with (recorder.run_directory / "metrics.csv").open(
            encoding="utf-8",
            newline="",
        ) as handle:
            assert list(csv.DictReader(handle)) == [{"iteration": "1", "environment_steps": "25"}]
    finally:
        recorder.close()
