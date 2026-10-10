"""Run identity, provenance, paired saves and append-only records."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from io import BufferedIOBase
from unittest.mock import Mock

import pytest

from assetto_corsa_ai import training_run as module
from assetto_corsa_ai.training_run import METRIC_FIELDS, TrainingRun, create_run_directory


@pytest.fixture
def run(tmp_path: Path) -> TrainingRun:
    return create_run_directory(tmp_path / "runs", track="monza", car="bmw_z4_gt3")


def test_unique_runs_and_timezone_aware_names(tmp_path: Path) -> None:
    first = create_run_directory(tmp_path, track="Monza", car="bmw_z4_gt3")
    second = create_run_directory(tmp_path, track="Monza", car="bmw_z4_gt3")
    assert first.directory != second.directory
    assert first.started_at.utcoffset() is not None
    assert first.directory.name == f"{first.started_at:%Y%m%d_%H%M%S}_monza_bmw_z4_gt3_{first.run_id}"
    assert len(first.run_id) == 6
    assert set(path.name for path in first.directory.iterdir()) == {"checkpoints", "metrics.csv"}


def test_collision_never_overwrites_a_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Mock()
    clock.now.return_value.astimezone.return_value = datetime(2026, 10, 10, 12, 3, 15, tzinfo=timezone.utc)
    monkeypatch.setattr(module, "datetime", clock)
    monkeypatch.setattr(module, "uuid4", Mock(return_value=Mock(hex="a81f3c" * 6)))
    first = create_run_directory(tmp_path, track="monza", car="car")
    first.save_config("[training]\nseed = 42\n")
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, track="monza", car="car")
    assert (first.directory / "config.toml").read_text() == "[training]\nseed = 42\n"


def test_names_cannot_escape_run_root(tmp_path: Path) -> None:
    result = create_run_directory(tmp_path, track="../../Monza", car="Car / Name")
    assert result.directory.parent == tmp_path
    assert "monza_car_name" in result.directory.name
    with pytest.raises(ValueError):
        create_run_directory(tmp_path, track="...", car="car")


def test_config_is_exact_valid_toml_and_written_once(run: TrainingRun) -> None:
    content = "# Experiment\n[training]\nseed = 17\n"
    run.save_config(content)
    assert (run.directory / "config.toml").read_text(encoding="utf-8") == content
    with pytest.raises(FileExistsError):
        run.save_config(content)


def test_invalid_config_does_not_create_file(run: TrainingRun) -> None:
    with pytest.raises(ValueError):
        run.save_config("[invalid")
    assert not (run.directory / "config.toml").exists()


@pytest.mark.parametrize("dirty", ["", " M src/agent.py\n?? source.py\n"])
def test_metadata_contains_only_automatic_provenance(
    run: TrainingRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirty: str,
) -> None:
    git = Mock(side_effect=[Mock(stdout="a" * 40 + "\n"), Mock(stdout=dirty)])
    monkeypatch.setattr(module.subprocess, "run", git)
    def library_version(name: str) -> str:
        return f"version-of-{name}"
    monkeypatch.setattr(module, "version", library_version)
    monkeypatch.setattr(module.platform, "platform", Mock(return_value="Windows-test"))
    run.save_metadata(tmp_path)
    data = json.loads((run.directory / "metadata.json").read_text())
    assert data["run_id"] == run.run_id
    assert datetime.fromisoformat(data["started_at"]) == run.started_at
    assert data["git_commit"] == "a" * 40
    assert data["git_dirty"] is bool(dirty)
    assert data["torch_version"] == "version-of-torch"
    assert set(data) == {
        "run_id", "started_at", "git_commit", "git_dirty", "python_version",
        "stable_baselines3_version", "torch_version", "operating_system", "project_version",
    }
    assert git.call_args.kwargs["check"] is True
    assert "--untracked-files=all" in git.call_args.args[0]


def test_missing_git_provenance_fails_instead_of_inventing_commit(
    run: TrainingRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module.subprocess, "run", Mock(side_effect=subprocess.CalledProcessError(1, "git")))
    with pytest.raises(subprocess.CalledProcessError):
        run.save_metadata(tmp_path)
    assert not (run.directory / "metadata.json").exists()


def test_sparse_metrics_append_with_one_header(run: TrainingRun) -> None:
    run.append_metrics({"step": 3, "episode": 1, "episode_reward": 22.5, "episode_length": 3})
    run.append_metrics({"step": 4, "value_loss": 1.5, "predictor_loss": None})
    with (run.directory / "metrics.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames == list(METRIC_FIELDS)
    assert len(rows) == 2
    assert rows[0]["episode_reward"] == "22.5"
    assert rows[0]["value_loss"] == rows[1]["predictor_loss"] == ""
    assert rows[1]["step"] == "4"


@pytest.mark.parametrize("values", [
    {}, {"step": -1}, {"step": True}, {"step": 0, "unknown": 1},
    {"step": 0, "value_loss": float("nan")}, {"step": 0, "value_loss": float("inf")},
])
def test_bad_metrics_leave_history_unchanged(run: TrainingRun, values: dict[str, int | float]) -> None:
    original = (run.directory / "metrics.csv").read_bytes()
    with pytest.raises(ValueError):
        run.append_metrics(values)
    assert (run.directory / "metrics.csv").read_bytes() == original


class FakeModel:
    def save(self, path: BufferedIOBase) -> None:
        path.write(b"model")


class FakeNormalizer:
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"normalizer")


@pytest.mark.parametrize("final", [False, True])
def test_snapshot_pair_is_saved_once(run: TrainingRun, final: bool) -> None:
    def save() -> Path:
        return run.save_final_model(FakeModel(), FakeNormalizer()) if final else run.save_checkpoint(10000, FakeModel(), FakeNormalizer())
    target = save()
    assert target.name == ("final" if final else "step_010000")
    assert {path.name for path in target.iterdir()} == {"model.zip", "vecnormalize.pkl"}
    assert (target / "model.zip").read_bytes() == b"model"
    with pytest.raises(FileExistsError):
        save()
    assert (target / "vecnormalize.pkl").read_bytes() == b"normalizer"


@pytest.mark.parametrize("fails", ["model", "normalizer"])
def test_failed_save_removes_only_its_incomplete_pair(run: TrainingRun, fails: str) -> None:
    run.save_checkpoint(10, FakeModel(), FakeNormalizer())
    model, normalizer = Mock(), Mock()
    if fails == "model":
        model.save.side_effect = RuntimeError("save failed")
    else:
        model.save.side_effect = FakeModel().save
        normalizer.save.side_effect = RuntimeError("save failed")
    with pytest.raises(RuntimeError, match="save failed"):
        run.save_checkpoint(20, model, normalizer)
    assert not (run.directory / "checkpoints/step_000020").exists()
    assert (run.directory / "checkpoints/step_000010/model.zip").read_bytes() == b"model"


@pytest.mark.parametrize("step", [-1, True, 1.5])
def test_invalid_checkpoint_step_is_rejected(run: TrainingRun, step: int) -> None:
    with pytest.raises(ValueError):
        run.save_checkpoint(step, FakeModel(), FakeNormalizer())
