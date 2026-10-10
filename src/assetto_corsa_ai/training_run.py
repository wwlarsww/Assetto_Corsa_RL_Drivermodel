"""Append-only run records and paired snapshots; no training logic."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from io import BufferedIOBase
import json
from math import isfinite
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tomllib
from typing import Protocol
from uuid import uuid4


METRIC_FIELDS = (
    "step", "episode", "episode_reward", "episode_length", "policy_loss",
    "value_loss", "entropy_loss", "predictor_loss", "learning_rate",
)
MetricValue = int | float | None


class ModelSaver(Protocol):
    def save(self, path: BufferedIOBase, /) -> None: ...


class NormalizerSaver(Protocol):
    def save(self, path: str, /) -> None: ...


def _name_component(value: str) -> str:
    component = re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_")
    if not component:
        raise ValueError("Track and car must contain a usable name.")
    return component


@dataclass(frozen=True)
class TrainingRun:
    """One exclusively created directory. Existing artifacts are never replaced.

    Config and provenance are written once, metrics are appended, and each
    checkpoint/final snapshot can be saved once. Call from one training process.
    """

    directory: Path
    run_id: str
    started_at: datetime

    def save_config(self, content: str) -> None:
        """Preserve the exact validated TOML used by the training entry point."""
        tomllib.loads(content)
        with (self.directory / "config.toml").open("x", encoding="utf-8") as stream:
            stream.write(content)

    def save_metadata(self, project_directory: Path) -> None:
        """Collect provenance only; fail if the implementation cannot be identified."""
        project_directory = project_directory.resolve()
        git = ["git", "-c", f"safe.directory={project_directory.as_posix()}"]
        commit = subprocess.run(
            [*git, "rev-parse", "HEAD"], cwd=project_directory,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [*git, "status", "--porcelain", "--untracked-files=all"],
            cwd=project_directory, check=True, capture_output=True, text=True,
        ).stdout.strip()
        metadata = {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "git_commit": commit,
            "git_dirty": bool(dirty),
            "python_version": platform.python_version(),
            "stable_baselines3_version": version("stable-baselines3"),
            "torch_version": version("torch"),
            "operating_system": platform.platform(),
            "project_version": version("assetto-corsa-ai"),
        }
        with (self.directory / "metadata.json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
            stream.write("\n")

    def append_metrics(self, values: Mapping[str, MetricValue]) -> None:
        """Append sparse event rows; unavailable losses remain empty, not zero."""
        if "step" not in values or type(values["step"]) is not int or values["step"] < 0:
            raise ValueError("Metrics need a non-negative integer step.")
        for key, value in values.items():
            if key not in METRIC_FIELDS:
                raise ValueError(f"Unknown metric: {key}")
            if value is not None and (type(value) not in (int, float) or not isfinite(value)):
                raise ValueError(f"Metric {key} must be finite or None.")
        with (self.directory / "metrics.csv").open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=list[str](METRIC_FIELDS)).writerow(values)

    def save_checkpoint(self, step: int, model: ModelSaver, normalizer: NormalizerSaver) -> Path:
        if type(step) is not int or step < 0:
            raise ValueError("Checkpoint step must be a non-negative integer.")
        return self._save_pair(self.directory / "checkpoints" / f"step_{step:06d}", model, normalizer)

    def save_final_model(self, model: ModelSaver, normalizer: NormalizerSaver) -> Path:
        return self._save_pair(self.directory / "final", model, normalizer)

    def _save_pair(self, target: Path, model: ModelSaver, normalizer: NormalizerSaver) -> Path:
        target.mkdir(exist_ok=False)
        try:
            # Write hidden temporary files first. A loader requires both final
            # names, so interrupted writes cannot masquerade as a complete pair.
            with (target / ".model.tmp").open("xb") as stream:
                model.save(stream)
            normalizer.save(str(target / ".vecnormalize.tmp"))
            (target / ".model.tmp").rename(target / "model.zip")
            (target / ".vecnormalize.tmp").rename(target / "vecnormalize.pkl")
        except BaseException as error:
            # target was exclusively created above and belongs to this call.
            try:
                shutil.rmtree(target)
            except OSError as cleanup_error:
                error.add_note(f"Incomplete snapshot cleanup failed: {cleanup_error}")
            raise
        return target


def create_run_directory(root: Path, *, track: str, car: str) -> TrainingRun:
    started_at = datetime.now().astimezone()
    run_id = uuid4().hex[:6]
    name = f"{started_at:%Y%m%d_%H%M%S}_{_name_component(track)}_{_name_component(car)}_{run_id}"
    directory = root / name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "checkpoints").mkdir(exist_ok=False)
    with (directory / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(METRIC_FIELDS)
    return TrainingRun(directory, run_id, started_at)
