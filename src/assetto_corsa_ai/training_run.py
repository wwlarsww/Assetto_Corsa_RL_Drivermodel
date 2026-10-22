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
import os
import re
import shutil
import subprocess
import tomllib
from typing import Protocol
from uuid import uuid4


METRIC_FIELDS = (
    "step", "episode", "episode_reward", "episode_length", "policy_loss",
    "value_loss", "entropy_loss", "predictor_loss", "learning_rate",
    "predictor_valid_samples", "predictor_forward_mae", "predictor_lateral_mae",
    "performance_step", "performance_prediction", "style_step", "style_prediction",
    "termination_penalty", "total_reward", "future_weight_scale",
    "mean_predicted_progress", "mean_abs_lateral_offset",
    "policy_updates",
    "requested_longitudinal_abs", "applied_throttle", "applied_brake",
    "observed_longitudinal_abs",
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

    def save_latest(self, step: int, policy_updates: int, model: ModelSaver, normalizer: NormalizerSaver) -> Path:
        """Publish an immutable pair, then atomically replace its small pointer.

        This rotating diagnostics cache keeps the two newest versions; regular
        checkpoints and final artifacts retain their append-only semantics.
        """
        if any(type(value) is not int or value < 0 for value in (step, policy_updates)):
            raise ValueError("Snapshot step and policy updates must be non-negative integers.")
        latest = self.directory / "latest"
        latest.mkdir(exist_ok=True)
        target = latest / f"snapshot_{uuid4().hex}"
        self._save_pair(target, model, normalizer)
        pointer = latest / f".current_{uuid4().hex}.tmp"
        try:
            with pointer.open("x", encoding="utf-8") as stream:
                json.dump({"directory": target.name, "step": step, "policy_updates": policy_updates,
                           "created_at": datetime.now().astimezone().isoformat()}, stream)
            os.replace(pointer, latest / "current.json")
        finally:
            pointer.unlink(missing_ok=True)
        older = sorted((p for p in latest.glob("snapshot_*") if p != target),
                       key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for directory in older[1:]:
            # Cleanup is confined to versions exclusively owned by this cache.
            if (re.fullmatch(r"snapshot_[0-9a-f]{32}", directory.name)
                    and directory.resolve().is_relative_to(latest.resolve()) and not directory.is_symlink()):
                try:
                    shutil.rmtree(directory)
                except OSError:
                    pass  # A reader may still have a Windows file handle open.
        return target

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
