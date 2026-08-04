"""No-operation implementation of experiment tracking."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from assetto_corsa_rl.training.tracking import (
    MetricValue,
    ParameterValue,
)


class NullTracker:
    """Tracker implementation that silently discards all data."""

    @property
    def active(self) -> bool:
        """A null tracker never has an active run."""

        return False

    @property
    def run_id(self) -> str | None:
        """A null tracker never owns a run identifier."""

        return None

    def start_run(
        self,
        *,
        run_name: str | None = None,
        parameters: Mapping[str, ParameterValue] | None = None,
        tags: Mapping[str, str] | None = None,
        nested: bool = False,
    ) -> None:
        """Discard run metadata."""

        del run_name, parameters, tags, nested

    def log_parameters(
        self,
        parameters: Mapping[str, ParameterValue],
    ) -> None:
        """Discard parameters."""

        del parameters

    def log_metrics(
        self,
        metrics: Mapping[str, MetricValue],
        *,
        step: int,
    ) -> None:
        """Discard metrics."""

        del metrics, step

    def log_artifact(
        self,
        path: str | Path,
        *,
        artifact_path: str | None = None,
    ) -> None:
        """Discard an artifact."""

        del path, artifact_path

    def set_tags(
        self,
        tags: Mapping[str, str],
    ) -> None:
        """Discard tags."""

        del tags

    def end_run(
        self,
        *,
        status: str = "FINISHED",
    ) -> None:
        """Discard the terminal status."""

        del status
