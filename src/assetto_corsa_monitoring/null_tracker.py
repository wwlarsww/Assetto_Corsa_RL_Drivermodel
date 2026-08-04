"""No operation implementation of experiment tracking."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from assetto_corsa_rl.training.tracking import (
    MetricValue,
    ParameterValue,
)


class NullTracker:
    """Silently discard all experiment tracking data."""

    @property
    def active(self) -> bool:
        """A null tracker never has an active run."""
        return False

    def start_run(
        self,
        *,
        run_name: str | None = None,
        parameters: Mapping[str, ParameterValue] | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> str | None:
        """Discard run metadata."""
        del run_name, parameters, tags
        return None

    def log_parameters(
        self,
        parameters: Mapping[str, ParameterValue],
    ) -> None:
        """Discard run parameters."""
        del parameters

    def log_metrics(
        self,
        metrics: Mapping[str, MetricValue],
        *,
        step: int,
    ) -> None:
        """Discard training metrics."""
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
        """Discard run tags."""
        del tags

    def end_run(
        self,
        *,
        status: str = "FINISHED",
    ) -> None:
        """Discard the final run status."""
        del status
