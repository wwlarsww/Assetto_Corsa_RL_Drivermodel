"""Backend-neutral experiment tracking contracts and parameter helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

ParameterValue = str | int | float | bool | None
MetricValue = int | float


class ExperimentTracker(Protocol):
    """Minimal tracking interface consumed by the RL training core."""

    @property
    def active(self) -> bool:
        """Return whether at least one run is active."""

    @property
    def run_id(self) -> str | None:
        """Return the currently active run identifier, if available."""

    def start_run(
        self,
        *,
        run_name: str | None = None,
        parameters: Mapping[str, ParameterValue] | None = None,
        tags: Mapping[str, str] | None = None,
        nested: bool = False,
    ) -> str | None:
        """Start a top-level or nested experiment run."""

    def log_parameters(
        self,
        parameters: Mapping[str, ParameterValue],
    ) -> None:
        """Log immutable run parameters."""

    def log_metrics(
        self,
        metrics: Mapping[str, MetricValue],
        *,
        step: int,
    ) -> None:
        """Log numeric metrics at one nonnegative step."""

    def log_artifact(
        self,
        path: str | Path,
        *,
        artifact_path: str | None = None,
    ) -> None:
        """Upload one existing local file."""

    def set_tags(
        self,
        tags: Mapping[str, str],
    ) -> None:
        """Add or update run tags."""

    def end_run(
        self,
        *,
        status: str = "FINISHED",
    ) -> None:
        """Finish the currently active run."""


def flatten_parameters(
    values: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, ParameterValue]:
    """Flatten nested mappings into stable dot-separated parameter names."""

    flattened: dict[str, ParameterValue] = {}
    for raw_name, value in values.items():
        name = f"{prefix}.{raw_name}" if prefix else str(raw_name)
        if isinstance(value, Mapping):
            flattened.update(flatten_parameters(value, prefix=name))
            continue
        flattened[name] = _parameter_value(value)
    return flattened


def _parameter_value(value: Any) -> ParameterValue:
    """Convert structured values to deterministic tracker-safe scalars."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list | tuple):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return str(value)
