"""Backend independent experiment tracking contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, TypeAlias

ParameterValue: TypeAlias = str | int | float | bool | None
MetricValue: TypeAlias = int | float


class ExperimentTracker(Protocol):
    """Interface used by the PPO training process."""

    @property
    def active(self) -> bool:
        """Return whether a tracking run is active."""
        ...

    def start_run(
        self,
        *,
        run_name: str | None = None,
        parameters: Mapping[str, ParameterValue] | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> str | None:
        """Start one experiment run."""
        ...

    def log_parameters(
        self,
        parameters: Mapping[str, ParameterValue],
    ) -> None:
        """Store immutable run parameters."""
        ...

    def log_metrics(
        self,
        metrics: Mapping[str, MetricValue],
        *,
        step: int,
    ) -> None:
        """Store numeric metrics for one PPO update."""
        ...

    def log_artifact(
        self,
        path: str | Path,
        *,
        artifact_path: str | None = None,
    ) -> None:
        """Store a local file as an experiment artifact."""
        ...

    def set_tags(
        self,
        tags: Mapping[str, str],
    ) -> None:
        """Add or update run tags."""
        ...

    def end_run(
        self,
        *,
        status: str = "FINISHED",
    ) -> None:
        """Finish the active tracking run."""
        ...


def flatten_parameters(
    values: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, ParameterValue]:
    """Flatten nested configuration values into dotted parameter names."""

    flattened: dict[str, ParameterValue] = {}

    for raw_name, value in values.items():
        name = f"{prefix}.{raw_name}" if prefix else str(raw_name)

        if isinstance(value, Mapping):
            flattened.update(
                flatten_parameters(
                    value,
                    prefix=name,
                )
            )
            continue

        if value is None:
            flattened[name] = None
            continue

        if isinstance(value, str | int | float | bool):
            flattened[name] = value
            continue

        if isinstance(value, list | tuple):
            flattened[name] = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            continue

        flattened[name] = str(value)

    return flattened
