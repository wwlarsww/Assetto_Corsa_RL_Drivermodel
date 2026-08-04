"""Configuration models for optional experiment monitoring."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Self


@dataclass(frozen=True, slots=True)
class MonitoringConfig:
    """Settings for the optional experiment tracking backend."""

    enabled: bool = False
    backend: str = "none"
    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "Assetto Corsa PPO"
    run_name: str | None = None
    log_interval_updates: int = 1
    fail_on_error: bool = False

    def __post_init__(self) -> None:
        """Validate monitoring settings independently of the RL config."""

        backend = self.normalized_backend
        if backend not in {"none", "mlflow"}:
            raise ValueError(
                "monitoring.backend must be 'none' or 'mlflow', "
                f"got {self.backend!r}"
            )
        if self.log_interval_updates < 1:
            raise ValueError("monitoring.log_interval_updates must be at least 1")
        if self.enabled and backend == "mlflow":
            if not self.tracking_uri.strip():
                raise ValueError("monitoring.tracking_uri must not be empty")
            if not self.experiment_name.strip():
                raise ValueError("monitoring.experiment_name must not be empty")

    @property
    def normalized_backend(self) -> str:
        """Return the normalized backend identifier."""

        return self.backend.strip().lower()

    @classmethod
    def from_toml(cls, path: str | Path) -> Self:
        """Load only the ``[monitoring]`` table from a TOML file."""

        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file does not exist: {config_path}"
            )
        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)

        raw = payload.get("monitoring", {})
        if not isinstance(raw, dict):
            raise ValueError("[monitoring] must be a TOML table")

        allowed = {field.name for field in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown key(s) in [monitoring]: {names}")

        values: dict[str, Any] = dict(raw)
        run_name = values.get("run_name")
        if isinstance(run_name, str) and not run_name.strip():
            values["run_name"] = None
        return cls(**values)
