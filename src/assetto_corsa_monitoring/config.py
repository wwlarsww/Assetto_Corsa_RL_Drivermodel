"""Configuration models for experiment monitoring."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Self


@dataclass(frozen=True, slots=True)
class MonitoringConfig:
    """Configuration for optional experiment tracking."""

    enabled: bool = False
    backend: str = "none"
    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "Assetto Corsa PPO"
    run_name: str | None = None
    log_interval_updates: int = 1
    fail_on_error: bool = False

    def __post_init__(self) -> None:
        """Validate the monitoring configuration."""

        backend = self.normalized_backend

        if backend not in {"none", "mlflow"}:
            raise ValueError(
                "Monitoring backend must be 'none' or 'mlflow', "
                f"but received {self.backend!r}."
            )

        if self.log_interval_updates < 1:
            raise ValueError("log_interval_updates must be at least 1.")

        if self.enabled and backend == "mlflow":
            if not self.tracking_uri.strip():
                raise ValueError(
                    "tracking_uri must not be empty when MLflow is enabled."
                )

            if not self.experiment_name.strip():
                raise ValueError(
                    "experiment_name must not be empty when MLflow is enabled."
                )

    @property
    def normalized_backend(self) -> str:
        """Return the normalized monitoring backend."""

        return self.backend.strip().lower()

    @classmethod
    def from_toml(
        cls,
        path: str | Path,
    ) -> Self:
        """Load the monitoring section from a TOML configuration file."""

        config_path = Path(path)

        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

        with config_path.open("rb") as handle:
            raw_config = tomllib.load(handle)

        raw_monitoring = raw_config.get("monitoring", {})

        if not isinstance(raw_monitoring, dict):
            raise ValueError("[monitoring] must be a TOML table.")

        allowed_keys = {field.name for field in fields(cls)}
        unknown_keys = set(raw_monitoring) - allowed_keys

        if unknown_keys:
            names = ", ".join(sorted(unknown_keys))
            raise ValueError(f"Unknown key(s) in [monitoring]: {names}")

        values: dict[str, Any] = dict(raw_monitoring)

        run_name = values.get("run_name")

        if isinstance(run_name, str) and not run_name.strip():
            values["run_name"] = None

        return cls(**values)
