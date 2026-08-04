"""MLflow implementation of the backend-neutral experiment tracker."""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from assetto_corsa_rl.training.tracking import (
    MetricValue,
    ParameterValue,
)

LOGGER = logging.getLogger("assetto_corsa_rl")


class MLflowTracker:
    """Track parent training runs and nested evaluation runs in MLflow."""

    _VALID_END_STATUSES = frozenset({"FINISHED", "FAILED", "KILLED"})

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        fail_on_error: bool = False,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._fail_on_error = fail_on_error
        self._mlflow: Any | None = None
        self._run_ids: list[str] = []

    @property
    def active(self) -> bool:
        """Return whether at least one MLflow run is active."""

        return bool(self._run_ids)

    @property
    def run_id(self) -> str | None:
        """Return the currently active run identifier."""

        return self._run_ids[-1] if self._run_ids else None

    def start_run(
        self,
        *,
        run_name: str | None = None,
        parameters: Mapping[str, ParameterValue] | None = None,
        tags: Mapping[str, str] | None = None,
        nested: bool = False,
    ) -> str | None:
        """Start a top-level run or a child of the current active run."""

        if nested and not self.active:
            raise RuntimeError("Cannot start a nested MLflow run without a parent run.")
        if self.active and not nested:
            raise RuntimeError(f"MLflow run {self.run_id!r} is already active.")

        try:
            mlflow = self._get_mlflow()
            mlflow.set_tracking_uri(self._tracking_uri)
            if not nested:
                mlflow.set_experiment(self._experiment_name)
            run = mlflow.start_run(
                run_name=run_name,
                tags=dict(tags) if tags else None,
                nested=nested,
            )
            run_id = str(run.info.run_id)
            self._run_ids.append(run_id)
            if parameters:
                self.log_parameters(parameters)
            run_kind = "nested MLflow run" if nested else "MLflow run"
            LOGGER.info(
                "Started %s %s in experiment %s.",
                run_kind,
                run_id,
                self._experiment_name,
            )
            return run_id
        except Exception as error:
            self._handle_error("Could not start the MLflow run.", error)
            return None

    def log_parameters(
        self,
        parameters: Mapping[str, ParameterValue],
    ) -> None:
        """Log run parameters after converting unsupported values to strings."""

        if not parameters or not self._require_active_run():
            return
        prepared = {
            str(name): self._prepare_parameter_value(value)
            for name, value in parameters.items()
        }
        try:
            self._get_mlflow().log_params(prepared)
        except Exception as error:
            self._handle_error("Could not log MLflow parameters.", error)

    def log_metrics(
        self,
        metrics: Mapping[str, MetricValue],
        *,
        step: int,
    ) -> None:
        """Log finite numeric metrics at one nonnegative step."""

        if step < 0:
            raise ValueError("Metric step must not be negative.")
        if not metrics or not self._require_active_run():
            return

        prepared: dict[str, float] = {}
        for name, value in metrics.items():
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                LOGGER.warning(
                    "Skipping non-finite metric %s=%r at step %d.",
                    name,
                    value,
                    step,
                )
                continue
            prepared[str(name)] = numeric_value
        if not prepared:
            return

        try:
            self._get_mlflow().log_metrics(prepared, step=step)
        except Exception as error:
            self._handle_error(
                f"Could not log MLflow metrics at step {step}.",
                error,
            )

    def log_artifact(
        self,
        path: str | Path,
        *,
        artifact_path: str | None = None,
    ) -> None:
        """Log one existing local file as an MLflow artifact."""

        if not self._require_active_run():
            return
        resolved_path = Path(path).expanduser().resolve()
        if not resolved_path.is_file():
            message = f"Artifact path is not an existing file: {resolved_path}"
            if self._fail_on_error:
                raise FileNotFoundError(message)
            LOGGER.warning(message)
            return

        try:
            self._get_mlflow().log_artifact(
                str(resolved_path),
                artifact_path=artifact_path,
            )
        except Exception as error:
            self._handle_error(
                f"Could not log artifact {resolved_path}.",
                error,
            )

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Add or update tags on the active run."""

        if not tags or not self._require_active_run():
            return
        prepared = {str(name): str(value) for name, value in tags.items()}
        try:
            self._get_mlflow().set_tags(prepared)
        except Exception as error:
            self._handle_error("Could not set MLflow tags.", error)

    def end_run(self, *, status: str = "FINISHED") -> None:
        """Finish the active child or parent run and restore its parent."""

        if not self.active:
            return
        normalized_status = status.strip().upper()
        if normalized_status not in self._VALID_END_STATUSES:
            raise ValueError(
                "MLflow run status must be one of "
                f"{sorted(self._VALID_END_STATUSES)}, "
                f"but received {status!r}."
            )

        run_id = self.run_id
        try:
            self._get_mlflow().end_run(status=normalized_status)
            LOGGER.info(
                "Finished MLflow run %s with status %s.",
                run_id,
                normalized_status,
            )
        except Exception as error:
            self._handle_error(
                f"Could not finish MLflow run {run_id}.",
                error,
            )
        finally:
            self._run_ids.pop()

    def _get_mlflow(self) -> Any:
        """Import MLflow lazily so the core installation stays optional."""

        if self._mlflow is not None:
            return self._mlflow
        try:
            self._mlflow = importlib.import_module("mlflow")
        except ImportError as error:
            raise RuntimeError(
                "MLflow monitoring is enabled, but the 'mlflow' package is not "
                "installed. Install the monitoring dependencies with: "
                'python -m pip install -e ".[dev,monitoring]"'
            ) from error
        return self._mlflow

    def _require_active_run(self) -> bool:
        """Return whether logging is currently possible."""

        if self.active:
            return True
        message = "Cannot log because no MLflow run is active."
        if self._fail_on_error:
            raise RuntimeError(message)
        LOGGER.warning(message)
        return False

    def _handle_error(self, message: str, error: Exception) -> None:
        """Either raise or log an MLflow-related error."""

        if self._fail_on_error:
            raise RuntimeError(message) from error
        LOGGER.exception("%s Training will continue.", message)

    @staticmethod
    def _prepare_parameter_value(
        value: ParameterValue,
    ) -> str | int | float | bool:
        """Convert parameter values into MLflow-compatible values."""

        if value is None:
            return "None"
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float | str):
            return value
        return str(value)
