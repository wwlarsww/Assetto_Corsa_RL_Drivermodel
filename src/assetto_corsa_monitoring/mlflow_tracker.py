"""MLflow implementation of experiment tracking."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar

from assetto_corsa_rl.training.tracking import (
    MetricValue,
    ParameterValue,
)

LOGGER = logging.getLogger(__name__)


class MLflowTracker:
    """Track training runs using an MLflow tracking server."""

    _VALID_END_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {
            "FINISHED",
            "FAILED",
            "KILLED",
        }
    )

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
        self._run_id: str | None = None

    @property
    def active(self) -> bool:
        """Return whether an MLflow run is active."""
        return self._run_id is not None

    def start_run(
        self,
        *,
        run_name: str | None = None,
        parameters: Mapping[str, ParameterValue] | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> str | None:
        """Start a new MLflow run."""

        if self.active:
            raise RuntimeError(f"MLflow run {self._run_id!r} is already active.")

        try:
            mlflow = self._get_mlflow()

            mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment_name)

            run = mlflow.start_run(
                run_name=run_name,
                tags=dict(tags) if tags else None,
            )

            self._run_id = run.info.run_id

            if parameters:
                self.log_parameters(parameters)

            LOGGER.info(
                "Started MLflow run %s in experiment %s.",
                self._run_id,
                self._experiment_name,
            )

            return self._run_id

        except Exception as error:
            self._handle_error(
                "Could not start the MLflow run.",
                error,
            )
            return None

    def log_parameters(
        self,
        parameters: Mapping[str, ParameterValue],
    ) -> None:
        """Log run parameters."""

        if not parameters:
            return

        if not self._require_active_run():
            return

        prepared_parameters = {
            str(name): self._prepare_parameter_value(value)
            for name, value in parameters.items()
        }

        try:
            self._get_mlflow().log_params(prepared_parameters)

        except Exception as error:
            self._handle_error(
                "Could not log MLflow parameters.",
                error,
            )

    def log_metrics(
        self,
        metrics: Mapping[str, MetricValue],
        *,
        step: int,
    ) -> None:
        """Log finite numeric metrics."""

        if step < 0:
            raise ValueError("Metric step must not be negative.")

        if not metrics:
            return

        if not self._require_active_run():
            return

        prepared_metrics: dict[str, float] = {}

        for name, value in metrics.items():
            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                LOGGER.warning(
                    "Skipping non finite metric %s=%r at step %d.",
                    name,
                    value,
                    step,
                )
                continue

            prepared_metrics[str(name)] = numeric_value

        if not prepared_metrics:
            return

        try:
            self._get_mlflow().log_metrics(
                prepared_metrics,
                step=step,
            )

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
        """Log one local file as an MLflow artifact."""

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

    def set_tags(
        self,
        tags: Mapping[str, str],
    ) -> None:
        """Add or update tags on the active run."""

        if not tags:
            return

        if not self._require_active_run():
            return

        prepared_tags = {str(name): str(value) for name, value in tags.items()}

        try:
            self._get_mlflow().set_tags(prepared_tags)

        except Exception as error:
            self._handle_error(
                "Could not set MLflow tags.",
                error,
            )

    def end_run(
        self,
        *,
        status: str = "FINISHED",
    ) -> None:
        """Finish the active MLflow run."""

        if not self.active:
            return

        normalized_status = status.strip().upper()

        if normalized_status not in self._VALID_END_STATUSES:
            raise ValueError(
                "MLflow run status must be one of "
                f"{sorted(self._VALID_END_STATUSES)}, "
                f"but received {status!r}."
            )

        run_id = self._run_id

        try:
            self._get_mlflow().end_run(
                status=normalized_status,
            )

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
            self._run_id = None

    def _get_mlflow(self) -> Any:
        """Import MLflow only when the tracker is actually used."""

        if self._mlflow is not None:
            return self._mlflow

        try:
            self._mlflow = import_module("mlflow")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "MLflow monitoring is enabled, but the 'mlflow' package "
                "is not installed. Install the monitoring dependencies "
                'with: python -m pip install -e ".[dev,monitoring]"'
            ) from error
        return self._mlflow

    def _require_active_run(self) -> bool:
        """Check whether logging is currently possible."""

        if self.active:
            return True

        message = "Cannot log because no MLflow run is active."

        if self._fail_on_error:
            raise RuntimeError(message)

        LOGGER.warning(message)
        return False

    def _handle_error(
        self,
        message: str,
        error: Exception,
    ) -> None:
        """Either raise or log an MLflow related error."""

        if self._fail_on_error:
            raise RuntimeError(message) from error

        LOGGER.exception(
            "%s Training will continue without failing.",
            message,
        )

    @staticmethod
    def _prepare_parameter_value(
        value: ParameterValue,
    ) -> str | int | float | bool:
        """Convert parameter values into MLflow compatible values."""

        if value is None:
            return "None"

        if isinstance(value, bool):
            return value

        return value
