"""Factory for experiment tracker implementations."""

from __future__ import annotations

from assetto_corsa_monitoring.config import MonitoringConfig
from assetto_corsa_monitoring.mlflow_tracker import MLflowTracker
from assetto_corsa_monitoring.null_tracker import NullTracker
from assetto_corsa_rl.training.tracking import ExperimentTracker


def create_tracker(config: MonitoringConfig) -> ExperimentTracker:
    """Create the configured experiment tracker."""

    if not config.enabled or config.normalized_backend == "none":
        return NullTracker()
    if config.normalized_backend == "mlflow":
        return MLflowTracker(
            tracking_uri=config.tracking_uri,
            experiment_name=config.experiment_name,
            fail_on_error=config.fail_on_error,
        )
    raise ValueError(f"Unsupported monitoring backend: {config.backend!r}")
