"""Portable checkpoint helpers for resumable training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True, slots=True)
class TrainingState:
    """Minimal progress state required to continue an on-policy run."""

    iteration: int
    environment_steps: int
    policy_version: int

    def __post_init__(self) -> None:
        if min(self.iteration, self.environment_steps, self.policy_version) < 0:
            raise ValueError("training state counters must be non-negative")


class _TrainingStatePayload(TypedDict):
    iteration: int
    environment_steps: int
    policy_version: int


class _CheckpointPayload(TypedDict):
    schema_version: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    training_state: _TrainingStatePayload
    configuration: object


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    training_state: TrainingState,
    configuration: dict[str, Any],
) -> None:
    """Atomically persist model, optimiser, progress and configuration."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "training_state": asdict(training_state),
        "configuration": configuration,
    }
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[TrainingState, dict[str, Any]]:
    """Load a trusted project checkpoint and restore supplied components."""

    payload = cast(
        _CheckpointPayload,
        torch.load(
            Path(path),
            map_location=map_location,
            weights_only=True,
        ),
    )
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    state_payload = payload["training_state"]
    state = TrainingState(
        iteration=state_payload["iteration"],
        environment_steps=state_payload["environment_steps"],
        policy_version=state_payload["policy_version"],
    )
    configuration_object = payload["configuration"]
    if not isinstance(configuration_object, dict):
        raise ValueError("checkpoint configuration must be a dictionary")
    configuration_mapping = cast(dict[object, object], configuration_object)
    if not all(isinstance(key, str) for key in configuration_mapping):
        raise ValueError("checkpoint configuration keys must be strings")
    configuration = cast(dict[str, Any], configuration_mapping)
    return state, configuration
