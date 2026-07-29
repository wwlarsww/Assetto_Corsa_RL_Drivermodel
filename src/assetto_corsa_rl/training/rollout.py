"""Policy-versioned trajectory fragments for hybrid on-policy sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.types import FloatArray


class StalePolicyFragmentError(ValueError):
    """Raised when a collector receives experience from an old policy."""


@dataclass(frozen=True, slots=True)
class TrajectoryContext:
    """Telemetry sidecar required by predictor targets and reference rewards.

    ``trajectory_id`` must be unique across actors and episodes. Equal IDs may
    only describe temporally contiguous samples from the same episode.
    """

    positions_xy_m: FloatArray
    vehicle_heading_rad: FloatArray
    simulation_time_s: FloatArray
    trajectory_id: npt.NDArray[np.int64]
    reference_progress_m: FloatArray

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_xy_m, dtype=np.float32)
        headings = np.asarray(self.vehicle_heading_rad, dtype=np.float32)
        times = np.asarray(self.simulation_time_s, dtype=np.float32)
        trajectory_ids = np.asarray(self.trajectory_id, dtype=np.int64)
        progress = np.asarray(self.reference_progress_m, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions_xy_m must have shape (steps, 2)")
        step_count = positions.shape[0]
        for name, shape in {
            "vehicle_heading_rad": headings.shape,
            "simulation_time_s": times.shape,
            "trajectory_id": trajectory_ids.shape,
            "reference_progress_m": progress.shape,
        }.items():
            if shape != (step_count,):
                raise ValueError(f"{name} must have shape ({step_count},)")
        for name, array in {
            "positions_xy_m": positions,
            "vehicle_heading_rad": headings,
            "simulation_time_s": times,
            "reference_progress_m": progress,
        }.items():
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")
        object.__setattr__(self, "positions_xy_m", positions)
        object.__setattr__(self, "vehicle_heading_rad", headings)
        object.__setattr__(self, "simulation_time_s", times)
        object.__setattr__(self, "trajectory_id", trajectory_ids)
        object.__setattr__(self, "reference_progress_m", progress)

    @property
    def step_count(self) -> int:
        return int(self.positions_xy_m.shape[0])


@dataclass(frozen=True, slots=True)
class TrajectoryFragment:
    """Fixed-policy fragment emitted by one independent actor."""

    policy_version: int
    observations: FloatArray
    actions: FloatArray
    rewards: FloatArray
    terminated: npt.NDArray[np.bool_]
    truncated: npt.NDArray[np.bool_]
    log_probabilities: FloatArray
    values: FloatArray
    next_values: FloatArray
    context: TrajectoryContext | None = None

    def __post_init__(self) -> None:
        observations = np.asarray(self.observations, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        terminated = np.asarray(self.terminated, dtype=np.bool_)
        truncated = np.asarray(self.truncated, dtype=np.bool_)
        log_probabilities = np.asarray(self.log_probabilities, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        next_values = np.asarray(self.next_values, dtype=np.float32)

        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError("observations and actions must be two-dimensional")
        step_count = observations.shape[0]
        if actions.shape[0] != step_count:
            raise ValueError("observations and actions have different step counts")
        for name, array in {
            "rewards": rewards,
            "terminated": terminated,
            "truncated": truncated,
            "log_probabilities": log_probabilities,
            "values": values,
            "next_values": next_values,
        }.items():
            if array.shape != (step_count,):
                raise ValueError(f"{name} must have shape ({step_count},)")
        for name, array in {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "log_probabilities": log_probabilities,
            "values": values,
            "next_values": next_values,
        }.items():
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")

        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "terminated", terminated)
        object.__setattr__(self, "truncated", truncated)
        object.__setattr__(self, "log_probabilities", log_probabilities)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "next_values", next_values)
        if self.context is not None and self.context.step_count != step_count:
            raise ValueError("context and observations have different step counts")

    @property
    def step_count(self) -> int:
        return int(self.rewards.size)


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """Flat batch ready for prediction shaping, GAE and PPO."""

    policy_version: int
    observations: FloatArray
    actions: FloatArray
    rewards: FloatArray
    terminated: npt.NDArray[np.bool_]
    truncated: npt.NDArray[np.bool_]
    log_probabilities: FloatArray
    values: FloatArray
    next_values: FloatArray
    context: TrajectoryContext | None = None

    @property
    def step_count(self) -> int:
        return int(self.rewards.size)


class OnPolicyCollector:
    """Validate policy versions and assemble complete actor fragments."""

    def __init__(self, *, policy_version: int, target_step_count: int) -> None:
        if policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if target_step_count <= 0:
            raise ValueError("target_step_count must be positive")
        self.policy_version = policy_version
        self.target_step_count = target_step_count
        self._fragments: list[TrajectoryFragment] = []
        self._step_count = 0

    @property
    def is_full(self) -> bool:
        return self._step_count >= self.target_step_count

    @property
    def step_count(self) -> int:
        return self._step_count

    def add(self, fragment: TrajectoryFragment) -> None:
        """Accept only complete data generated by the current policy."""

        if fragment.policy_version != self.policy_version:
            raise StalePolicyFragmentError(
                f"expected policy version {self.policy_version}, received {fragment.policy_version}"
            )
        if self.is_full:
            raise RuntimeError("collector is already full")
        self._fragments.append(fragment)
        self._step_count += fragment.step_count

    def build(self) -> RolloutBatch:
        """Flatten accepted fragments without splitting actor trajectories."""

        if not self.is_full:
            raise RuntimeError(
                f"collector has {self._step_count} steps; target is {self.target_step_count}"
            )
        contexts = [fragment.context for fragment in self._fragments]
        if any(context is None for context in contexts) and any(
            context is not None for context in contexts
        ):
            raise ValueError("all fragments must either provide trajectory context or omit it")
        context = (
            _concatenate_contexts(
                [item for item in contexts if item is not None],
            )
            if contexts and contexts[0] is not None
            else None
        )
        return RolloutBatch(
            policy_version=self.policy_version,
            observations=np.concatenate(
                [fragment.observations for fragment in self._fragments],
                axis=0,
            ),
            actions=np.concatenate(
                [fragment.actions for fragment in self._fragments],
                axis=0,
            ),
            rewards=np.concatenate(
                [fragment.rewards for fragment in self._fragments],
                axis=0,
            ),
            terminated=np.concatenate(
                [fragment.terminated for fragment in self._fragments],
                axis=0,
            ),
            truncated=np.concatenate(
                [fragment.truncated for fragment in self._fragments],
                axis=0,
            ),
            log_probabilities=np.concatenate(
                [fragment.log_probabilities for fragment in self._fragments],
                axis=0,
            ),
            values=np.concatenate(
                [fragment.values for fragment in self._fragments],
                axis=0,
            ),
            next_values=np.concatenate(
                [fragment.next_values for fragment in self._fragments],
                axis=0,
            ),
            context=context,
        )


def _concatenate_contexts(contexts: list[TrajectoryContext]) -> TrajectoryContext:
    return TrajectoryContext(
        positions_xy_m=np.concatenate([context.positions_xy_m for context in contexts], axis=0),
        vehicle_heading_rad=np.concatenate(
            [context.vehicle_heading_rad for context in contexts],
            axis=0,
        ),
        simulation_time_s=np.concatenate(
            [context.simulation_time_s for context in contexts],
            axis=0,
        ),
        trajectory_id=np.concatenate([context.trajectory_id for context in contexts], axis=0),
        reference_progress_m=np.concatenate(
            [context.reference_progress_m for context in contexts],
            axis=0,
        ),
    )
