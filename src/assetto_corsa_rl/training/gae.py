"""Generalised advantage estimation with explicit terminal semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.types import FloatArray


@dataclass(frozen=True, slots=True)
class AdvantageBatch:
    """Advantages and value targets computed from one policy rollout."""

    advantages: FloatArray
    returns: FloatArray


def compute_gae(
    *,
    rewards: npt.ArrayLike,
    values: npt.ArrayLike,
    next_values: npt.ArrayLike,
    terminated: npt.ArrayLike,
    truncated: npt.ArrayLike,
    gamma: float,
    gae_lambda: float,
) -> AdvantageBatch:
    """Compute GAE while distinguishing termination from time-limit truncation.

    A true task termination has no value bootstrap. A time-limit truncation
    bootstraps from its final observation, but ends the recursive advantage
    trace so the following reset episode is never mixed into it.
    """

    reward = np.asarray(rewards, dtype=np.float32)
    value = np.asarray(values, dtype=np.float32)
    next_value = np.asarray(next_values, dtype=np.float32)
    terminal = np.asarray(terminated, dtype=np.bool_)
    time_limit = np.asarray(truncated, dtype=np.bool_)
    if reward.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    expected_shape = reward.shape
    for name, array in {
        "values": value,
        "next_values": next_value,
        "terminated": terminal,
        "truncated": time_limit,
    }.items():
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    if reward.size == 0:
        raise ValueError("GAE requires at least one rollout step")
    if not all(np.all(np.isfinite(array)) for array in (reward, value, next_value)):
        raise ValueError("GAE inputs must contain only finite values")
    if np.any(terminal & time_limit):
        raise ValueError("a step cannot be both terminated and truncated")
    if not 0 < gamma <= 1 or not 0 < gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must lie in (0, 1]")

    bootstrap = (~terminal).astype(np.float32)
    trace_continues = (~(terminal | time_limit)).astype(np.float32)
    deltas = reward + gamma * next_value * bootstrap - value
    advantages = np.zeros_like(reward)
    running_advantage = 0.0
    for index in range(reward.size - 1, -1, -1):
        running_advantage = float(
            deltas[index] + gamma * gae_lambda * trace_continues[index] * running_advantage
        )
        advantages[index] = running_advantage
    return AdvantageBatch(
        advantages=advantages,
        returns=np.asarray(advantages + value, dtype=np.float32),
    )
