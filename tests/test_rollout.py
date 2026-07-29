from __future__ import annotations

import numpy as np
import pytest

from assetto_corsa_rl.features.schema import OBSERVATION_SIZE
from assetto_corsa_rl.training.rollout import (
    OnPolicyCollector,
    StalePolicyFragmentError,
    TrajectoryContext,
    TrajectoryFragment,
)


def _fragment(
    policy_version: int,
    steps: int = 3,
    *,
    with_context: bool = False,
) -> TrajectoryFragment:
    context = (
        TrajectoryContext(
            positions_xy_m=np.zeros((steps, 2), dtype=np.float32),
            vehicle_heading_rad=np.zeros(steps, dtype=np.float32),
            simulation_time_s=np.arange(steps, dtype=np.float32),
            trajectory_id=np.zeros(steps, dtype=np.int64),
            reference_progress_m=np.arange(steps, dtype=np.float32),
        )
        if with_context
        else None
    )
    return TrajectoryFragment(
        policy_version=policy_version,
        observations=np.zeros((steps, OBSERVATION_SIZE), dtype=np.float32),
        actions=np.zeros((steps, 2), dtype=np.float32),
        rewards=np.ones(steps, dtype=np.float32),
        terminated=np.zeros(steps, dtype=np.bool_),
        truncated=np.zeros(steps, dtype=np.bool_),
        log_probabilities=np.zeros(steps, dtype=np.float32),
        values=np.zeros(steps, dtype=np.float32),
        next_values=np.zeros(steps, dtype=np.float32),
        context=context,
    )


def test_collector_rejects_stale_policy_and_builds_flat_batch() -> None:
    collector = OnPolicyCollector(policy_version=4, target_step_count=5)
    with pytest.raises(StalePolicyFragmentError, match="received 3"):
        collector.add(_fragment(3))
    collector.add(_fragment(4))
    collector.add(_fragment(4))
    assert collector.is_full
    batch = collector.build()
    assert batch.policy_version == 4
    assert batch.step_count == 6
    assert batch.observations.shape == (6, OBSERVATION_SIZE)


def test_incomplete_collector_and_invalid_fragment_fail() -> None:
    collector = OnPolicyCollector(policy_version=0, target_step_count=4)
    collector.add(_fragment(0, steps=2))
    with pytest.raises(RuntimeError, match="target"):
        collector.build()
    with pytest.raises(ValueError, match="rewards"):
        TrajectoryFragment(
            policy_version=0,
            observations=np.zeros((2, OBSERVATION_SIZE), dtype=np.float32),
            actions=np.zeros((2, 2), dtype=np.float32),
            rewards=np.zeros(1, dtype=np.float32),
            terminated=np.zeros(2, dtype=np.bool_),
            truncated=np.zeros(2, dtype=np.bool_),
            log_probabilities=np.zeros(2, dtype=np.float32),
            values=np.zeros(2, dtype=np.float32),
            next_values=np.zeros(2, dtype=np.float32),
        )


def test_full_collector_rejects_additional_fragment() -> None:
    collector = OnPolicyCollector(policy_version=1, target_step_count=1)
    collector.add(_fragment(1, steps=1))
    with pytest.raises(RuntimeError, match="already full"):
        collector.add(_fragment(1, steps=1))


def test_collector_concatenates_predictor_context_and_rejects_mixed_fragments() -> None:
    collector = OnPolicyCollector(policy_version=2, target_step_count=4)
    collector.add(_fragment(2, steps=2, with_context=True))
    collector.add(_fragment(2, steps=2, with_context=True))
    batch = collector.build()
    assert batch.context is not None
    assert batch.context.step_count == 4

    mixed = OnPolicyCollector(policy_version=2, target_step_count=4)
    mixed.add(_fragment(2, steps=2, with_context=True))
    mixed.add(_fragment(2, steps=2))
    with pytest.raises(ValueError, match="either provide"):
        mixed.build()


def test_trajectory_context_validates_shapes() -> None:
    with pytest.raises(ValueError, match="vehicle_heading"):
        TrajectoryContext(
            positions_xy_m=np.zeros((2, 2), dtype=np.float32),
            vehicle_heading_rad=np.zeros(1, dtype=np.float32),
            simulation_time_s=np.zeros(2, dtype=np.float32),
            trajectory_id=np.zeros(2, dtype=np.int64),
            reference_progress_m=np.zeros(2, dtype=np.float32),
        )
