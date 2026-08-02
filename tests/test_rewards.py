from __future__ import annotations

import numpy as np
import pytest

from assetto_corsa_rl.rewards.components import (
    EnvironmentReward,
    PredictedProgressReward,
)
from assetto_corsa_rl.types import TerminationReason


def test_environment_reward_penalises_only_safety_termination() -> None:
    reward = EnvironmentReward(progress_scale=2.0, termination_penalty=-10.0)
    safe = reward.compute(3.0, TerminationReason.LAP_COMPLETE)
    unsafe = reward.compute(3.0, TerminationReason.OFF_TRACK)

    assert safe.total == 6.0
    assert unsafe.total == -4.0
    assert unsafe.as_dict()["reward_termination"] == -10.0


def test_baseline_variant_adds_predicted_progress_without_imitation() -> None:
    predicted = np.zeros((2, 3, 2), dtype=np.float32)
    predicted[0] = [[0.0, 1.0], [0.0, 3.0], [2.0, 3.0]]
    reference = np.zeros((2, 3, 2), dtype=np.float32)
    reference[0] = [[0.0, 0.0], [0.0, 3.0], [4.0, 3.0]]
    reference_progress = np.zeros((2, 3), dtype=np.float32)
    reference_progress[0] = [0.0, 5.0, 11.0]

    terms = PredictedProgressReward().compute(
        predicted,
        reference,
        reference_progress,
        [True, False],
    )
    # The endpoint z is only 3 m. Its projection is halfway along a reference
    # segment whose exact arc coordinate spans 5..11 m.
    np.testing.assert_allclose(terms.predicted_progress, [8.0, 0.0])
    shaped = PredictedProgressReward.shape_rollout_rewards(
        [1.0, 2.0],
        terms,
        predicted_progress_scale=0.5,
    )
    np.testing.assert_allclose(shaped, [5.0, 2.0])
    assert not hasattr(terms, "imitation_reward")


def test_rollout_shaping_rejects_inconsistent_coefficients() -> None:
    reference = np.zeros((1, 2, 2), dtype=np.float32)
    reference[0, 1, 1] = 1.0
    terms = PredictedProgressReward().compute(
        np.zeros((1, 2, 2), dtype=np.float32),
        reference,
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="equal shape"):
        PredictedProgressReward.shape_rollout_rewards(
            [0.0, 0.0],
            terms,
            predicted_progress_scale=1.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        PredictedProgressReward.shape_rollout_rewards(
            [0.0],
            terms,
            predicted_progress_scale=-1.0,
        )
