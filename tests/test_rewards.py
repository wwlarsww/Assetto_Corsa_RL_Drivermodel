from __future__ import annotations

import numpy as np
import pytest

from assetto_corsa_rl.rewards.components import (
    EnvironmentReward,
    PredictedProgressReward,
    PredictionReward,
    StyleLagrangeController,
)
from assetto_corsa_rl.types import TerminationReason


def test_environment_reward_penalises_only_safety_termination() -> None:
    reward = EnvironmentReward(progress_scale=2.0, termination_penalty=-10.0)
    safe = reward.compute(3.0, TerminationReason.LAP_COMPLETE)
    unsafe = reward.compute(3.0, TerminationReason.OFF_TRACK)

    assert safe.total == 6.0
    assert unsafe.total == -4.0
    assert unsafe.as_dict()["reward_termination"] == -10.0


def test_prediction_reward_and_rollout_shaping() -> None:
    reference = np.zeros((2, 3, 2), dtype=np.float32)
    reference[:, :, 1] = [0.0, 1.0, 2.0]
    predicted = reference.copy()
    predicted[1, :, 0] = 1.0

    calculator = PredictionReward(decay_rate_per_s=0.0)
    terms = calculator.compute(predicted, reference)
    np.testing.assert_allclose(terms.imitation_cost, [0.0, 1.0])
    np.testing.assert_allclose(terms.predicted_progress, [2.0, 2.0])
    shaped = calculator.shape_rollout_rewards(
        [1.0, 1.0],
        terms,
        style_coefficient=2.0,
        predicted_progress_scale=0.5,
    )
    np.testing.assert_allclose(shaped, [2.0, 0.0])


def test_prediction_valid_mask_and_validation() -> None:
    trajectories = np.zeros((2, 2, 2), dtype=np.float32)
    terms = PredictionReward().compute(trajectories, trajectories, [True, False])
    np.testing.assert_allclose(terms.imitation_cost, [0.0, 0.0])
    np.testing.assert_array_equal(terms.valid_mask, [True, False])
    with pytest.raises(ValueError, match="share shape"):
        PredictionReward().compute(trajectories, np.zeros((1, 2, 2)))
    with pytest.raises(ValueError, match="valid_mask"):
        PredictionReward().compute(trajectories, trajectories, [True])
    with pytest.raises(ValueError, match="time_offsets"):
        PredictionReward().compute(
            trajectories,
            trajectories,
            time_offsets_s=[0.5, 0.25],
        )
    with pytest.raises(ValueError, match="decay_rate"):
        PredictionReward(decay_rate_per_s=-1.0)


def test_fixed_reference_variant_adds_predicted_progress_without_imitation() -> None:
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


def test_predicted_progress_is_relative_to_current_vehicle_origin() -> None:
    predicted = np.zeros((1, 3, 2), dtype=np.float32)
    predicted[0, :, 1] = [2.0, 4.0, 7.0]
    terms = PredictionReward(decay_rate_per_s=0.0).compute(
        predicted,
        predicted,
        time_offsets_s=[0.25, 0.5, 1.0],
    )
    np.testing.assert_allclose(terms.predicted_progress, [7.0])


def test_lagrange_controller_projects_coefficient() -> None:
    controller = StyleLagrangeController(
        coefficient=1.0,
        target_cost=0.25,
        learning_rate=2.0,
    )
    assert controller.update([1.25]) == 3.0
    controller.coefficient = 0.1
    assert controller.update([0.0, 100.0], [True, False]) == 0.0
    with pytest.raises(ValueError, match="select"):
        controller.update([1.0], [False])


def test_rollout_shaping_rejects_inconsistent_coefficients() -> None:
    trajectories = np.zeros((1, 2, 2), dtype=np.float32)
    terms = PredictionReward().compute(trajectories, trajectories)
    with pytest.raises(ValueError, match="equal shape"):
        PredictionReward.shape_rollout_rewards(
            [0.0, 0.0],
            terms,
            style_coefficient=1.0,
            predicted_progress_scale=1.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        PredictionReward.shape_rollout_rewards(
            [0.0],
            terms,
            style_coefficient=-1.0,
            predicted_progress_scale=1.0,
        )
