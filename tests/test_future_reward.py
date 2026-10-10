"""Reference projection, horizon weighting and central reward composition."""

from dataclasses import replace

import numpy as np
import pytest

from assetto_corsa_ai.future_reward import FuturePredictionRewardConfig, FuturePredictorRewardCalculator, FutureReward
from assetto_corsa_ai.predictor import trajectory_to_world
from assetto_corsa_ai.reward import RewardCalculator
from assetto_corsa_ai.termination import TerminationReason, TerminationResult
from assetto_corsa_ai.track_geometry import TrackGeometry, TrackPoint
from assetto_corsa_ai.track_projection import TrackProjector


def reference() -> TrackProjector:
    return TrackProjector(TrackGeometry([
        TrackPoint(p, 0.0, 5.0, 5.0, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
        for p in [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0),
                  (100.0, 0.0, 100.0), (0.0, 0.0, 100.0)]
    ]))


def settings(**changes: object) -> FuturePredictionRewardConfig:
    return replace(FuturePredictionRewardConfig(True, 1.0, 1.0, 1.0, 0, 0), **changes)


def evaluate(points: list[list[float]], config: FuturePredictionRewardConfig | None = None) -> FutureReward:
    return FuturePredictorRewardCalculator(config or settings(), RewardCalculator()).calculate(
        np.asarray(points, dtype=np.float32), position=np.array([10, 0, 0], dtype=np.float64),
        forward=np.array([1, 0, 0], dtype=np.float64), current_s=10, projector=reference(), step=0,
    )


def test_points_on_reference_have_unit_style_and_signed_progress() -> None:
    forward = evaluate([[2, 0], [4, 0], [6, 0]])
    backward = evaluate([[-1, 0], [-3, 0], [-6, 0]])
    assert forward.style_score == forward.style_prediction == 1.0
    assert forward.performance_prediction == 6.0
    assert backward.performance_prediction == -6.0
    np.testing.assert_allclose(forward.progress_deltas, [2, 2, 2])


def test_style_is_a_weighted_mean_and_decreases_with_lateral_distance() -> None:
    config = settings(horizon_discount=0.5)
    on_line = evaluate([[2, 0], [4, 0], [6, 0]], config)
    away = evaluate([[2, 0], [4, 1], [6, 2]], config)
    expected = (1 + 0.5 * np.exp(-0.5) + 0.25 * np.exp(-2)) / 1.75
    assert away.style_prediction == pytest.approx(expected)
    assert away.style_prediction < on_line.style_prediction
    assert evaluate([[2, 0]]).style_prediction == evaluate([[2, 0]] * 20).style_prediction == 1.0


def test_each_horizon_contributes_and_nearer_increments_have_more_weight() -> None:
    first = evaluate([[2, 0], [4, 0], [6, 0]], settings(horizon_discount=0.5))
    changed = evaluate([[2, 0], [5, 1], [6, 0]], settings(horizon_discount=0.5))
    np.testing.assert_array_equal(first.horizon_weights, [1, 0.5, 0.25])
    np.testing.assert_allclose(first.progress_contributions, [2, 1, 0.5])
    assert first.performance_prediction == pytest.approx(3.5)
    assert changed.performance_prediction == pytest.approx(3.75)
    assert changed.style_prediction < first.style_prediction


@pytest.mark.parametrize("position,forward,points,current,expected", [
    ([0, 0, 2], [0, 0, -1], [[1, 0], [2, -2], [2, -4]], 398, [1, 3, 2]),
    ([2, 0, 0], [1, 0, 0], [[-1, 0], [-2, -2], [-2, -4]], 2, [-1, -3, -2]),
])
def test_both_wraparound_directions_use_the_existing_progress_rule(
    position: list[float], forward: list[float], points: list[list[float]], current: float, expected: list[float],
) -> None:
    result = FuturePredictorRewardCalculator(settings(), RewardCalculator()).calculate(
        np.asarray(points, dtype=np.float32), position=np.asarray(position, dtype=np.float64),
        forward=np.asarray(forward, dtype=np.float64), current_s=current, projector=reference(), step=0,
    )
    np.testing.assert_allclose(result.progress_deltas, expected)
    assert result.performance_prediction == sum(expected)


def test_local_inverse_uses_origin_heading_and_keeps_height() -> None:
    world = trajectory_to_world(
        np.asarray([[5, 3], [-2, -1]], dtype=np.float32),
        np.asarray([100, 7, 200], dtype=np.float64), np.asarray([0, 2, -3], dtype=np.float64),
    )
    np.testing.assert_allclose(world, [[97, 7, 195], [101, 7, 202]])


def test_future_weights_scale_terms_but_preserve_step_and_negative_penalty() -> None:
    calculator = RewardCalculator(performance_weight=2, style_weight=3, collision_penalty=-100)
    base = calculator.calculate(2, 0, TerminationResult(True, TerminationReason.STRONG_COLLISION))
    future = evaluate([[2, 0], [4, 0]], settings(performance_weight=0.05, style_weight=0.1))
    reward = calculator.with_prediction(
        base, performance_prediction=future.performance_prediction, style_prediction=future.style_prediction,
    )
    assert reward.performance_step == base.performance_step == 4
    assert reward.style_step == base.style_step == 1
    assert reward.termination_penalty == -100
    assert reward.total == pytest.approx(4 + 0.2 + 3 * (1 + 0.1) - 100)


def test_disabled_prediction_preserves_the_identical_base_reward_object() -> None:
    calculator = RewardCalculator()
    base = calculator.calculate(2.5, 0.8, TerminationResult(False, None))
    future = evaluate([[2, 3], [5, 4]], settings(enabled=False))
    assert future.performance_prediction == future.style_prediction == future.weight_scale == 0
    assert calculator.with_prediction(base, performance_prediction=0, style_prediction=0) is base


@pytest.mark.parametrize("step,expected", [(0, 0), (19999, 0), (20000, 0), (35000, 0.5), (50000, 1), (90000, 1)])
def test_warmup_and_ramp_scale_only_future_reward_weights(step: int, expected: float) -> None:
    config = settings(warmup_steps=20000, ramp_steps=30000)
    assert config.weight_scale(step) == expected
    result = FuturePredictorRewardCalculator(config, RewardCalculator()).calculate(
        np.asarray([[2, 0], [4, 0]], dtype=np.float32), position=np.asarray([10, 0, 0], dtype=np.float64),
        forward=np.asarray([1, 0, 0], dtype=np.float64), current_s=10, projector=reference(), step=step,
    )
    assert result.performance_prediction == pytest.approx(4 * expected)
    assert result.style_prediction == expected
    assert result.style_score == 1.0


@pytest.mark.parametrize("changes", [
    {"horizon_discount": 0}, {"horizon_discount": 1.1}, {"horizon_discount": float("nan")},
    {"performance_weight": -1}, {"style_weight": float("inf")}, {"warmup_steps": -1}, {"ramp_steps": 1.5},
])
def test_invalid_future_reward_settings_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        settings(**changes).validate()
