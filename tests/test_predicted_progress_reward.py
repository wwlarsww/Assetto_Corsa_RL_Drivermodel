import numpy as np
import pytest

import assetto_corsa_rl.rewards.components as reward_components
from assetto_corsa_rl.rewards.components import PredictedProgressReward


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((1, 3, 2), dtype=np.float32),
        np.zeros((1, 4, 2), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
    )


def test_horizon_discount_preserves_endpoint_reward_at_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reward_components,
        "project_points_to_polyline_progress",
        lambda *_args: np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )
    predicted, reference, progress = _inputs()

    undiscounted = PredictedProgressReward(horizon_discount=1.0).compute(
        predicted,
        reference,
        progress,
    )
    discounted = PredictedProgressReward(horizon_discount=0.5).compute(
        predicted,
        reference,
        progress,
    )

    assert undiscounted.predicted_progress[0] == pytest.approx(3.0)
    assert discounted.predicted_progress[0] == pytest.approx(1.75)


def test_horizon_discount_prefers_early_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    predicted, reference, progress = _inputs()
    projection = iter(
        (
            np.asarray([3.0, 3.0, 3.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 3.0], dtype=np.float32),
        )
    )
    monkeypatch.setattr(
        reward_components,
        "project_points_to_polyline_progress",
        lambda *_args: next(projection),
    )
    reward = PredictedProgressReward(horizon_discount=0.5)

    early = reward.compute(predicted, reference, progress)
    late = reward.compute(predicted, reference, progress)

    assert early.predicted_progress[0] == pytest.approx(3.0)
    assert late.predicted_progress[0] == pytest.approx(0.75)


@pytest.mark.parametrize("discount", [0.0, -0.1, 1.1])
def test_invalid_horizon_discount_is_rejected(discount: float) -> None:
    with pytest.raises(ValueError, match="horizon_discount"):
        PredictedProgressReward(horizon_discount=discount)
