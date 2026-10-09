"""Verify signed progress, reward components and episode penalties."""

import pytest

from assetto_corsa_ai.reward import RewardCalculator, calculate_progress_delta
from assetto_corsa_ai.termination import TerminationReason, TerminationResult


@pytest.mark.parametrize(
    "previous_s, current_s, expected",
    [
        (10.0, 15.0, 5.0),
        (15.0, 10.0, -5.0),
        (10.0, 10.0, 0.0),
        (98.0, 3.0, 5.0),
        (3.0, 98.0, -5.0),
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 50.0, 50.0),
        (50.0, 0.0, -50.0),
    ],
)
def test_progress_delta_handles_direction_and_start_finish(
    previous_s: float,
    current_s: float,
    expected: float,
) -> None:
    assert calculate_progress_delta(current_s, previous_s, 100.0) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("track_length", [0.0, -100.0])
def test_progress_delta_rejects_nonpositive_track_length(track_length: float) -> None:
    with pytest.raises(ValueError, match="track_length"):
        calculate_progress_delta(3.0, 98.0, track_length)


@pytest.mark.parametrize("delta_s", [3.0, 0.0, -3.0])
def test_weighted_reward_preserves_signed_progress_and_component_breakdown(
    delta_s: float,
) -> None:
    calculator = RewardCalculator(
        performance_weight=2.0,
        style_weight=3.0,
        style_sigma=2.0,
    )
    reward = calculator.calculate(delta_s, 2.0, TerminationResult(False, None))

    assert reward.performance_step == pytest.approx(2.0 * delta_s)
    assert reward.style_step == pytest.approx(0.6065306597126334)
    assert reward.performance_prediction == 0.0
    assert reward.style_prediction == 0.0
    assert reward.termination_penalty == 0.0
    assert reward.total == pytest.approx(2.0 * delta_s + 1.8195919791379003)


@pytest.mark.parametrize(
    "offset, expected",
    [(0.0, 1.0), (1.0, 0.6065306597126334), (2.0, 0.1353352832366127)],
)
def test_style_reward_decreases_symmetrically_away_from_reference_line(
    offset: float,
    expected: float,
) -> None:
    calculator = RewardCalculator()
    for signed_offset in (offset, -offset):
        reward = calculator.calculate(0.0, signed_offset, TerminationResult(False, None))
        assert reward.style_step == pytest.approx(expected)
        assert reward.total == pytest.approx(expected)


def test_larger_style_sigma_tolerates_larger_lateral_offsets() -> None:
    termination = TerminationResult(False, None)
    narrow = RewardCalculator(style_sigma=0.5).calculate(0.0, 1.0, termination)
    wide = RewardCalculator(style_sigma=2.0).calculate(0.0, 1.0, termination)

    assert narrow.style_step < wide.style_step < 1.0


@pytest.mark.parametrize(
    "reason, expected_penalty",
    [
        (TerminationReason.OFF_TRACK, -10.0),
        (TerminationReason.STUCK_OR_WRONG_WAY, -20.0),
        (TerminationReason.STRONG_COLLISION, -30.0),
    ],
)
def test_each_termination_reason_adds_its_configured_penalty(
    reason: TerminationReason,
    expected_penalty: float,
) -> None:
    calculator = RewardCalculator(
        off_track_penalty=-10.0,
        stuck_penalty=-20.0,
        collision_penalty=-30.0,
    )
    reward = calculator.calculate(2.0, 0.0, TerminationResult(True, reason))

    assert reward.termination_penalty == expected_penalty
    assert reward.performance_step == 2.0
    assert reward.style_step == 1.0
    assert reward.total == pytest.approx(3.0 + expected_penalty)


@pytest.mark.parametrize(
    "performance_weight, style_weight, expected_total",
    [(0.0, 1.0, 1.0), (1.0, 0.0, 5.0), (0.0, 0.0, 0.0)],
)
def test_zero_weights_disable_only_their_contribution(
    performance_weight: float,
    style_weight: float,
    expected_total: float,
) -> None:
    calculator = RewardCalculator(
        performance_weight=performance_weight,
        style_weight=style_weight,
    )
    reward = calculator.calculate(5.0, 0.0, TerminationResult(False, None))

    assert reward.total == expected_total
    # The breakdown retains the unweighted style term.
    assert reward.style_step == 1.0


def test_termination_penalty_remains_when_both_reward_weights_are_zero() -> None:
    calculator = RewardCalculator(
        performance_weight=0.0,
        style_weight=0.0,
        collision_penalty=-42.0,
    )
    reward = calculator.calculate(
        5.0, 0.0, TerminationResult(True, TerminationReason.STRONG_COLLISION)
    )
    assert reward.total == -42.0


@pytest.mark.parametrize(
    "kwargs, parameter",
    [
        ({"performance_weight": -1.0}, "performance_weight"),
        ({"style_weight": -1.0}, "style_weight"),
        ({"style_sigma": 0.0}, "style_sigma"),
        ({"style_sigma": -1.0}, "style_sigma"),
        ({"off_track_penalty": 1.0}, "off_track_penalty"),
        ({"stuck_penalty": 1.0}, "stuck_penalty"),
        ({"collision_penalty": 1.0}, "collision_penalty"),
    ],
)
def test_invalid_reward_configuration_is_rejected(
    kwargs: dict[str, float],
    parameter: str,
) -> None:
    with pytest.raises(ValueError, match=parameter):
        RewardCalculator(**kwargs)


def test_zero_termination_penalties_are_allowed() -> None:
    calculator = RewardCalculator(
        off_track_penalty=0.0, stuck_penalty=0.0, collision_penalty=0.0
    )
    for reason in TerminationReason:
        reward = calculator.calculate(0.0, 0.0, TerminationResult(True, reason))
        assert reward.termination_penalty == 0.0
        assert reward.total == 1.0
