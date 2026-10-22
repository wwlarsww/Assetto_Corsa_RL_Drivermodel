"""Observer reward and reset history use the same core math, independently."""

from dataclasses import replace

import numpy as np
import pytest

from tools.diagnostics.checks import reward_is_consistent
from tools.diagnostics.frame import DiagnosticsFrame
from tools.diagnostics.state import DiagnosticsState
from assetto_corsa_ai.termination import TerminationResult
from test_diagnostics import frame
from test_diagnostics_training import diagnostics_config
from tools.validation.validate_predictor import synthetic_reference


def sample(timestamp: float, x: float = 0.0, speed: float = 72.0) -> DiagnosticsFrame:
    original = frame(timestamp)
    assert original.telemetry is not None
    telemetry = replace(original.telemetry, speed_kmh=speed, car_coordinates=(x, 0.0, 0.0),
                        tyre_contact_points=((x + 1, 0.0, -1.0), (x + 1, 0.0, 1.0),
                                             (x - 1, 0.0, -1.0), (x - 1, 0.0, 1.0)))
    return DiagnosticsFrame(timestamp, telemetry_timestamp=timestamp, telemetry=telemetry,
                            position=(x, 0.0, 0.0), forward=(1.0, 0.0, 0.0),
                            progress=x + 100, lateral_offset=0.0)


def test_weighted_reward_matches_core_math_and_only_observed_controls_are_reported() -> None:
    config = diagnostics_config()
    config = replace(config, reward=replace(config.reward, style_weight=2.5))
    state = DiagnosticsState(config, synthetic_reference())
    first = state.evaluate(sample(0.0), packet=1, laps=0)
    assert first.reset and first.reward is None
    second = sample(0.05, x=1.0)
    assert second.telemetry is not None
    second = replace(second, telemetry=replace(second.telemetry, gas=0.7, brake=0.0, steering_angle=0.2))
    result = state.evaluate(second, packet=2, laps=0)
    expected = state.reward.calculate(1.0, 0.0, TerminationResult(False, None))
    assert result.reward == expected and reward_is_consistent(result)
    assert result.previous_observed_action is not None and result.observed_action_delta is not None
    np.testing.assert_allclose(result.previous_observed_action, (0.1, 0.5))
    np.testing.assert_allclose(result.observed_action_delta, (0.1, 0.2))
    assert "sampled_action" not in result.to_bytes().decode()
    assert result.reward_style_weight == 2.5
    assert "Local reconstruction" in result.reward_source
    assert not result.terminated and not result.truncated


def test_future_reward_uses_previous_pose_prediction_and_checkpoint_step() -> None:
    config = diagnostics_config()
    config = replace(config, reward=replace(config.reward,
                     future_prediction=replace(config.reward.future_prediction, enabled=True)))
    projector = synthetic_reference()
    state = DiagnosticsState(config, projector)
    first = replace(sample(0.0), predicted_trajectory=np.asarray([[1.0, 0.0], [2.0, 0.0]]), step=64)
    state.evaluate(first, 1, 0)
    expected = state.future.calculate(np.asarray(first.predicted_trajectory, dtype=np.float32),
                                     position=np.zeros(3, dtype=np.float64), forward=np.asarray([1., 0., 0.]),
                                     current_s=100.0, projector=projector, step=64)
    # A very different current prediction must not replace the transition's origin.
    result = state.evaluate(replace(sample(0.05, 1.0),
                            predicted_trajectory=np.asarray([[90., 0.], [100., 0.]]), step=96), 2, 0)
    assert result.reward is not None
    assert result.reward.performance_prediction == pytest.approx(expected.performance_prediction)
    assert result.reward.style_prediction == pytest.approx(expected.style_prediction)
    assert result.mean_predicted_progress == pytest.approx(1.5)
    assert reward_is_consistent(result)


@pytest.mark.parametrize("cause", ["packet", "lap_counter", "teleport", "progress", "gap", "damage"])
def test_resets_clear_only_observer_history_and_do_not_reward_the_jump(cause: str) -> None:
    state = DiagnosticsState(diagnostics_config(), synthetic_reference())
    initial = sample(0.0)
    assert initial.telemetry is not None
    initial = replace(initial, telemetry=replace(initial.telemetry, car_damage=(5., 0., 0., 0., 0.)))
    telemetry = initial.telemetry
    assert telemetry is not None
    state.evaluate(initial, packet=10, laps=2)
    second = replace(sample(0.05, 1.0), telemetry=replace(telemetry, car_coordinates=(1., 0., 0.)))
    state.evaluate(second, packet=11, laps=2)
    third = replace(sample(0.1, 2.0), telemetry=replace(telemetry, car_coordinates=(2., 0., 0.)))
    packet, laps = 12, 2
    if cause == "packet":
        packet = 0
    elif cause == "lap_counter":
        laps = 0
    elif cause == "teleport":
        third = replace(third, position=(2000., 0., 0.))
    elif cause == "progress":
        third = replace(third, progress=900.)
    elif cause == "gap":
        third = replace(third, timestamp=2.0)
    else:
        third = replace(third, telemetry=replace(telemetry, car_damage=(0., 0., 0., 0., 0.)))
    result = state.evaluate(third, packet=packet, laps=laps)
    assert result.reset and result.reset_reason
    assert result.episode_id == 1 and result.reward is None
    assert state.elapsed == 0.0 and state.samples == 0
    assert not result.terminated


def test_duplicate_packets_do_not_advance_stuck_history_and_instances_are_independent() -> None:
    config = diagnostics_config()
    config = replace(config, termination=replace(config.termination, stuck_time=0.25))
    first = DiagnosticsState(config, synthetic_reference())
    second = DiagnosticsState(config, synthetic_reference())
    assert first.termination is not second.termination and first.reward is not second.reward
    first.evaluate(sample(0., speed=0.), 1)
    second.evaluate(sample(0., speed=0.), 1)
    moving = first.evaluate(sample(0.125, speed=0.), 2)
    for _ in range(10):
        assert first.evaluate(sample(0.125, speed=0.), 2).reward == moving.reward
    assert first.samples == 1 and first.elapsed == 0.125
    assert second.samples == 0 and second.elapsed == 0.
    ended = first.evaluate(sample(0.25, speed=0.), 3)
    assert ended.terminated and ended.termination_reason == "STUCK_OR_WRONG_WAY"
    reset = first.evaluate(replace(sample(0.375, speed=0.), position=(1000., 0., 0.)), 4)
    assert reset.reset
    after = first.evaluate(replace(sample(0.5, speed=0.), position=(1000., 0., 0.)), 5)
    assert not after.terminated  # A fresh observation window after the reset.


def test_start_finish_wrap_is_progress_and_not_a_reset() -> None:
    projector = synthetic_reference()
    state = DiagnosticsState(diagnostics_config(), projector)
    state.evaluate(replace(sample(0.), progress=projector.track_length - 0.5), 1)
    result = state.evaluate(replace(sample(0.05), progress=0.5), 2)
    assert not result.reset and result.reward is not None
    assert result.reward.performance_step == pytest.approx(state.config.reward.performance_weight)


def test_local_step_limit_is_truncation_and_remains_so_while_waiting() -> None:
    config = diagnostics_config()
    config = replace(config, environment=replace(config.environment, max_episode_steps=1))
    state = DiagnosticsState(config, synthetic_reference())
    state.evaluate(sample(0., speed=0.), 1)
    result = state.evaluate(sample(0.05, speed=0.), 2)
    assert result.truncated and not result.terminated
    waiting = state.evaluate(sample(0.1, speed=0.), 3)
    assert waiting.truncated and not waiting.terminated
    assert waiting.reward is None
