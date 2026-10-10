"""Test real PPO and normalization using a small in-memory racing environment."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike
import pytest
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd

from assetto_corsa_ai import agent as module
from assetto_corsa_ai.agent import PPOAgent, PPOConfig
from assetto_corsa_ai.RacingEnvironment import FloatArray, RacingEnvironment


class MemoryRacingEnvironment(RacingEnvironment):
    """Provide the racing API without starting AC or opening shared memory."""

    def __init__(self, observation_size: int = 3) -> None:
        gym.Env.__init__(self)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(observation_size,), dtype=np.float32)
        self.step_calls = 0
        self.reset_calls = 0
        self.close_calls = 0
        self._episode_steps = 0

    def raw_observation(self) -> FloatArray:
        assert self.observation_space.shape is not None
        return np.arange(self.observation_space.shape[0], dtype=np.float32) * 10.0 + 100.0 + self._episode_steps

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        self.reset_calls += 1
        self._episode_steps = 0
        return self.raw_observation(), {}

    def step(self, action: ArrayLike) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(np.asarray(action, dtype=np.float32))
        self.step_calls += 1
        self._episode_steps += 1
        return self.raw_observation(), 7.5, False, self._episode_steps == 3, {}

    def close(self, *, stop_session: bool = True) -> None:
        self.close_calls += 1


@pytest.fixture
def environment() -> MemoryRacingEnvironment:
    return MemoryRacingEnvironment()


@pytest.fixture
def config() -> PPOConfig:
    return PPOConfig(n_steps=4, batch_size=2, n_epochs=1, seed=17, verbose=0, device="cpu")


@pytest.fixture
def agent(environment: MemoryRacingEnvironment, config: PPOConfig) -> Iterator[PPOAgent]:
    instance = PPOAgent(environment, config=config, policy_kwargs={"net_arch": [8]})
    yield instance
    instance.close()


def advance_normalizer(agent: PPOAgent) -> FloatArray:
    # SB3's VecEnvStepReturn leaves its info dictionaries unparameterized.
    return agent.vec_env.step(np.zeros((1, 2), dtype=np.float32))[1]  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize("parameter", [
    "learning_rate", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
    "clip_observations", "normalization_epsilon", "gamma", "gae_lambda",
])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0])
def test_invalid_float_configuration_is_rejected(config: PPOConfig, parameter: str, invalid: float) -> None:
    with pytest.raises(ValueError, match=parameter):
        replace(config, **{parameter: invalid}).validate()


@pytest.mark.parametrize("parameter", ["n_steps", "batch_size", "n_epochs"])
@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_count_settings_require_positive_integers(config: PPOConfig, parameter: str, invalid: Any) -> None:
    with pytest.raises(ValueError, match=parameter):
        replace(config, **{parameter: invalid}).validate()


def test_config_rejects_single_sample_minibatch_and_incomplete_batches(config: PPOConfig) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        replace(config, batch_size=1).validate()
    with pytest.raises(ValueError, match="divisible"):
        replace(config, batch_size=3).validate()


def test_invalid_config_does_not_take_ownership_of_environment(environment: MemoryRacingEnvironment) -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        PPOAgent(environment, config=PPOConfig(learning_rate=0.0))
    assert environment.reset_calls == environment.close_calls == 0


def test_wrappers_preserve_raw_observations_and_reward_scale(agent: PPOAgent, environment: MemoryRacingEnvironment) -> None:
    assert agent.vec_env.num_envs == 1
    assert agent.vec_env.env_is_wrapped(Monitor) == [True]
    assert agent.vec_env.norm_obs and agent.vec_env.training
    assert not agent.vec_env.norm_reward
    normalized = agent.vec_env.reset()
    assert isinstance(normalized, np.ndarray)
    raw = environment.raw_observation()
    np.testing.assert_array_equal(raw, [100.0, 110.0, 120.0])
    np.testing.assert_array_equal(agent.vec_env.get_original_obs(), raw[None, :])
    assert not np.allclose(normalized, raw[None, :])
    rewards = advance_normalizer(agent)
    np.testing.assert_array_equal(rewards, [7.5])
    rms = agent.vec_env.obs_rms
    assert isinstance(rms, RunningMeanStd) and rms.count > 1.0


def test_predict_normalizes_once_and_does_not_mutate_inputs_or_statistics(
    agent: PPOAgent, environment: MemoryRacingEnvironment, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rms = agent.vec_env.obs_rms
    assert isinstance(rms, RunningMeanStd)
    rms.mean = np.array([10.0, 20.0, 30.0])
    rms.var = np.array([4.0, 9.0, 16.0])
    rms.count = 50.0
    observation = np.array([12.0, 23.0, 34.0], dtype=np.float32)
    original = observation.copy()
    predict = Mock(return_value=(np.array([0.25, -0.5]), None))
    monkeypatch.setattr(agent.model, "predict", predict)
    command = agent.predict(observation, deterministic=False)
    np.testing.assert_allclose(predict.call_args.args[0], [1.0, 1.0, 1.0], rtol=1e-6)
    assert predict.call_args.kwargs == {"deterministic": False}
    np.testing.assert_array_equal(observation, original)
    np.testing.assert_array_equal(rms.mean, [10.0, 20.0, 30.0])
    assert rms.count == 50.0
    assert command.dtype == np.dtype(np.float32)
    assert environment.action_space.contains(command)


@pytest.mark.parametrize("observation", [
    np.zeros(2, dtype=np.float32), np.zeros((1, 3), dtype=np.float32),
    np.array([float("nan"), 0.0, 0.0], dtype=np.float32),
    np.array([float("inf"), 0.0, 0.0], dtype=np.float32),
])
def test_invalid_prediction_input_never_reaches_policy(
    agent: PPOAgent, observation: FloatArray, monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict = Mock()
    monkeypatch.setattr(agent.model, "predict", predict)
    with pytest.raises(ValueError, match="[Oo]bservation|shape"):
        agent.predict(observation)
    predict.assert_not_called()


@pytest.mark.parametrize("command", [
    np.zeros(3), np.zeros((1, 2)), np.array([float("nan"), 0.0]),
    np.array([float("inf"), 0.0]),
])
def test_invalid_policy_outputs_are_rejected(
    agent: PPOAgent, command: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent.model, "predict", Mock(return_value=(command, None)))
    with pytest.raises(ValueError, match="[Aa]ction"):
        agent.predict(np.zeros(3, dtype=np.float32))


def test_prediction_clips_policy_output_to_simulator_range(agent: PPOAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent.model, "predict", Mock(return_value=(np.array([-1.5, 1.5]), None)))
    np.testing.assert_array_equal(agent.predict(np.zeros(3, dtype=np.float32)), [-1.0, 1.0])


def test_evaluation_freezes_statistics_and_training_enables_them_again(agent: PPOAgent) -> None:
    agent.vec_env.reset()
    rms = agent.vec_env.obs_rms
    assert isinstance(rms, RunningMeanStd)
    mean, variance, count = rms.mean.copy(), rms.var.copy(), rms.count
    agent.set_evaluation_mode()
    advance_normalizer(agent)
    np.testing.assert_array_equal(rms.mean, mean)
    np.testing.assert_array_equal(rms.var, variance)
    assert rms.count == count
    assert not agent.vec_env.training and not agent.vec_env.norm_reward
    agent.set_training_mode()
    advance_normalizer(agent)
    assert agent.vec_env.training and rms.count > count


@pytest.mark.parametrize("invalid", [0, -1, 2.5, True])
def test_invalid_training_length_does_not_start_rollouts(
    agent: PPOAgent, environment: MemoryRacingEnvironment, invalid: Any,
) -> None:
    with pytest.raises(ValueError, match="total_timesteps"):
        agent.learn(invalid)
    assert environment.step_calls == environment.reset_calls == 0


class StepCounter(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.steps = 0

    def _on_step(self) -> bool:
        self.steps += 1
        return True


def test_real_ppo_training_prediction_and_continuation(agent: PPOAgent, environment: MemoryRacingEnvironment) -> None:
    callback = StepCounter()
    agent.set_evaluation_mode()
    assert agent.learn(8, callback=callback) is agent
    assert callback.steps == environment.step_calls == 8
    assert environment.reset_calls > 1
    assert agent.vec_env.training
    observation, _ = environment.reset()
    assert environment.action_space.contains(agent.predict(observation))
    previous_resets = environment.reset_calls
    agent.learn(4, reset_num_timesteps=False)
    assert environment.step_calls == 12
    assert environment.reset_calls - previous_resets < 2


def test_save_load_preserves_predictions_statistics_and_configuration(
    agent: PPOAgent, environment: MemoryRacingEnvironment, tmp_path: Path,
) -> None:
    agent.learn(8)
    agent.set_evaluation_mode()
    raw = environment.raw_observation()
    expected_action = agent.predict(raw)
    directory = tmp_path / "nested" / "checkpoint"
    agent.save(directory)
    assert (directory / "ppo_model.zip").is_file()
    assert (directory / "vec_normalize.pkl").is_file()
    other_environment = MemoryRacingEnvironment()
    loaded = PPOAgent.load(directory, other_environment, device="cpu")
    try:
        assert loaded.config == agent.config
        assert not loaded.vec_env.training and not loaded.vec_env.norm_reward
        np.testing.assert_array_equal(loaded.predict(raw), expected_action)
        original_rms, loaded_rms = agent.vec_env.obs_rms, loaded.vec_env.obs_rms
        assert isinstance(original_rms, RunningMeanStd) and isinstance(loaded_rms, RunningMeanStd)
        np.testing.assert_array_equal(original_rms.mean, loaded_rms.mean)
        np.testing.assert_array_equal(original_rms.var, loaded_rms.var)
        assert original_rms.count == loaded_rms.count
        loaded.learn(4, reset_num_timesteps=False)
        assert other_environment.step_calls == 4
        assert loaded_rms.count > original_rms.count
    finally:
        loaded.close()
    assert other_environment.close_calls == 1


@pytest.mark.parametrize("training", [False, True])
def test_load_preserves_optional_normalization_settings(
    environment: MemoryRacingEnvironment, config: PPOConfig, tmp_path: Path, training: bool,
) -> None:
    custom = replace(config, normalize_observations=False, normalize_reward=True, clip_observations=4.0,
                     normalization_epsilon=1e-5, gamma=0.9, learning_rate=1e-4, clip_range=0.1)
    original = PPOAgent(environment, config=custom, policy_kwargs={"net_arch": [8]})
    try:
        original.set_evaluation_mode()
        original.save(tmp_path)
    finally:
        original.close()
    loaded = PPOAgent.load(tmp_path, MemoryRacingEnvironment(), training=training, device="cpu")
    try:
        assert loaded.config == custom
        assert loaded.vec_env.training is training
        assert loaded.vec_env.norm_reward is training
        assert not loaded.vec_env.norm_obs
        loaded.set_training_mode()
        assert loaded.vec_env.norm_reward
    finally:
        loaded.close()


@pytest.mark.parametrize("missing", ["model", "normalizer", "config"])
def test_load_missing_files_does_not_touch_environment(
    environment: MemoryRacingEnvironment, tmp_path: Path, missing: str,
) -> None:
    if missing != "model":
        (tmp_path / "ppo_model.zip").touch()
    if missing == "config":
        (tmp_path / "vec_normalize.pkl").touch()
    with pytest.raises(FileNotFoundError):
        PPOAgent.load(tmp_path, environment)
    assert environment.reset_calls == environment.close_calls == 0


@pytest.mark.parametrize("failure", ["normalizer", "model", "observation_shape"])
def test_failed_load_closes_environment(
    agent: PPOAgent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    agent.save(tmp_path)
    environment = MemoryRacingEnvironment(observation_size=4 if failure == "observation_shape" else 3)
    if failure == "normalizer":
        monkeypatch.setattr(module.VecNormalize, "load", Mock(side_effect=ValueError("bad normalizer")))
    elif failure == "model":
        monkeypatch.setattr(module.PPO, "load", Mock(side_effect=ValueError("bad model")))
    with pytest.raises((ValueError, AssertionError)):
        PPOAgent.load(tmp_path, environment, device="cpu")
    assert environment.close_calls == 1


def test_failed_model_creation_closes_environment(
    environment: MemoryRacingEnvironment, config: PPOConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "PPO", Mock(side_effect=RuntimeError("model creation failed")))
    with pytest.raises(RuntimeError, match="model creation failed"):
        PPOAgent(environment, config=config)
    assert environment.close_calls == 1


def test_close_releases_underlying_environment(agent: PPOAgent, environment: MemoryRacingEnvironment) -> None:
    agent.close()
    agent.close()
    assert environment.close_calls == 1


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_training_failure_releases_environment_and_preserves_error(
    agent: PPOAgent, environment: MemoryRacingEnvironment, monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    error = failure("training failed")
    monkeypatch.setattr(agent.model, "learn", Mock(side_effect=error))
    with pytest.raises(failure) as caught:
        agent.learn(4)
    assert caught.value is error
    assert environment.close_calls == 1
    agent.close()
    assert environment.close_calls == 1


def test_failed_training_preserves_error_when_cleanup_also_fails(
    agent: PPOAgent, monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("training failed")
    monkeypatch.setattr(agent.model, "learn", Mock(side_effect=error))
    monkeypatch.setattr(agent.vec_env, "close", Mock(side_effect=OSError("close failed")))
    with pytest.raises(RuntimeError) as caught:
        agent.learn(4)
    assert caught.value is error
    assert any("close failed" in note for note in error.__notes__)


@pytest.mark.parametrize("operation", ["learn", "predict", "save", "evaluation", "training"])
def test_closed_agent_cannot_reactivate_environment(
    agent: PPOAgent, environment: MemoryRacingEnvironment, tmp_path: Path, operation: str,
) -> None:
    agent.close()
    with pytest.raises(RuntimeError, match="closed"):
        if operation == "learn":
            agent.learn(4)
        elif operation == "predict":
            agent.predict(np.zeros(3, dtype=np.float32))
        elif operation == "save":
            agent.save(tmp_path / "checkpoint")
        elif operation == "evaluation":
            agent.set_evaluation_mode()
        else:
            agent.set_training_mode()
    assert environment.reset_calls == environment.step_calls == 0


def test_corrupt_config_is_rejected_before_wrapping_environment(
    agent: PPOAgent, tmp_path: Path, environment: MemoryRacingEnvironment,
) -> None:
    agent.save(tmp_path)
    (tmp_path / PPOAgent.CONFIG_FILENAME).write_text("{ invalid json", encoding="utf-8")
    other_environment = MemoryRacingEnvironment()
    with pytest.raises(ValueError):
        PPOAgent.load(tmp_path, other_environment)
    assert other_environment.close_calls == other_environment.reset_calls == 0


def test_prediction_can_bypass_observation_normalization(
    environment: MemoryRacingEnvironment, config: PPOConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = PPOAgent(environment, config=replace(config, normalize_observations=False),
                        policy_kwargs={"net_arch": [8]})
    try:
        predict = Mock(return_value=(np.zeros(2), None))
        monkeypatch.setattr(instance.model, "predict", predict)
        raw = environment.raw_observation()
        instance.predict(raw)
        np.testing.assert_array_equal(predict.call_args.args[0], raw)
    finally:
        instance.close()


def test_failed_normalizer_creation_releases_environment(
    environment: MemoryRacingEnvironment, config: PPOConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "VecNormalize", Mock(side_effect=ValueError("normalizer failed")))
    with pytest.raises(ValueError, match="normalizer failed"):
        PPOAgent(environment, config=config)
    assert environment.close_calls == 1
