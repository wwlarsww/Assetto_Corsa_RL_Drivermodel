"""Frozen rollout rewards, normalization, shared features and resume timing."""

import csv
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import Logger
import torch
from torch import nn

from assetto_corsa_ai.agent import PPOAgent
from assetto_corsa_ai.future_reward import FuturePredictorRewardCalculator
from assetto_corsa_ai.main import create_agent
from assetto_corsa_ai.racing_policy import RacingActorCriticPolicy
from assetto_corsa_ai.reward import Reward, RewardCalculator
from assetto_corsa_ai.reward_wrapper import FutureRewardWrapper
from assetto_corsa_ai.training_callbacks import RunMetricsWriter, TrainingRunCallback
from assetto_corsa_ai.training_config import TrainingConfig
from assetto_corsa_ai.training_run import create_run_directory
from test_racing_policy import small_config
from tools.validate_predictor import ValidationEnvironment, synthetic_rollout


class Probe(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.infos: list[dict[str, Any]] = []
        self.rewards: list[float] = []
        self.raw_rewards: list[float] = []

    def _on_step(self) -> bool:
        self.infos.append(dict(self.locals["infos"][0]))
        self.rewards.append(float(self.locals["rewards"][0]))
        normalizer = self.model.get_vec_normalize_env()
        assert normalizer is not None
        self.raw_rewards.append(float(normalizer.get_original_reward()[0]))
        return True


def enabled_config(**changes: Any) -> TrainingConfig:
    config = small_config()
    future = replace(config.reward.future_prediction, enabled=True, warmup_steps=0, ramp_steps=0)
    return replace(config, reward=replace(config.reward, future_prediction=replace(future, **changes)))


def calculator(config: TrainingConfig) -> RewardCalculator:
    settings = asdict(config.reward)
    settings.pop("future_prediction")
    return RewardCalculator(**settings)


def test_rollout_snapshot_uses_action_features_is_detached_and_consumed_once() -> None:
    config = enabled_config()
    data = synthetic_rollout(config)
    agent = create_agent(ValidationEnvironment(data, reward_calculator=calculator(config)), config)
    try:
        policy = agent.model.policy
        assert isinstance(policy, RacingActorCriticPolicy)
        shared: list[torch.Tensor] = []
        head_inputs: list[torch.Tensor] = []
        def features(module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            shared.append(output)
        def head(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            head_inputs.append(inputs[0])
        hooks = [policy.features_extractor.register_forward_hook(features),
                 policy.mlp_extractor.register_forward_pre_hook(head),
                 policy.predictor_head.register_forward_pre_hook(head)]
        policy(torch.ones((1, data.observations.shape[-1]), requires_grad=True))
        prediction = policy.take_reward_prediction()
        assert len(shared) == 1 and len(head_inputs) == 2
        assert all(value is shared[0] for value in head_inputs)
        assert not prediction.requires_grad and prediction.grad_fn is None
        environment = agent.environment.unwrapped
        assert isinstance(environment, ValidationEnvironment)
        reward = FuturePredictorRewardCalculator(config.reward.future_prediction, calculator(config)).calculate(
            prediction[0].numpy(), position=data.positions[0, 0], forward=data.forwards[0, 0],
            current_s=environment.reference_projector.project((0.0, 0.0, 0.0)).distance_along_track,
            projector=environment.reference_projector, step=0,
        )
        assert isinstance(reward.performance_prediction, float)
        assert all(parameter.grad is None for parameter in policy.parameters())
        with pytest.raises(RuntimeError, match="no detached"):
            policy.take_reward_prediction()
        for hook in hooks:
            hook.remove()
    finally:
        agent.close()


@pytest.mark.parametrize("normalize_reward", [False, True])
def test_predicted_reward_is_frozen_before_normalization_and_logged_per_update(
    tmp_path: Path, normalize_reward: bool,
) -> None:
    config = enabled_config()
    config = replace(config, normalization=replace(config.normalization, normalize_reward=normalize_reward))
    data = synthetic_rollout(config, steps=128)
    environment = ValidationEnvironment(data, reward_calculator=calculator(config))
    agent = create_agent(environment, config)
    run = create_run_directory(tmp_path, track="test", car="test")
    logger = Logger(None, [RunMetricsWriter(run)])
    try:
        agent.model.set_logger(logger)
        policy = agent.model.policy
        assert isinstance(policy, RacingActorCriticPolicy)
        last = policy.predictor_head[-1]
        assert isinstance(last, nn.Linear)
        constant = torch.stack((torch.arange(1, 11) * 0.8, torch.zeros(10)), dim=-1)
        last.load_state_dict({"weight": torch.zeros_like(last.weight), "bias": constant.flatten()})
        before = {name: value.detach().clone() for name, value in policy.named_parameters()}
        probe = Probe()
        agent.learn(32, callback=CallbackList([TrainingRunCallback(run, 1), probe]))
        future_calculator = FuturePredictorRewardCalculator(config.reward.future_prediction, calculator(config))
        for index, info in enumerate(probe.infos):
            origin = environment.reference_projector.project(tuple(data.positions[index, 0]))
            expected = future_calculator.calculate(
                constant.numpy(), position=data.positions[index, 0], forward=data.forwards[index, 0],
                current_s=origin.distance_along_track, projector=environment.reference_projector, step=index,
            )
            reward = info["reward"]
            assert isinstance(reward, Reward)
            assert reward.performance_prediction == pytest.approx(expected.performance_prediction)
            assert reward.style_prediction == pytest.approx(expected.style_prediction)
            assert probe.raw_rewards[index] == float(np.float32(reward.total))
        buffer = agent.model.rollout_buffer
        np.testing.assert_array_equal(buffer.rewards[:, 0], np.asarray(probe.rewards, dtype=np.float32))
        if normalize_reward:
            assert not np.allclose(probe.rewards, probe.raw_rewards)
        else:
            np.testing.assert_array_equal(probe.rewards, probe.raw_rewards)
        with (run.directory / "metrics.csv").open(newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["policy_loss"]]
        assert len(rows) == 1 and int(rows[0]["step"]) == 32
        for name in ("performance_step", "performance_prediction", "style_step", "style_prediction", "termination_penalty"):
            assert float(rows[0][name]) == pytest.approx(np.mean([getattr(info["reward"], name) for info in probe.infos]))
        assert float(rows[0]["total_reward"]) == pytest.approx(np.mean([info["reward"].total for info in probe.infos]))
        assert float(rows[0]["future_weight_scale"]) == 1
        assert rows[0]["mean_predicted_progress"] and rows[0]["mean_abs_lateral_offset"]
        assert float(rows[0]["predictor_loss"]) > 0
        assert any(not torch.equal(value, before[name]) for name, value in policy.named_parameters() if name.startswith("predictor_head."))
        frozen = buffer.rewards.copy()
        # An additional optimizer update cannot regenerate stored rewards.
        agent.model.train()
        np.testing.assert_array_equal(buffer.rewards, frozen)
    finally:
        agent.close()
        logger.close()


def test_monitor_episode_sum_includes_future_terms_and_auto_reset_never_reuses_a_snapshot() -> None:
    config = enabled_config()
    data = synthetic_rollout(config)
    agent = create_agent(ValidationEnvironment(data, reward_calculator=calculator(config)), config)
    try:
        probe = Probe()
        agent.learn(64, callback=probe)
        assert len(probe.infos) == 64
        assert probe.infos[31]["episode"]["r"] == pytest.approx(sum(probe.raw_rewards[:32]), abs=2e-6)
        assert probe.infos[63]["episode"]["r"] == pytest.approx(sum(probe.raw_rewards[32:]), abs=2e-6)
        assert all(isinstance(info["reward"], Reward) and info["reward"].style_prediction > 0 for info in probe.infos)
    finally:
        agent.close()


def test_warmup_keeps_auxiliary_training_and_resumes_global_reward_schedule(tmp_path: Path) -> None:
    config = small_config()
    config = replace(config, reward=replace(config.reward, future_prediction=replace(
        config.reward.future_prediction, enabled=True, warmup_steps=16, ramp_steps=16,
    )))
    data = synthetic_rollout(config, steps=128)
    agent = create_agent(ValidationEnvironment(data, reward_calculator=calculator(config)), config)
    run = create_run_directory(tmp_path, track="test", car="test")
    try:
        probe = Probe()
        agent.learn(32, callback=probe)
        assert [info["future_weight_scale"] for info in probe.infos] == [0] * 17 + [i / 16 for i in range(1, 16)]
        assert all(info["reward"].performance_prediction == info["reward"].style_prediction == 0 for info in probe.infos[:17])
        assert agent.model.logger.name_to_value["train/predictor_loss"] > 0
        final = run.save_final_model(agent.model, agent.vec_env)
        restored = PPOAgent.load_snapshot(final, ValidationEnvironment(data, reward_calculator=calculator(config)),
                                          config=config.agent_config(), training=True)
        try:
            assert isinstance(restored.environment, FutureRewardWrapper)
            assert restored.environment.config == config.reward.future_prediction
            restored.set_evaluation_mode()
            observation, _ = restored.environment.reset()
            _, reward, _, _, info = restored.environment.step(restored.predict(observation))
            assert info["future_weight_scale"] == 1
            assert info["reward"].style_prediction > 0
            assert reward == info["reward"].total
            resumed = Probe()
            restored.learn(32, callback=resumed, reset_num_timesteps=False)
            assert [info["future_weight_scale"] for info in resumed.infos] == [1] * 32
        finally:
            restored.close()
    finally:
        agent.close()


def test_disabled_future_reward_preserves_ppo_actions_returns_and_updated_weights_exactly() -> None:
    config = small_config()
    data = synthetic_rollout(config, steps=128)
    template = create_agent(ValidationEnvironment(data), config)
    kwargs = template.model.policy_kwargs.copy()
    template.close()
    results: list[tuple[list[NDArray[np.float32]], dict[str, torch.Tensor]]] = []
    for future_config in (None, config.reward.future_prediction):
        agent = PPOAgent(ValidationEnvironment(data), config=config.agent_config(), policy_kwargs=kwargs,
                         predictor_config=config.predictor, control_interval=config.environment.control_interval,
                         future_reward_config=future_config)
        try:
            assert not isinstance(agent.environment, FutureRewardWrapper)
            agent.learn(32)
            buffer = agent.model.rollout_buffer
            results.append(([buffer.actions.copy(), buffer.rewards.copy(), buffer.returns.copy(), buffer.advantages.copy()],
                            {name: value.detach().clone() for name, value in agent.model.policy.state_dict().items()}))
        finally:
            agent.close()
    for baseline, disabled in zip(results[0][0], results[1][0]):
        np.testing.assert_array_equal(baseline, disabled)
    for name, baseline in results[0][1].items():
        torch.testing.assert_close(baseline, results[1][1][name], rtol=0, atol=0)


def test_component_means_are_attached_to_the_matching_update_not_the_next_rollout(tmp_path: Path) -> None:
    config = enabled_config(warmup_steps=16, ramp_steps=16)
    data = synthetic_rollout(config, steps=256)
    agent = create_agent(ValidationEnvironment(data, reward_calculator=calculator(config)), config)
    run = create_run_directory(tmp_path, track="test", car="test")
    logger = Logger(None, [RunMetricsWriter(run)])
    try:
        agent.model.set_logger(logger)
        probe = Probe()
        agent.learn(64, callback=CallbackList([TrainingRunCallback(run, 1), probe]))
        with (run.directory / "metrics.csv").open(newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["policy_loss"]]
        assert [int(row["step"]) for row in rows] == [32, 64]
        for index, row in enumerate(rows):
            infos = probe.infos[index * 32:(index + 1) * 32]
            assert float(row["future_weight_scale"]) == pytest.approx(np.mean([info["future_weight_scale"] for info in infos]))
            assert float(row["performance_prediction"]) == pytest.approx(np.mean([info["reward"].performance_prediction for info in infos]))
            assert float(row["total_reward"]) == pytest.approx(np.mean([info["reward"].total for info in infos]))
        assert float(rows[0]["future_weight_scale"]) < float(rows[1]["future_weight_scale"]) == 1
    finally:
        agent.close()
        logger.close()
