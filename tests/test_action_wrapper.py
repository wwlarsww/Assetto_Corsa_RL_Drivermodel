"""Check executed controls and PPO likelihoods through the real wrapper stack."""

from dataclasses import replace
from typing import Any
from pathlib import Path
import csv

import numpy as np
import pytest
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
import torch
from torch import nn

from assetto_corsa_ai.action_wrapper import ActionFilterWrapper
from assetto_corsa_ai.main import create_agent
from assetto_corsa_ai.racing_policy import RacingActorCriticPolicy
from assetto_corsa_ai.racing_ppo import RacingPPO
from assetto_corsa_ai.training_callbacks import RunMetricsWriter, TrainingRunCallback
from assetto_corsa_ai.training_run import create_run_directory
from test_racing_policy import small_config
from tools.validation.validate_predictor import ValidationEnvironment, synthetic_rollout


def test_filter_tracks_applied_controls_and_resets_episode_history() -> None:
    config = small_config()
    environment = ValidationEnvironment(synthetic_rollout(config))
    wrapper = ActionFilterWrapper(environment, config.action)
    try:
        wrapper.reset()
        previous = np.zeros(2, dtype=np.float32)
        for _ in range(32):
            observation, _, _, _, info = wrapper.step([1, -1])
            delta = np.abs(info["applied_action"] - previous)
            assert np.all(delta <= np.array([0.05, 0.10]) + 1e-7)
            np.testing.assert_array_equal(observation[-2:], info["applied_action"])
            np.testing.assert_array_equal(info["observed_action"], observation[-2:])
            np.testing.assert_array_equal(info["requested_action"], [1, -1])
            previous = observation[-2:].copy()
        with pytest.raises(RuntimeError, match="reset"):
            wrapper.step([1, -1])
        wrapper.reset()
        np.testing.assert_allclose(wrapper.step([1, -1])[0][-2:], [0.05, -0.1])
    finally:
        wrapper.close()


@pytest.mark.parametrize("action", [[1.1, 0], [0], [float("nan"), 0], [[0, 0]]])
def test_invalid_actions_never_reach_the_underlying_environment(action: Any) -> None:
    config = small_config()
    environment = ValidationEnvironment(synthetic_rollout(config))
    wrapper = ActionFilterWrapper(environment, config.action)
    try:
        wrapper.reset()
        with pytest.raises(ValueError, match="Action"):
            wrapper.step(action)
        assert environment.index == 0
        np.testing.assert_array_equal(environment.controls, [0, 0])
    finally:
        wrapper.close()


def test_initial_training_explores_pedals_and_records_control_stages(tmp_path: Path) -> None:
    config = small_config()
    run = create_run_directory(tmp_path, track="test", car="car")
    logger = Logger(None, [RunMetricsWriter(run)])
    agent = create_agent(ValidationEnvironment(synthetic_rollout(config)), config)
    try:
        policy = agent.model.policy
        assert isinstance(policy.action_net, nn.Linear)
        # A neutral actor mean must still explore both pedals during rollouts.
        with torch.no_grad():
            policy.action_net.weight.zero_()
            policy.action_net.bias.zero_()
        agent.model.set_logger(logger)
        agent.learn(32, callback=TrainingRunCallback(run, 0))
        with (run.directory / "metrics.csv").open(newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["policy_updates"]]
        assert len(rows) == 1
        row = rows[0]
        assert float(row["requested_longitudinal_abs"]) > 0.05
        assert float(row["applied_throttle"]) > 0.0
        assert float(row["applied_brake"]) > 0.0
        assert float(row["observed_longitudinal_abs"]) == pytest.approx(
            float(row["applied_throttle"]) + float(row["applied_brake"])
        )
    finally:
        agent.close()
        logger.close()


class LikelihoodProbe(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.previous = np.zeros(2, dtype=np.float32)
        self.filtered = False

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        applied = np.asarray(info["applied_action"], dtype=np.float32)
        assert np.all(np.abs(applied - self.previous) <= np.array([0.05, 0.10]) + 1e-7)
        self.filtered |= not np.allclose(applied, info["requested_action"])
        self.previous = np.zeros(2, dtype=np.float32) if self.locals["dones"][0] else applied.copy()
        return True

    def _on_rollout_end(self) -> None:
        assert isinstance(self.model, RacingPPO)
        buffer = self.model.rollout_buffer
        obs = torch.as_tensor(buffer.observations[:, 0], device=self.model.device)
        actions = torch.as_tensor(buffer.actions[:, 0], device=self.model.device)
        with torch.no_grad():
            _, likelihood, _ = self.model.policy.evaluate_actions(obs, actions)
        np.testing.assert_allclose(likelihood.cpu().numpy(), buffer.log_probs[:, 0], atol=1e-6)


@pytest.mark.parametrize("future_enabled", [False, True])
def test_training_preserves_sample_likelihoods_and_filters_executed_controls(future_enabled: bool) -> None:
    config = small_config()
    config = replace(config, reward=replace(config.reward, future_prediction=replace(
        config.reward.future_prediction, enabled=future_enabled, warmup_steps=0, ramp_steps=0,
    )))
    agent = create_agent(ValidationEnvironment(synthetic_rollout(config)), config)
    try:
        policy = agent.model.policy
        assert isinstance(policy, RacingActorCriticPolicy) and isinstance(policy.action_net, nn.Linear)
        np.testing.assert_allclose(policy.log_std.detach().exp().cpu().numpy(), [0.2, 0.2])
        with torch.no_grad():
            policy.action_net.weight.zero_()
            policy.action_net.bias.fill_(0.8)
        probe = LikelihoodProbe()
        agent.learn(32, callback=probe)
        assert probe.filtered
    finally:
        agent.close()
