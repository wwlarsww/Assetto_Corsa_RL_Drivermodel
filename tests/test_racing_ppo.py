"""Real auxiliary PPO updates, episode separation and complete checkpoints."""

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
import torch

from assetto_corsa_ai.agent import PPOAgent
from assetto_corsa_ai.main import create_agent
from assetto_corsa_ai.racing_ppo import RacingPPO, RacingRolloutBuffer
from assetto_corsa_ai.training_callbacks import RunMetricsWriter, TrainingRunCallback
from assetto_corsa_ai.training_run import create_run_directory
from test_racing_policy import small_config
from tools.validate_predictor import ValidationEnvironment, synthetic_rollout


def test_short_predictor_training_logs_and_restores_all_heads(tmp_path: Path) -> None:
    config = small_config()
    data = synthetic_rollout(config)
    agent = create_agent(ValidationEnvironment(data), config)
    run = create_run_directory(tmp_path, track="test", car="test")
    logger = Logger(None, [RunMetricsWriter(run)])
    try:
        agent.model.set_logger(logger)
        before = {name: value.detach().clone() for name, value in agent.model.policy.named_parameters()}
        agent.learn(64, callback=TrainingRunCallback(run, 1))
        assert isinstance(agent.model, RacingPPO)
        assert agent.model.completed_policy_updates == 2
        assert isinstance(agent.model.rollout_buffer, RacingRolloutBuffer)
        with (run.directory / "metrics.csv").open(newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["predictor_loss"]]
        assert [int(row["step"]) for row in rows] == [32, 64]
        assert all(float(row["predictor_loss"]) > 0 and int(row["predictor_valid_samples"]) > 0 for row in rows)
        assert all(float(row["predictor_forward_mae"]) >= 0 and float(row["predictor_lateral_mae"]) >= 0 for row in rows)
        for prefix in ("features_extractor.", "predictor_head.", "action_net.", "value_net."):
            assert any(not torch.equal(value, before[name]) for name, value in agent.model.policy.named_parameters() if name.startswith(prefix))
        observation = data.observations[0, 0]
        prediction, action = agent.predict_trajectory(observation), agent.predict(observation)
        final = run.save_final_model(agent.model, agent.vec_env)
        assert {path.name for path in final.iterdir()} == {"model.zip", "vecnormalize.pkl"}
        restored = PPOAgent.load_snapshot(final, ValidationEnvironment(data), config=config.agent_config(), training=True)
        try:
            np.testing.assert_array_equal(restored.predict_trajectory(observation), prediction)
            np.testing.assert_array_equal(restored.predict(observation), action)
            assert isinstance(restored.model, RacingPPO)
            assert restored.model.predictor_config == config.predictor
            assert restored.model.completed_policy_updates == 2
            restored.learn(32, reset_num_timesteps=False)
            assert getattr(restored.model, "num_timesteps") == 96
            assert restored.model.completed_policy_updates == 3
            assert restored.model.logger.name_to_value["train/predictor_valid_samples"] > 0
        finally:
            restored.close()
    finally:
        agent.close()
        logger.close()


def test_collection_keeps_terminal_endpoint_and_never_uses_auto_reset_pose() -> None:
    config = small_config()
    config = replace(config, ppo=replace(config.ppo, n_steps=64))
    data = synthetic_rollout(config)
    agent = create_agent(ValidationEnvironment(data), config)
    try:
        agent.learn(64)
        assert isinstance(agent.model, RacingPPO)
        buffer = agent.model.rollout_buffer
        assert isinstance(buffer, RacingRolloutBuffer)
        assert buffer.episode_ends[31, 0]
        np.testing.assert_array_equal(buffer.vehicle_positions[32, 0], data.positions[0, 0])
        np.testing.assert_array_equal(buffer.next_vehicle_positions[31, 0], data.next_positions[31, 0])
        # Buffer samples are now flattened, with one environment in this test.
        assert buffer.trajectory_mask[30, 0]
        assert not buffer.trajectory_mask[30, 1:].any()
        assert buffer.trajectory_mask[32].all()
        assert np.abs(buffer.trajectory_targets).max() < 20
    finally:
        agent.close()


class StopAfterOne(BaseCallback):
    def _on_step(self) -> bool:
        return False


def test_original_callback_can_stop_collection_before_any_update() -> None:
    config = small_config()
    agent = create_agent(ValidationEnvironment(synthetic_rollout(config)), config)
    try:
        before = {name: value.detach().clone() for name, value in agent.model.policy.named_parameters()}
        callback = StopAfterOne()
        agent.learn(64, callback=callback)
        assert callback.n_calls == 1
        assert getattr(agent.model, "num_timesteps") == 1
        assert all(torch.equal(value, before[name]) for name, value in agent.model.policy.named_parameters())
    finally:
        agent.close()


@pytest.mark.parametrize("future_enabled", [False, True])
def test_future_reward_is_configurable_with_an_enabled_predictor(future_enabled: bool) -> None:
    config = small_config()
    config = replace(config, reward=replace(config.reward, future_prediction=replace(config.reward.future_prediction, enabled=future_enabled)))
    config.validate()
    if future_enabled:
        with pytest.raises(ValueError, match="enabled predictor"):
            replace(config, predictor=replace(config.predictor, enabled=False)).validate()
