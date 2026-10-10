"""Actual PPO update logging and run snapshot roundtrips without the game."""

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.running_mean_std import RunningMeanStd
import torch

from assetto_corsa_ai.agent import PPOAgent, PPOConfig
from assetto_corsa_ai.training_callbacks import RunMetricsWriter, TrainingRunCallback
from assetto_corsa_ai.training_run import create_run_directory
from assetto_corsa_ai.racing_ppo import RacingPPO
from test_agent import MemoryRacingEnvironment


def test_ppo_update_metrics_and_paired_checkpoint_roundtrip(tmp_path: Path) -> None:
    run = create_run_directory(tmp_path, track="monza", car="car")
    config = PPOConfig(n_steps=4, batch_size=2, n_epochs=1, seed=17, device="cpu", verbose=0)
    agent = PPOAgent(MemoryRacingEnvironment(), config=config, policy_kwargs={"net_arch": [8]})
    logger = Logger(None, [RunMetricsWriter(run)])
    try:
        agent.model.set_logger(logger)
        agent.learn(8, callback=TrainingRunCallback(run, 1))
        final = run.save_final_model(agent.model, agent.vec_env)
        observation = np.array([101, 111, 121], dtype=np.float32)
        expected = agent.predict(observation)
        with (run.directory / "metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [int(row["step"]) for row in rows if row["policy_loss"]] == [4, 8]
        assert [int(row["step"]) for row in rows if row["episode_reward"]] == [3, 6]
        restored = PPOAgent.load_snapshot(final, MemoryRacingEnvironment(), config=config)
        try:
            np.testing.assert_array_equal(restored.predict(observation), expected)
            original_rms, restored_rms = agent.vec_env.obs_rms, restored.vec_env.obs_rms
            assert isinstance(original_rms, RunningMeanStd) and isinstance(restored_rms, RunningMeanStd)
            np.testing.assert_array_equal(restored_rms.mean, original_rms.mean)
            np.testing.assert_array_equal(restored_rms.var, original_rms.var)
            assert restored_rms.count == original_rms.count
            steps = getattr(restored.model, "num_timesteps")
            assert isinstance(steps, int) and steps == 8
            assert not restored.vec_env.training
            assert not restored.vec_env.norm_reward
        finally:
            restored.close()
    finally:
        agent.close()
        logger.close()


def test_missing_normalizer_cannot_be_loaded_as_complete_snapshot(tmp_path: Path) -> None:
    (tmp_path / "model.zip").touch()
    environment = MemoryRacingEnvironment()
    with pytest.raises(FileNotFoundError, match="vecnormalize"):
        PPOAgent.load_snapshot(tmp_path, environment, config=PPOConfig())
    assert environment.reset_calls == environment.close_calls == 0


def test_fourth_completed_phase_saves_optimized_model_and_normalizer(tmp_path: Path) -> None:
    run = create_run_directory(tmp_path, track="test", car="car")
    config = PPOConfig(n_steps=4, batch_size=2, n_epochs=3, seed=17, device="cpu", verbose=0)
    agent = PPOAgent(MemoryRacingEnvironment(), config=config, policy_kwargs={"net_arch": [8]})
    logger = Logger(None, [RunMetricsWriter(run)])
    try:
        agent.model.set_logger(logger)
        agent.learn(16, callback=TrainingRunCallback(run, 4))
        assert [path.name for path in (run.directory / "checkpoints").iterdir()] == ["step_000016"]
        with (run.directory / "metrics.csv").open(newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["policy_loss"]]
        assert [int(row["policy_updates"]) for row in rows] == [1, 2, 3, 4]
        restored = PPOAgent.load_snapshot(run.directory / "checkpoints/step_000016", MemoryRacingEnvironment(), config=config)
        try:
            assert isinstance(restored.model, RacingPPO)
            assert restored.model.completed_policy_updates == 4
            assert getattr(restored.model, "_n_updates") == 12
            for name, expected in agent.model.policy.state_dict().items():
                torch.testing.assert_close(restored.model.policy.state_dict()[name], expected, rtol=0, atol=0)
            original_rms, restored_rms = agent.vec_env.obs_rms, restored.vec_env.obs_rms
            assert isinstance(original_rms, RunningMeanStd) and isinstance(restored_rms, RunningMeanStd)
            np.testing.assert_array_equal(original_rms.mean, restored_rms.mean)
            np.testing.assert_array_equal(original_rms.var, restored_rms.var)
            assert original_rms.count == restored_rms.count
        finally:
            restored.close()
    finally:
        agent.close()
        logger.close()


def test_resume_continues_global_checkpoint_phase_count(tmp_path: Path) -> None:
    config = PPOConfig(n_steps=4, batch_size=2, n_epochs=2, seed=17, device="cpu", verbose=0)
    first = create_run_directory(tmp_path, track="test", car="car")
    agent = PPOAgent(MemoryRacingEnvironment(), config=config, policy_kwargs={"net_arch": [8]})
    try:
        agent.learn(12, callback=TrainingRunCallback(first, 4))
        final = first.save_final_model(agent.model, agent.vec_env)
    finally:
        agent.close()
    original = {path.name: path.read_bytes() for path in final.iterdir()}
    second = create_run_directory(tmp_path, track="test", car="car")
    restored = PPOAgent.load_snapshot(final, MemoryRacingEnvironment(), config=config, training=True)
    try:
        assert isinstance(restored.model, RacingPPO) and restored.model.completed_policy_updates == 3
        restored.learn(8, callback=TrainingRunCallback(second, 4), reset_num_timesteps=False)
        assert restored.model.completed_policy_updates == 5
        assert [path.name for path in (second.directory / "checkpoints").iterdir()] == ["step_000016"]
        checkpoint = PPOAgent.load_snapshot(second.directory / "checkpoints/step_000016", MemoryRacingEnvironment(), config=config)
        try:
            assert isinstance(checkpoint.model, RacingPPO) and checkpoint.model.completed_policy_updates == 4
        finally:
            checkpoint.close()
        assert original == {path.name: path.read_bytes() for path in final.iterdir()}
    finally:
        restored.close()


class StopAtStep(BaseCallback):
    def __init__(self, step: int) -> None:
        super().__init__()
        self.stop_step = step

    def _on_step(self) -> bool:
        return self.num_timesteps < self.stop_step


@pytest.mark.parametrize("stop_step,updates", [(1, 0), (13, 3)])
def test_cancelled_rollout_never_counts_as_a_completed_policy_update(tmp_path: Path, stop_step: int, updates: int) -> None:
    run = create_run_directory(tmp_path, track="test", car="car")
    config = PPOConfig(n_steps=4, batch_size=2, n_epochs=2, seed=17, device="cpu", verbose=0)
    agent = PPOAgent(MemoryRacingEnvironment(), config=config, policy_kwargs={"net_arch": [8]})
    logger = Logger(None, [RunMetricsWriter(run)])
    try:
        agent.model.set_logger(logger)
        agent.learn(16, callback=CallbackList([TrainingRunCallback(run, 4), StopAtStep(stop_step)]))
        assert isinstance(agent.model, RacingPPO) and agent.model.completed_policy_updates == updates
        assert not list((run.directory / "checkpoints").iterdir())
        with (run.directory / "metrics.csv").open(newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["policy_loss"]]
        assert [int(row["step"]) for row in rows] == list(range(4, updates * 4 + 1, 4))
    finally:
        agent.close()
        logger.close()


def test_zero_interval_disables_periodic_checkpoints_but_allows_final_snapshot(tmp_path: Path) -> None:
    run = create_run_directory(tmp_path, track="test", car="car")
    config = PPOConfig(n_steps=4, batch_size=2, n_epochs=1, seed=17, device="cpu", verbose=0)
    agent = PPOAgent(MemoryRacingEnvironment(), config=config, policy_kwargs={"net_arch": [8]})
    try:
        agent.learn(8, callback=TrainingRunCallback(run, 0))
        assert not list((run.directory / "checkpoints").iterdir())
        assert {path.name for path in run.save_final_model(agent.model, agent.vec_env).iterdir()} == {"model.zip", "vecnormalize.pkl"}
    finally:
        agent.close()


@pytest.mark.parametrize("interval", [-1, 1.5, True])
def test_invalid_checkpoint_intervals_are_rejected(tmp_path: Path, interval: Any) -> None:
    with pytest.raises(ValueError, match="interval_updates"):
        TrainingRunCallback(create_run_directory(tmp_path, track="test", car="car"), interval)
