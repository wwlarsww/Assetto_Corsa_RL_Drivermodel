"""Actual PPO update logging and run snapshot roundtrips without the game."""

import csv
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.running_mean_std import RunningMeanStd

from assetto_corsa_ai.agent import PPOAgent, PPOConfig
from assetto_corsa_ai.training_callbacks import RunMetricsWriter, TrainingRunCallback
from assetto_corsa_ai.training_run import create_run_directory
from test_agent import MemoryRacingEnvironment


def test_ppo_update_metrics_and_paired_checkpoint_roundtrip(tmp_path: Path) -> None:
    run = create_run_directory(tmp_path, track="monza", car="car")
    config = PPOConfig(n_steps=4, batch_size=2, n_epochs=1, seed=17, device="cpu", verbose=0)
    agent = PPOAgent(MemoryRacingEnvironment(), config=config, policy_kwargs={"net_arch": [8]})
    logger = Logger(None, [RunMetricsWriter(run)])
    try:
        agent.model.set_logger(logger)
        agent.learn(8, callback=TrainingRunCallback(run, 4))
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
