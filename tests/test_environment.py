from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.envs.factory import create_mock_environment
from assetto_corsa_rl.policies import MockDriverPolicy


def test_mock_environment_passes_gymnasium_checker() -> None:
    environment = create_mock_environment(ProjectConfig())
    try:
        # The thesis policy is Gaussian: only its mean is tanh-bounded. The
        # unbounded Box is deliberate so sampled actions and PPO log
        # probabilities remain identical; absolute commands are clipped later.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Box action space.*",
                category=UserWarning,
            )
            check_env(environment, skip_render_check=True)
    finally:
        environment.close()


def test_mock_policy_completes_a_lap_reproducibly() -> None:
    config = ProjectConfig()
    outcomes: list[tuple[int, float, Any]] = []
    for _ in range(2):
        environment = create_mock_environment(config)
        policy = MockDriverPolicy(
            config.actions.pedal_delta_limit,
            config.actions.steering_delta_limit,
        )
        observation, _ = environment.reset(seed=config.experiment.seed)
        total_reward = 0.0
        final_info: dict[str, Any] = {}
        step_count = 0
        for _ in range(config.experiment.max_episode_steps):
            observation, reward, terminated, truncated, final_info = environment.step(
                policy.act(observation)
            )
            total_reward += reward
            step_count += 1
            if terminated or truncated:
                break
        environment.close()
        outcomes.append((step_count, total_reward, final_info["termination_reason"]))

    assert outcomes[0] == outcomes[1]
    assert outcomes[0][2] == "lap_complete"


def test_time_limit_is_truncation_not_termination() -> None:
    config = ProjectConfig()
    environment = create_mock_environment(config)
    environment.max_episode_steps = 1
    environment.reset(seed=1)
    _, reward, terminated, truncated, info = environment.step(np.zeros(2, dtype=np.float32))
    environment.close()

    assert not terminated
    assert truncated
    assert info["termination_reason"] is None
    assert reward == pytest.approx(info["reference_progress_delta_m"])
    assert info["reward_terms"]["reward_progress"] == pytest.approx(
        info["reference_progress_delta_m"]
    )


def test_environment_requires_reset_and_rejects_reset_options() -> None:
    environment = create_mock_environment(ProjectConfig())
    with pytest.raises(RuntimeError, match="has not been reset"):
        _ = environment.active_reference
    with pytest.raises(RuntimeError, match="reset must be called"):
        environment.step(np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="unsupported reset option"):
        environment.reset(seed=1, options={"unknown": True})
    environment.close()
