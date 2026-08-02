from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.envs.factory import create_mock_environment
from assetto_corsa_rl.envs.mock_backend import MockAssettoCorsaBackend
from assetto_corsa_rl.policies import MockDriverPolicy
from assetto_corsa_rl.references.fixed_reference import ReferenceTrajectory
from assetto_corsa_rl.types import TerminationReason


def test_mock_environment_passes_gymnasium_checker() -> None:
    environment = create_mock_environment(ProjectConfig())
    try:
        check_env(environment, skip_render_check=True)
    finally:
        environment.close()


def test_reset_starts_with_full_throttle_and_centred_steering() -> None:
    environment = create_mock_environment(ProjectConfig())

    observation, _ = environment.reset(seed=1)

    np.testing.assert_allclose(environment.action_adapter.current, [1.0, 0.0])
    np.testing.assert_allclose(observation[-2:], [1.0, 0.0])
    _, _, _, _, info = environment.step(np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(info["action_change"], [-1.0, 0.0])
    environment.close()


def test_mock_policy_completes_a_lap_reproducibly() -> None:
    config = ProjectConfig()
    outcomes: list[tuple[int, float, Any]] = []
    for _ in range(2):
        environment = create_mock_environment(config)
        policy = MockDriverPolicy()
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


def test_time_limit_can_be_disabled_for_lap_or_stall_episodes() -> None:
    environment = create_mock_environment(ProjectConfig())
    environment.max_episode_steps = 1
    environment.truncate_on_max_episode_steps = False
    environment.reset(seed=1)
    _, _, terminated, truncated, _ = environment.step(np.zeros(2, dtype=np.float32))
    environment.close()

    assert not terminated
    assert not truncated


def test_environment_requires_reset_and_rejects_reset_options() -> None:
    environment = create_mock_environment(ProjectConfig())
    with pytest.raises(RuntimeError, match="has not been reset"):
        _ = environment.active_reference
    with pytest.raises(RuntimeError, match="reset must be called"):
        environment.step(np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="unsupported reset option"):
        environment.reset(seed=1, options={"unknown": True})
    environment.close()


def test_body_sideslip_termination_uses_speed_gate_and_configured_angle() -> None:
    reference = ReferenceTrajectory.constant(track_length_m=1000.0, speed_mps=20.0)
    low_speed = MockAssettoCorsaBackend(
        initial_speed_mps=1.0,
        minimum_speed_mps=0.1,
        minimum_speed_grace_steps=100,
        critical_body_sideslip_angle_deg=0.1,
        minimum_body_sideslip_termination_speed_mps=5.0,
    )
    low_speed.reset(seed=None, reference=reference)
    low_transition = low_speed.step(np.asarray([0.0, 1.0], dtype=np.float32))
    assert low_transition.termination_reason is None

    normal_speed = MockAssettoCorsaBackend(
        initial_speed_mps=20.0,
        critical_body_sideslip_angle_deg=0.1,
        minimum_body_sideslip_termination_speed_mps=5.0,
    )
    normal_speed.reset(seed=None, reference=reference)
    normal_transition = normal_speed.step(np.asarray([0.0, 1.0], dtype=np.float32))
    assert normal_transition.termination_reason is TerminationReason.EXCESSIVE_BODY_SIDESLIP


def test_stall_termination_uses_the_configured_five_kmh_threshold() -> None:
    reference = ReferenceTrajectory.constant(track_length_m=1000.0, speed_mps=20.0)
    backend = MockAssettoCorsaBackend(
        initial_speed_mps=1.5,
        minimum_speed_mps=5.0 / 3.6,
        minimum_speed_grace_steps=0,
    )
    backend.reset(seed=None, reference=reference)
    moving = backend.step(np.zeros(2, dtype=np.float32))
    assert moving.termination_reason is None

    backend.reset(seed=None, reference=reference)
    too_slow = backend.step(np.asarray([-1.0, 0.0], dtype=np.float32))
    assert too_slow.telemetry.vehicle_state.speed_mps < 5.0 / 3.6
    assert too_slow.termination_reason is TerminationReason.STALLED
