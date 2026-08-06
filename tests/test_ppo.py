from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from assetto_corsa_rl.config import (
    ACTION_METHOD_VARIANT,
    STABILIZED_PPO_METHOD_VARIANT,
    EnvironmentConfig,
    ExperimentConfig,
    ModelConfig,
    PPOConfig,
    ProjectConfig,
)
from assetto_corsa_rl.envs.factory import create_mock_environment
from assetto_corsa_rl.features.schema import OBSERVATION_FEATURE_NAMES
from assetto_corsa_rl.training.checkpoints import load_checkpoint
from assetto_corsa_rl.training.ppo import (
    _validate_initial_checkpoint,
    build_driver_network,
    train_ppo,
)


def test_legacy_delta_action_checkpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="not compatible"):
        _validate_initial_checkpoint(
            {
                "method_variant": "fixed_reference_v1",
                "observation_feature_names": list(OBSERVATION_FEATURE_NAMES),
            }
        )


def test_short_mock_ppo_run_updates_versioned_policy(tmp_path: Path) -> None:
    base = ProjectConfig()
    config = replace(
        base,
        experiment=ExperimentConfig(
            seed=7,
            max_episode_steps=200,
            output_directory=str(tmp_path),
        ),
        environment=EnvironmentConfig(
            control_frequency_hz=20.0,
            mock_track_length_m=10_000.0,
            mock_track_half_width_m=1_000.0,
            initial_speed_mps=20.0,
            minimum_speed_mps=1.0,
            stall_debounce_steps=10_000,
            critical_body_sideslip_angle_deg=90.0,
        ),
        model=ModelConfig(
            feature_hidden_sizes=(16,),
            policy_hidden_sizes=(8,),
            value_hidden_sizes=(8,),
            predictor_hidden_sizes=(8,),
        ),
        ppo=PPOConfig(
            total_environment_steps=104,
            rollout_steps=104,
            update_epochs=1,
            minibatch_size=52,
            predictor_loss_coefficient=0.01,
            use_predicted_progress_reward=True,
        ),
    )
    environment = create_mock_environment(config)
    try:
        result = train_ppo(
            config,
            environment,
            command=["test"],
            backend_name="mock",
        )
    finally:
        environment.close()

    assert result.environment_steps == 104
    assert result.iterations == 1
    assert result.policy_version == 1
    assert result.checkpoint_path.is_file()
    model = build_driver_network(config)
    state, checkpoint_config = load_checkpoint(result.checkpoint_path, model=model)
    assert state.policy_version == 1
    assert checkpoint_config["method_variant"] == STABILIZED_PPO_METHOD_VARIANT
    assert checkpoint_config["action_method_variant"] == ACTION_METHOD_VARIANT

    with (result.run_directory / "metrics.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["policy_version"] == "1"
    assert rows[0]["environment_steps"] == "104"
    expected_metric_columns = {
        "iteration",
        "policy_version",
        "environment_steps",
        "rollout_steps",
        "episodes_completed",
        "environment_reward_mean",
        "predicted_progress_mean_m",
        "shaped_reward_mean",
        "predictor_valid_samples",
        "policy_loss",
        "value_loss",
        "entropy",
        "predictor_loss",
        "approximate_kl",
        "clip_fraction",
    }
    assert expected_metric_columns <= rows[0].keys()
    assert float(rows[0]["predicted_progress_mean_m"]) > 0.0
    assert float(rows[0]["shaped_reward_mean"]) > float(rows[0]["environment_reward_mean"])
    assert int(rows[0]["predictor_valid_samples"]) > 0
    summary = json.loads((result.run_directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["predicted_progress_reward_enabled"] is True
