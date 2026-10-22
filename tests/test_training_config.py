"""Strict settings prevent hidden defaults and silently ignored features."""

from dataclasses import replace
from pathlib import Path
import re

import pytest

from assetto_corsa_ai.training_config import TrainingConfig, checkpoint_config_path
from config_helpers import replace_setting


TEMPLATE = Path(__file__).resolve().parents[1] / "configs/training.toml"


def test_complete_template_roundtrips_without_hidden_agent_settings() -> None:
    config = TrainingConfig.load(TEMPLATE)
    assert config.toml_text == TEMPLATE.read_text(encoding="utf-8")
    agent = config.agent_config()
    assert agent.seed == 42
    assert agent.n_steps == 1024
    assert config.network.feature_layers == (512, 512)
    assert config.predictor.enabled
    assert agent.delta_actions
    assert config.reward.future_prediction.enabled
    assert config.reward.future_prediction.horizon_discount == 0.95
    assert not config.trajectories.enabled
    assert agent.initial_action_std == config.ppo.initial_action_std
    assert agent.initial_longitudinal_mean == config.ppo.initial_longitudinal_mean
    assert agent.max_steering_delta == 0.05
    assert agent.max_longitudinal_delta == 0.10
    assert agent.longitudinal_deadband == 0.05
    assert config.training.checkpoint_interval_updates == 4


@pytest.mark.parametrize("value", ["0", "0.0", "0.1", "120", "120.0"])
def test_episode_time_limit_accepts_zero_or_positive_seconds(value: str) -> None:
    content = TEMPLATE.read_text(encoding="utf-8")
    config = TrainingConfig.from_toml(replace_setting(content, "max_episode_time", f"max_episode_time = {value}"))
    assert config.environment.max_episode_time == float(value)


@pytest.mark.parametrize("before,after", [
    ("gamma = 0.99", "gamma = nan"),
    ("gamma = 0.99", "gamma = inf"),
    ("gamma = 0.99", "gamam = 0.99"),
    ("seed = 42", "seed = true"),
    ("seed = 42", "seed = -1"),
    ("seed = 42", "seed = 4294967296"),
    ("batch_size = 256", "batch_size = 300"),
    ("feature_layers = [512, 512]", "feature_layers = [512, -1]"),
    ("feature_dim = 512", "feature_dim = 0"),
    ("feature_dim = 512", "feature_dim = 1.5"),
    ("activation = \"tanh\"", "activation = \"unsupported\""),
    ("optimizer = \"adam\"", "optimizer = \"unsupported\""),
    ("optimizer_betas = [0.9, 0.999]", "optimizer_betas = [1.0, 0.999]"),
    ("output_dimensions = 2", "output_dimensions = 3"),
    ("prediction_points = 10", "prediction_points = 9"),
    ("horizon_discount = 0.95", "horizon_discount = 1.1"),
    ("warmup_steps = 20000", "warmup_steps = -1"),
    ("ramp_steps = 30000", "ramp_steps = -1"),
    ("[predictor]\nenabled = true", "[predictor]\nenabled = false"),
    ("action_mode = \"delta\"", "action_mode = \"unsupported\""),
    ("enabled = false # Reserved", "enabled = true # Reserved"),
    ("car_index = 0", "car_index = 1"),
    ("abs_mode = 1", "abs_mode = 3"),
    ("car = \"bmw_z4_gt3\"", "car = \"../car\""),
    ("control_interval = 0.05", "control_interval = 0.0"),
    ("max_episode_time = 0", "max_episode_time = -1"),
    ("max_episode_time = 0", "max_episode_time = nan"),
    ("max_episode_time = 0", "max_episode_time = inf"),
    ("max_episode_time = 0", "max_episode_time = true"),
    ("collision_penalty = -100.0", "collision_penalty = 1.0"),
    ("reset_speed_tolerance_kmh = 1.0", "reset_speed_tolerance_kmh = -1.0"),
    ("lookahead_distances = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]", "lookahead_distances = [10.0, 5.0]"),
    ("n_epochs = 10", ""),
    ("initial_action_std = 0.20", "initial_action_std = 0"),
    ("initial_action_std = 0.20", "initial_action_std = nan"),
    ("initial_action_std = 0.20", "initial_action_std = -1"),
    ("initial_action_std = 0.20", "initial_action_std = true"),
    ("max_steering_delta = 0.05", "max_steering_delta = -0.1"),
    ("max_steering_delta = 0.05", "max_steering_delta = 2.1"),
    ("max_longitudinal_delta = 0.10", "max_longitudinal_delta = inf"),
    ("longitudinal_deadband = 0.05", "longitudinal_deadband = 1.0"),
    ("longitudinal_deadband = 0.05", "longitudinal_deadband = -0.01"),
    ("checkpoint_interval_updates = 4", "checkpoint_interval_updates = -1"),
    ("checkpoint_interval_updates = 4", "checkpoint_interval_updates = 1.5"),
    ("checkpoint_interval_updates = 4", "checkpoint_interval_updates = true"),
])
def test_invalid_settings_fail_before_training(before: str, after: str) -> None:
    content = TEMPLATE.read_text(encoding="utf-8")
    if before.startswith("[predictor]"):
        content = replace_setting(content, "enabled", "enabled = false", section="predictor")
    elif before.startswith("enabled = false"):
        content = replace_setting(content, "enabled", "enabled = true", section="trajectories")
    else:
        content = replace_setting(content, before.split("=", 1)[0].strip(), after or None)
    with pytest.raises(ValueError):
        TrainingConfig.from_toml(content)


def test_unknown_section_is_rejected() -> None:
    with pytest.raises(ValueError, match="sections"):
        TrainingConfig.from_toml(TEMPLATE.read_text() + "\n[unused]\nvalue = 1\n")


def test_resume_allows_budget_changes_but_not_different_training_semantics() -> None:
    previous = TrainingConfig.load(TEMPLATE)
    current = replace(previous, training=replace(previous.training, total_timesteps=10000, resume_from="run/final"))
    current.require_compatible_resume(previous)
    with pytest.raises(ValueError, match="network"):
        replace(current, network=replace(current.network, feature_dim=64)).require_compatible_resume(previous)
    with pytest.raises(ValueError, match="seed"):
        replace(current, training=replace(current.training, seed=23)).require_compatible_resume(previous)
    with pytest.raises(ValueError, match="action"):
        replace(current, action=replace(current.action, max_steering_delta=0.1)).require_compatible_resume(previous)
    replace(current, training=replace(current.training, checkpoint_interval_updates=8)).require_compatible_resume(previous)


def test_periodic_checkpoints_can_be_disabled() -> None:
    content = replace_setting(TEMPLATE.read_text(encoding="utf-8"), "checkpoint_interval_updates", "checkpoint_interval_updates = 0")
    assert TrainingConfig.from_toml(content).training.checkpoint_interval_updates == 0


@pytest.mark.parametrize("value", ["nan", "inf", "1.1", "-1.1", "true"])
def test_invalid_initial_longitudinal_mean_is_rejected(value: str) -> None:
    content = replace_setting(TEMPLATE.read_text(), "initial_longitudinal_mean", f"initial_longitudinal_mean = {value}")
    with pytest.raises(ValueError, match="initial_longitudinal_mean"):
        TrainingConfig.from_toml(content)


def test_archived_configs_keep_unbiased_initial_exploration() -> None:
    content = replace_setting(TEMPLATE.read_text(), "initial_longitudinal_mean", None)
    config = TrainingConfig.from_toml(content)
    assert config.ppo.initial_longitudinal_mean == 0.0
    assert config.toml_text == content


def test_legacy_run_config_keeps_original_action_semantics_and_exact_text() -> None:
    content = TEMPLATE.read_text(encoding="utf-8")
    content = re.sub(r"(?ms)^\[action\]\n.*?(?=^\[)", "", content)
    content = replace_setting(content, "initial_action_std", None)
    content = content.replace("[network]\n", "[network]\nlog_std_init = 0.0\n")
    content = replace_setting(content, "checkpoint_interval_updates", "checkpoint_interval = 10000")
    config = TrainingConfig.from_toml(content)
    assert config.toml_text == content
    assert config.ppo.initial_action_std == 1.0
    assert config.action.max_steering_delta == config.action.max_longitudinal_delta == config.action.longitudinal_deadband == 0.0
    assert config.training.checkpoint_interval_updates == 10


def test_new_config_requires_an_explicit_action_section() -> None:
    content = re.sub(r"(?ms)^\[action\]\n.*?(?=^\[)", "", TEMPLATE.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="sections"):
        TrainingConfig.from_toml(content)


@pytest.mark.parametrize("assignment", ["checkpoint_interval = 10000", "log_std_init = 0.0"])
def test_legacy_and_new_settings_cannot_silently_override_each_other(assignment: str) -> None:
    section = "training" if assignment.startswith("checkpoint") else "network"
    content = TEMPLATE.read_text(encoding="utf-8").replace(f"[{section}]\n", f"[{section}]\n{assignment}\n")
    with pytest.raises(ValueError, match="instead|only"):
        TrainingConfig.from_toml(content)


def test_checkpoint_config_resolution_is_limited_to_run_layout(tmp_path: Path) -> None:
    assert checkpoint_config_path(tmp_path / "run/final") == tmp_path / "run/config.toml"
    assert checkpoint_config_path(tmp_path / "run/checkpoints/step_010000") == tmp_path / "run/config.toml"
    with pytest.raises(ValueError):
        checkpoint_config_path(tmp_path / "random/model")
