"""Strict settings prevent hidden defaults and silently ignored features."""

from dataclasses import replace
from pathlib import Path

import pytest

from assetto_corsa_ai.training_config import TrainingConfig, checkpoint_config_path


TEMPLATE = Path(__file__).resolve().parents[1] / "configs/training.toml"


def test_complete_template_roundtrips_without_hidden_agent_settings() -> None:
    config = TrainingConfig.load(TEMPLATE)
    assert config.toml_text == TEMPLATE.read_text(encoding="utf-8")
    agent = config.agent_config()
    assert agent.seed == 42
    assert agent.n_steps == 1024
    assert config.network.feature_layers == (512, 512)
    assert not config.predictor.enabled
    assert not config.trajectories.enabled


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
    ("enabled = false # Auxiliary", "enabled = true # Auxiliary"),
    ("enabled = false # Reserved", "enabled = true # Reserved"),
    ("car_index = 0", "car_index = 1"),
    ("abs_mode = 1", "abs_mode = 3"),
    ("car = \"bmw_z4_gt3\"", "car = \"../car\""),
    ("control_interval = 0.05", "control_interval = 0.0"),
    ("collision_penalty = -100.0", "collision_penalty = 1.0"),
    ("reset_speed_tolerance_kmh = 1.0", "reset_speed_tolerance_kmh = -1.0"),
    ("lookahead_distances = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]", "lookahead_distances = [10.0, 5.0]"),
    ("n_epochs = 10\n", ""),
])
def test_invalid_settings_fail_before_training(before: str, after: str) -> None:
    content = TEMPLATE.read_text(encoding="utf-8")
    assert before in content
    with pytest.raises(ValueError):
        TrainingConfig.from_toml(content.replace(before, after))


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


def test_checkpoint_config_resolution_is_limited_to_run_layout(tmp_path: Path) -> None:
    assert checkpoint_config_path(tmp_path / "run/final") == tmp_path / "run/config.toml"
    assert checkpoint_config_path(tmp_path / "run/checkpoints/step_010000") == tmp_path / "run/config.toml"
    with pytest.raises(ValueError):
        checkpoint_config_path(tmp_path / "random/model")
