from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from assetto_corsa_rl.config import (
    ActionConfig,
    BehavioralCloningConfig,
    EnvironmentConfig,
    ModelConfig,
    PPOConfig,
    ProjectConfig,
    ReferenceConfig,
)


def test_load_project_config() -> None:
    config = ProjectConfig.from_toml(Path("configs/mock.toml"))
    assert config.experiment.seed == 769913
    assert config.model.feature_hidden_sizes == (512, 512)
    assert config.model.trajectory_codec == "bezier-5x2"
    assert config.model.trajectory_output_size == 10
    assert config.to_dict()["environment"]["control_frequency_hz"] == 20.0

    fixed = ProjectConfig.from_toml(Path("configs/f317_barcelona_fixed.toml"))
    assert fixed.dataset.car_id == "dallara_f317"
    assert fixed.reference.mode == "fixed"
    assert fixed.behavioral_cloning.method_variant == "fixed_reference_no_promp_v1"
    assert fixed.environment.track_length_m == pytest.approx(4591.88720703125)


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[experiment]\nunknown = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown key"):
        ProjectConfig.from_toml(path)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ActionConfig(pedal_delta_limit=0.0), "positive"),
        (
            lambda: EnvironmentConfig(minimum_speed_mps=30.0),
            "minimum speed",
        ),
        (lambda: ReferenceConfig(mode="unknown"), "reference.mode"),
        (lambda: ModelConfig(trajectory_codec="unknown"), "trajectory_codec"),
        (lambda: ModelConfig(trajectory_output_size=20), "output_size = 10"),
        (
            lambda: BehavioralCloningConfig(method_variant="promp"),
            "method_variant",
        ),
        (
            lambda: PPOConfig(rollout_steps=10, minibatch_size=11),
            "minibatch_size",
        ),
    ],
)
def test_invalid_configuration_is_rejected(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
