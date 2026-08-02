from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from assetto_corsa_rl.config import (
    ActionConfig,
    AssettoCorsaRuntimeConfig,
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

    fixed = ProjectConfig.from_toml(Path("configs/bmw_z4_gt3_monza_baseline.toml"))
    assert fixed.dataset.car_id == "bmw_z4_gt3"
    assert fixed.dataset.track_id == "monza"
    assert fixed.reference.mode == "fixed"
    assert fixed.behavioral_cloning.method_variant == "ai_spline_absolute_v1"
    assert fixed.actions.method_variant == "absolute_bounded_v1"
    assert fixed.environment.mock_track_length_m == pytest.approx(1000.0)
    assert fixed.environment.minimum_speed_mps * 3.6 == pytest.approx(5.0)
    assert fixed.environment.initial_combined_pedal == pytest.approx(1.0)
    assert fixed.environment.critical_body_sideslip_angle_deg == pytest.approx(20.0)
    assert not fixed.experiment.truncate_on_max_episode_steps

    live_smoke = ProjectConfig.from_toml(Path("configs/bmw_z4_gt3_monza_live_smoke.toml"))
    assert live_smoke.rewards == fixed.rewards


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[experiment]\nunknown = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown key"):
        ProjectConfig.from_toml(path)


def test_removed_slip_ratio_termination_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "removed-slip-termination.toml"
    path.write_text("[environment]\ncritical_slip_ratio = 0.35\n", encoding="utf-8")
    with pytest.raises(ValueError, match="critical_slip_ratio"):
        ProjectConfig.from_toml(path)


def test_removed_control_loss_variant_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "removed-control-loss-variant.toml"
    path.write_text(
        '[environment]\ncontrol_loss_termination = "legacy_variant"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="control_loss_termination"):
        ProjectConfig.from_toml(path)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ActionConfig(method_variant="delta"), "method_variant"),
        (
            lambda: EnvironmentConfig(minimum_speed_mps=30.0),
            "minimum speed",
        ),
        (
            lambda: EnvironmentConfig(critical_body_sideslip_angle_deg=91.0),
            "sideslip angle",
        ),
        (
            lambda: EnvironmentConfig(initial_combined_pedal=1.1),
            "initial_combined_pedal",
        ),
        (lambda: ReferenceConfig(mode="unknown"), "reference.mode"),
        (lambda: ModelConfig(trajectory_codec="unknown"), "trajectory_codec"),
        (lambda: ModelConfig(trajectory_output_size=20), "output_size = 10"),
        (
            lambda: BehavioralCloningConfig(method_variant="unknown"),
            "method_variant",
        ),
        (
            lambda: PPOConfig(rollout_steps=10, minibatch_size=11),
            "minibatch_size",
        ),
        (
            lambda: AssettoCorsaRuntimeConfig(steering_output_sign=0.0),
            "steering_output_sign",
        ),
        (
            lambda: AssettoCorsaRuntimeConfig(launch_timeout_s=0.0),
            "launch_timeout_s",
        ),
        (
            lambda: AssettoCorsaRuntimeConfig(reset_collision_quiescence_s=-0.1),
            "reset_collision_quiescence_s",
        ),
        (
            lambda: AssettoCorsaRuntimeConfig(reset_backend="unknown"),
            "reset_backend",
        ),
    ],
)
def test_invalid_configuration_is_rejected(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
