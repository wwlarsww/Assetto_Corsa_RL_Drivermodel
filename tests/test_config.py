from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from assetto_corsa_rl.config import (
    KL_CONTROL_METHOD_VARIANT,
    PERMUTED_FEEDBACK_BC_STEERING_ROBUSTNESS_METHOD_VARIANT,
    POLICY_PROTECTED_PPO_METHOD_VARIANT,
    PRE_ACTION_PROGRESS_BASELINE_METHOD_VARIANT,
    PREDICTION_REWARD_METHOD_VARIANT,
    STABILIZED_PPO_METHOD_VARIANT,
    STEERING_WEIGHTED_BC_ACTION_LOSS_METHOD_VARIANT,
    STRAIGHT_STEERING_BC_ACTION_LOSS_METHOD_VARIANT,
    TERMINATION_METHOD_VARIANT,
    UNRECOVERABLE_PROGRESS_STALL_TERMINATION_METHOD_VARIANT,
    ActionConfig,
    AssettoCorsaRuntimeConfig,
    BehavioralCloningConfig,
    EnvironmentConfig,
    ModelConfig,
    PPOConfig,
    ProjectConfig,
    ReferenceConfig,
    RewardConfig,
)


def test_load_project_config() -> None:
    config = ProjectConfig.from_toml(Path("configs/mock.toml"))
    assert config.experiment.seed == 769913
    assert config.model.feature_hidden_sizes == (512, 512)
    assert config.model.trajectory_codec == "bezier-origin-5x2-bounded-monotonic-10x0p1s-v2"
    assert config.model.trajectory_output_size == 10
    assert config.to_dict()["environment"]["control_frequency_hz"] == 20.0

    fixed = ProjectConfig.from_toml(Path("configs/bmw_z4_gt3_monza_baseline.toml"))
    assert fixed.dataset.car_id == "bmw_z4_gt3"
    assert fixed.dataset.track_id == "monza"
    assert fixed.reference.mode == "fixed"
    assert fixed.behavioral_cloning.method_variant == "ai_spline_absolute_v1"
    assert fixed.behavioral_cloning.mirror_augmentation is True
    assert fixed.actions.method_variant == "absolute_bounded_v1"
    assert fixed.environment.termination_method_variant == TERMINATION_METHOD_VARIANT
    assert fixed.rewards.prediction_method_variant == PREDICTION_REWARD_METHOD_VARIANT
    assert fixed.ppo.method_variant == STABILIZED_PPO_METHOD_VARIANT
    assert fixed.ppo.kl_control_method_variant == KL_CONTROL_METHOD_VARIANT
    assert fixed.environment.mock_track_length_m == pytest.approx(1000.0)
    assert fixed.environment.minimum_speed_mps * 3.6 == pytest.approx(5.0)
    assert fixed.environment.stall_startup_grace_steps == 25
    assert fixed.environment.stall_debounce_steps == 5
    assert fixed.environment.initial_combined_pedal == pytest.approx(1.0)
    assert fixed.environment.critical_body_sideslip_angle_deg == pytest.approx(20.0)
    assert fixed.rewards.prediction_progress_scale == pytest.approx(0.01)
    assert fixed.rewards.prediction_horizon_discount == pytest.approx(0.5)
    assert fixed.rewards.reference_tracking_penalty_scale == pytest.approx(0.10)
    assert fixed.rewards.reference_tracking_free_distance_m == pytest.approx(2.5)
    assert fixed.rewards.reference_tracking_distance_scale_m == pytest.approx(2.0)
    assert fixed.ppo.entropy_coefficient == pytest.approx(0.01)
    assert fixed.ppo.initial_log_std_after_bc == pytest.approx(-1.0)
    assert fixed.ppo.kl_target == pytest.approx(0.01)
    assert fixed.ppo.kl_tolerance == pytest.approx(0.003)
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
        (
            lambda: EnvironmentConfig(termination_method_variant="unknown"),
            "termination_method_variant",
        ),
        (
            lambda: EnvironmentConfig(progress_stall_window_s=0.0),
            "progress_stall_window_s",
        ),
        (
            lambda: EnvironmentConfig(progress_stall_minimum_progress_m=0.0),
            "progress_stall_minimum_progress_m",
        ),
        (lambda: ReferenceConfig(mode="unknown"), "reference.mode"),
        (
            lambda: RewardConfig(reference_tracking_penalty_scale=-0.1),
            "reference_tracking_penalty_scale",
        ),
        (
            lambda: RewardConfig(reference_tracking_distance_scale_m=0.0),
            "reference_tracking_distance_scale_m",
        ),
        (lambda: ModelConfig(trajectory_codec="unknown"), "trajectory_codec"),
        (lambda: ModelConfig(trajectory_output_size=20), "output_size = 10"),
        (
            lambda: BehavioralCloningConfig(method_variant="unknown"),
            "method_variant",
        ),
        (
            lambda: BehavioralCloningConfig(action_loss_method_variant="unknown"),
            "action_loss_method_variant",
        ),
        (
            lambda: BehavioralCloningConfig(
                predictor_target_normalization_method_variant="unknown"
            ),
            "predictor_target_normalization_method_variant",
        ),
        (
            lambda: BehavioralCloningConfig(steering_loss_weight=0.0),
            "steering_loss_weight",
        ),
        (
            lambda: BehavioralCloningConfig(steering_robustness_method_variant="unknown"),
            "steering_robustness_method_variant",
        ),
        (
            lambda: BehavioralCloningConfig(steering_robustness_loss_weight=0.0),
            "steering_robustness_loss_weight",
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
            lambda: AssettoCorsaRuntimeConfig(steering_output_gain=0.0),
            "steering_output_gain",
        ),
        (
            lambda: AssettoCorsaRuntimeConfig(x360_control_profile="unknown"),
            "x360_control_profile",
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
            lambda: AssettoCorsaRuntimeConfig(reset_reference_speed_tolerance_mps=0.0),
            "reset_reference_speed_tolerance_mps",
        ),
        (
            lambda: AssettoCorsaRuntimeConfig(reset_reference_heading_tolerance_rad=0.0),
            "reset_reference_heading_tolerance_rad",
        ),
        (
            lambda: AssettoCorsaRuntimeConfig(reset_reference_lateral_offset_m=5.1),
            "reset_reference_lateral_offset_m",
        ),
        (
            lambda: AssettoCorsaRuntimeConfig(reset_max_abs_slip_ratio=0.0),
            "reset_max_abs_slip_ratio",
        ),
        (
            lambda: AssettoCorsaRuntimeConfig(reset_slip_stable_frames=0),
            "reset_slip_stable_frames",
        ),
        (
            lambda: AssettoCorsaRuntimeConfig(reset_reference_speed_max_attempts=0),
            "reset_reference_speed_max_attempts",
        ),
        (
            lambda: AssettoCorsaRuntimeConfig(reset_backend="unknown"),
            "reset_backend",
        ),
        (
            lambda: AssettoCorsaRuntimeConfig(step_timing_method_variant="unknown"),
            "step_timing_method_variant",
        ),
    ],
)
def test_invalid_configuration_is_rejected(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_pre_action_progress_timing_is_an_explicit_runtime_variant() -> None:
    runtime = AssettoCorsaRuntimeConfig(
        step_timing_method_variant=PRE_ACTION_PROGRESS_BASELINE_METHOD_VARIANT
    )
    assert runtime.step_timing_method_variant == PRE_ACTION_PROGRESS_BASELINE_METHOD_VARIANT


def test_steering_weighted_bc_loss_is_an_explicit_variant() -> None:
    bc = BehavioralCloningConfig(
        action_loss_method_variant=STEERING_WEIGHTED_BC_ACTION_LOSS_METHOD_VARIANT,
        steering_loss_weight=20.0,
    )
    assert bc.steering_loss_weight == pytest.approx(20.0)

    straight = BehavioralCloningConfig(
        action_loss_method_variant=STRAIGHT_STEERING_BC_ACTION_LOSS_METHOD_VARIANT,
        steering_loss_weight=20.0,
    )
    assert straight.action_loss_method_variant == STRAIGHT_STEERING_BC_ACTION_LOSS_METHOD_VARIANT


def test_closed_loop_bc_and_policy_protected_ppo_variants_are_accepted() -> None:
    behavioral_cloning = BehavioralCloningConfig(
        steering_robustness_method_variant=(
            PERMUTED_FEEDBACK_BC_STEERING_ROBUSTNESS_METHOD_VARIANT
        ),
        steering_robustness_loss_weight=1.5,
    )
    ppo = PPOConfig(method_variant=POLICY_PROTECTED_PPO_METHOD_VARIANT)

    assert (
        behavioral_cloning.steering_robustness_method_variant
        == PERMUTED_FEEDBACK_BC_STEERING_ROBUSTNESS_METHOD_VARIANT
    )
    assert behavioral_cloning.steering_robustness_loss_weight == pytest.approx(1.5)
    assert ppo.method_variant == POLICY_PROTECTED_PPO_METHOD_VARIANT


def test_unrecoverable_progress_stall_is_an_explicit_termination_variant() -> None:
    environment = EnvironmentConfig(
        termination_method_variant=UNRECOVERABLE_PROGRESS_STALL_TERMINATION_METHOD_VARIANT,
        progress_stall_window_s=1.0,
        progress_stall_minimum_progress_m=2.0,
    )

    assert environment.progress_stall_window_s == pytest.approx(1.0)
    assert environment.progress_stall_minimum_progress_m == pytest.approx(2.0)

    active = ProjectConfig.from_toml(Path("configs/assetto_corsa_real_extended_smoke.toml"))
    assert (
        active.environment.termination_method_variant
        == UNRECOVERABLE_PROGRESS_STALL_TERMINATION_METHOD_VARIANT
    )
