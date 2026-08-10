"""Validated, dependency-free configuration loaded from TOML artefacts."""

from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Self, TypeVar

from assetto_corsa_rl.models.trajectory_contract import (
    BEZIER_TRAJECTORY_CODEC,
    LEGACY_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT,
    TRAIN_SPLIT_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT,
)

ConfigSection = TypeVar("ConfigSection")
BASELINE_METHOD_VARIANT = "ai_spline_absolute_v1"
STABILIZED_PPO_METHOD_VARIANT = "ai_spline_absolute_stabilized_v2"
POLICY_PROTECTED_PPO_METHOD_VARIANT = "ai_spline_absolute_policy_protected_v3"
ACTION_METHOD_VARIANT = "absolute_bounded_v1"
TERMINATION_METHOD_VARIANT = "any_tyre_armed_stall_v1"
UNRECOVERABLE_PROGRESS_STALL_TERMINATION_METHOD_VARIANT = "unrecoverable_progress_stall_v2"
PREDICTION_REWARD_METHOD_VARIANT = "discounted_progress_increments_v1"
KL_CONTROL_METHOD_VARIANT = "target_band_learning_rate_v1"
CURRENT_REFERENCE_TRACKING_METHOD_VARIANT = "current_distance_quadratic_v1"
SUCCESSOR_REFERENCE_TRACKING_METHOD_VARIANT = "successor_progress_alignment_v1"
LEGACY_LIVE_STEP_TIMING_METHOD_VARIANT = "observation_to_post_hold_v1"
PRE_ACTION_PROGRESS_BASELINE_METHOD_VARIANT = "pre_action_progress_baseline_v1"
PRESERVE_USER_X360_CONTROL_PROFILE = "preserve_user_x360_v1"
LINEAR_BC_X360_CONTROL_PROFILE = "linear_bc_calibrated_x360_v2"
UNIFORM_BC_ACTION_LOSS_METHOD_VARIANT = "uniform_action_mse_v1"
STEERING_WEIGHTED_BC_ACTION_LOSS_METHOD_VARIANT = "steering_weighted_action_mse_v1"
STRAIGHT_STEERING_BC_ACTION_LOSS_METHOD_VARIANT = "straight_steering_weighted_action_mse_v2"
NO_BC_STEERING_ROBUSTNESS_METHOD_VARIANT = "none_v1"
PERMUTED_FEEDBACK_BC_STEERING_ROBUSTNESS_METHOD_VARIANT = "permuted_causal_feedback_steering_v1"


def _positive(name: str, value: float | int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _non_negative(name: str, value: float | int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r} ({type(value).__name__})")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _non_negative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r} ({type(value).__name__})")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _is_finite_real(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Run-level settings that influence reproducibility and persistence."""

    seed: int = 769913
    max_episode_steps: int = 1200
    truncate_on_max_episode_steps: bool = True
    output_directory: str = "runs"

    def __post_init__(self) -> None:
        _positive_integer("experiment.max_episode_steps", self.max_episode_steps)
        _non_negative_integer("experiment.seed", self.seed)


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Control rate plus explicitly mock-only vehicle and track settings."""

    control_frequency_hz: float = 20.0
    mock_track_length_m: float = 1000.0
    mock_track_half_width_m: float = 5.0
    initial_speed_mps: float = 20.0
    termination_method_variant: str = TERMINATION_METHOD_VARIANT
    minimum_speed_mps: float = 5.0 / 3.6
    stall_startup_grace_steps: int = 25
    stall_debounce_steps: int = 5
    progress_stall_window_s: float = 1.0
    progress_stall_minimum_progress_m: float = 2.0
    initial_combined_pedal: float = 1.0
    critical_body_sideslip_angle_deg: float = 20.0
    minimum_body_sideslip_termination_speed_mps: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "control_frequency_hz",
            "mock_track_length_m",
            "mock_track_half_width_m",
            "initial_speed_mps",
            "minimum_speed_mps",
            "critical_body_sideslip_angle_deg",
            "minimum_body_sideslip_termination_speed_mps",
        ):
            _positive(f"environment.{name}", getattr(self, name))
        if self.minimum_speed_mps >= self.initial_speed_mps:
            raise ValueError("minimum speed must be lower than the initial speed")
        if self.termination_method_variant not in {
            TERMINATION_METHOD_VARIANT,
            UNRECOVERABLE_PROGRESS_STALL_TERMINATION_METHOD_VARIANT,
        }:
            raise ValueError(
                "environment.termination_method_variant must be "
                f"{TERMINATION_METHOD_VARIANT!r} or "
                f"{UNRECOVERABLE_PROGRESS_STALL_TERMINATION_METHOD_VARIANT!r}"
            )
        _positive_integer("environment.stall_debounce_steps", self.stall_debounce_steps)
        _positive_integer(
            "environment.stall_startup_grace_steps",
            self.stall_startup_grace_steps,
        )
        _positive("environment.progress_stall_window_s", self.progress_stall_window_s)
        _positive(
            "environment.progress_stall_minimum_progress_m",
            self.progress_stall_minimum_progress_m,
        )
        if not -1.0 <= self.initial_combined_pedal <= 1.0:
            raise ValueError("initial_combined_pedal must lie in [-1, 1]")
        if self.critical_body_sideslip_angle_deg > 90.0:
            raise ValueError("critical body sideslip angle must not exceed 90 degrees")


@dataclass(frozen=True, slots=True)
class ActionConfig:
    """Named bounded absolute-action contract."""

    method_variant: str = ACTION_METHOD_VARIANT

    def __post_init__(self) -> None:
        if self.method_variant != ACTION_METHOD_VARIANT:
            raise ValueError(f"actions.method_variant must be {ACTION_METHOD_VARIANT!r}")


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Progress, reference tracking and prediction reward configuration."""

    progress_scale: float = 1.0
    termination_penalty: float = -100.0
    prediction_method_variant: str = PREDICTION_REWARD_METHOD_VARIANT
    prediction_progress_scale: float = 0.01
    prediction_horizon_discount: float = 0.5
    reference_tracking_method_variant: str = CURRENT_REFERENCE_TRACKING_METHOD_VARIANT
    reference_tracking_penalty_scale: float = 0.0
    reference_tracking_free_distance_m: float = 2.5
    reference_tracking_distance_scale_m: float = 2.0
    reference_heading_penalty_scale: float = 0.0
    reference_heading_scale_rad: float = math.pi / 18.0

    def __post_init__(self) -> None:
        _positive("rewards.progress_scale", self.progress_scale)
        if self.termination_penalty > 0:
            raise ValueError("termination_penalty must be non-positive")
        if self.prediction_method_variant != PREDICTION_REWARD_METHOD_VARIANT:
            raise ValueError(
                f"rewards.prediction_method_variant must be {PREDICTION_REWARD_METHOD_VARIANT!r}"
            )
        if self.prediction_progress_scale < 0:
            raise ValueError("prediction_progress_scale must be non-negative")
        if not 0.0 < self.prediction_horizon_discount <= 1.0:
            raise ValueError("prediction_horizon_discount must lie in (0, 1]")
        if self.reference_tracking_method_variant not in {
            CURRENT_REFERENCE_TRACKING_METHOD_VARIANT,
            SUCCESSOR_REFERENCE_TRACKING_METHOD_VARIANT,
        }:
            raise ValueError(
                "rewards.reference_tracking_method_variant must be "
                f"{CURRENT_REFERENCE_TRACKING_METHOD_VARIANT!r} or "
                f"{SUCCESSOR_REFERENCE_TRACKING_METHOD_VARIANT!r}"
            )
        _non_negative(
            "rewards.reference_tracking_penalty_scale",
            self.reference_tracking_penalty_scale,
        )
        _non_negative(
            "rewards.reference_tracking_free_distance_m",
            self.reference_tracking_free_distance_m,
        )
        _positive(
            "rewards.reference_tracking_distance_scale_m",
            self.reference_tracking_distance_scale_m,
        )
        _non_negative(
            "rewards.reference_heading_penalty_scale",
            self.reference_heading_penalty_scale,
        )
        _positive(
            "rewards.reference_heading_scale_rad",
            self.reference_heading_scale_rad,
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Figure 2.3 dimensions plus explicit public implementation defaults."""

    feature_hidden_sizes: tuple[int, ...] = (512, 512)
    policy_hidden_sizes: tuple[int, ...] = (128, 128)
    value_hidden_sizes: tuple[int, ...] = (512, 512)
    predictor_hidden_sizes: tuple[int, ...] = (128, 128)
    trajectory_output_size: int = 10
    trajectory_codec: str = BEZIER_TRAJECTORY_CODEC
    initial_log_std: float = -0.5

    def __post_init__(self) -> None:
        for name in (
            "feature_hidden_sizes",
            "policy_hidden_sizes",
            "value_hidden_sizes",
            "predictor_hidden_sizes",
        ):
            sizes = getattr(self, name)
            if not sizes or any(size <= 0 for size in sizes):
                raise ValueError(f"model.{name} must contain positive layer sizes")
        _positive("model.trajectory_output_size", self.trajectory_output_size)
        if self.trajectory_codec != BEZIER_TRAJECTORY_CODEC:
            raise ValueError(f"model.trajectory_codec must be {BEZIER_TRAJECTORY_CODEC!r}")
        if self.trajectory_output_size != 10:
            raise ValueError(
                f"the {BEZIER_TRAJECTORY_CODEC} codec requires trajectory_output_size = 10"
            )


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Paths and stable identifiers for the initial public dataset subset."""

    track_id: str = "monza"
    track_layout: str = ""
    car_id: str = "bmw_z4_gt3"
    root_directory: str = "data/external/assetto_corsa_gym"
    steering_lock_degrees: float = 320.0
    steering_lock_normalized: float = 0.7119140625

    def __post_init__(self) -> None:
        for name in (
            "track_id",
            "car_id",
            "root_directory",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"dataset.{name} must not be empty")
        _positive("dataset.steering_lock_degrees", self.steering_lock_degrees)
        _positive("dataset.steering_lock_normalized", self.steering_lock_normalized)
        if self.steering_lock_normalized > 1.0:
            raise ValueError("dataset.steering_lock_normalized may not exceed one")

    @property
    def recordings_directory(self) -> Path:
        """Return the configured raw recording directory."""

        return Path(self.root_directory) / "data_sets" / self.track_id / self.car_id


@dataclass(frozen=True, slots=True)
class TrackAssetsConfig:
    """Automatic Assetto Corsa track-asset generation and caching."""

    source: str = "assetto_corsa_fast_lane"
    auto_build: bool = True
    cache_directory: str = "data/generated/assetto_corsa_tracks"
    reference_speed_source: str = "human_if_available_else_ai"
    human_reference_artifact_path: str = ""
    minimum_reference_speed_mps: float = 5.0

    def __post_init__(self) -> None:
        if self.source != "assetto_corsa_fast_lane":
            raise ValueError("track_assets.source must be 'assetto_corsa_fast_lane'")
        if not self.cache_directory.strip():
            raise ValueError("track_assets.cache_directory must not be empty")
        if self.reference_speed_source not in {
            "ai",
            "human_if_available_else_ai",
        }:
            raise ValueError(
                "track_assets.reference_speed_source must be 'ai' or 'human_if_available_else_ai'"
            )
        _positive(
            "track_assets.minimum_reference_speed_mps",
            self.minimum_reference_speed_mps,
        )

    def generated_directory(self, dataset: DatasetConfig) -> Path:
        """Return the deterministic cache directory for a configured track."""

        layout = dataset.track_layout.strip() or "default"
        return Path(self.cache_directory) / dataset.track_id / layout

    def generated_geometry_path(self, dataset: DatasetConfig) -> Path:
        """Return the generated static geometry artefact path."""

        return self.generated_directory(dataset) / "track_geometry.npz"

    def generated_reference_path(self, dataset: DatasetConfig) -> Path:
        """Return the generated reference artefact path."""

        return self.generated_directory(dataset) / "reference.npz"


@dataclass(frozen=True, slots=True)
class BehavioralCloningConfig:
    """Explicit public defaults for the fixed-reference warm start."""

    method_variant: str = BASELINE_METHOD_VARIANT
    dataset_artifact_path: str = "data/processed/bmw_z4_gt3_monza/bc_dataset.npz"
    epochs: int = 3
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    predictor_loss_weight: float = 0.0
    predictor_target_normalization_method_variant: str = (
        LEGACY_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT
    )
    action_loss_method_variant: str = UNIFORM_BC_ACTION_LOSS_METHOD_VARIANT
    steering_loss_weight: float = 1.0
    steering_robustness_method_variant: str = NO_BC_STEERING_ROBUSTNESS_METHOD_VARIANT
    steering_robustness_loss_weight: float = 1.0
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    minimum_speed_mps: float = 0.1
    mirror_augmentation: bool = True

    def __post_init__(self) -> None:
        if self.method_variant != BASELINE_METHOD_VARIANT:
            raise ValueError(
                f"behavioral_cloning.method_variant must be {BASELINE_METHOD_VARIANT!r}"
            )
        if not self.dataset_artifact_path.strip():
            raise ValueError("behavioral_cloning.dataset_artifact_path must not be empty")
        for name in ("epochs", "batch_size"):
            _positive_integer(
                f"behavioral_cloning.{name}",
                getattr(self, name),
            )
        _positive("behavioral_cloning.learning_rate", self.learning_rate)
        _non_negative(
            "behavioral_cloning.minimum_speed_mps",
            self.minimum_speed_mps,
        )
        _non_negative("behavioral_cloning.predictor_loss_weight", self.predictor_loss_weight)
        if self.predictor_target_normalization_method_variant not in {
            LEGACY_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT,
            TRAIN_SPLIT_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT,
        }:
            raise ValueError(
                "behavioral_cloning.predictor_target_normalization_method_variant must be "
                f"{LEGACY_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT!r} or "
                f"{TRAIN_SPLIT_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT!r}"
            )
        if self.action_loss_method_variant not in {
            UNIFORM_BC_ACTION_LOSS_METHOD_VARIANT,
            STEERING_WEIGHTED_BC_ACTION_LOSS_METHOD_VARIANT,
            STRAIGHT_STEERING_BC_ACTION_LOSS_METHOD_VARIANT,
        }:
            raise ValueError(
                "behavioral_cloning.action_loss_method_variant must be "
                f"{UNIFORM_BC_ACTION_LOSS_METHOD_VARIANT!r} or "
                f"{STEERING_WEIGHTED_BC_ACTION_LOSS_METHOD_VARIANT!r} or "
                f"{STRAIGHT_STEERING_BC_ACTION_LOSS_METHOD_VARIANT!r}"
            )
        _positive(
            "behavioral_cloning.steering_loss_weight",
            self.steering_loss_weight,
        )
        if self.steering_robustness_method_variant not in {
            NO_BC_STEERING_ROBUSTNESS_METHOD_VARIANT,
            PERMUTED_FEEDBACK_BC_STEERING_ROBUSTNESS_METHOD_VARIANT,
        }:
            raise ValueError(
                "behavioral_cloning.steering_robustness_method_variant must be "
                f"{NO_BC_STEERING_ROBUSTNESS_METHOD_VARIANT!r} or "
                f"{PERMUTED_FEEDBACK_BC_STEERING_ROBUSTNESS_METHOD_VARIANT!r}"
            )
        _positive(
            "behavioral_cloning.steering_robustness_loss_weight",
            self.steering_robustness_loss_weight,
        )
        if self.weight_decay < 0:
            raise ValueError("behavioral_cloning.weight_decay must be non-negative")
        for name in ("validation_fraction", "test_fraction"):
            value = getattr(self, name)
            if not 0 < value < 1:
                raise ValueError(f"behavioral_cloning.{name} must lie in (0, 1)")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("behavioral-cloning split fractions must leave training sessions")


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Named synchronous PPO variant for the fixed-reference setup."""

    method_variant: str = STABILIZED_PPO_METHOD_VARIANT
    total_environment_steps: int = 1024
    rollout_steps: int = 256
    checkpoint_interval_updates: int = 5
    update_epochs: int = 4
    minibatch_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    predictor_loss_coefficient: float = 0.0
    initial_log_std_after_bc: float = -1.0
    kl_control_method_variant: str = KL_CONTROL_METHOD_VARIANT
    kl_target: float = 0.01
    kl_tolerance: float = 0.003
    kl_learning_rate_decrease_factor: float = 0.5
    kl_learning_rate_increase_factor: float = 1.1
    minimum_learning_rate: float = 3e-5
    max_gradient_norm: float = 0.5
    use_predicted_progress_reward: bool = False
    initial_checkpoint_path: str = ""
    initial_checkpoint_metrics_path: str = ""
    maximum_initial_predictor_ade_m: float = 15.0
    maximum_initial_predictor_fde_m: float = 40.0

    def __post_init__(self) -> None:
        _validate_ppo_identity_and_sizes(self)
        _validate_ppo_objective_ranges(self)
        _validate_ppo_kl_control(self)


def _validate_ppo_identity_and_sizes(config: PPOConfig) -> None:
    if config.method_variant not in {
        STABILIZED_PPO_METHOD_VARIANT,
        POLICY_PROTECTED_PPO_METHOD_VARIANT,
    }:
        raise ValueError(
            "ppo.method_variant must be "
            f"{STABILIZED_PPO_METHOD_VARIANT!r} or "
            f"{POLICY_PROTECTED_PPO_METHOD_VARIANT!r}"
        )
    for name in (
        "total_environment_steps",
        "rollout_steps",
        "checkpoint_interval_updates",
        "update_epochs",
        "minibatch_size",
    ):
        _positive_integer(f"ppo.{name}", getattr(config, name))
    for name in (
        "learning_rate",
        "value_loss_coefficient",
        "max_gradient_norm",
        "maximum_initial_predictor_ade_m",
        "maximum_initial_predictor_fde_m",
    ):
        _positive(f"ppo.{name}", getattr(config, name))
    if config.minibatch_size > config.rollout_steps:
        raise ValueError("ppo.minibatch_size may not exceed rollout_steps")


def _validate_ppo_objective_ranges(config: PPOConfig) -> None:
    for name in ("gamma", "gae_lambda"):
        value = getattr(config, name)
        if not 0 < value <= 1:
            raise ValueError(f"ppo.{name} must lie in (0, 1]")
    if not 0 < config.clip_range < 1:
        raise ValueError("ppo.clip_range must lie in (0, 1)")
    _non_negative("ppo.entropy_coefficient", config.entropy_coefficient)
    _non_negative("ppo.predictor_loss_coefficient", config.predictor_loss_coefficient)
    if not math.isfinite(config.initial_log_std_after_bc):
        raise ValueError("ppo.initial_log_std_after_bc must be finite")


def _validate_ppo_kl_control(config: PPOConfig) -> None:
    if config.kl_control_method_variant != KL_CONTROL_METHOD_VARIANT:
        raise ValueError(f"ppo.kl_control_method_variant must be {KL_CONTROL_METHOD_VARIANT!r}")
    _positive("ppo.kl_target", config.kl_target)
    _positive("ppo.kl_tolerance", config.kl_tolerance)
    if config.kl_tolerance >= config.kl_target:
        raise ValueError("ppo.kl_tolerance must be lower than kl_target")
    if not 0.0 < config.kl_learning_rate_decrease_factor < 1.0:
        raise ValueError("ppo.kl_learning_rate_decrease_factor must lie in (0, 1)")
    if config.kl_learning_rate_increase_factor <= 1.0:
        raise ValueError("ppo.kl_learning_rate_increase_factor must exceed one")
    _positive("ppo.minimum_learning_rate", config.minimum_learning_rate)
    if config.minimum_learning_rate > config.learning_rate:
        raise ValueError("ppo.minimum_learning_rate may not exceed learning_rate")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Periodic policy evaluation settings for reproducible checkpoints."""

    enabled: bool = False
    interval_updates: int = 5
    episodes: int = 3
    deterministic: bool = True
    seed_offset: int = 10_000
    record_trajectories: bool = True
    evaluate_final_update: bool = True

    def __post_init__(self) -> None:
        _positive_integer("evaluation.interval_updates", self.interval_updates)
        _positive_integer("evaluation.episodes", self.episodes)
        _non_negative_integer("evaluation.seed_offset", self.seed_offset)


@dataclass(frozen=True, slots=True)
class AssettoCorsaRuntimeConfig:
    """Local Windows paths and explicit, auditable live adapter settings."""

    installation_directory: str = r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
    controls_configuration_path: str = ""
    control_backend: str = "vgamepad"
    x360_control_profile: str = PRESERVE_USER_X360_CONTROL_PROFILE
    reset_backend: str = "csp_hotlap_spawn_v1"
    crash_termination: str = "csp_collision_latched_v1"
    telemetry_calibration: str = "assetto_corsa_gym_world_v1"
    step_timing_method_variant: str = LEGACY_LIVE_STEP_TIMING_METHOD_VARIANT
    automatic_shifting: bool = True
    steering_output_sign: float = -1.0
    steering_output_gain: float = 1.0
    frame_timeout_s: float = 2.0
    reset_timeout_s: float = 10.0
    reset_position_tolerance: float = 0.05
    reset_reference_lateral_offset_m: float = 0.0
    reset_reference_heading_tolerance_rad: float = 0.05
    reset_speed_threshold_mps: float = 5.0
    reset_reference_speed_tolerance_mps: float = 2.0
    reset_stable_frames: int = 3
    reset_max_abs_slip_ratio: float = 0.2
    reset_slip_stable_frames: int = 1
    reset_reference_speed_max_attempts: int = 1
    reset_collision_quiescence_s: float = 0.5
    poll_interval_s: float = 0.001
    launch_timeout_s: float = 120.0
    launch_poll_interval_s: float = 0.25

    def __post_init__(self) -> None:
        if not self.installation_directory.strip():
            raise ValueError("assetto_corsa.installation_directory must not be empty")
        if self.control_backend not in {"vgamepad"}:
            raise ValueError("assetto_corsa.control_backend must be 'vgamepad'")
        if self.x360_control_profile not in {
            PRESERVE_USER_X360_CONTROL_PROFILE,
            LINEAR_BC_X360_CONTROL_PROFILE,
        }:
            raise ValueError(
                "assetto_corsa.x360_control_profile must be "
                f"{PRESERVE_USER_X360_CONTROL_PROFILE!r} or "
                f"{LINEAR_BC_X360_CONTROL_PROFILE!r}"
            )
        if self.reset_backend not in {
            "csp_hotlap_spawn_v1",
            "csp_hotlap_reference_speed_v1",
            "csp_reference_pose_speed_v2",
            "python_ext_reset_car_v1",
        }:
            raise ValueError(
                "assetto_corsa.reset_backend must be 'csp_hotlap_spawn_v1', "
                "'csp_hotlap_reference_speed_v1', 'csp_reference_pose_speed_v2' "
                "or 'python_ext_reset_car_v1'"
            )
        if self.crash_termination != "csp_collision_latched_v1":
            raise ValueError("assetto_corsa.crash_termination must be 'csp_collision_latched_v1'")
        if self.telemetry_calibration != "assetto_corsa_gym_world_v1":
            raise ValueError(
                "assetto_corsa.telemetry_calibration must be 'assetto_corsa_gym_world_v1'"
            )
        if self.step_timing_method_variant not in {
            LEGACY_LIVE_STEP_TIMING_METHOD_VARIANT,
            PRE_ACTION_PROGRESS_BASELINE_METHOD_VARIANT,
        }:
            raise ValueError(
                "assetto_corsa.step_timing_method_variant must be "
                f"{LEGACY_LIVE_STEP_TIMING_METHOD_VARIANT!r} or "
                f"{PRE_ACTION_PROGRESS_BASELINE_METHOD_VARIANT!r}"
            )
        if self.steering_output_sign not in {-1.0, 1.0}:
            raise ValueError("assetto_corsa.steering_output_sign must be -1.0 or 1.0")
        if (
            not _is_finite_real(self.steering_output_gain)
            or not 0.0 < self.steering_output_gain <= 2.0
        ):
            raise ValueError("assetto_corsa.steering_output_gain must be finite and lie in (0, 2]")
        if (
            not math.isfinite(self.reset_reference_lateral_offset_m)
            or abs(self.reset_reference_lateral_offset_m) > 5.0
        ):
            raise ValueError(
                "assetto_corsa.reset_reference_lateral_offset_m must be finite and lie in [-5, 5]"
            )
        for name in (
            "frame_timeout_s",
            "reset_timeout_s",
            "reset_position_tolerance",
            "reset_reference_heading_tolerance_rad",
            "reset_speed_threshold_mps",
            "reset_reference_speed_tolerance_mps",
            "reset_max_abs_slip_ratio",
            "poll_interval_s",
            "launch_timeout_s",
            "launch_poll_interval_s",
        ):
            _positive(f"assetto_corsa.{name}", getattr(self, name))
        _positive_integer(
            "assetto_corsa.reset_stable_frames",
            self.reset_stable_frames,
        )
        _positive_integer(
            "assetto_corsa.reset_slip_stable_frames",
            self.reset_slip_stable_frames,
        )
        _positive_integer(
            "assetto_corsa.reset_reference_speed_max_attempts",
            self.reset_reference_speed_max_attempts,
        )
        _non_negative(
            "assetto_corsa.reset_collision_quiescence_s",
            self.reset_collision_quiescence_s,
        )

    @property
    def controls_path(self) -> Path:
        """Return the explicit or standard per-user controls configuration."""

        if self.controls_configuration_path:
            return Path(self.controls_configuration_path)
        return Path.home() / "Documents" / "Assetto Corsa" / "cfg" / "controls.ini"


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    """Episode-level reference selection and its safe local artefact."""

    mode: str = "fixed"
    artifact_path: str = "auto"

    def __post_init__(self) -> None:
        if self.mode != "fixed":
            raise ValueError("reference.mode must be 'fixed'")
        if not self.artifact_path.strip():
            raise ValueError("reference.artifact_path must not be empty")


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Complete immutable experiment configuration."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    track_assets: TrackAssetsConfig = field(default_factory=TrackAssetsConfig)
    behavioral_cloning: BehavioralCloningConfig = field(default_factory=BehavioralCloningConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    assetto_corsa: AssettoCorsaRuntimeConfig = field(default_factory=AssettoCorsaRuntimeConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> Self:
        """Load configuration while rejecting unknown sections and keys."""

        config_path = Path(path)
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)

        allowed_sections = {
            "experiment",
            "monitoring",
            "environment",
            "actions",
            "rewards",
            "model",
            "dataset",
            "track_assets",
            "behavioral_cloning",
            "ppo",
            "evaluation",
            "assetto_corsa",
            "reference",
        }
        unknown_sections = set(raw) - allowed_sections
        if unknown_sections:
            names = ", ".join(sorted(unknown_sections))
            raise ValueError(f"Unknown configuration section(s): {names}")

        dataset = _load_section(DatasetConfig, raw.get("dataset", {}))
        track_assets = _load_section(
            TrackAssetsConfig,
            raw.get("track_assets", {}),
        )
        reference = _load_section(ReferenceConfig, raw.get("reference", {}))
        if reference.artifact_path.strip().lower() == "auto":
            reference = replace(
                reference,
                artifact_path=str(track_assets.generated_reference_path(dataset)),
            )

        return cls(
            experiment=_load_section(ExperimentConfig, raw.get("experiment", {})),
            environment=_load_section(EnvironmentConfig, raw.get("environment", {})),
            actions=_load_section(ActionConfig, raw.get("actions", {})),
            rewards=_load_section(RewardConfig, raw.get("rewards", {})),
            model=_load_model_config(raw.get("model", {})),
            dataset=dataset,
            track_assets=track_assets,
            behavioral_cloning=_load_section(
                BehavioralCloningConfig,
                raw.get("behavioral_cloning", {}),
            ),
            ppo=_load_section(PPOConfig, raw.get("ppo", {})),
            evaluation=_load_section(
                EvaluationConfig,
                raw.get("evaluation", {}),
            ),
            assetto_corsa=_load_section(
                AssettoCorsaRuntimeConfig,
                raw.get("assetto_corsa", {}),
            ),
            reference=reference,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot."""

        return asdict(self)


def _load_section(
    section_type: type[ConfigSection],
    values: dict[str, Any],
) -> ConfigSection:
    allowed = set(getattr(section_type, "__dataclass_fields__"))  # noqa: B009
    unknown = set(values) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown key(s) in [{section_type.__name__}]: {names}")
    return section_type(**values)


def _load_model_config(values: dict[str, Any]) -> ModelConfig:
    converted = dict(values)
    for key in (
        "feature_hidden_sizes",
        "policy_hidden_sizes",
        "value_hidden_sizes",
        "predictor_hidden_sizes",
    ):
        if key in converted:
            converted[key] = tuple(int(value) for value in converted[key])
    return _load_section(ModelConfig, converted)
