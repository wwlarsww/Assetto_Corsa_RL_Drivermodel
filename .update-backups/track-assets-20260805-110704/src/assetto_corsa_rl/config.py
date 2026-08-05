"""Validated, dependency-free configuration loaded from TOML artefacts."""

from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Self, TypeVar

ConfigSection = TypeVar("ConfigSection")
BASELINE_METHOD_VARIANT = "ai_spline_absolute_v1"
STABILIZED_PPO_METHOD_VARIANT = "ai_spline_absolute_stabilized_v2"
ACTION_METHOD_VARIANT = "absolute_bounded_v1"
TERMINATION_METHOD_VARIANT = "any_tyre_armed_stall_v1"
PREDICTION_REWARD_METHOD_VARIANT = "discounted_progress_increments_v1"
KL_CONTROL_METHOD_VARIANT = "target_band_learning_rate_v1"


def _positive(name: str, value: float | int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _non_negative(name: str, value: float | int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be an integer, got {value!r} ({type(value).__name__})"
        )
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Run-level settings that influence reproducibility and persistence."""

    seed: int = 769913
    max_episode_steps: int = 1200
    truncate_on_max_episode_steps: bool = True
    output_directory: str = "runs"

    def __post_init__(self) -> None:
        _positive_integer("experiment.max_episode_steps", self.max_episode_steps)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("experiment.seed must be an integer")
        if self.seed < 0:
            raise ValueError("experiment.seed must be non-negative")


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
        if self.termination_method_variant != TERMINATION_METHOD_VARIANT:
            raise ValueError(
                f"environment.termination_method_variant must be {TERMINATION_METHOD_VARIANT!r}"
            )
        _positive_integer("environment.stall_debounce_steps", self.stall_debounce_steps)
        _positive_integer(
            "environment.stall_startup_grace_steps",
            self.stall_startup_grace_steps,
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
    """Progress and named rollout prediction-reward configuration."""

    progress_scale: float = 1.0
    termination_penalty: float = -100.0
    prediction_method_variant: str = PREDICTION_REWARD_METHOD_VARIANT
    prediction_progress_scale: float = 0.01
    prediction_horizon_discount: float = 0.5

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


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Figure 2.3 dimensions plus explicit public implementation defaults."""

    feature_hidden_sizes: tuple[int, ...] = (512, 512)
    policy_hidden_sizes: tuple[int, ...] = (128, 128)
    value_hidden_sizes: tuple[int, ...] = (512, 512)
    predictor_hidden_sizes: tuple[int, ...] = (128, 128)
    trajectory_output_size: int = 10
    trajectory_codec: str = "bezier-5x2"
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
        if self.trajectory_codec != "bezier-5x2":
            raise ValueError("model.trajectory_codec must be 'bezier-5x2'")
        if self.trajectory_output_size != 10:
            raise ValueError("the bezier-5x2 codec requires trajectory_output_size = 10")


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
            raise ValueError(
                "track_assets.source must be 'assetto_corsa_fast_lane'"
            )
        if not self.cache_directory.strip():
            raise ValueError("track_assets.cache_directory must not be empty")
        if self.reference_speed_source not in {
            "ai",
            "human_if_available_else_ai",
        }:
            raise ValueError(
                "track_assets.reference_speed_source must be 'ai' or "
                "'human_if_available_else_ai'"
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
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    minimum_speed_mps: float = 5.0

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
        for name in ("learning_rate", "minimum_speed_mps"):
            _positive(f"behavioral_cloning.{name}", getattr(self, name))
        _non_negative("behavioral_cloning.predictor_loss_weight", self.predictor_loss_weight)
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
    entropy_coefficient: float = 0.0
    predictor_loss_coefficient: float = 0.0
    initial_log_std_after_bc: float = -1.5
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
        if self.method_variant != STABILIZED_PPO_METHOD_VARIANT:
            raise ValueError(f"ppo.method_variant must be {STABILIZED_PPO_METHOD_VARIANT!r}")
        for name in (
            "total_environment_steps",
            "rollout_steps",
            "checkpoint_interval_updates",
            "update_epochs",
            "minibatch_size",
        ):
            _positive_integer(f"ppo.{name}", getattr(self, name))
        for name in (
            "learning_rate",
            "value_loss_coefficient",
            "max_gradient_norm",
            "maximum_initial_predictor_ade_m",
            "maximum_initial_predictor_fde_m",
        ):
            _positive(f"ppo.{name}", getattr(self, name))
        if self.minibatch_size > self.rollout_steps:
            raise ValueError("ppo.minibatch_size may not exceed rollout_steps")
        for name in ("gamma", "gae_lambda"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"ppo.{name} must lie in (0, 1]")
        if not 0 < self.clip_range < 1:
            raise ValueError("ppo.clip_range must lie in (0, 1)")
        _non_negative("ppo.entropy_coefficient", self.entropy_coefficient)
        _non_negative("ppo.predictor_loss_coefficient", self.predictor_loss_coefficient)
        if not math.isfinite(self.initial_log_std_after_bc):
            raise ValueError("ppo.initial_log_std_after_bc must be finite")
        if self.kl_control_method_variant != KL_CONTROL_METHOD_VARIANT:
            raise ValueError(f"ppo.kl_control_method_variant must be {KL_CONTROL_METHOD_VARIANT!r}")
        _positive("ppo.kl_target", self.kl_target)
        _positive("ppo.kl_tolerance", self.kl_tolerance)
        if self.kl_tolerance >= self.kl_target:
            raise ValueError("ppo.kl_tolerance must be lower than kl_target")
        if not 0.0 < self.kl_learning_rate_decrease_factor < 1.0:
            raise ValueError("ppo.kl_learning_rate_decrease_factor must lie in (0, 1)")
        if self.kl_learning_rate_increase_factor <= 1.0:
            raise ValueError("ppo.kl_learning_rate_increase_factor must exceed one")
        _positive("ppo.minimum_learning_rate", self.minimum_learning_rate)
        if self.minimum_learning_rate > self.learning_rate:
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
        if isinstance(self.seed_offset, bool) or not isinstance(self.seed_offset, int):
            raise TypeError("evaluation.seed_offset must be an integer")
        if self.seed_offset < 0:
            raise ValueError("evaluation.seed_offset must be non-negative")


@dataclass(frozen=True, slots=True)
class AssettoCorsaRuntimeConfig:
    """Local Windows paths and explicit, auditable live adapter settings."""

    installation_directory: str = r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
    controls_configuration_path: str = ""
    control_backend: str = "vgamepad"
    reset_backend: str = "csp_hotlap_spawn_v1"
    crash_termination: str = "csp_collision_latched_v1"
    telemetry_calibration: str = "assetto_corsa_gym_world_v1"
    automatic_shifting: bool = True
    steering_output_sign: float = -1.0
    frame_timeout_s: float = 2.0
    reset_timeout_s: float = 10.0
    reset_position_tolerance: float = 0.05
    reset_speed_threshold_mps: float = 5.0
    reset_stable_frames: int = 3
    reset_collision_quiescence_s: float = 0.5
    poll_interval_s: float = 0.001
    launch_timeout_s: float = 120.0
    launch_poll_interval_s: float = 0.25

    def __post_init__(self) -> None:
        if not self.installation_directory.strip():
            raise ValueError("assetto_corsa.installation_directory must not be empty")
        if self.control_backend not in {"vgamepad"}:
            raise ValueError("assetto_corsa.control_backend must be 'vgamepad'")
        if self.reset_backend not in {
            "csp_hotlap_spawn_v1",
            "python_ext_reset_car_v1",
        }:
            raise ValueError(
                "assetto_corsa.reset_backend must be 'csp_hotlap_spawn_v1' or "
                "'python_ext_reset_car_v1'"
            )
        if self.crash_termination != "csp_collision_latched_v1":
            raise ValueError("assetto_corsa.crash_termination must be 'csp_collision_latched_v1'")
        if self.telemetry_calibration != "assetto_corsa_gym_world_v1":
            raise ValueError(
                "assetto_corsa.telemetry_calibration must be 'assetto_corsa_gym_world_v1'"
            )
        if self.steering_output_sign not in {-1.0, 1.0}:
            raise ValueError("assetto_corsa.steering_output_sign must be -1.0 or 1.0")
        for name in (
            "frame_timeout_s",
            "reset_timeout_s",
            "reset_position_tolerance",
            "reset_speed_threshold_mps",
            "poll_interval_s",
            "launch_timeout_s",
            "launch_poll_interval_s",
        ):
            _positive(f"assetto_corsa.{name}", getattr(self, name))
        _positive_integer(
            "assetto_corsa.reset_stable_frames",
            self.reset_stable_frames,
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
