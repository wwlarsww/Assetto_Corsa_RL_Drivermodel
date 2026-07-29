"""Validated, dependency-free configuration loaded from TOML artefacts."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self, TypeVar

ConfigSection = TypeVar("ConfigSection")


def _positive(name: str, value: float | int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Run-level settings that influence reproducibility and persistence."""

    seed: int = 769913
    max_episode_steps: int = 1200
    output_directory: str = "runs"

    def __post_init__(self) -> None:
        _positive("experiment.max_episode_steps", self.max_episode_steps)
        if self.seed < 0:
            raise ValueError("experiment.seed must be non-negative")


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Mock vehicle and safety-boundary settings."""

    control_frequency_hz: float = 20.0
    track_length_m: float = 1000.0
    track_half_width_m: float = 5.0
    initial_speed_mps: float = 20.0
    minimum_speed_mps: float = 2.0
    minimum_speed_grace_steps: int = 100
    critical_slip_ratio: float = 0.35

    def __post_init__(self) -> None:
        for name in (
            "control_frequency_hz",
            "track_length_m",
            "track_half_width_m",
            "initial_speed_mps",
            "minimum_speed_mps",
            "critical_slip_ratio",
        ):
            _positive(f"environment.{name}", getattr(self, name))
        if self.minimum_speed_mps >= self.initial_speed_mps:
            raise ValueError("minimum speed must be lower than the initial speed")
        if self.minimum_speed_grace_steps < 0:
            raise ValueError("minimum_speed_grace_steps must be non-negative")


@dataclass(frozen=True, slots=True)
class ActionConfig:
    """Scaling from policy output in ``[-1, 1]`` to physical action deltas."""

    pedal_delta_limit: float = 0.10
    steering_delta_limit: float = 0.05

    def __post_init__(self) -> None:
        _positive("actions.pedal_delta_limit", self.pedal_delta_limit)
        _positive("actions.steering_delta_limit", self.steering_delta_limit)
        if self.pedal_delta_limit > 1.0 or self.steering_delta_limit > 1.0:
            raise ValueError("action delta limits may not exceed the normalised actuator range")


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Coefficients for step-level and post-rollout reward terms.

    ``prediction_decay`` is the exponential horizon decay rate in ``1/s``.
    Style/Lagrange values are inactive compatibility settings in
    ``fixed_reference_no_promp_v1``.
    The mock defaults are examples, not calibrated thesis hyperparameters.
    """

    progress_scale: float = 1.0
    termination_penalty: float = -100.0
    prediction_progress_scale: float = 0.05
    style_initial_coefficient: float = 1.0
    style_target_cost: float = 0.25
    style_lagrange_learning_rate: float = 0.01
    prediction_decay: float = 0.15

    def __post_init__(self) -> None:
        _positive("rewards.progress_scale", self.progress_scale)
        _positive("rewards.style_target_cost", self.style_target_cost)
        _positive("rewards.style_lagrange_learning_rate", self.style_lagrange_learning_rate)
        if self.termination_penalty > 0:
            raise ValueError("termination_penalty must be non-positive")
        if self.prediction_progress_scale < 0 or self.style_initial_coefficient < 0:
            raise ValueError("prediction and style coefficients must be non-negative")
        if self.prediction_decay < 0:
            raise ValueError("prediction_decay must be non-negative")


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

    track_id: str = "ks_barcelona-layout_gp"
    car_id: str = "dallara_f317"
    root_directory: str = "data/external/assetto_corsa_gym"
    racing_line_relative_path: str = (
        "source/AssettoCorsaConfigs/tracks/ks_barcelona-layout_gp-racing_line_fixed.csv"
    )
    track_geometry_relative_path: str = (
        "source/AssettoCorsaConfigs/tracks/ks_barcelona-layout_gp.csv"
    )
    steering_lock_degrees: float = 240.0
    steering_lock_normalized: float = 0.5341796875

    def __post_init__(self) -> None:
        for name in (
            "track_id",
            "car_id",
            "root_directory",
            "racing_line_relative_path",
            "track_geometry_relative_path",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"dataset.{name} must not be empty")
        _positive("dataset.steering_lock_degrees", self.steering_lock_degrees)
        _positive("dataset.steering_lock_normalized", self.steering_lock_normalized)
        if self.steering_lock_normalized > 1.0:
            raise ValueError("dataset.steering_lock_normalized may not exceed one")

    @property
    def recordings_directory(self) -> Path:
        return Path(self.root_directory) / "data_sets" / self.track_id / self.car_id

    @property
    def racing_line_path(self) -> Path:
        return Path(self.root_directory) / self.racing_line_relative_path

    @property
    def track_geometry_path(self) -> Path:
        return Path(self.root_directory) / self.track_geometry_relative_path


@dataclass(frozen=True, slots=True)
class BehavioralCloningConfig:
    """Explicit public defaults for the joint policy/predictor warm start."""

    method_variant: str = "fixed_reference_no_promp_v1"
    dataset_artifact_path: str = "data/processed/f317_barcelona/bc_dataset.npz"
    epochs: int = 3
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    predictor_loss_weight: float = 0.01
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    minimum_speed_mps: float = 5.0

    def __post_init__(self) -> None:
        if self.method_variant != "fixed_reference_no_promp_v1":
            raise ValueError(
                "behavioral_cloning.method_variant must be 'fixed_reference_no_promp_v1'"
            )
        if not self.dataset_artifact_path.strip():
            raise ValueError("behavioral_cloning.dataset_artifact_path must not be empty")
        for name in (
            "epochs",
            "batch_size",
            "learning_rate",
            "predictor_loss_weight",
            "minimum_speed_mps",
        ):
            _positive(f"behavioral_cloning.{name}", getattr(self, name))
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
    """Named synchronous PPO baseline with auxiliary predictor updates."""

    method_variant: str = "fixed_reference_no_promp_v1"
    total_environment_steps: int = 1024
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    predictor_loss_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    use_predicted_progress_reward: bool = False
    initial_checkpoint_path: str = ""
    initial_checkpoint_metrics_path: str = ""
    maximum_initial_predictor_ade_m: float = 15.0
    maximum_initial_predictor_fde_m: float = 40.0

    def __post_init__(self) -> None:
        if self.method_variant != "fixed_reference_no_promp_v1":
            raise ValueError("ppo.method_variant must be 'fixed_reference_no_promp_v1'")
        for name in (
            "total_environment_steps",
            "rollout_steps",
            "update_epochs",
            "minibatch_size",
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
        if self.entropy_coefficient < 0 or self.predictor_loss_coefficient < 0:
            raise ValueError("PPO entropy and predictor coefficients must be non-negative")


@dataclass(frozen=True, slots=True)
class AssettoCorsaRuntimeConfig:
    """Local Windows paths and the explicit live-control backend."""

    installation_directory: str = r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
    controls_configuration_path: str = ""
    control_backend: str = "vgamepad"

    def __post_init__(self) -> None:
        if not self.installation_directory.strip():
            raise ValueError("assetto_corsa.installation_directory must not be empty")
        if self.control_backend not in {"vgamepad"}:
            raise ValueError("assetto_corsa.control_backend must be 'vgamepad'")

    @property
    def controls_path(self) -> Path:
        if self.controls_configuration_path:
            return Path(self.controls_configuration_path)
        return Path.home() / "Documents" / "Assetto Corsa" / "cfg" / "controls.ini"


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    """Episode-level reference selection and its safe local artefact."""

    mode: str = "fixed"
    artifact_path: str = "data/processed/f317_barcelona/fixed_reference.npz"

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "promp"}:
            raise ValueError("reference.mode must be 'fixed' or 'promp'")
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
    behavioral_cloning: BehavioralCloningConfig = field(default_factory=BehavioralCloningConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
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
            "environment",
            "actions",
            "rewards",
            "model",
            "dataset",
            "behavioral_cloning",
            "ppo",
            "assetto_corsa",
            "reference",
        }
        unknown_sections = set(raw) - allowed_sections
        if unknown_sections:
            names = ", ".join(sorted(unknown_sections))
            raise ValueError(f"Unknown configuration section(s): {names}")

        return cls(
            experiment=_load_section(ExperimentConfig, raw.get("experiment", {})),
            environment=_load_section(EnvironmentConfig, raw.get("environment", {})),
            actions=_load_section(ActionConfig, raw.get("actions", {})),
            rewards=_load_section(RewardConfig, raw.get("rewards", {})),
            model=_load_model_config(raw.get("model", {})),
            dataset=_load_section(DatasetConfig, raw.get("dataset", {})),
            behavioral_cloning=_load_section(
                BehavioralCloningConfig,
                raw.get("behavioral_cloning", {}),
            ),
            ppo=_load_section(PPOConfig, raw.get("ppo", {})),
            assetto_corsa=_load_section(
                AssettoCorsaRuntimeConfig,
                raw.get("assetto_corsa", {}),
            ),
            reference=_load_section(ReferenceConfig, raw.get("reference", {})),
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
