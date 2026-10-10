"""Strict, complete TOML settings for a training run, without implicit defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
import re
import tomllib
from typing import Any, TypeVar, cast, get_args, get_origin, get_type_hints

from assetto_corsa_ai.agent import PPOConfig
from assetto_corsa_ai.predictor import PredictorConfig, horizon_indices


@dataclass(frozen=True)
class EnvironmentConfig:
    car: str
    track: str
    track_layout: str
    skin: str
    abs_mode: int
    traction_control_mode: int
    control_interval: float
    max_episode_time: float
    max_episode_steps: int
    reset_position_tolerance: float
    reset_heading_tolerance_degrees: float
    reset_speed_tolerance_kmh: float
    reset_timeout: float
    reset_poll_interval: float
    start_sampler: str
    action_mode: str
    front_axle_distance: float
    rear_axle_distance: float
    minimum_speed: float
    lookahead_distances: tuple[float, ...]
    reference_distances: tuple[float, ...]


@dataclass(frozen=True)
class SimulatorConfig:
    assetto_corsa_directory: str
    documents_directory: str
    startup_timeout: float
    drive_ready_timeout: float
    poll_interval: float
    car_index: int
    activation_timeout: float
    pulse_duration: float


@dataclass(frozen=True)
class FuturePredictionRewardConfig:
    enabled: bool
    performance_weight: float
    style_weight: float


@dataclass(frozen=True)
class RewardConfig:
    performance_weight: float
    style_weight: float
    style_sigma: float
    off_track_penalty: float
    stuck_penalty: float
    collision_penalty: float
    future_prediction: FuturePredictionRewardConfig


@dataclass(frozen=True)
class TerminationConfig:
    stuck_time: float
    minimum_progress: float


@dataclass(frozen=True)
class NetworkConfig:
    feature_layers: tuple[int, ...]
    feature_dim: int
    policy_layers: tuple[int, ...]
    value_layers: tuple[int, ...]
    activation: str
    ortho_init: bool
    log_std_init: float
    optimizer: str
    optimizer_epsilon: float
    optimizer_betas: tuple[float, ...]
    optimizer_weight_decay: float
    optimizer_amsgrad: bool


@dataclass(frozen=True)
class TrainingSettings:
    seed: int
    total_timesteps: int
    checkpoint_interval: int
    device: str
    verbose: int
    resume_from: str


@dataclass(frozen=True)
class TrajectoryConfig:
    enabled: bool


@dataclass(frozen=True)
class PPOSettings:
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    clip_range_vf: float
    normalize_advantage: bool
    n_steps: int
    batch_size: int
    n_epochs: int
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    use_sde: bool
    sde_sample_freq: int
    target_kl: float
    stats_window_size: int


@dataclass(frozen=True)
class NormalizationConfig:
    normalize_observations: bool
    normalize_reward: bool
    clip_observations: float
    clip_reward: float
    normalization_epsilon: float


Section = TypeVar("Section")


def _typed_value(value: object, expected: Any, name: str) -> object:
    if isinstance(expected, type) and hasattr(expected, "__dataclass_fields__"):
        return _section({name: value}, name, cast(type[Any], expected))
    if get_origin(expected) is tuple:
        if not isinstance(value, list):
            raise ValueError(f"{name} must be an array.")
        element_type = get_args(expected)[0]
        return tuple(_typed_value(item, element_type, name) for item in cast(list[object], value))
    if expected is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        return float(value)
    if type(value) is not expected:
        raise ValueError(f"{name} must have type {expected.__name__}.")
    return value


def _section(data: dict[str, Any], name: str, cls: type[Section]) -> Section:
    section = data[name]
    hints = get_type_hints(cls)
    if not isinstance(section, dict):
        raise ValueError(f"[{name}] must be a table.")
    section = cast(dict[str, Any], section)
    if set(section) != set(hints):
        raise ValueError(f"[{name}] requires exactly these settings: {', '.join(hints)}")
    values = {key: _typed_value(section[key], expected, f"{name}.{key}") for key, expected in hints.items()}
    return cls(**values)


@dataclass(frozen=True)
class TrainingConfig:
    environment: EnvironmentConfig
    simulator: SimulatorConfig
    reward: RewardConfig
    termination: TerminationConfig
    ppo: PPOSettings
    normalization: NormalizationConfig
    network: NetworkConfig
    predictor: PredictorConfig
    training: TrainingSettings
    trajectories: TrajectoryConfig
    toml_text: str

    @classmethod
    def load(cls, path: Path) -> TrainingConfig:
        return cls.from_toml(path.read_text(encoding="utf-8"))

    @classmethod
    def from_toml(cls, content: str) -> TrainingConfig:
        data = tomllib.loads(content)
        sections = get_type_hints(cls)
        sections.pop("toml_text")
        if set(data) != set(sections):
            raise ValueError(f"Config requires exactly these sections: {', '.join(sections)}")
        config = cls(**{name: _section(data, name, kind) for name, kind in sections.items()}, toml_text=content)
        config.validate()
        return config

    def agent_config(self) -> PPOConfig:
        return PPOConfig(
            **asdict(self.ppo), **asdict(self.normalization),
            seed=self.training.seed, device=self.training.device,
            verbose=self.training.verbose, tensorboard_log=None,
            delta_actions=self.environment.action_mode == "delta",
        )

    def validate(self) -> None:
        self.agent_config().validate()
        env, sim, net, train = self.environment, self.simulator, self.network, self.training
        for name, value in (
            ("control_interval", env.control_interval), ("max_episode_time", env.max_episode_time),
            ("reset_position_tolerance", env.reset_position_tolerance),
            ("reset_timeout", env.reset_timeout), ("reset_poll_interval", env.reset_poll_interval),
            ("minimum_speed", env.minimum_speed), ("startup_timeout", sim.startup_timeout),
            ("drive_ready_timeout", sim.drive_ready_timeout), ("poll_interval", sim.poll_interval),
            ("activation_timeout", sim.activation_timeout), ("pulse_duration", sim.pulse_duration),
            ("style_sigma", self.reward.style_sigma), ("stuck_time", self.termination.stuck_time),
            ("feature_dim", net.feature_dim), ("optimizer_epsilon", net.optimizer_epsilon),
            ("total_timesteps", train.total_timesteps), ("checkpoint_interval", train.checkpoint_interval),
            ("prediction_points", self.predictor.prediction_points),
            ("output_dimensions", self.predictor.output_dimensions),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        for name, value in (
            ("max_episode_steps", env.max_episode_steps), ("front_axle_distance", env.front_axle_distance),
            ("rear_axle_distance", env.rear_axle_distance), ("seed", train.seed),
            ("reset_speed_tolerance_kmh", env.reset_speed_tolerance_kmh),
            ("minimum_progress", self.termination.minimum_progress),
            ("performance_weight", self.reward.performance_weight), ("style_weight", self.reward.style_weight),
            ("loss_weight", self.predictor.loss_weight),
            ("optimizer_weight_decay", net.optimizer_weight_decay),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if not 0 < env.reset_heading_tolerance_degrees <= 180:
            raise ValueError("reset_heading_tolerance_degrees must be in (0, 180].")
        if sim.car_index != 0:
            raise ValueError("The single-player environment supports car_index = 0 only.")
        if train.verbose not in (0, 1, 2) or not train.device:
            raise ValueError("Use verbose 0, 1 or 2 and a non-empty device.")
        if train.seed >= 2**32:
            raise ValueError("seed must be below 2**32 for SB3/NumPy.")
        if not sim.assetto_corsa_directory:
            raise ValueError("assetto_corsa_directory must not be empty.")
        for name in ("car", "track", "track_layout", "skin"):
            value = getattr(env, name)
            if (name in ("car", "track") or value) and not re.fullmatch(r"[A-Za-z0-9_-]+", value):
                raise ValueError(f"Invalid environment.{name} identifier.")
        if env.abs_mode not in (0, 1, 2) or env.traction_control_mode not in (0, 1, 2):
            raise ValueError("Assist modes must be 0 (off), 1 (factory) or 2 (on).")
        if env.start_sampler != "uniform_track_point":
            raise ValueError("Only the uniform_track_point start sampler is implemented.")
        if env.action_mode not in ("delta", "absolute"):
            raise ValueError("action_mode must be delta or absolute.")
        for name in ("off_track_penalty", "stuck_penalty", "collision_penalty"):
            if getattr(self.reward, name) > 0:
                raise ValueError(f"{name} must be non-positive.")
        for name in ("lookahead_distances", "reference_distances"):
            distances = getattr(env, name)
            if not distances or any(value <= 0 for value in distances) or tuple(sorted(set(distances))) != distances:
                raise ValueError(f"{name} must be positive and strictly increasing.")
        for layers in (net.feature_layers, net.policy_layers, net.value_layers, self.predictor.layers):
            if any(size <= 0 for size in layers):
                raise ValueError("Network layer sizes must be positive integers.")
        if net.activation not in ("tanh", "relu") or net.optimizer != "adam":
            raise ValueError("Supported network settings: activation tanh/relu, optimizer adam.")
        if len(net.optimizer_betas) != 2 or any(not 0 <= value < 1 for value in net.optimizer_betas):
            raise ValueError("optimizer_betas must contain two values in [0, 1).")
        if self.predictor.output_dimensions != 2:
            raise ValueError("Local trajectory points require output_dimensions = 2 (forward, left).")
        if len(self.predictor.horizons_seconds) != self.predictor.prediction_points:
            raise ValueError("prediction_points must match the number of horizons_seconds.")
        offsets = horizon_indices(self.predictor.horizons_seconds, env.control_interval)
        if self.predictor.enabled and min(offsets) > self.ppo.n_steps:
            raise ValueError("The PPO rollout must cover at least the first prediction horizon.")
        future = self.reward.future_prediction
        if future.performance_weight < 0 or future.style_weight < 0:
            raise ValueError("Future prediction reward weights must be non-negative.")
        if future.enabled:
            raise ValueError("Future prediction reward is not validated yet; leave it disabled.")
        if self.trajectories.enabled:
            raise ValueError("Trajectory export is not implemented; set trajectories.enabled = false.")

    def require_compatible_resume(self, previous: TrainingConfig) -> None:
        """Only the new budget, checkpoint cadence and runtime verbosity may change."""
        for name in ("environment", "simulator", "reward", "termination", "ppo", "normalization", "network", "predictor", "trajectories"):
            if getattr(self, name) != getattr(previous, name):
                raise ValueError(f"Resuming with different [{name}] settings is unsupported.")
        if self.training.seed != previous.training.seed or self.training.device != previous.training.device:
            raise ValueError("Resuming requires the saved training seed and device.")


def checkpoint_config_path(directory: Path) -> Path:
    """Resolve only the defined run layout; never guess a config from unrelated ancestors."""
    if directory.name == "final":
        return directory.parent / "config.toml"
    if directory.parent.name == "checkpoints" and re.fullmatch(r"step_\d{6,}", directory.name):
        return directory.parent.parent / "config.toml"
    raise ValueError("Expected a run's final/ or checkpoints/step_XXXXXX directory.")
