"""PPO agent for the Assetto Corsa racing environment.

The environment exposes raw observations in physical/SI units. This module
keeps that representation unchanged and applies Stable-Baselines3
``VecNormalize`` immediately before PPO sees the observations.

The training entry point supplies normalization and PPO settings from the
run's TOML. The default experiment keeps rewards on their original scale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from math import isfinite
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize
import torch
from torch import nn

from assetto_corsa_ai.RacingEnvironment import RacingEnvironment
from assetto_corsa_ai.predictor import PredictorConfig
from assetto_corsa_ai.future_reward import FuturePredictionRewardConfig
from assetto_corsa_ai.reward_wrapper import FutureRewardWrapper
from assetto_corsa_ai.racing_policy import ActionAwareVecNormalize, RacingActorCriticPolicy
from assetto_corsa_ai.racing_ppo import RacingPPO


FloatArray = NDArray[np.float32]


class SharedFeatureExtractor(BaseFeaturesExtractor):
    """Shared dense features; sizes and activation come from the run config."""

    def __init__(
        self, observation_space: Box[np.float32], *, layers: tuple[int, ...],
        features_dim: int, activation: type[nn.Module],
    ) -> None:
        # SB3 annotates its base constructor with an unparameterized Space.
        super().__init__(observation_space, features_dim)  # pyright: ignore[reportUnknownMemberType]
        sizes = (int(np.prod(observation_space.shape)), *layers, features_dim)
        modules: list[nn.Module] = [nn.Flatten()]
        for source, target in zip(sizes, sizes[1:]):
            modules.extend((nn.Linear(source, target), activation()))
        self.network = nn.Sequential(*modules)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


def _close_after_failure(vec_env: VecEnv, error: BaseException) -> None:
    """Release the simulator while preserving the original exception."""
    try:
        vec_env.close()
    except Exception as cleanup_error:
        error.add_note(f"Environment cleanup also failed: {cleanup_error}")


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters and preprocessing settings for PPO.

    These values are a sensible starting point, not final tuned parameters.
    With one real-time simulator environment, ``n_steps`` also determines how
    much simulator time is collected before each PPO update.
    """

    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_range_vf: float = 0.0
    normalize_advantage: bool = True
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: float = 0.0
    stats_window_size: int = 100

    # Observation normalization. Raw environment observations remain in SI
    # units; only PPO receives the normalized representation.
    normalize_observations: bool = True
    clip_observations: float = 10.0
    normalization_epsilon: float = 1e-8

    # Keep the custom reward on its original scale for now.
    normalize_reward: bool = False
    clip_reward: float = 10.0

    seed: int | None = None
    verbose: int = 1
    device: str = "auto"
    tensorboard_log: str | None = None
    delta_actions: bool = False

    def validate(self) -> None:
        """Validate configuration values that would otherwise fail later."""

        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if type(self.n_steps) is not int or self.n_steps <= 1:
            raise ValueError("n_steps must be an integer greater than 1.")
        if type(self.batch_size) is not int or self.batch_size <= 1:
            raise ValueError("batch_size must be an integer greater than 1.")
        if self.n_steps % self.batch_size != 0:
            raise ValueError(
                "For a single racing environment, n_steps must be divisible "
                "by batch_size so each rollout splits into complete minibatches."
            )
        if type(self.n_epochs) is not int or self.n_epochs <= 0:
            raise ValueError("n_epochs must be a positive integer.")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1].")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1].")
        if not isfinite(self.clip_range) or self.clip_range <= 0.0:
            raise ValueError("clip_range must be positive.")
        if not isfinite(self.ent_coef) or self.ent_coef < 0.0:
            raise ValueError("ent_coef must be non-negative.")
        if not isfinite(self.vf_coef) or self.vf_coef < 0.0:
            raise ValueError("vf_coef must be non-negative.")
        if not isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive.")
        if not isfinite(self.clip_observations) or self.clip_observations <= 0.0:
            raise ValueError("clip_observations must be positive.")
        if not isfinite(self.normalization_epsilon) or self.normalization_epsilon <= 0.0:
            raise ValueError("normalization_epsilon must be positive.")
        for name in ("clip_range_vf", "target_kl"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not isfinite(self.clip_reward) or self.clip_reward <= 0:
            raise ValueError("clip_reward must be finite and positive.")
        if type(self.stats_window_size) is not int or self.stats_window_size <= 0:
            raise ValueError("stats_window_size must be a positive integer.")
        if type(self.sde_sample_freq) is not int or self.sde_sample_freq == 0 or self.sde_sample_freq < -1:
            raise ValueError("sde_sample_freq must be -1 or a positive integer.")


class PPOAgent:
    """Thin project-specific wrapper around Stable-Baselines3 PPO.

    Responsibilities:

    - wrap ``RacingEnvironment`` as the vectorized environment expected by SB3,
    - normalize observations using running mean and variance,
    - configure PPO,
    - expose training and inference helpers,
    - save/load the PPO model together with normalization statistics.

    The class does not know the semantic meaning or ordering of individual
    observation features. ``Observation.to_array()`` and ``RacingEnvironment``
    define that interface.
    """

    MODEL_FILENAME = "ppo_model"
    NORMALIZER_FILENAME = "vec_normalize.pkl"
    CONFIG_FILENAME = "ppo_config.json"

    def __init__(
        self,
        environment: RacingEnvironment,
        *,
        config: PPOConfig | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        predictor_config: PredictorConfig | None = None,
        future_reward_config: FuturePredictionRewardConfig | None = None,
        control_interval: float | None = None,
    ) -> None:
        self._config = config if config is not None else PPOConfig()
        self._config.validate()
        if future_reward_config is not None:
            future_reward_config.validate()
            if future_reward_config.enabled and (predictor_config is None or not predictor_config.enabled):
                raise ValueError("Future prediction reward requires an enabled predictor.")

        self._environment = environment
        self._vec_env = self._build_vec_env(environment)
        self._closed = False

        try:
            auxiliary = predictor_config is not None and predictor_config.enabled
            custom_policy = auxiliary or self._config.delta_actions
            model_type = RacingPPO if custom_policy else PPO
            options = dict(policy_kwargs or {})
            extra: dict[str, Any] = {}
            if custom_policy:
                options.update(
                    delta_actions=self._config.delta_actions,
                    predictor_layers=predictor_config.layers if auxiliary and predictor_config else (),
                    prediction_points=predictor_config.prediction_points if auxiliary and predictor_config else 0,
                    output_dimensions=predictor_config.output_dimensions if predictor_config else 2,
                )
                extra.update(predictor_config=predictor_config, control_interval=control_interval,
                             future_reward_config=future_reward_config)
            self._model = model_type(
                policy=RacingActorCriticPolicy if custom_policy else "MlpPolicy",
                env=self._vec_env,
                learning_rate=self._config.learning_rate,
                n_steps=self._config.n_steps,
                batch_size=self._config.batch_size,
                n_epochs=self._config.n_epochs,
                gamma=self._config.gamma,
                gae_lambda=self._config.gae_lambda,
                clip_range=self._config.clip_range,
                ent_coef=self._config.ent_coef,
                vf_coef=self._config.vf_coef,
                max_grad_norm=self._config.max_grad_norm,
                clip_range_vf=self._config.clip_range_vf or None,
                normalize_advantage=self._config.normalize_advantage,
                use_sde=self._config.use_sde,
                sde_sample_freq=self._config.sde_sample_freq,
                target_kl=self._config.target_kl or None,
                stats_window_size=self._config.stats_window_size,
                seed=self._config.seed,
                verbose=self._config.verbose,
                device=self._config.device,
                tensorboard_log=self._config.tensorboard_log,
                policy_kwargs=options,
                **extra,
            )
            self._acting_environment = self._install_future_reward(self._vec_env, self._model, environment)
        except BaseException as error:
            _close_after_failure(self._vec_env, error)
            self._closed = True
            raise

    @property
    def model(self) -> PPO:
        """Return the underlying Stable-Baselines3 PPO model."""

        return self._model

    @property
    def vec_env(self) -> VecNormalize:
        """Return the normalized vector environment used by PPO."""

        return self._vec_env

    @property
    def config(self) -> PPOConfig:
        return self._config

    @property
    def environment(self) -> gym.Env[FloatArray, FloatArray]:
        """Raw observations with the same optional reward wrapper as training."""
        return self._acting_environment

    def learn(
        self,
        total_timesteps: int,
        *,
        callback: BaseCallback | None = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "PPOAgent":
        """Train PPO using live rollouts from Assetto Corsa."""

        self._ensure_open()
        if type(total_timesteps) is not int or total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer.")

        # Running observation statistics are updated only during training.
        self._vec_env.training = True
        self._vec_env.norm_reward = self._config.normalize_reward

        try:
            # SB3's callback alias contains an unparameterized Callable.
            self._model.learn(  # pyright: ignore[reportUnknownMemberType]
                total_timesteps=total_timesteps,
                callback=callback,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=progress_bar,
            )
        except BaseException as error:
            _close_after_failure(self._vec_env, error)
            self._closed = True
            raise

        return self

    def predict(
        self,
        observation: FloatArray,
        *,
        deterministic: bool = True,
    ) -> FloatArray:
        """Predict an action from one raw, unnormalized environment observation.

        ``observation`` must be the same raw float32 vector returned by
        ``RacingEnvironment``. The saved/running VecNormalize statistics are
        applied here before the policy is evaluated.
        """

        self._ensure_open()
        raw_observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        expected_shape = self._environment.observation_space.shape
        if raw_observation.shape != expected_shape:
            raise ValueError(
                f"Expected observation shape {expected_shape}, "
                f"got {raw_observation.shape}."
            )

        if not np.isfinite(raw_observation).all():
            raise ValueError("Observation contains non-finite values.")

        normalized_observation = self._vec_env.normalize_obs(
            raw_observation.copy()
        )
        assert isinstance(normalized_observation, np.ndarray)

        action, _ = self._model.predict(
            normalized_observation,
            deterministic=deterministic,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        )
        if action.shape != self._environment.action_space.shape or not np.isfinite(action).all():
            raise ValueError("Predicted action must have the action-space shape and finite values.")

        # PPO's squashed/clipped output should already respect the Box space.
        # Keep this final clip as a numerical safety boundary for the simulator.
        # RacingEnvironment defines a float32 Box for its controls.
        action_space = cast(Box[np.float32], self._environment.action_space)
        action = np.clip(
            action,
            action_space.low,
            action_space.high,
        ).astype(np.float32, copy=False)

        return action

    def predict_trajectory(self, observation: FloatArray) -> FloatArray:
        """Detached (horizon, forward/left) metre coordinates for validation/reward."""
        self._ensure_open()
        if not isinstance(self._model.policy, RacingActorCriticPolicy):
            raise RuntimeError("Trajectory predictor is disabled.")
        raw = np.asarray(observation, dtype=np.float32)
        if raw.shape != self._environment.observation_space.shape or not np.isfinite(raw).all():
            raise ValueError("Observation must have the declared shape and finite values.")
        normalized = self._vec_env.normalize_obs(raw.copy())
        assert isinstance(normalized, np.ndarray)
        tensor = torch.as_tensor(normalized, device=self._model.device).unsqueeze(0)
        prediction = self._model.policy.prediction_for_reward(tensor)
        points = prediction[0].cpu().numpy().astype(np.float32)
        if not np.isfinite(points).all():
            raise ValueError("Predicted trajectory contains non-finite values.")
        return points

    def save(
        self,
        directory: str | Path,
    ) -> None:
        """Save a legacy standalone bundle; training runs use TrainingRun instead."""

        self._ensure_open()
        output_directory = Path(directory)
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._model.save(
            output_directory / self.MODEL_FILENAME
        )
        self._vec_env.save(
            str(output_directory / self.NORMALIZER_FILENAME)
        )
        (output_directory / self.CONFIG_FILENAME).write_text(
            json.dumps(asdict(self._config), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        environment: RacingEnvironment,
        *,
        training: bool = False,
        device: str = "auto",
    ) -> "PPOAgent":
        """Load PPO and its matching observation-normalization statistics.

        Evaluation should use the default ``training=False`` so running mean and
        variance remain frozen. To resume training, pass ``training=True``.
        A checkpoint contains the model, normalizer and ``ppo_config.json``;
        ``device`` overrides only the saved device setting.
        """

        input_directory = Path(directory)
        model_path = input_directory / f"{cls.MODEL_FILENAME}.zip"
        normalizer_path = input_directory / cls.NORMALIZER_FILENAME
        config_path = input_directory / cls.CONFIG_FILENAME

        if not model_path.is_file():
            raise FileNotFoundError(
                f"PPO model not found: {model_path}"
            )

        if not normalizer_path.is_file():
            raise FileNotFoundError(
                f"VecNormalize statistics not found: {normalizer_path}"
            )

        if not config_path.is_file():
            raise FileNotFoundError(f"PPO configuration not found: {config_path}")
        config_data: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        config = replace(PPOConfig(**config_data), device=device)
        config.validate()

        return cls._load_pair(model_path, normalizer_path, environment, config, training)

    @classmethod
    def load_snapshot(
        cls, directory: Path, environment: RacingEnvironment, *,
        config: PPOConfig, training: bool = False,
    ) -> "PPOAgent":
        """Restore a run snapshot using its validated TOML-derived settings."""
        config.validate()
        model_path = directory / "model.zip"
        normalizer_path = directory / "vecnormalize.pkl"
        for path in (model_path, normalizer_path):
            if not path.is_file():
                raise FileNotFoundError(f"Incomplete training snapshot: {path}")
        return cls._load_pair(model_path, normalizer_path, environment, config, training, model_type=RacingPPO)

    @classmethod
    def _load_pair(
        cls, model_path: Path, normalizer_path: Path, environment: RacingEnvironment,
        config: PPOConfig, training: bool,
        *, model_type: type[PPO] = RacingPPO,
    ) -> "PPOAgent":

        base_env = cls._make_base_vec_env(
            environment
        )

        try:
            vec_env = VecNormalize.load(str(normalizer_path), base_env)
            vec_env.training = training
            vec_env.norm_reward = config.normalize_reward if training else False

            # SB3's load signature leaves **kwargs untyped.
            model = model_type.load(  # pyright: ignore[reportUnknownMemberType]
                model_path, env=vec_env, device=config.device,
            )
            acting_environment = cls._install_future_reward(vec_env, model, environment)
        except BaseException as error:
            _close_after_failure(base_env, error)
            raise

        agent = cls.__new__(cls)
        agent._config = config
        agent._environment = environment
        agent._vec_env = vec_env
        agent._model = model
        agent._acting_environment = acting_environment
        agent._closed = False

        return agent

    @staticmethod
    def _install_future_reward(
        vec_env: VecNormalize, model: PPO, environment: RacingEnvironment,
    ) -> gym.Env[FloatArray, FloatArray]:
        settings = getattr(model, "future_reward_config", None)
        if not isinstance(settings, FuturePredictionRewardConfig) or not settings.enabled:
            return environment
        policy = model.policy
        if not isinstance(policy, RacingActorCriticPolicy) or not policy.prediction_points:
            raise ValueError("Future reward requires the saved shared trajectory policy.")
        base = vec_env.unwrapped
        if not isinstance(base, DummyVecEnv) or base.num_envs != 1:
            raise ValueError("Future reward currently supports one live racing environment.")
        monitor = base.envs[0]
        if not isinstance(monitor, Monitor):
            raise ValueError("Future reward must be installed before Monitor.")
        policy.configure_reward_capture(True)

        def snapshot() -> tuple[FloatArray, int]:
            prediction = policy.take_reward_prediction()
            step = getattr(model, "num_timesteps")
            assert isinstance(step, int)
            return prediction[0].cpu().numpy().astype(np.float32), step

        wrapped = FutureRewardWrapper(environment, settings, snapshot)
        monitor.env = wrapped
        return wrapped

    def set_evaluation_mode(self) -> None:
        """Freeze normalization statistics for deterministic evaluation."""

        self._ensure_open()
        self._vec_env.training = False
        self._vec_env.norm_reward = False

    def set_training_mode(self) -> None:
        """Allow observation statistics to continue updating during training."""

        self._ensure_open()
        self._vec_env.training = True
        self._vec_env.norm_reward = self._config.normalize_reward

    def close(self) -> None:
        """Close the vector environment and therefore the racing environment."""

        if not self._closed:
            self._closed = True
            self._vec_env.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PPOAgent is closed. Create or load a new agent.")

    def _build_vec_env(
        self,
        environment: RacingEnvironment,
    ) -> VecNormalize:
        base_env = self._make_base_vec_env(
            environment
        )

        try:
            normalizer_type = ActionAwareVecNormalize if self._config.delta_actions else VecNormalize
            return normalizer_type(
                base_env,
                training=True,
                norm_obs=self._config.normalize_observations,
                norm_reward=self._config.normalize_reward,
                clip_obs=self._config.clip_observations,
                clip_reward=self._config.clip_reward,
                gamma=self._config.gamma,
                epsilon=self._config.normalization_epsilon,
            )
        except BaseException as error:
            _close_after_failure(base_env, error)
            raise

    @staticmethod
    def _make_base_vec_env(
        environment: RacingEnvironment,
    ) -> DummyVecEnv:
        """Wrap the single live simulator instance for Stable-Baselines3."""

        return DummyVecEnv(
            [
                lambda: Monitor(
                    environment
                )
            ]
        )
