"""One shared feature tensor for actor, critic and auxiliary trajectory head."""

from typing import Any

from gymnasium.spaces import Box
import numpy as np
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution, Distribution, StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.vec_env import VecNormalize
import torch
from torch import nn


class ActionAwareVecNormalize(VecNormalize):
    """Normalize physical features; controls already have their [-1, 1] scale.

    Keeping the last two features in control units lets the residual actor add
    its delta without recovering clipped raw values from running statistics.
    Running statistics and serialization remain owned by VecNormalize.
    """

    def normalize_obs(self, obs: np.ndarray | dict[str, np.ndarray]) -> np.ndarray | dict[str, np.ndarray]:
        normalized = super().normalize_obs(obs)
        if not isinstance(obs, np.ndarray) or not isinstance(normalized, np.ndarray):
            raise ValueError("Racing controls require a flat Box observation.")
        normalized[..., -2:] = obs[..., -2:]
        return normalized

    def unnormalize_obs(self, obs: np.ndarray | dict[str, np.ndarray]) -> np.ndarray | dict[str, np.ndarray]:
        original = super().unnormalize_obs(obs)
        if not isinstance(obs, np.ndarray) or not isinstance(original, np.ndarray):
            raise ValueError("Racing controls require a flat Box observation.")
        original[..., -2:] = obs[..., -2:]
        return original


class RacingActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self, observation_space: Box[np.float32], action_space: Box[np.float32],
        lr_schedule: Schedule, *, predictor_layers: tuple[int, ...] = (),
        prediction_points: int = 0, output_dimensions: int = 2,
        delta_actions: bool = False, initial_longitudinal_mean: float = 0.0, **kwargs: Any,
    ) -> None:
        if kwargs.get("share_features_extractor", True) is not True:
            raise ValueError("Racing policy requires one shared feature extractor.")
        if action_space.shape != (2,) or len(observation_space.shape) != 1 or observation_space.shape[0] < 2:
            raise ValueError("Racing policy needs two controls and a flat observation ending in controls.")
        if prediction_points < 0 or output_dimensions != 2 or any(size <= 0 for size in predictor_layers):
            raise ValueError("Invalid local trajectory head dimensions.")
        self.prediction_points = prediction_points
        self.output_dimensions = output_dimensions
        self.predictor_layers = predictor_layers
        self.delta_actions = delta_actions
        if not np.isfinite(initial_longitudinal_mean) or not -1.0 <= initial_longitudinal_mean <= 1.0:
            raise ValueError("initial_longitudinal_mean must be finite and in [-1, 1].")
        self.initial_longitudinal_mean = initial_longitudinal_mean
        self.capture_reward_prediction = False
        self._reward_prediction: torch.Tensor | None = None
        # SB3's constructor uses unparameterized Gymnasium Space annotations.
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)  # pyright: ignore[reportUnknownMemberType]

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        if not isinstance(self.action_net, nn.Linear):
            raise ValueError("Racing policy requires a linear Gaussian action head.")
        # A learnable initial preference, applied once. PPO still samples and
        # evaluates the same distribution; loading restores the trained bias.
        with torch.no_grad():
            self.action_net.bias[1] = self.initial_longitudinal_mean
        modules: list[nn.Module] = []
        size = self.features_dim
        if self.prediction_points:
            for hidden in self.predictor_layers:
                modules.extend((nn.Linear(size, hidden), self.activation_fn()))
                size = hidden
            modules.append(nn.Linear(size, self.prediction_points * self.output_dimensions))
        self.predictor_head = nn.Sequential(*modules)
        # ActorCriticPolicy already owns one optimizer. Extend that instance
        # with the new head; the shared feature parameters are never duplicated.
        if self.prediction_points:
            self.optimizer.add_param_group({"params": self.predictor_head.parameters()})

    def _shared_features(self, obs: PyTorchObs) -> torch.Tensor:
        features = self.extract_features(obs)
        if not isinstance(features, torch.Tensor):
            raise RuntimeError("Expected one shared feature tensor.")
        return features

    def _distribution(self, latent: torch.Tensor, obs: PyTorchObs) -> Distribution:
        if not self.delta_actions:
            return self._get_action_dist_from_latent(latent)
        if not isinstance(obs, torch.Tensor):
            raise ValueError("Delta actions need a tensor observation.")
        # Shifting a Gaussian by the observed controls preserves likelihoods.
        # SB3 stores the sampled, unbounded FINAL action and clips only the
        # command sent to the environment, including during evaluation.
        mean_actions = obs[..., -2:] + self.action_net(latent)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        if isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent)
        raise ValueError("Racing policy requires a continuous Gaussian distribution.")

    def _trajectory(self, features: torch.Tensor) -> torch.Tensor:
        if not self.prediction_points:
            raise RuntimeError("Trajectory predictor is disabled.")
        return self.predictor_head(features).reshape(-1, self.prediction_points, self.output_dimensions)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._shared_features(obs)
        self._capture_prediction(features)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._distribution(latent_pi, obs)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.value_net(latent_vf), distribution.log_prob(actions)

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        features = self._shared_features(obs)
        self._capture_prediction(features)
        return self._distribution(self.mlp_extractor.forward_actor(features), obs)

    def _capture_prediction(self, features: torch.Tensor) -> None:
        if self.capture_reward_prediction:
            with torch.no_grad():
                self._reward_prediction = self._trajectory(features).detach()

    def configure_reward_capture(self, enabled: bool) -> None:
        self.capture_reward_prediction = enabled
        self._reward_prediction = None

    def take_reward_prediction(self) -> torch.Tensor:
        """Consume one rollout snapshot; never reuse a previous transition."""
        prediction, self._reward_prediction = self._reward_prediction, None
        if prediction is None:
            raise RuntimeError("The current action has no detached trajectory prediction.")
        return prediction

    def evaluate_with_prediction(
        self, obs: torch.Tensor, actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        features = self._shared_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._distribution(latent_pi, obs)
        return (
            self.value_net(latent_vf), distribution.log_prob(actions),
            distribution.entropy(), self._trajectory(features),
        )

    def evaluate_actions(self, obs: PyTorchObs, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        features = self._shared_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._distribution(latent_pi, obs)
        return self.value_net(latent_vf), distribution.log_prob(actions), distribution.entropy()

    def prediction_for_reward(self, observation: torch.Tensor) -> torch.Tensor:
        """Local metre coordinates, detached for validation or external reward use."""
        with torch.no_grad():
            return self._trajectory(self._shared_features(observation)).detach()

    def _get_constructor_parameters(self) -> dict[str, Any]:
        return {
            **super()._get_constructor_parameters(),
            "predictor_layers": self.predictor_layers, "prediction_points": self.prediction_points,
            "output_dimensions": self.output_dimensions, "delta_actions": self.delta_actions,
            "initial_longitudinal_mean": self.initial_longitudinal_mean,
        }
