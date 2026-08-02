"""Shared-backbone policy, value and auxiliary trajectory network."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from assetto_corsa_rl.features.schema import OBSERVATION_SIZE


def _mlp(
    input_size: int,
    hidden_sizes: tuple[int, ...],
    output_size: int | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_size
    for size in hidden_sizes:
        layers.extend((nn.Linear(previous, size), nn.Tanh()))
        previous = size
    if output_size is not None:
        layers.append(nn.Linear(previous, output_size))
    return nn.Sequential(*layers)


@dataclass(frozen=True, slots=True)
class NetworkOutput:
    """All head outputs from one shared feature pass."""

    action_location: Tensor
    action_mean: Tensor
    action_log_std: Tensor
    value: Tensor
    trajectory_parameters: Tensor


class SquashedNormal:
    """Tanh-squashed diagonal Gaussian with bounded samples and exact log density."""

    _epsilon = 1e-6
    _log_two_pi = math.log(2.0 * math.pi)

    def __init__(self, location: Tensor, log_standard_deviation: Tensor) -> None:
        if location.shape != log_standard_deviation.shape:
            raise ValueError("location and log standard deviation shapes must match")
        self.location = location
        self.log_standard_deviation = log_standard_deviation

    @property
    def batch_shape(self) -> torch.Size:
        """Return the non-action dimensions of the distribution."""

        return self.location.shape[:-1]

    def _raw_sample(self) -> Tensor:
        noise = torch.randn_like(self.location)
        return self.location + self.log_standard_deviation.exp() * noise

    def _bounded(self, raw_action: Tensor) -> Tensor:
        return torch.tanh(raw_action).clamp(-1.0 + self._epsilon, 1.0 - self._epsilon)

    def sample(self) -> Tensor:
        """Draw a bounded, non-reparameterised action."""

        return self._bounded(self._raw_sample()).detach()

    def rsample(self) -> Tensor:
        """Draw a bounded reparameterised action."""

        return self._bounded(self._raw_sample())

    def log_prob(self, action: Tensor) -> Tensor:
        """Evaluate the transformed density including the tanh Jacobian."""

        bounded = action.clamp(-1.0 + self._epsilon, 1.0 - self._epsilon)
        raw_action = torch.atanh(bounded)
        standardised = (raw_action - self.location) / self.log_standard_deviation.exp()
        base_log_probability = -0.5 * (
            standardised.square() + 2.0 * self.log_standard_deviation + self._log_two_pi
        ).sum(dim=-1)
        log_jacobian = torch.log1p(-bounded.square() + self._epsilon).sum(dim=-1)
        return base_log_probability - log_jacobian

    def entropy(self) -> Tensor:
        """Return a one-sample Monte Carlo estimate of squashed entropy."""

        sample = self.rsample()
        return -self.log_prob(sample)


class DriverNetwork(nn.Module):
    """Feedforward baseline from Section 2.2.4 of the thesis.

    Default dimensions:

    - feature extractor: ``[512, 512]``
    - Gaussian policy head: ``[128, 128, 2]``
    - value head: ``[512, 512, 1]``
    - trajectory predictor: ``[128, 128, 10]``
    """

    def __init__(
        self,
        observation_size: int = OBSERVATION_SIZE,
        action_size: int = 2,
        *,
        feature_hidden_sizes: tuple[int, ...] = (512, 512),
        policy_hidden_sizes: tuple[int, ...] = (128, 128),
        value_hidden_sizes: tuple[int, ...] = (512, 512),
        predictor_hidden_sizes: tuple[int, ...] = (128, 128),
        trajectory_output_size: int = 10,
        initial_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        if observation_size <= 0 or action_size <= 0 or trajectory_output_size <= 0:
            raise ValueError("network input and output sizes must be positive")
        if not feature_hidden_sizes:
            raise ValueError("feature_hidden_sizes may not be empty")

        self.observation_size = observation_size
        self.action_size = action_size
        self.trajectory_output_size = trajectory_output_size
        self.feature_extractor = _mlp(
            observation_size,
            feature_hidden_sizes,
        )
        latent_size = feature_hidden_sizes[-1]
        self.policy_head = _mlp(
            latent_size,
            policy_hidden_sizes,
            output_size=action_size,
        )
        self.value_head = _mlp(
            latent_size,
            value_hidden_sizes,
            output_size=1,
        )
        self.trajectory_predictor = _mlp(
            latent_size,
            predictor_hidden_sizes,
            output_size=trajectory_output_size,
        )
        self.action_log_std = nn.Parameter(torch.full((action_size,), initial_log_std))
        self.apply(self._initialise_linear)

    @staticmethod
    def _initialise_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(module.bias)

    def forward(self, observation: Tensor) -> NetworkOutput:
        """Evaluate every head from the same latent representation."""

        if observation.shape[-1] != self.observation_size:
            raise ValueError(
                f"last observation dimension must be {self.observation_size}, "
                f"got {observation.shape[-1]}"
            )
        latent = self.feature_extractor(observation)
        action_location = self.policy_head(latent)
        action_mean = torch.tanh(action_location)
        value = self.value_head(latent).squeeze(-1)
        trajectory_parameters = self.trajectory_predictor(latent)
        log_std = self.action_log_std.expand_as(action_mean)
        return NetworkOutput(
            action_location=action_location,
            action_mean=action_mean,
            action_log_std=log_std,
            value=value,
            trajectory_parameters=trajectory_parameters,
        )

    def action_distribution(self, observation: Tensor) -> SquashedNormal:
        """Return the bounded state-dependent distribution used by PPO."""

        output = self.forward(observation)
        return SquashedNormal(output.action_location, output.action_log_std)

    @torch.no_grad()
    def act(self, observation: Tensor, *, deterministic: bool) -> Tensor:
        """Choose mean actions for evaluation or samples for collection."""

        output = self.forward(observation)
        if deterministic:
            return output.action_mean
        return SquashedNormal(output.action_location, output.action_log_std).sample()
