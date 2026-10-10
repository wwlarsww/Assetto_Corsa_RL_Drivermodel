"""Inspect a saved trajectory predictor on test data or a selected NPZ rollout.

The default prints predictions from a fresh, untrained TOML-driven model using
synthetic data. --checkpoint loads a run snapshot and its VecNormalize instead.
No AC process, live driving, reward changes or persistent rollout capture.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch

from assetto_corsa_ai.agent import PPOAgent
from assetto_corsa_ai.main import create_agent
from assetto_corsa_ai.predictor import build_trajectory_targets, horizon_indices, masked_predictor_loss
from assetto_corsa_ai.RacingEnvironment import FloatArray, RacingEnvironment
from assetto_corsa_ai.training_config import TrainingConfig, checkpoint_config_path


@dataclass(frozen=True)
class RolloutData:
    observations: NDArray[np.float32]
    positions: NDArray[np.float64]
    forwards: NDArray[np.float64]
    next_positions: NDArray[np.float64]
    episode_ends: NDArray[np.bool_]

    def validate(self, observation_size: int) -> None:
        shape = self.positions.shape[:2]
        if len(shape) != 2 or not shape[0] or not shape[1]:
            raise ValueError("Rollout must contain steps and environments.")
        if self.positions.shape != (*shape, 3) or self.forwards.shape != self.positions.shape or self.next_positions.shape != self.positions.shape:
            raise ValueError("Pose arrays must have shape (steps, environments, 3).")
        if self.observations.shape != (*shape, observation_size) or not np.isfinite(self.observations).all():
            raise ValueError("Raw observations must be finite and match the saved observation size.")
        if self.episode_ends.shape != shape or self.episode_ends.dtype != np.bool_:
            raise ValueError("episode_ends must be a boolean (steps, environments) array.")


def load_rollout(path: Path) -> RolloutData:
    with np.load(path, allow_pickle=False) as data:
        required = {"observations", "vehicle_positions", "vehicle_forwards", "next_vehicle_positions", "episode_ends"}
        if not required.issubset(data.files):
            raise ValueError(f"Selected rollout is missing arrays: {sorted(required.difference(data.files))}")
        ends: NDArray[Any] = np.asarray(data["episode_ends"])
        if ends.dtype != np.bool_:
            raise ValueError("episode_ends must contain boolean flags.")
        return RolloutData(
            np.asarray(data["observations"], dtype=np.float32),
            np.asarray(data["vehicle_positions"], dtype=np.float64),
            np.asarray(data["vehicle_forwards"], dtype=np.float64),
            np.asarray(data["next_vehicle_positions"], dtype=np.float64),
            cast(NDArray[np.bool_], ends),
        )


def synthetic_rollout(config: TrainingConfig, steps: int = 64) -> RolloutData:
    """A smooth test curve, with an episode boundary halfway through."""
    if steps < 2:
        raise ValueError("Synthetic data needs at least two transitions.")
    interval = config.environment.control_interval
    times = np.arange(steps + 1, dtype=np.float64) * interval
    world = np.stack((8 * times, np.zeros_like(times), 0.4 * times**2), axis=-1)
    forwards = np.stack((np.full(steps, 8.0), np.zeros(steps), 0.8 * times[:-1]), axis=-1)
    positions, endpoints = world[:-1].copy(), world[1:].copy()
    midpoint = steps // 2
    # Make an auto-reset leak unmistakable in a plot or transformation test.
    positions[midpoint:] += (1000.0, 0.0, -500.0)
    endpoints[midpoint:] += (1000.0, 0.0, -500.0)
    dimensions = 10 + 4 * len(config.environment.lookahead_distances) + 2 * len(config.environment.reference_distances)
    observations = np.zeros((steps, 1, dimensions), dtype=np.float32)
    observations[:, 0, 0] = 8.0
    episode_ends = np.zeros((steps, 1), dtype=np.bool_)
    episode_ends[midpoint - 1] = True
    episode_ends[-1] = True
    return RolloutData(observations, positions[:, None], forwards[:, None], endpoints[:, None], episode_ends)


class ValidationEnvironment(RacingEnvironment):
    """A test-data Gymnasium adapter, independent of AC, for policy loading."""

    def __init__(self, data: RolloutData) -> None:
        gym.Env.__init__(self)
        self.data = data
        self.index = 0
        self.controls = np.zeros(2, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(data.observations.shape[-1],), dtype=np.float32)

    def _observation(self) -> FloatArray:
        observation = self.data.observations[self.index, 0].copy()
        observation[-2:] = self.controls
        return observation

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[FloatArray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        self.index = 0
        self.controls[:] = 0
        return self._observation(), {
            "vehicle_position": self.data.positions[0, 0], "vehicle_forward": self.data.forwards[0, 0],
        }

    def step(self, action: ArrayLike) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        controls = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(controls):
            raise ValueError("Invalid validation controls.")
        index = self.index
        self.controls[:] = controls
        ended = bool(self.data.episode_ends[index, 0])
        self.index = min(index + 1, len(self.data.observations) - 1)
        return self._observation(), 0.0, False, ended, {
            "vehicle_position": self.data.next_positions[index, 0],
            "vehicle_forward": self.data.forwards[self.index, 0],
        }

    def close(self, *, stop_session: bool = True) -> None:
        pass


@dataclass(frozen=True)
class ValidationResult:
    prediction: FloatArray
    actual: FloatArray
    mask: NDArray[np.bool_]
    forward_error: FloatArray
    lateral_error: FloatArray
    loss: float


def validate_sample(agent: PPOAgent, data: RolloutData, config: TrainingConfig, sample: int) -> ValidationResult:
    if not 0 <= sample < len(data.observations):
        raise ValueError("Sample index is outside the rollout.")
    shape = agent.vec_env.observation_space.shape
    assert shape is not None
    data.validate(shape[0])
    offsets = horizon_indices(config.predictor.horizons_seconds, config.environment.control_interval)
    targets, masks = build_trajectory_targets(data.positions, data.forwards, data.next_positions, data.episode_ends, offsets)
    prediction = agent.predict_trajectory(data.observations[sample, 0])
    actual, mask = targets[sample, 0], masks[sample, 0]
    loss = masked_predictor_loss(torch.as_tensor(prediction), torch.as_tensor(actual), torch.as_tensor(mask))
    errors = np.abs(prediction - actual)
    return ValidationResult(prediction, actual, mask, errors[:, 0], errors[:, 1], loss.item())


def print_result(result: ValidationResult, horizons: tuple[float, ...]) -> None:
    print("horizon | predicted trajectory (forward, left) | actual trajectory | forward error | lateral error")
    for horizon, predicted, actual, valid, forward, lateral in zip(
        horizons, result.prediction, result.actual, result.mask, result.forward_error, result.lateral_error,
    ):
        measured = f"({actual[0]:+.3f}, {actual[1]:+.3f}) | {forward:.3f} m | {lateral:.3f} m" if valid else "masked | -- | --"
        print(f"{horizon:.2f}s | ({predicted[0]:+.3f}, {predicted[1]:+.3f}) | {measured}")
    print(f"Valid horizon points: {int(result.mask.sum())}/{len(result.mask)}")
    print(f"Total predictor loss (masked SmoothL1): {result.loss:.6f}")


def plot_result(result: ValidationResult, *, destination: Path | None, show: bool) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    valid = result.mask
    # Matplotlib's stubs leave stylistic **kwargs untyped. Keep that boundary
    # local to rendering; numerical data and validation remain fully typed.
    axes.plot(result.prediction[valid, 0], result.prediction[valid, 1], "o-", label="Predicted")  # pyright: ignore[reportUnknownMemberType]
    axes.plot(result.actual[valid, 0], result.actual[valid, 1], "x-", label="Actual")  # pyright: ignore[reportUnknownMemberType]
    axes.scatter([0], [0], marker="s", label="Car at t")  # pyright: ignore[reportUnknownMemberType]
    axes.set(xlabel="Forward [m]", ylabel="Left [m]", title="Local future trajectory")
    axes.set_aspect("equal", adjustable="datalim")
    axes.grid(True)  # pyright: ignore[reportUnknownMemberType]
    axes.legend()  # pyright: ignore[reportUnknownMemberType]
    figure.tight_layout()
    try:
        if destination is not None:
            figure.savefig(destination)  # pyright: ignore[reportUnknownMemberType]
        if show:
            plt.show()  # pyright: ignore[reportUnknownMemberType]
    finally:
        plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--checkpoint", type=Path, help="Run final/ or checkpoints/step_XXXXXX directory.")
    model_source.add_argument("--config", type=Path, default=Path("configs/training.toml"))
    parser.add_argument("--data", type=Path, help="Selected rollout NPZ containing raw observations and poses.")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-plot", type=Path)
    args = parser.parse_args(argv)
    config = TrainingConfig.load(checkpoint_config_path(args.checkpoint) if args.checkpoint else args.config)
    if not config.predictor.enabled:
        parser.error("Predictor must be enabled in the model's configuration.")
    data = load_rollout(args.data) if args.data else synthetic_rollout(config)
    dimensions = 10 + 4 * len(config.environment.lookahead_distances) + 2 * len(config.environment.reference_distances)
    data.validate(dimensions)
    environment = ValidationEnvironment(data)
    agent: PPOAgent | None = None
    try:
        if args.checkpoint:
            agent = PPOAgent.load_snapshot(args.checkpoint, environment, config=config.agent_config())
            print(f"Saved predictor: {args.checkpoint}")
        else:
            agent = create_agent(environment, config)
            print("Fresh, UNTRAINED predictor; these errors validate plumbing, not learned accuracy.")
        agent.set_evaluation_mode()
        print("Selected rollout data." if args.data else "Synthetic test data; no live simulator.")
        result = validate_sample(agent, data, config, args.sample)
        print_result(result, config.predictor.horizons_seconds)
        if args.plot or args.save_plot:
            plot_result(result, destination=args.save_plot, show=args.plot)
    finally:
        if agent is not None:
            agent.close()
        else:
            environment.close()


if __name__ == "__main__":
    main()
