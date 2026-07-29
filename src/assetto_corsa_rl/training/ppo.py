"""Synchronous PPO baseline for the fixed-reference, no-ProMP method."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Independent, Normal

from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.envs.environment import AssettoCorsaEnv
from assetto_corsa_rl.experiment import ExperimentRecorder
from assetto_corsa_rl.features.schema import OBSERVATION_FEATURE_NAMES, OBSERVATION_SIZE
from assetto_corsa_rl.models.bezier import decode_bezier_trajectory
from assetto_corsa_rl.models.driver_network import DriverNetwork
from assetto_corsa_rl.rewards.components import PredictedProgressReward
from assetto_corsa_rl.training.checkpoints import (
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from assetto_corsa_rl.training.gae import AdvantageBatch, compute_gae
from assetto_corsa_rl.training.losses import bezier_trajectory_prediction_loss
from assetto_corsa_rl.training.rollout import (
    OnPolicyCollector,
    RolloutBatch,
    TrajectoryContext,
    TrajectoryFragment,
)
from assetto_corsa_rl.training.targets import (
    FutureTrajectoryTargets,
    build_observed_trajectory_targets,
    build_reference_projection_paths,
)
from assetto_corsa_rl.types import REFERENCE_TIME_OFFSETS_S, FloatArray


@dataclass(frozen=True, slots=True)
class PPOIterationMetrics:
    """One update's collection, reward and optimisation diagnostics."""

    iteration: int
    policy_version: int
    environment_steps: int
    rollout_steps: int
    episodes_completed: int
    environment_reward_mean: float
    predicted_progress_mean_m: float
    shaped_reward_mean: float
    predictor_valid_samples: int
    policy_loss: float
    value_loss: float
    entropy: float
    predictor_loss: float
    approximate_kl: float
    clip_fraction: float


@dataclass(frozen=True, slots=True)
class PPOTrainingResult:
    """Completed PPO run and its reproducible local artefacts."""

    run_directory: Path
    checkpoint_path: Path
    iterations: int
    environment_steps: int
    policy_version: int
    episodes_completed: int


@dataclass(slots=True)
class _CollectionCursor:
    observation: FloatArray
    info: dict[str, Any]
    trajectory_id: int
    episodes_completed: int = 0


@dataclass(frozen=True, slots=True)
class _ProcessedRollout:
    batch: RolloutBatch
    rewards: FloatArray
    advantages: AdvantageBatch
    predictor_targets: FutureTrajectoryTargets
    predicted_progress_m: FloatArray


@dataclass(frozen=True, slots=True)
class _OptimisationMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    predictor_loss: float
    approximate_kl: float
    clip_fraction: float


def build_driver_network(config: ProjectConfig) -> DriverNetwork:
    """Construct the exact network architecture declared by the configuration."""

    return DriverNetwork(
        observation_size=OBSERVATION_SIZE,
        feature_hidden_sizes=config.model.feature_hidden_sizes,
        policy_hidden_sizes=config.model.policy_hidden_sizes,
        value_hidden_sizes=config.model.value_hidden_sizes,
        predictor_hidden_sizes=config.model.predictor_hidden_sizes,
        trajectory_output_size=config.model.trajectory_output_size,
        initial_log_std=config.model.initial_log_std,
    )


def train_ppo(
    config: ProjectConfig,
    environment: AssettoCorsaEnv,
    *,
    command: list[str],
    backend_name: str,
    device: str | torch.device = "cpu",
) -> PPOTrainingResult:
    """Train the named synchronous PPO baseline.

    Prediction reward shaping is intentionally completed after collection and
    before GAE. The same rollout then supplies actual future paths for the
    auxiliary predictor loss. This keeps simulator dynamics independent of
    learned reward terms.
    """

    if config.ppo.method_variant != "fixed_reference_no_promp_v1":
        raise ValueError("train_ppo only supports fixed_reference_no_promp_v1")
    if config.reference.mode != "fixed":
        raise ValueError("the current PPO baseline requires one fixed reference")

    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    torch.use_deterministic_algorithms(True)
    target_device = torch.device(device)
    model = build_driver_network(config).to(target_device)
    initial_checkpoint = config.ppo.initial_checkpoint_path.strip()
    if initial_checkpoint:
        source = Path(initial_checkpoint)
        if not source.is_file():
            raise FileNotFoundError(f"initial PPO checkpoint does not exist: {source}")
        _, checkpoint_configuration = load_checkpoint(
            source,
            model=model,
            map_location=target_device,
        )
        _validate_initial_checkpoint(checkpoint_configuration)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.learning_rate)
    recorder = ExperimentRecorder.create(
        output_directory=Path(config.experiment.output_directory) / f"ppo-{backend_name}",
        config=config,
        command=command,
    )
    checkpoint_path = recorder.run_directory / "checkpoint.pt"
    iteration = 0
    environment_steps = 0
    policy_version = 0
    random = np.random.default_rng(config.experiment.seed)
    observation, info = environment.reset(seed=config.experiment.seed)
    cursor = _CollectionCursor(
        observation=observation,
        info=info,
        trajectory_id=0,
    )

    try:
        while environment_steps < config.ppo.total_environment_steps:
            iteration += 1
            rollout_step_count = min(
                config.ppo.rollout_steps,
                config.ppo.total_environment_steps - environment_steps,
            )
            batch, cursor = _collect_rollout(
                environment,
                model,
                cursor,
                policy_version=policy_version,
                step_count=rollout_step_count,
                device=target_device,
            )
            processed = _postprocess_rollout(config, environment, model, batch, target_device)
            optimisation = _optimise_ppo(
                config,
                model,
                optimizer,
                processed,
                random=random,
                device=target_device,
            )
            environment_steps += batch.step_count
            policy_version += 1
            metrics = PPOIterationMetrics(
                iteration=iteration,
                policy_version=policy_version,
                environment_steps=environment_steps,
                rollout_steps=batch.step_count,
                episodes_completed=cursor.episodes_completed,
                environment_reward_mean=float(batch.rewards.mean()),
                predicted_progress_mean_m=float(processed.predicted_progress_m.mean()),
                shaped_reward_mean=float(processed.rewards.mean()),
                predictor_valid_samples=int(
                    np.count_nonzero(processed.predictor_targets.valid_mask)
                ),
                **asdict(optimisation),
            )
            recorder.record_step(asdict(metrics))
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                training_state=TrainingState(
                    iteration=iteration,
                    environment_steps=environment_steps,
                    policy_version=policy_version,
                ),
                configuration={
                    "method_variant": config.ppo.method_variant,
                    "project": config.to_dict(),
                    "observation_feature_names": list(OBSERVATION_FEATURE_NAMES),
                    "initial_checkpoint": initial_checkpoint or None,
                    "backend": backend_name,
                },
            )
    finally:
        recorder.close()

    result = PPOTrainingResult(
        run_directory=recorder.run_directory,
        checkpoint_path=checkpoint_path,
        iterations=iteration,
        environment_steps=environment_steps,
        policy_version=policy_version,
        episodes_completed=cursor.episodes_completed,
    )
    recorder.write_summary(
        {
            "method_variant": config.ppo.method_variant,
            "backend": backend_name,
            "iterations": result.iterations,
            "environment_steps": result.environment_steps,
            "policy_version": result.policy_version,
            "episodes_completed": result.episodes_completed,
            "predicted_progress_reward_enabled": config.ppo.use_predicted_progress_reward,
            "checkpoint": checkpoint_path.name,
        }
    )
    return result


def _collect_rollout(  # noqa: PLR0915
    environment: AssettoCorsaEnv,
    model: DriverNetwork,
    cursor: _CollectionCursor,
    *,
    policy_version: int,
    step_count: int,
    device: torch.device,
) -> tuple[RolloutBatch, _CollectionCursor]:
    observations: list[FloatArray] = []
    actions: list[FloatArray] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    log_probabilities: list[float] = []
    values: list[float] = []
    next_values: list[float] = []
    positions: list[FloatArray] = []
    headings: list[float] = []
    simulation_times: list[float] = []
    trajectory_ids: list[int] = []
    reference_progress: list[float] = []
    observation = cursor.observation
    info = cursor.info
    trajectory_id = cursor.trajectory_id
    episodes_completed = cursor.episodes_completed
    model.eval()

    for _ in range(step_count):
        observations.append(observation.copy())
        positions.append(np.asarray(info["position_xy_m"], dtype=np.float32))
        headings.append(float(info["vehicle_heading_rad"]))
        telemetry = info["telemetry"]
        simulation_times.append(float(telemetry["simulation_time_s"]))
        reference_progress.append(float(telemetry["progress_m"]))
        trajectory_ids.append(trajectory_id)

        observation_tensor = torch.as_tensor(observation, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(observation_tensor)
            distribution: Independent[Normal] = Independent(
                Normal(output.action_mean, output.action_log_std.exp()),
                1,
            )
            action_tensor = distribution.sample()
            log_probability = distribution.log_prob(  # type: ignore[no-untyped-call]
                action_tensor
            )
        action = np.asarray(action_tensor.squeeze(0).cpu(), dtype=np.float32)
        next_observation, reward, is_terminated, is_truncated, next_info = environment.step(action)
        with torch.no_grad():
            next_value = model(
                torch.as_tensor(next_observation, device=device).unsqueeze(0)
            ).value.item()

        actions.append(action)
        rewards.append(reward)
        terminated.append(is_terminated)
        truncated.append(is_truncated)
        log_probabilities.append(float(log_probability.item()))
        values.append(float(output.value.item()))
        next_values.append(float(next_value))

        if is_terminated or is_truncated:
            episodes_completed += 1
            trajectory_id += 1
            observation, info = environment.reset()
        else:
            observation, info = next_observation, next_info

    fragment = TrajectoryFragment(
        policy_version=policy_version,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated, dtype=np.bool_),
        truncated=np.asarray(truncated, dtype=np.bool_),
        log_probabilities=np.asarray(log_probabilities, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        next_values=np.asarray(next_values, dtype=np.float32),
        context=TrajectoryContext(
            positions_xy_m=np.asarray(positions, dtype=np.float32),
            vehicle_heading_rad=np.asarray(headings, dtype=np.float32),
            simulation_time_s=np.asarray(simulation_times, dtype=np.float32),
            trajectory_id=np.asarray(trajectory_ids, dtype=np.int64),
            reference_progress_m=np.asarray(reference_progress, dtype=np.float32),
        ),
    )
    collector = OnPolicyCollector(
        policy_version=policy_version,
        target_step_count=step_count,
    )
    collector.add(fragment)
    return collector.build(), _CollectionCursor(
        observation=observation,
        info=info,
        trajectory_id=trajectory_id,
        episodes_completed=episodes_completed,
    )


def _postprocess_rollout(
    config: ProjectConfig,
    environment: AssettoCorsaEnv,
    model: DriverNetwork,
    batch: RolloutBatch,
    device: torch.device,
) -> _ProcessedRollout:
    if batch.context is None:
        raise ValueError("PPO rollout requires trajectory context")
    context = batch.context
    targets = build_observed_trajectory_targets(
        positions_xy_m=context.positions_xy_m,
        heading_rad=context.vehicle_heading_rad,
        simulation_time_s=context.simulation_time_s,
        trajectory_id=context.trajectory_id,
    )
    predicted_progress = np.zeros(batch.step_count, dtype=np.float32)
    shaped_rewards = batch.rewards.copy()
    if config.ppo.use_predicted_progress_reward:
        with torch.no_grad():
            prediction = decode_bezier_trajectory(
                model(torch.as_tensor(batch.observations, device=device)).trajectory_parameters,
                sample_count=REFERENCE_TIME_OFFSETS_S.size,
            )
        reward_mask = np.ones(batch.step_count, dtype=np.bool_)
        reference_paths = build_reference_projection_paths(
            reference=environment.active_reference,
            progress_m=context.reference_progress_m,
            positions_xy_m=context.positions_xy_m,
            heading_rad=context.vehicle_heading_rad,
            valid_mask=reward_mask,
        )
        prediction_terms = PredictedProgressReward().compute(
            np.asarray(prediction.cpu(), dtype=np.float32),
            reference_paths.points_xz_m,
            reference_paths.progress_m,
            reference_paths.valid_mask,
        )
        predicted_progress = prediction_terms.predicted_progress
        shaped_rewards = PredictedProgressReward.shape_rollout_rewards(
            batch.rewards,
            prediction_terms,
            predicted_progress_scale=config.rewards.prediction_progress_scale,
        )
    advantages = compute_gae(
        rewards=shaped_rewards,
        values=batch.values,
        next_values=batch.next_values,
        terminated=batch.terminated,
        truncated=batch.truncated,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
    )
    return _ProcessedRollout(
        batch=batch,
        rewards=shaped_rewards,
        advantages=advantages,
        predictor_targets=targets,
        predicted_progress_m=predicted_progress,
    )


def _optimise_ppo(
    config: ProjectConfig,
    model: DriverNetwork,
    optimizer: torch.optim.Optimizer,
    rollout: _ProcessedRollout,
    *,
    random: np.random.Generator,
    device: torch.device,
) -> _OptimisationMetrics:
    batch = rollout.batch
    observation = torch.as_tensor(batch.observations, device=device)
    action = torch.as_tensor(batch.actions, device=device)
    old_log_probability = torch.as_tensor(batch.log_probabilities, device=device)
    returns = torch.as_tensor(rollout.advantages.returns, device=device)
    advantage_values = rollout.advantages.advantages
    normalised_advantage = (advantage_values - advantage_values.mean()) / (
        advantage_values.std() + 1e-8
    )
    advantage = torch.as_tensor(normalised_advantage, device=device)
    predictor_target = torch.as_tensor(
        rollout.predictor_targets.trajectory_xz_m,
        device=device,
    )
    predictor_valid = torch.as_tensor(
        rollout.predictor_targets.valid_mask,
        device=device,
    )
    metric_values: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "predictor_loss": [],
        "approximate_kl": [],
        "clip_fraction": [],
    }
    model.train()
    for _ in range(config.ppo.update_epochs):
        permutation = random.permutation(batch.step_count)
        for start in range(0, batch.step_count, config.ppo.minibatch_size):
            indices = torch.as_tensor(
                permutation[start : start + config.ppo.minibatch_size],
                device=device,
            )
            output = model(observation[indices])
            distribution: Independent[Normal] = Independent(
                Normal(output.action_mean, output.action_log_std.exp()),
                1,
            )
            new_log_probability = distribution.log_prob(  # type: ignore[no-untyped-call]
                action[indices]
            )
            log_ratio = new_log_probability - old_log_probability[indices]
            ratio = log_ratio.exp()
            minibatch_advantage = advantage[indices]
            unclipped = ratio * minibatch_advantage
            clipped = (
                ratio.clamp(1.0 - config.ppo.clip_range, 1.0 + config.ppo.clip_range)
                * minibatch_advantage
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(output.value, returns[indices])
            entropy = distribution.entropy().mean()  # type: ignore[no-untyped-call]
            minibatch_valid = predictor_valid[indices]
            predictor_loss = _masked_predictor_loss(
                output.trajectory_parameters,
                predictor_target[indices],
                minibatch_valid,
            )
            total_loss = (
                policy_loss
                + config.ppo.value_loss_coefficient * value_loss
                - config.ppo.entropy_coefficient * entropy
                + config.ppo.predictor_loss_coefficient * predictor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.ppo.max_gradient_norm)
            optimizer.step()

            with torch.no_grad():
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (torch.abs(ratio - 1.0) > config.ppo.clip_range).float().mean()
            for name, value in {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "predictor_loss": predictor_loss,
                "approximate_kl": approximate_kl,
                "clip_fraction": clip_fraction,
            }.items():
                metric_values[name].append(float(value.detach().item()))
    return _OptimisationMetrics(
        **{name: float(np.mean(values)) for name, values in metric_values.items()}
    )


def _masked_predictor_loss(
    flat_prediction: Tensor,
    target_xz_m: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    if bool(valid_mask.any()):
        return bezier_trajectory_prediction_loss(
            flat_prediction,
            target_xz_m,
            valid_mask,
        )
    return flat_prediction.sum() * 0.0


def _validate_initial_checkpoint(configuration: dict[str, Any]) -> None:
    method = configuration.get("method_variant")
    if method != "fixed_reference_no_promp_v1":
        raise ValueError("initial checkpoint was not produced by fixed_reference_no_promp_v1")
    feature_names = configuration.get("observation_feature_names")
    if feature_names != list(OBSERVATION_FEATURE_NAMES):
        raise ValueError("initial checkpoint observation feature schema does not match")
