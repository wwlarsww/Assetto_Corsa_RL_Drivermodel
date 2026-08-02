"""Synchronous PPO baseline for the fixed-reference setup.

The training flow is intentionally simple so that new readers can follow the
complete pipeline from rollout collection to policy updates without having to
understand the earlier reference-specific machinery.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from assetto_corsa_rl.config import ACTION_METHOD_VARIANT, BASELINE_METHOD_VARIANT, ProjectConfig
from assetto_corsa_rl.envs.environment import AssettoCorsaEnv
from assetto_corsa_rl.experiment import ExperimentRecorder
from assetto_corsa_rl.features.schema import OBSERVATION_FEATURE_NAMES, OBSERVATION_SIZE
from assetto_corsa_rl.models.bezier import decode_bezier_trajectory
from assetto_corsa_rl.models.driver_network import DriverNetwork, SquashedNormal
from assetto_corsa_rl.references.fixed_reference import ReferenceTrajectory
from assetto_corsa_rl.rewards.components import PredictedProgressReward
from assetto_corsa_rl.training.checkpoints import TrainingState, load_checkpoint, save_checkpoint
from assetto_corsa_rl.training.gae import AdvantageBatch, compute_gae
from assetto_corsa_rl.training.losses import bezier_trajectory_prediction_loss
from assetto_corsa_rl.training.monitor import launch_training_monitor
from assetto_corsa_rl.training.rollout import (
    OnPolicyCollector,
    RolloutBatch,
    TrajectoryContext,
    TrajectoryFragment,
)
from assetto_corsa_rl.training.targets import (
    build_observed_trajectory_targets,
    build_reference_projection_paths,
)
from assetto_corsa_rl.types import FloatArray, TerminationReason

LOGGER = logging.getLogger("assetto_corsa_rl")


@dataclass(frozen=True, slots=True)
class PPOIterationMetrics:
    """One update's collection, reward and optimisation diagnostics."""

    iteration: int
    policy_version: int
    environment_steps: int
    rollout_steps: int
    episodes_completed: int
    crash_terminations: int
    off_track_terminations: int
    stalled_terminations: int
    excessive_body_sideslip_terminations: int
    lap_completions: int
    time_limit_truncations: int
    environment_reward_mean: float
    shaped_reward_mean: float
    predicted_progress_reward_mean: float
    predictor_valid_samples: int
    predictor_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
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
    termination_counts: dict[str, int]


@dataclass(slots=True)
class _CollectionCursor:
    observation: FloatArray
    info: dict[str, Any]
    trajectory_id: int
    episodes_completed: int = 0
    termination_counts: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True, slots=True)
class _ProcessedRollout:
    batch: RolloutBatch
    rewards: FloatArray
    advantages: AdvantageBatch
    predictor_targets: FloatArray | None = None
    predictor_valid_mask: Any | None = None


@dataclass(frozen=True, slots=True)
class _OptimisationMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    predictor_loss: float = 0.0


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
    monitor_ui: bool = False,
) -> PPOTrainingResult:
    """Train the fixed-reference PPO baseline end to end.

    Training alternates between rollout collection and PPO optimisation until
    the configured environment step budget is exhausted. Each iteration:

    1. collect an on-policy fragment from the environment,
    2. compute GAE advantages for the fragment,
    3. apply clipped PPO updates over shuffled minibatches,
    4. persist a checkpoint and record metrics.
    """

    if config.ppo.method_variant != BASELINE_METHOD_VARIANT:
        raise ValueError(f"train_ppo only supports {BASELINE_METHOD_VARIANT}")
    if config.reference.mode != "fixed":
        raise ValueError("the current PPO baseline requires one fixed reference")

    torch.manual_seed(config.experiment.seed)  # pyright: ignore[reportUnknownMemberType]
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
    expected_iterations = (
        config.ppo.total_environment_steps + config.ppo.rollout_steps - 1
    ) // config.ppo.rollout_steps
    LOGGER.info(
        "Starting PPO training: backend=%s, run=%s, target_steps=%s, "
        "rollout_steps=%s, expected_iterations=%s",
        backend_name,
        recorder.run_directory,
        config.ppo.total_environment_steps,
        config.ppo.rollout_steps,
        expected_iterations,
    )
    if monitor_ui:
        launch_training_monitor(recorder.run_directory)

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
            processed = _postprocess_rollout(
                config,
                batch,
                model=model,
                reference=environment.active_reference,
                device=target_device,
            )
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
                crash_terminations=cursor.termination_counts[TerminationReason.CRASH.value],
                off_track_terminations=cursor.termination_counts[TerminationReason.OFF_TRACK.value],
                stalled_terminations=cursor.termination_counts[TerminationReason.STALLED.value],
                excessive_body_sideslip_terminations=cursor.termination_counts[
                    TerminationReason.EXCESSIVE_BODY_SIDESLIP.value
                ],
                lap_completions=cursor.termination_counts[TerminationReason.LAP_COMPLETE.value],
                time_limit_truncations=cursor.termination_counts["time_limit"],
                environment_reward_mean=float(batch.rewards.mean()),
                shaped_reward_mean=float(processed.rewards.mean()),
                predicted_progress_reward_mean=float((processed.rewards - batch.rewards).mean()),
                predictor_valid_samples=(
                    int(np.count_nonzero(processed.predictor_valid_mask))
                    if processed.predictor_valid_mask is not None
                    else 0
                ),
                **asdict(optimisation),
            )
            recorder.record_step(asdict(metrics))
            LOGGER.info(
                "Iteration %s/%s | steps=%s | env_reward=%.4f | shaped_reward=%.4f | "
                "policy_loss=%.6f | value_loss=%.6f | entropy=%.6f | kl=%.6f | clip=%.4f | "
                "predictor_loss=%.6f | episodes=%s | "
                "ends=crash:%s/off_track:%s/body_sideslip:%s/stalled:%s/lap:%s/time:%s",
                iteration,
                expected_iterations,
                environment_steps,
                metrics.environment_reward_mean,
                metrics.shaped_reward_mean,
                metrics.policy_loss,
                metrics.value_loss,
                metrics.entropy,
                metrics.approximate_kl,
                metrics.clip_fraction,
                metrics.predictor_loss,
                metrics.episodes_completed,
                metrics.crash_terminations,
                metrics.off_track_terminations,
                metrics.excessive_body_sideslip_terminations,
                metrics.stalled_terminations,
                metrics.lap_completions,
                metrics.time_limit_truncations,
            )
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
                    "action_method_variant": config.actions.method_variant,
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
        termination_counts=dict(cursor.termination_counts),
    )
    recorder.write_summary(
        {
            "method_variant": config.ppo.method_variant,
            "action_method_variant": config.actions.method_variant,
            "backend": backend_name,
            "iterations": result.iterations,
            "environment_steps": result.environment_steps,
            "policy_version": result.policy_version,
            "episodes_completed": result.episodes_completed,
            "termination_counts": result.termination_counts,
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
    """Collect one on-policy rollout fragment from the environment.

    The fragment may span multiple episodes, and the cursor preserves the last
    observation and episode state across iterations. The returned batch is
    ready for advantage estimation and PPO optimisation.
    """
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
    termination_counts = cursor.termination_counts.copy()
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
            distribution = SquashedNormal(output.action_location, output.action_log_std)
            action_tensor = distribution.sample()
            log_probability = distribution.log_prob(action_tensor)
        action = np.asarray(action_tensor.squeeze(0).cpu(), dtype=np.float32)
        # step() applies the action and returns the next feature vector computed
        # by AssettoCorsaEnv from the latest simulator telemetry.
        next_observation, reward, is_terminated, is_truncated, next_info = environment.step(action)
        with torch.no_grad():
            # Bootstrap from the actual successor before a possible episode reset.
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
            # A new trajectory id prevents future processing across episode boundaries.
            episodes_completed += 1
            reason = next_info["termination_reason"] if is_terminated else "time_limit"
            if not isinstance(reason, str):
                raise RuntimeError("completed episode has no termination reason")
            termination_counts[reason] += 1
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
    # The collector validates fragment length and rejects stale policy versions.
    collector.add(fragment)
    return collector.build(), _CollectionCursor(
        observation=observation,
        info=info,
        trajectory_id=trajectory_id,
        episodes_completed=episodes_completed,
        termination_counts=termination_counts,
    )


def _postprocess_rollout(
    config: ProjectConfig,
    batch: RolloutBatch,
    *,
    model: DriverNetwork,
    reference: ReferenceTrajectory,
    device: torch.device,
) -> _ProcessedRollout:
    """Apply prediction shaping and targets before termination-aware GAE."""

    shaped_rewards = batch.rewards.copy()
    context = batch.context
    if context is None and (
        config.ppo.use_predicted_progress_reward or config.ppo.predictor_loss_coefficient > 0
    ):
        raise ValueError("trajectory context is required for prediction post-processing")

    if config.ppo.use_predicted_progress_reward:
        if context is None:  # pragma: no cover - guarded above for type narrowing
            raise RuntimeError("trajectory context is unavailable")
        model.eval()
        with torch.no_grad():
            output = model(torch.as_tensor(batch.observations, device=device))
            predicted_trajectory = np.asarray(
                decode_bezier_trajectory(output.trajectory_parameters).cpu(),
                dtype=np.float32,
            )
        reference_paths = build_reference_projection_paths(
            reference=reference,
            progress_m=context.reference_progress_m,
            positions_xy_m=context.positions_xy_m,
            heading_rad=context.vehicle_heading_rad,
            valid_mask=np.ones(context.step_count, dtype=np.bool_),
        )
        prediction_reward = PredictedProgressReward().compute(
            predicted_trajectory,
            reference_paths.points_xz_m,
            reference_paths.progress_m,
            reference_paths.valid_mask,
        )
        shaped_rewards = PredictedProgressReward.shape_rollout_rewards(
            batch.rewards,
            prediction_reward,
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
    predictor_targets = None
    predictor_valid_mask = None
    if config.ppo.predictor_loss_coefficient > 0:
        if context is None:  # pragma: no cover - guarded above for type narrowing
            raise RuntimeError("trajectory context is unavailable")
        observed_targets = build_observed_trajectory_targets(
            positions_xy_m=context.positions_xy_m,
            heading_rad=context.vehicle_heading_rad,
            simulation_time_s=context.simulation_time_s.astype(np.float64),
            trajectory_id=context.trajectory_id,
        )
        predictor_targets = observed_targets.trajectory_xz_m
        predictor_valid_mask = observed_targets.valid_mask

    return _ProcessedRollout(
        batch=batch,
        rewards=shaped_rewards,
        advantages=advantages,
        predictor_targets=predictor_targets,
        predictor_valid_mask=predictor_valid_mask,
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
    """Perform PPO updates over the collected rollout batch.

    The rollout is shuffled and split into minibatches for multiple epochs.
    The clipped surrogate objective is combined with value regression and
    entropy regularisation to stabilise policy improvement.
    """
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
    metric_values: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
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
            distribution = SquashedNormal(output.action_location, output.action_log_std)
            new_log_probability = distribution.log_prob(action[indices])
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
            entropy = distribution.entropy().mean()
            total_loss = (
                policy_loss
                + config.ppo.value_loss_coefficient * value_loss
                - config.ppo.entropy_coefficient * entropy
            )
            predictor_loss_value = 0.0
            # Auxiliary predictor loss (bezier decoded) when targets are available
            if rollout.predictor_targets is not None and rollout.predictor_valid_mask is not None:
                # Extract target slice matching indices.
                target_batch = torch.as_tensor(
                    rollout.predictor_targets[indices.cpu().numpy()],
                    device=device,
                )
                valid_mask = torch.as_tensor(
                    rollout.predictor_valid_mask[indices.cpu().numpy()],
                    device=device,
                    dtype=torch.bool,
                )
                if valid_mask.numel() > 0 and valid_mask.any():
                    predictor_loss = bezier_trajectory_prediction_loss(
                        output.trajectory_parameters[valid_mask],
                        target_batch[valid_mask],
                    )
                    predictor_loss_value = float(predictor_loss.detach())
                    total_loss = total_loss + config.ppo.predictor_loss_coefficient * predictor_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.ppo.max_gradient_norm)
            optimizer.step()

            with torch.no_grad():
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (torch.abs(ratio - 1.0) > config.ppo.clip_range).float().mean()
            for name, value in {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approximate_kl": approximate_kl,
                "clip_fraction": clip_fraction,
            }.items():
                metric_values[name].append(float(value.detach().item()))
            metric_values.setdefault("predictor_loss", []).append(float(predictor_loss_value))
    return _OptimisationMetrics(
        **{name: float(np.mean(values)) for name, values in metric_values.items()}
    )


def _validate_initial_checkpoint(configuration: dict[str, Any]) -> None:
    """Verify any provided checkpoint matches the current action/feature schema."""
    method = configuration.get("method_variant")
    if method != BASELINE_METHOD_VARIANT:
        raise ValueError(f"initial checkpoint is not compatible with {BASELINE_METHOD_VARIANT}")
    if configuration.get("action_method_variant") != ACTION_METHOD_VARIANT:
        raise ValueError("initial checkpoint action schema does not match")
    feature_names = configuration.get("observation_feature_names")
    if feature_names != list(OBSERVATION_FEATURE_NAMES):
        raise ValueError("initial checkpoint observation feature schema does not match")
