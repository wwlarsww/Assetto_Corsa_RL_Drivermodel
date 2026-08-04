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
import numpy.typing as npt
import torch

from assetto_corsa_rl.config import (
    ACTION_METHOD_VARIANT,
    BASELINE_METHOD_VARIANT,
    STABILIZED_PPO_METHOD_VARIANT,
    ProjectConfig,
)
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
    crash_terminations_delta: int
    off_track_terminations_delta: int
    stalled_terminations_delta: int
    excessive_body_sideslip_terminations_delta: int
    lap_completions_delta: int
    time_limit_truncations_delta: int
    environment_reward_mean: float
    progress_reward_mean: float
    shaped_reward_mean: float
    predicted_progress_reward_mean: float
    predicted_reward_share: float
    predicted_to_progress_ratio: float
    pedal_mean: float
    pedal_std: float
    pedal_p10: float
    pedal_p50: float
    pedal_p90: float
    throttle_mean: float
    brake_mean: float
    throttle_fraction: float
    brake_fraction: float
    coast_fraction: float
    pedal_switches: int
    pedal_switch_rate: float
    speed_mean_mps: float
    speed_min_mps: float
    speed_max_mps: float
    tyres_out_mean: float
    tyres_out_max: int
    tyres_out_any_fraction: float
    predictor_valid_samples: int
    predictor_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    kl_target: float
    kl_lower_bound: float
    kl_upper_bound: float
    kl_status: str
    kl_adjustment: str
    learning_rate: float
    update_epochs_completed: int
    action_sigma_mean: float


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
    predicted_rewards: FloatArray
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
    update_epochs_completed: int = 0


@dataclass(frozen=True, slots=True)
class _RolloutDiagnostics:
    progress_rewards: FloatArray
    speed_mps: FloatArray
    tyres_out: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class KLBandUpdate:
    """Result of adapting the optimiser learning rate to the KL target band."""

    status: str
    adjustment: str
    learning_rate: float


class KLBandController:
    """Bound PPO learning rate using a deterministic approximate-KL band."""

    def __init__(self, config: ProjectConfig) -> None:
        self.target = config.ppo.kl_target
        self.tolerance = config.ppo.kl_tolerance
        self.minimum_learning_rate = config.ppo.minimum_learning_rate
        self.maximum_learning_rate = config.ppo.learning_rate
        self.decrease_factor = config.ppo.kl_learning_rate_decrease_factor
        self.increase_factor = config.ppo.kl_learning_rate_increase_factor

    @property
    def lower_bound(self) -> float:
        """Return the lower acceptable approximate-KL boundary."""

        return self.target - self.tolerance

    @property
    def upper_bound(self) -> float:
        """Return the upper acceptable approximate-KL boundary."""

        return self.target + self.tolerance

    def update(
        self,
        approximate_kl: float,
        optimizer: torch.optim.Optimizer,
    ) -> KLBandUpdate:
        """Adjust all optimiser parameter groups for the next PPO iteration."""

        current = float(optimizer.param_groups[0]["lr"])
        if approximate_kl > self.upper_bound:
            status = "above_band"
            adjustment = "decrease"
            updated = max(self.minimum_learning_rate, current * self.decrease_factor)
        elif approximate_kl < self.lower_bound:
            status = "below_band"
            adjustment = "increase"
            updated = min(self.maximum_learning_rate, current * self.increase_factor)
        else:
            status = "in_band"
            adjustment = "hold"
            updated = current
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = updated
        return KLBandUpdate(status, adjustment, updated)


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


def train_ppo(  # noqa: PLR0915
    config: ProjectConfig,
    environment: AssettoCorsaEnv,
    *,
    command: list[str],
    backend_name: str,
    device: str | torch.device = "cpu",
    monitor_ui: bool = False,
    monitor_backend: str = "aim",
) -> PPOTrainingResult:
    """Train the fixed-reference PPO baseline end to end.

    Training alternates between rollout collection and PPO optimisation until
    the configured environment step budget is exhausted. Each iteration:

    1. collect an on-policy fragment from the environment,
    2. compute GAE advantages for the fragment,
    3. apply clipped PPO updates over shuffled minibatches,
    4. persist a checkpoint and record metrics.
    """

    if config.ppo.method_variant != STABILIZED_PPO_METHOD_VARIANT:
        raise ValueError(f"train_ppo only supports {STABILIZED_PPO_METHOD_VARIANT}")
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
        _apply_post_bc_exploration(model, config, checkpoint_configuration)

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
    kl_controller = KLBandController(config)
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
    if monitor_ui and monitor_backend != "none":
        launch_training_monitor(
            recorder.run_directory,
            monitor_backend=monitor_backend,
            start_ui=True,
        )

    try:
        while environment_steps < config.ppo.total_environment_steps:
            iteration += 1
            rollout_step_count = min(
                config.ppo.rollout_steps,
                config.ppo.total_environment_steps - environment_steps,
            )
            previous_terminations = cursor.termination_counts.copy()
            batch, cursor, rollout_diagnostics = _collect_rollout(
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
            kl_update = kl_controller.update(optimisation.approximate_kl, optimizer)
            environment_steps += batch.step_count
            policy_version += 1
            termination_delta = cursor.termination_counts - previous_terminations
            diagnostic_metrics = _rollout_metric_values(
                batch,
                rollout_diagnostics,
                processed.predicted_rewards,
            )
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
                crash_terminations_delta=termination_delta[TerminationReason.CRASH.value],
                off_track_terminations_delta=termination_delta[TerminationReason.OFF_TRACK.value],
                stalled_terminations_delta=termination_delta[TerminationReason.STALLED.value],
                excessive_body_sideslip_terminations_delta=termination_delta[
                    TerminationReason.EXCESSIVE_BODY_SIDESLIP.value
                ],
                lap_completions_delta=termination_delta[TerminationReason.LAP_COMPLETE.value],
                time_limit_truncations_delta=termination_delta["time_limit"],
                environment_reward_mean=float(batch.rewards.mean()),
                shaped_reward_mean=float(processed.rewards.mean()),
                predictor_valid_samples=(
                    int(np.count_nonzero(processed.predictor_valid_mask))
                    if processed.predictor_valid_mask is not None
                    else 0
                ),
                kl_target=kl_controller.target,
                kl_lower_bound=kl_controller.lower_bound,
                kl_upper_bound=kl_controller.upper_bound,
                kl_status=kl_update.status,
                kl_adjustment=kl_update.adjustment,
                learning_rate=kl_update.learning_rate,
                action_sigma_mean=float(model.action_log_std.detach().exp().mean().cpu()),
                **diagnostic_metrics,
                **asdict(optimisation),
            )
            recorder.record_step(asdict(metrics))
            LOGGER.info(
                "Iteration %s/%s | steps=%s | env_reward=%.4f | shaped_reward=%.4f | "
                "policy_loss=%.6f | value_loss=%.6f | entropy=%.6f | kl=%.6f | clip=%.4f | "
                "predictor_loss=%.6f | lr=%.2e | kl_status=%s | episodes=%s | "
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
                metrics.learning_rate,
                metrics.kl_status,
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
) -> tuple[RolloutBatch, _CollectionCursor, _RolloutDiagnostics]:
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
    progress_rewards: list[float] = []
    speed_mps: list[float] = []
    tyres_out: list[int] = []
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
        progress_rewards.append(float(next_info["reward_terms"]["reward_progress"]))
        speed_mps.append(float(next_info["telemetry"]["speed_mps"]))
        backend_diagnostics = next_info["backend"]
        tyres_out.append(int(backend_diagnostics.get("number_of_tyres_out", 0)))

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
    return (
        collector.build(),
        _CollectionCursor(
            observation=observation,
            info=info,
            trajectory_id=trajectory_id,
            episodes_completed=episodes_completed,
            termination_counts=termination_counts,
        ),
        _RolloutDiagnostics(
            progress_rewards=np.asarray(progress_rewards, dtype=np.float32),
            speed_mps=np.asarray(speed_mps, dtype=np.float32),
            tyres_out=np.asarray(tyres_out, dtype=np.int64),
        ),
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
    predicted_rewards = np.zeros_like(batch.rewards)
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
        prediction_reward = PredictedProgressReward(
            horizon_discount=config.rewards.prediction_horizon_discount,
        ).compute(
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
        predicted_rewards = np.asarray(shaped_rewards - batch.rewards, dtype=np.float32)
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
        predicted_rewards=predicted_rewards,
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
        "predictor_loss": [],
    }
    approximate_kl = 0.0
    clip_fraction = 0.0
    update_epochs_completed = 0
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
            if rollout.predictor_targets is not None and rollout.predictor_valid_mask is not None:
                target_batch = torch.as_tensor(
                    rollout.predictor_targets[indices.cpu().numpy()],
                    device=device,
                )
                valid_mask = torch.as_tensor(
                    rollout.predictor_valid_mask[indices.cpu().numpy()],
                    device=device,
                    dtype=torch.bool,
                )
                if valid_mask.any():
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

            for name, value in {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
            }.items():
                metric_values[name].append(float(value.detach().item()))
            metric_values["predictor_loss"].append(predictor_loss_value)
        update_epochs_completed += 1
        approximate_kl, clip_fraction = _policy_change_metrics(
            model,
            observation,
            action,
            old_log_probability,
            clip_range=config.ppo.clip_range,
        )
        if approximate_kl > config.ppo.kl_target + config.ppo.kl_tolerance:
            break
    return _OptimisationMetrics(
        policy_loss=float(np.mean(metric_values["policy_loss"])),
        value_loss=float(np.mean(metric_values["value_loss"])),
        entropy=float(np.mean(metric_values["entropy"])),
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        predictor_loss=float(np.mean(metric_values["predictor_loss"])),
        update_epochs_completed=update_epochs_completed,
    )


def _policy_change_metrics(
    model: DriverNetwork,
    observation: torch.Tensor,
    action: torch.Tensor,
    old_log_probability: torch.Tensor,
    *,
    clip_range: float,
) -> tuple[float, float]:
    """Measure the complete rollout against the policy that collected it."""

    with torch.no_grad():
        output = model(observation)
        distribution = SquashedNormal(output.action_location, output.action_log_std)
        log_ratio = distribution.log_prob(action) - old_log_probability
        ratio = log_ratio.exp()
        approximate_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (torch.abs(ratio - 1.0) > clip_range).float().mean()
    return float(approximate_kl), float(clip_fraction)


def _rollout_metric_values(
    batch: RolloutBatch,
    diagnostics: _RolloutDiagnostics,
    predicted_rewards: FloatArray,
) -> dict[str, Any]:
    """Aggregate flat rollout diagnostics for CSV and Aim monitoring."""

    pedal = batch.actions[:, 0]
    throttle = np.clip(pedal, 0.0, 1.0)
    brake = np.clip(-pedal, 0.0, 1.0)
    pedal_state = np.zeros(pedal.size, dtype=np.int8)
    pedal_state[pedal > 0.05] = 1
    pedal_state[pedal < -0.05] = -1
    same_episode_transition = ~(batch.terminated[:-1] | batch.truncated[:-1])
    direct_switch = pedal_state[1:] * pedal_state[:-1] == -1
    switches = int(np.count_nonzero(direct_switch & same_episode_transition))
    switch_denominator = max(1, int(np.count_nonzero(same_episode_transition)))
    predicted_magnitude = float(np.abs(predicted_rewards).sum())
    progress_magnitude = float(np.abs(diagnostics.progress_rewards).sum())
    total_shaping_magnitude = progress_magnitude + predicted_magnitude
    return {
        "progress_reward_mean": float(diagnostics.progress_rewards.mean()),
        "predicted_progress_reward_mean": float(predicted_rewards.mean()),
        "predicted_reward_share": (
            predicted_magnitude / total_shaping_magnitude if total_shaping_magnitude > 0.0 else 0.0
        ),
        "predicted_to_progress_ratio": (
            predicted_magnitude / progress_magnitude if progress_magnitude > 0.0 else 0.0
        ),
        "pedal_mean": float(pedal.mean()),
        "pedal_std": float(pedal.std()),
        "pedal_p10": float(np.quantile(pedal, 0.1)),
        "pedal_p50": float(np.quantile(pedal, 0.5)),
        "pedal_p90": float(np.quantile(pedal, 0.9)),
        "throttle_mean": float(throttle.mean()),
        "brake_mean": float(brake.mean()),
        "throttle_fraction": float(np.mean(pedal_state == 1)),
        "brake_fraction": float(np.mean(pedal_state == -1)),
        "coast_fraction": float(np.mean(pedal_state == 0)),
        "pedal_switches": switches,
        "pedal_switch_rate": switches / switch_denominator,
        "speed_mean_mps": float(diagnostics.speed_mps.mean()),
        "speed_min_mps": float(diagnostics.speed_mps.min()),
        "speed_max_mps": float(diagnostics.speed_mps.max()),
        "tyres_out_mean": float(diagnostics.tyres_out.mean()),
        "tyres_out_max": int(diagnostics.tyres_out.max()),
        "tyres_out_any_fraction": float(np.mean(diagnostics.tyres_out > 0)),
    }


def _validate_initial_checkpoint(configuration: dict[str, Any]) -> None:
    """Verify any provided checkpoint matches the current action/feature schema."""

    method = configuration.get("method_variant")
    if method not in {BASELINE_METHOD_VARIANT, STABILIZED_PPO_METHOD_VARIANT}:
        raise ValueError(
            f"initial checkpoint is not compatible with {STABILIZED_PPO_METHOD_VARIANT}"
        )
    if configuration.get("action_method_variant") != ACTION_METHOD_VARIANT:
        raise ValueError("initial checkpoint action schema does not match")
    feature_names = configuration.get("observation_feature_names")
    if feature_names != list(OBSERVATION_FEATURE_NAMES):
        raise ValueError("initial checkpoint observation feature schema does not match")


def _apply_post_bc_exploration(
    model: DriverNetwork,
    config: ProjectConfig,
    checkpoint_configuration: dict[str, Any],
) -> bool:
    """Lower exploration only for checkpoints produced by behavioural cloning."""

    is_bc_checkpoint = (
        checkpoint_configuration.get("training_stage") == "behavioral_cloning"
        or "predictor_trained_with_bc" in checkpoint_configuration
    )
    if not is_bc_checkpoint:
        return False
    with torch.no_grad():
        model.action_log_std.fill_(config.ppo.initial_log_std_after_bc)
    LOGGER.info(
        "Applied post-BC action sigma %.4f (log_std=%.4f)",
        float(np.exp(config.ppo.initial_log_std_after_bc)),
        config.ppo.initial_log_std_after_bc,
    )
    return True
