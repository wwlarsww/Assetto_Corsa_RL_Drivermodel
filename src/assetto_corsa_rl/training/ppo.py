"""Synchronous PPO implementation for the configured stabilized PPO variant.

Each iteration collects one on-policy rollout, computes rewards and GAE targets,
optimises the clipped PPO objective, and persists the new policy version. The
simulator remains behind the Gymnasium environment; this module only owns
learning state and reproducible training artefacts.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from assetto_corsa_rl.ac_integration.track_assets import (
    TrackAssetBundle,
    ensure_track_assets,
    sha256_file,
)
from assetto_corsa_rl.config import (
    ACTION_METHOD_VARIANT,
    BASELINE_METHOD_VARIANT,
    POLICY_PROTECTED_PPO_METHOD_VARIANT,
    STABILIZED_PPO_METHOD_VARIANT,
    SUCCESSOR_REFERENCE_TRACKING_METHOD_VARIANT,
    ProjectConfig,
)
from assetto_corsa_rl.envs.environment import AssettoCorsaEnv
from assetto_corsa_rl.experiment import ExperimentRecorder
from assetto_corsa_rl.features.schema import OBSERVATION_FEATURE_NAMES, OBSERVATION_SIZE
from assetto_corsa_rl.models.bezier import decode_bezier_trajectory
from assetto_corsa_rl.models.driver_network import DriverNetwork, SquashedNormal
from assetto_corsa_rl.models.predictor_target_normalization import (
    load_predictor_target_scale_m,
)
from assetto_corsa_rl.models.trajectory_contract import BEZIER_TRAJECTORY_CODEC
from assetto_corsa_rl.rewards.components import PredictedProgressReward
from assetto_corsa_rl.rewards.reference_tracking import (
    ReferenceTrackingReward,
    SuccessorReferenceTrackingReward,
)
from assetto_corsa_rl.training.checkpoints import (
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from assetto_corsa_rl.training.evaluation import (
    EnvironmentFactory,
    evaluate_policy,
)
from assetto_corsa_rl.training.gae import AdvantageBatch, compute_gae
from assetto_corsa_rl.training.losses import masked_normalized_bezier_trajectory_loss
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
from assetto_corsa_rl.training.tracking import (
    ExperimentTracker,
    MetricValue,
    flatten_parameters,
)
from assetto_corsa_rl.types import PREDICTOR_TIME_OFFSETS_S, FloatArray


@dataclass(frozen=True, slots=True)
class PPOIterationMetrics:
    """One update's collection, reward and optimisation diagnostics."""

    iteration: int
    policy_version: int
    environment_steps: int
    rollout_steps: int
    episodes_completed: int
    environment_reward_mean: float
    reference_distance_mean_m: float
    reference_heading_error_mean_abs_rad: float
    reference_progress_alignment_mean: float
    reference_tracking_reward_mean: float
    predicted_progress_mean_m: float
    shaped_reward_mean: float
    predictor_valid_samples: int
    action_log_std_mean: float
    action_std_mean: float
    learning_rate: float
    next_learning_rate: float
    update_epochs_completed: int
    policy_loss: float
    value_loss: float
    entropy: float
    predictor_loss: float
    approximate_kl: float
    clip_fraction: float


def _tracking_metrics(
    metrics: PPOIterationMetrics,
) -> dict[str, MetricValue]:
    """Convert one PPO update into consistently named tracking metrics."""

    return {
        "train/progress/ppo_update": metrics.iteration,
        "train/progress/policy_version": metrics.policy_version,
        "train/progress/environment_steps": metrics.environment_steps,
        "train/progress/rollout_steps": metrics.rollout_steps,
        "train/progress/episodes_completed": metrics.episodes_completed,
        "train/reward/environment_mean": metrics.environment_reward_mean,
        "train/reward/reference_tracking_mean": metrics.reference_tracking_reward_mean,
        "train/reward/shaped_mean": metrics.shaped_reward_mean,
        "train/reference/distance_mean_m": metrics.reference_distance_mean_m,
        "train/reference/heading_error_mean_abs_rad": (
            metrics.reference_heading_error_mean_abs_rad
        ),
        "train/reference/progress_alignment_mean": metrics.reference_progress_alignment_mean,
        "train/predictor/predicted_progress_mean_m": (metrics.predicted_progress_mean_m),
        "train/predictor/valid_samples": metrics.predictor_valid_samples,
        "train/predictor/loss": metrics.predictor_loss,
        "train/ppo/policy_loss": metrics.policy_loss,
        "train/ppo/value_loss": metrics.value_loss,
        "train/ppo/entropy": metrics.entropy,
        "train/ppo/action_log_std_mean": metrics.action_log_std_mean,
        "train/ppo/action_std_mean": metrics.action_std_mean,
        "train/ppo/learning_rate": metrics.learning_rate,
        "train/ppo/next_learning_rate": metrics.next_learning_rate,
        "train/ppo/update_epochs_completed": metrics.update_epochs_completed,
        "train/ppo/approximate_kl": metrics.approximate_kl,
        "train/ppo/clip_fraction": metrics.clip_fraction,
    }


@dataclass(frozen=True, slots=True)
class PPOTrainingResult:
    """Completed PPO run and its reproducible local artefacts."""

    run_directory: Path
    checkpoint_path: Path
    iterations: int
    environment_steps: int
    policy_version: int
    episodes_completed: int
    evaluations_completed: int


@dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    """Immutable checkpoint files associated with one PPO update."""

    directory: Path
    checkpoint_path: Path
    metadata_path: Path
    iteration: int


@dataclass(slots=True)
class _CollectionCursor:
    """Environment state carried across fixed-size rollout fragments."""

    observation: FloatArray
    info: dict[str, Any]
    trajectory_id: int
    episodes_completed: int = 0


@dataclass(frozen=True, slots=True)
class _ProcessedRollout:
    """Immutable inputs consumed by one PPO optimisation phase."""

    batch: RolloutBatch
    rewards: FloatArray
    advantages: AdvantageBatch
    predictor_targets: FutureTrajectoryTargets
    reference_distance_m: FloatArray
    reference_heading_error_abs_rad: FloatArray
    reference_progress_alignment: FloatArray
    reference_tracking_reward: FloatArray
    predicted_progress_m: FloatArray


@dataclass(frozen=True, slots=True)
class _OptimisationMetrics:
    """Mean diagnostics over all minibatches in one PPO update."""

    policy_loss: float
    value_loss: float
    entropy: float
    predictor_loss: float
    approximate_kl: float
    clip_fraction: float
    learning_rate: float
    next_learning_rate: float
    update_epochs_completed: int


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
    tracker: ExperimentTracker | None = None,
    tracking_run_name: str | None = None,
    evaluation_environment_factory: EnvironmentFactory | None = None,
    evaluation_uses_training_environment: bool = False,
) -> PPOTrainingResult:
    """Train the stabilized PPO variant with synchronous, versioned updates.

    A policy version is never changed while its rollout is being collected.
    Only after post-processing and optimisation is the version incremented and
    written to the checkpoint.
    """

    _validate_training_configuration(
        config,
        evaluation_environment_factory=evaluation_environment_factory,
        evaluation_uses_training_environment=evaluation_uses_training_environment,
    )
    predictor_target_scale_m = load_predictor_target_scale_m(
        config.behavioral_cloning.dataset_artifact_path,
        expected_method_variant=(
            config.behavioral_cloning.predictor_target_normalization_method_variant
        ),
    )
    target_device = torch.device(device)
    model, optimizer, initial_checkpoint = _initialise_training_model(
        config,
        target_device,
    )
    recorder = ExperimentRecorder.create(
        output_directory=Path(config.experiment.output_directory) / f"ppo-{backend_name}",
        config=config,
        command=command,
    )
    checkpoint_path = recorder.run_directory / "checkpoint.pt"
    tracking_parameters, resolved_tracking_run_name = _prepare_tracking_run(
        config=config,
        recorder=recorder,
        command=command,
        backend_name=backend_name,
        target_device=target_device,
        initial_checkpoint=initial_checkpoint,
        tracking_run_name=tracking_run_name,
    )
    random = np.random.default_rng(config.experiment.seed)
    observation, info = environment.reset(seed=config.experiment.seed)
    cursor = _CollectionCursor(
        observation=observation,
        info=info,
        trajectory_id=0,
    )
    iteration = 0
    environment_steps = 0
    policy_version = 0
    evaluations_completed = 0
    run_status = "FAILED"
    track_asset_bundle = ensure_track_assets(config) if backend_name == "assetto-corsa" else None

    try:
        _start_tracking_run(
            tracker,
            run_name=resolved_tracking_run_name,
            parameters=tracking_parameters,
            config=config,
            backend_name=backend_name,
        )
        _log_track_assets(tracker, track_asset_bundle)

        while environment_steps < config.ppo.total_environment_steps:
            iteration += 1
            metrics, cursor = _run_training_update(
                config=config,
                environment=environment,
                model=model,
                optimizer=optimizer,
                cursor=cursor,
                random=random,
                device=target_device,
                iteration=iteration,
                environment_steps=environment_steps,
                policy_version=policy_version,
                predictor_target_scale_m=predictor_target_scale_m,
            )
            environment_steps = metrics.environment_steps
            policy_version = metrics.policy_version
            recorder.record_step(asdict(metrics))
            _log_training_metrics(tracker, metrics)
            snapshot = _save_training_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                metrics=metrics,
                initial_checkpoint=initial_checkpoint,
                backend_name=backend_name,
                tracker=tracker,
            )
            if _should_run_evaluation(config, metrics):
                if snapshot is None:
                    raise RuntimeError("evaluation requires a versioned checkpoint")
                evaluate_policy(
                    config=config,
                    model=model,
                    environment_factory=(
                        None
                        if evaluation_uses_training_environment
                        else evaluation_environment_factory
                    ),
                    environment=(environment if evaluation_uses_training_environment else None),
                    device=target_device,
                    training_run_directory=recorder.run_directory,
                    checkpoint_path=snapshot.checkpoint_path,
                    checkpoint_update=metrics.iteration,
                    environment_steps=metrics.environment_steps,
                    backend_name=backend_name,
                    tracker=tracker,
                )
                evaluations_completed += 1

                if evaluation_uses_training_environment:
                    observation, info = environment.reset(
                        seed=config.experiment.seed,
                    )
                    cursor = _CollectionCursor(
                        observation=observation,
                        info=info,
                        trajectory_id=cursor.trajectory_id + 1,
                        episodes_completed=cursor.episodes_completed,
                    )

        run_status = "FINISHED"

    except KeyboardInterrupt:
        run_status = "KILLED"
        raise

    finally:
        recorder.close()
        _end_tracking_run(tracker, status=run_status)

    result = PPOTrainingResult(
        run_directory=recorder.run_directory,
        checkpoint_path=checkpoint_path,
        iterations=iteration,
        environment_steps=environment_steps,
        policy_version=policy_version,
        episodes_completed=cursor.episodes_completed,
        evaluations_completed=evaluations_completed,
    )
    _write_training_summary(
        recorder=recorder,
        result=result,
        config=config,
        backend_name=backend_name,
    )
    return result


def _validate_training_configuration(
    config: ProjectConfig,
    *,
    evaluation_environment_factory: EnvironmentFactory | None,
    evaluation_uses_training_environment: bool,
) -> None:
    """Reject configurations that do not match the stabilized PPO contract."""

    supported_methods = {
        STABILIZED_PPO_METHOD_VARIANT,
        POLICY_PROTECTED_PPO_METHOD_VARIANT,
    }
    if config.ppo.method_variant not in supported_methods:
        raise ValueError(f"train_ppo only supports {sorted(supported_methods)!r}")
    if config.reference.mode != "fixed":
        raise ValueError(f"{config.ppo.method_variant!r} requires reference.mode='fixed'")
    if (
        config.evaluation.enabled
        and evaluation_environment_factory is None
        and not evaluation_uses_training_environment
    ):
        raise ValueError(
            "evaluation.enabled requires either a separate evaluation "
            "environment factory or sequential reuse of the training environment"
        )
    if evaluation_environment_factory is not None and evaluation_uses_training_environment:
        raise ValueError(
            "evaluation cannot use both a separate environment factory and the training environment"
        )


def _whole_step_count(name: str, value: object) -> int:
    """Validate and normalize one discrete step count."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be an integer step count, got {value!r} ({type(value).__name__})"
        )
    if not float(value).is_integer():
        raise ValueError(f"{name} must contain a whole number, got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _log_track_assets(
    tracker: ExperimentTracker | None,
    bundle: TrackAssetBundle | None,
) -> None:
    """Log static geometry and reference once on the parent training run."""

    if tracker is None or not tracker.active or bundle is None:
        return
    tracker.log_artifact(bundle.paths.geometry_path, artifact_path="track")
    tracker.log_artifact(bundle.paths.metadata_path, artifact_path="track")
    tracker.log_artifact(bundle.paths.reference_path, artifact_path="reference")


def _initialise_training_model(
    config: ProjectConfig,
    device: torch.device,
) -> tuple[DriverNetwork, torch.optim.Optimizer, str]:
    """Create the model and optimizer and apply an optional warm start."""

    torch.manual_seed(config.experiment.seed)  # pyright: ignore[reportUnknownMemberType]
    np.random.seed(config.experiment.seed)
    torch.use_deterministic_algorithms(True)
    model = build_driver_network(config).to(device)
    initial_checkpoint = config.ppo.initial_checkpoint_path.strip()

    if initial_checkpoint:
        source = Path(initial_checkpoint)
        if not source.is_file():
            raise FileNotFoundError(f"initial PPO checkpoint does not exist: {source}")
        _, checkpoint_configuration = load_checkpoint(
            source,
            model=model,
            map_location=device,
        )
        validate_initial_checkpoint(checkpoint_configuration, config=config)
        with torch.no_grad():
            # BC checkpoints contain the training-time log standard deviation.
            # PPO owns its exploration level explicitly, so reset it after the
            # policy weights have been loaded.
            model.action_log_std.fill_(config.ppo.initial_log_std_after_bc)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.ppo.learning_rate,
    )
    return model, optimizer, initial_checkpoint


def _prepare_tracking_run(
    *,
    config: ProjectConfig,
    recorder: ExperimentRecorder,
    command: list[str],
    backend_name: str,
    target_device: torch.device,
    initial_checkpoint: str,
    tracking_run_name: str | None,
) -> tuple[dict[str, Any], str]:
    """Build the immutable parameters and resolved name for one tracking run."""

    parameters = flatten_parameters(
        {
            **config.to_dict(),
            "runtime": {
                "backend": backend_name,
                "device": str(target_device),
                "command": command,
                "initial_checkpoint": initial_checkpoint or None,
            },
        }
    )
    run_name = (
        tracking_run_name.strip()
        if tracking_run_name and tracking_run_name.strip()
        else recorder.run_directory.name
    )
    return parameters, run_name


def _start_tracking_run(
    tracker: ExperimentTracker | None,
    *,
    run_name: str,
    parameters: dict[str, Any],
    config: ProjectConfig,
    backend_name: str,
) -> None:
    """Start the optional experiment tracking run."""

    if tracker is None:
        return

    tracker.start_run(
        run_name=run_name,
        parameters=parameters,
        tags={
            "run_type": "training",
            "algorithm": "PPO",
            "method_variant": config.ppo.method_variant,
            "backend": backend_name,
        },
    )


def _run_training_update(
    *,
    config: ProjectConfig,
    environment: AssettoCorsaEnv,
    model: DriverNetwork,
    optimizer: torch.optim.Optimizer,
    cursor: _CollectionCursor,
    random: np.random.Generator,
    device: torch.device,
    iteration: int,
    environment_steps: int,
    policy_version: int,
    predictor_target_scale_m: FloatArray,
) -> tuple[PPOIterationMetrics, _CollectionCursor]:
    """Collect, post-process and optimise one complete PPO update."""

    configured_rollout_steps = _whole_step_count(
        "ppo.rollout_steps",
        config.ppo.rollout_steps,
    )
    total_environment_steps = _whole_step_count(
        "ppo.total_environment_steps",
        config.ppo.total_environment_steps,
    )
    completed_environment_steps = _whole_step_count(
        "environment_steps",
        environment_steps,
    )
    remaining_steps = total_environment_steps - completed_environment_steps
    if remaining_steps <= 0:
        raise RuntimeError("a PPO update was requested although no environment steps remain")
    rollout_step_count = min(configured_rollout_steps, remaining_steps)
    batch, next_cursor = _collect_rollout(
        environment,
        model,
        cursor,
        policy_version=policy_version,
        step_count=rollout_step_count,
        device=device,
    )
    processed = _postprocess_rollout(
        config,
        environment,
        model,
        batch,
        device,
    )
    optimisation = _optimise_ppo(
        config,
        model,
        optimizer,
        processed,
        random=random,
        device=device,
        predictor_target_scale_m=predictor_target_scale_m,
    )
    next_environment_steps = environment_steps + batch.step_count
    next_policy_version = policy_version + 1
    with torch.no_grad():
        action_log_std = model.action_log_std.detach()
        action_log_std_mean = float(action_log_std.mean().item())
        action_std_mean = float(action_log_std.exp().mean().item())
    metrics = PPOIterationMetrics(
        iteration=iteration,
        policy_version=next_policy_version,
        environment_steps=next_environment_steps,
        rollout_steps=batch.step_count,
        episodes_completed=next_cursor.episodes_completed,
        environment_reward_mean=float(batch.rewards.mean()),
        reference_distance_mean_m=float(processed.reference_distance_m.mean()),
        reference_heading_error_mean_abs_rad=float(
            processed.reference_heading_error_abs_rad.mean()
        ),
        reference_progress_alignment_mean=float(processed.reference_progress_alignment.mean()),
        reference_tracking_reward_mean=float(processed.reference_tracking_reward.mean()),
        predicted_progress_mean_m=float(processed.predicted_progress_m.mean()),
        shaped_reward_mean=float(processed.rewards.mean()),
        predictor_valid_samples=int(np.count_nonzero(processed.predictor_targets.valid_mask)),
        action_log_std_mean=action_log_std_mean,
        action_std_mean=action_std_mean,
        **asdict(optimisation),
    )
    return metrics, next_cursor


def _log_training_metrics(
    tracker: ExperimentTracker | None,
    metrics: PPOIterationMetrics,
) -> None:
    """Forward one PPO update to the active experiment tracker."""

    if tracker is None or not tracker.active:
        return

    tracker.log_metrics(
        _tracking_metrics(metrics),
        step=metrics.environment_steps,
    )


def _save_training_checkpoint(
    *,
    checkpoint_path: Path,
    model: DriverNetwork,
    optimizer: torch.optim.Optimizer,
    config: ProjectConfig,
    metrics: PPOIterationMetrics,
    initial_checkpoint: str,
    backend_name: str,
    tracker: ExperimentTracker | None,
) -> CheckpointSnapshot | None:
    """Persist the latest state and periodically publish immutable snapshots."""

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        training_state=TrainingState(
            iteration=metrics.iteration,
            environment_steps=metrics.environment_steps,
            policy_version=metrics.policy_version,
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

    if not _should_save_versioned_checkpoint(config, metrics):
        return None

    snapshot_directory = checkpoint_path.parent / "checkpoints" / f"update_{metrics.iteration:06d}"
    snapshot_directory.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_directory / "checkpoint.pt"
    shutil.copy2(checkpoint_path, snapshot_path)

    metadata_path = snapshot_directory / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "backend": backend_name,
                "environment_steps": metrics.environment_steps,
                "final_update": (metrics.environment_steps >= config.ppo.total_environment_steps),
                "iteration": metrics.iteration,
                "method_variant": config.ppo.method_variant,
                "action_method_variant": config.actions.method_variant,
                "policy_version": metrics.policy_version,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _log_checkpoint_artifacts(
        tracker,
        snapshot_path=snapshot_path,
        metadata_path=metadata_path,
        iteration=metrics.iteration,
    )
    return CheckpointSnapshot(
        directory=snapshot_directory,
        checkpoint_path=snapshot_path,
        metadata_path=metadata_path,
        iteration=metrics.iteration,
    )


def _should_save_versioned_checkpoint(
    config: ProjectConfig,
    metrics: PPOIterationMetrics,
) -> bool:
    """Return whether this update requires an immutable checkpoint snapshot."""

    is_interval = metrics.iteration % config.ppo.checkpoint_interval_updates == 0
    is_evaluation = _should_run_evaluation(config, metrics)
    is_final = metrics.environment_steps >= config.ppo.total_environment_steps
    return is_interval or is_evaluation or is_final


def _should_run_evaluation(
    config: ProjectConfig,
    metrics: PPOIterationMetrics,
) -> bool:
    """Return whether this update should create an evaluation child run."""

    if not config.evaluation.enabled:
        return False
    is_interval = metrics.iteration % config.evaluation.interval_updates == 0
    is_final = (
        config.evaluation.evaluate_final_update
        and metrics.environment_steps >= config.ppo.total_environment_steps
    )
    return is_interval or is_final


def _log_checkpoint_artifacts(
    tracker: ExperimentTracker | None,
    *,
    snapshot_path: Path,
    metadata_path: Path,
    iteration: int,
) -> None:
    """Upload one immutable checkpoint directory to the active tracker."""

    if tracker is None or not tracker.active:
        return

    artifact_path = f"checkpoints/update_{iteration:06d}"
    tracker.log_artifact(snapshot_path, artifact_path=artifact_path)
    tracker.log_artifact(metadata_path, artifact_path=artifact_path)


def _end_tracking_run(
    tracker: ExperimentTracker | None,
    *,
    status: str,
) -> None:
    """Finish the optional tracking run if it was started successfully."""

    if tracker is not None and tracker.active:
        tracker.end_run(status=status)


def _write_training_summary(
    *,
    recorder: ExperimentRecorder,
    result: PPOTrainingResult,
    config: ProjectConfig,
    backend_name: str,
) -> None:
    """Write the local summary for a completed PPO invocation."""

    recorder.write_summary(
        {
            "method_variant": config.ppo.method_variant,
            "backend": backend_name,
            "iterations": result.iterations,
            "environment_steps": result.environment_steps,
            "policy_version": result.policy_version,
            "episodes_completed": result.episodes_completed,
            "predicted_progress_reward_enabled": (config.ppo.use_predicted_progress_reward),
            "checkpoint": result.checkpoint_path.name,
            "checkpoint_interval_updates": (config.ppo.checkpoint_interval_updates),
            "versioned_checkpoints_directory": "checkpoints",
            "evaluations_completed": result.evaluations_completed,
            "evaluations_directory": "evaluations",
        }
    )


def _collect_rollout(  # noqa: PLR0915
    environment: AssettoCorsaEnv,
    model: DriverNetwork,
    cursor: _CollectionCursor,
    *,
    policy_version: int,
    step_count: int,
    device: torch.device,
) -> tuple[RolloutBatch, _CollectionCursor]:
    """Collect exactly one fragment with a single frozen policy version.

    Log probabilities and value estimates are stored at sampling time because
    PPO compares the updated policy against these fixed rollout-time values.
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
    successor_lateral_offsets: list[float] = []
    successor_heading_errors: list[float] = []
    progress_rewards: list[float] = []
    observation = cursor.observation
    info = cursor.info
    trajectory_id = cursor.trajectory_id
    episodes_completed = cursor.episodes_completed
    model.eval()

    for _ in range(step_count):
        # Context is recorded with the transition for later reward/target processing.
        observations.append(observation.copy())
        positions.append(np.asarray(info["position_xy_m"], dtype=np.float32))
        headings.append(float(info["vehicle_heading_rad"]))
        telemetry = info["telemetry"]
        simulation_times.append(float(telemetry["simulation_time_s"]))
        reference_progress.append(float(telemetry["progress_m"]))
        trajectory_ids.append(trajectory_id)

        observation_tensor = torch.as_tensor(observation, device=device).unsqueeze(0)
        with torch.no_grad():
            # Sampling and optimisation must use the same transformed density.
            output = model(observation_tensor)
            distribution = SquashedNormal(
                output.action_location,
                output.action_log_std,
            )
            action_tensor = distribution.sample()
            log_probability = distribution.log_prob(action_tensor)
        action = np.asarray(action_tensor.squeeze(0).cpu(), dtype=np.float32)
        # step() applies the action and returns the next feature vector computed
        # by AssettoCorsaEnv from the latest simulator telemetry.
        next_observation, reward, is_terminated, is_truncated, next_info = environment.step(action)
        next_telemetry = next_info["telemetry"]
        next_reward_terms = next_info["reward_terms"]
        successor_lateral_offsets.append(float(next_telemetry["lateral_offset_m"]))
        successor_heading_errors.append(float(next_telemetry["heading_error_rad"]))
        progress_rewards.append(float(next_reward_terms["reward_progress"]))
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
            successor_lateral_offset_m=np.asarray(
                successor_lateral_offsets,
                dtype=np.float32,
            ),
            successor_heading_error_rad=np.asarray(
                successor_heading_errors,
                dtype=np.float32,
            ),
            progress_reward=np.asarray(progress_rewards, dtype=np.float32),
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
    )


def _postprocess_rollout(
    config: ProjectConfig,
    environment: AssettoCorsaEnv,
    model: DriverNetwork,
    batch: RolloutBatch,
    device: torch.device,
) -> _ProcessedRollout:
    """Build predictor targets, shape rewards, and then compute GAE.

    Learned reward terms deliberately live at this rollout boundary rather
    than in simulator dynamics. This ordering guarantees that GAE sees the
    complete baseline reward while the Gymnasium environment remains purely
    telemetry- and termination-driven.
    """

    if batch.context is None:
        raise ValueError("PPO rollout requires trajectory context")
    context = batch.context
    targets = build_observed_trajectory_targets(
        positions_xy_m=context.positions_xy_m,
        heading_rad=context.vehicle_heading_rad,
        simulation_time_s=context.simulation_time_s,
        trajectory_id=context.trajectory_id,
    )
    reward_mask = np.ones(batch.step_count, dtype=np.bool_)
    reference_paths = build_reference_projection_paths(
        reference=environment.active_reference,
        progress_m=context.reference_progress_m,
        positions_xy_m=context.positions_xy_m,
        heading_rad=context.vehicle_heading_rad,
        valid_mask=reward_mask,
        time_offsets_s=PREDICTOR_TIME_OFFSETS_S,
    )
    if (
        config.rewards.reference_tracking_method_variant
        == SUCCESSOR_REFERENCE_TRACKING_METHOD_VARIANT
    ):
        successor_tracking = SuccessorReferenceTrackingReward(
            lateral_penalty_scale=config.rewards.reference_tracking_penalty_scale,
            free_distance_m=config.rewards.reference_tracking_free_distance_m,
            distance_scale_m=config.rewards.reference_tracking_distance_scale_m,
            heading_penalty_scale=config.rewards.reference_heading_penalty_scale,
            heading_scale_rad=config.rewards.reference_heading_scale_rad,
        ).compute(
            successor_lateral_offset_m=context.successor_lateral_offset_m,
            successor_heading_error_rad=context.successor_heading_error_rad,
            progress_reward=context.progress_reward,
        )
        reference_distance = successor_tracking.distance_m
        reference_heading_error = successor_tracking.absolute_heading_error_rad
        reference_progress_alignment = successor_tracking.progress_alignment
        reference_reward = successor_tracking.reward
    else:
        current_tracking = ReferenceTrackingReward(
            penalty_scale=config.rewards.reference_tracking_penalty_scale,
            free_distance_m=config.rewards.reference_tracking_free_distance_m,
            distance_scale_m=config.rewards.reference_tracking_distance_scale_m,
        ).compute(
            reference_paths.points_xz_m,
            reference_paths.valid_mask,
        )
        reference_distance = current_tracking.distance_m
        reference_heading_error = np.zeros(batch.step_count, dtype=np.float32)
        reference_progress_alignment = np.ones(batch.step_count, dtype=np.float32)
        reference_reward = current_tracking.reward
    predicted_progress = np.zeros(batch.step_count, dtype=np.float32)
    shaped_rewards = np.asarray(
        batch.rewards + reference_reward,
        dtype=np.float32,
    )
    if config.ppo.use_predicted_progress_reward:
        with torch.no_grad():
            prediction = decode_bezier_trajectory(
                model(torch.as_tensor(batch.observations, device=device)).trajectory_parameters,
                sample_count=PREDICTOR_TIME_OFFSETS_S.size,
            )
        prediction_terms = PredictedProgressReward().compute(
            np.asarray(prediction.cpu(), dtype=np.float32),
            reference_paths.points_xz_m,
            reference_paths.progress_m,
            reference_paths.valid_mask,
        )
        predicted_progress = prediction_terms.predicted_progress
        shaped_rewards = PredictedProgressReward.shape_rollout_rewards(
            shaped_rewards,
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
        reference_distance_m=reference_distance,
        reference_heading_error_abs_rad=reference_heading_error,
        reference_progress_alignment=reference_progress_alignment,
        reference_tracking_reward=reference_reward,
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
    predictor_target_scale_m: FloatArray,
) -> _OptimisationMetrics:
    """Optimise the clipped policy objective and value head on one rollout."""

    batch = rollout.batch
    observation = torch.as_tensor(batch.observations, device=device)
    action = torch.as_tensor(batch.actions, device=device)
    old_log_probability = torch.as_tensor(batch.log_probabilities, device=device)
    returns = torch.as_tensor(rollout.advantages.returns, device=device)
    advantage_values = rollout.advantages.advantages
    # Normalisation changes optimisation scale, not the underlying return targets.
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
    }
    learning_rate = _optimizer_learning_rate(optimizer)
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
            policy_protected = config.ppo.method_variant == POLICY_PROTECTED_PPO_METHOD_VARIANT
            output = model(
                observation[indices],
                detach_auxiliary_backbone=policy_protected,
            )
            distribution = SquashedNormal(
                output.action_location,
                output.action_log_std,
            )
            new_log_probability = distribution.log_prob(action[indices])
            log_ratio = new_log_probability - old_log_probability[indices]
            ratio = log_ratio.exp()
            minibatch_advantage = advantage[indices]
            # PPO uses the pessimistic of the unclipped and clipped surrogates.
            unclipped = ratio * minibatch_advantage
            clipped = (
                ratio.clamp(1.0 - config.ppo.clip_range, 1.0 + config.ppo.clip_range)
                * minibatch_advantage
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(output.value, returns[indices])
            entropy = distribution.entropy().mean()
            minibatch_valid = predictor_valid[indices]
            predictor_loss = _masked_predictor_loss(
                output.trajectory_parameters,
                predictor_target[indices],
                minibatch_valid,
                predictor_target_scale_m,
            )
            total_loss = (
                policy_loss
                + config.ppo.value_loss_coefficient * value_loss
                - config.ppo.entropy_coefficient * entropy
                + config.ppo.predictor_loss_coefficient * predictor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()  # type: ignore[no-untyped-call]
            # The protected variant prevents the initially enormous value error
            # from both changing and globally suppressing the BC policy gradient.
            _clip_ppo_gradients(
                model,
                maximum_norm=config.ppo.max_gradient_norm,
                policy_protected=policy_protected,
            )
            optimizer.step()

            for name, value in {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "predictor_loss": predictor_loss,
            }.items():
                metric_values[name].append(float(value.detach().item()))
        update_epochs_completed += 1
        approximate_kl, clip_fraction = _full_rollout_policy_change(
            model=model,
            observation=observation,
            action=action,
            old_log_probability=old_log_probability,
            clip_range=config.ppo.clip_range,
        )
        if approximate_kl > config.ppo.kl_target + config.ppo.kl_tolerance:
            break

    next_learning_rate = _next_learning_rate(
        config,
        current_learning_rate=learning_rate,
        approximate_kl=approximate_kl,
    )
    _set_optimizer_learning_rate(optimizer, next_learning_rate)
    return _OptimisationMetrics(
        **{name: float(np.mean(values)) for name, values in metric_values.items()},
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        learning_rate=learning_rate,
        next_learning_rate=next_learning_rate,
        update_epochs_completed=update_epochs_completed,
    )


def _clip_ppo_gradients(
    model: DriverNetwork,
    *,
    maximum_norm: float,
    policy_protected: bool,
) -> None:
    """Clip PPO gradients globally or by protected optimisation role."""

    if not policy_protected:
        torch.nn.utils.clip_grad_norm_(model.parameters(), maximum_norm)
        return
    policy_parameters = [
        *model.feature_extractor.parameters(),
        *model.policy_head.parameters(),
        model.action_log_std,
    ]
    torch.nn.utils.clip_grad_norm_(policy_parameters, maximum_norm)
    torch.nn.utils.clip_grad_norm_(model.value_head.parameters(), maximum_norm)
    torch.nn.utils.clip_grad_norm_(model.trajectory_predictor.parameters(), maximum_norm)


def _full_rollout_policy_change(
    *,
    model: DriverNetwork,
    observation: Tensor,
    action: Tensor,
    old_log_probability: Tensor,
    clip_range: float,
) -> tuple[float, float]:
    """Measure the updated policy against all samples from the frozen rollout."""

    with torch.no_grad():
        output = model(observation)
        distribution = SquashedNormal(
            output.action_location,
            output.action_log_std,
        )
        log_ratio = distribution.log_prob(action) - old_log_probability
        ratio = log_ratio.exp()
        approximate_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (torch.abs(ratio - 1.0) > clip_range).float().mean()
    return float(approximate_kl.item()), float(clip_fraction.item())


def _optimizer_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Read one common learning rate from all optimizer parameter groups."""

    learning_rates = {float(group["lr"]) for group in optimizer.param_groups}
    if len(learning_rates) != 1:
        raise ValueError("PPO optimizer parameter groups must share one learning rate")
    return learning_rates.pop()


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    """Apply the controller output to the next PPO iteration."""

    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _next_learning_rate(
    config: ProjectConfig,
    *,
    current_learning_rate: float,
    approximate_kl: float,
) -> float:
    """Apply the named target-band KL controller for the next iteration."""

    lower_bound = config.ppo.kl_target - config.ppo.kl_tolerance
    upper_bound = config.ppo.kl_target + config.ppo.kl_tolerance
    if approximate_kl > upper_bound:
        return max(
            config.ppo.minimum_learning_rate,
            current_learning_rate * config.ppo.kl_learning_rate_decrease_factor,
        )
    if approximate_kl < lower_bound:
        return min(
            config.ppo.learning_rate,
            current_learning_rate * config.ppo.kl_learning_rate_increase_factor,
        )
    return current_learning_rate


def _masked_predictor_loss(
    flat_prediction: Tensor,
    target_xz_m: Tensor,
    valid_mask: Tensor,
    predictor_target_scale_m: FloatArray,
) -> Tensor:
    """Evaluate auxiliary trajectory loss only where a full future exists."""

    return masked_normalized_bezier_trajectory_loss(
        flat_prediction,
        target_xz_m,
        valid_mask,
        predictor_target_scale_m,
    )


def validate_initial_checkpoint(
    configuration: dict[str, Any],
    *,
    config: ProjectConfig,
) -> None:
    """Reject warm starts with a different method or observation contract."""

    method = configuration.get("method_variant")
    if method != BASELINE_METHOD_VARIANT:
        raise ValueError(
            "initial checkpoint is not compatible: "
            f"expected method variant {BASELINE_METHOD_VARIANT!r}, got {method!r}"
        )
    action_method = configuration.get("action_method_variant")
    if action_method != ACTION_METHOD_VARIANT:
        raise ValueError(
            "initial checkpoint is not compatible: "
            f"expected action method {ACTION_METHOD_VARIANT!r}, "
            f"got {action_method!r}"
        )
    feature_names = configuration.get("observation_feature_names")
    if feature_names != list(OBSERVATION_FEATURE_NAMES):
        raise ValueError("initial checkpoint observation feature schema does not match")
    dataset_metadata = configuration.get("dataset_metadata")
    if not isinstance(dataset_metadata, dict):
        raise ValueError("initial checkpoint has no BC dataset provenance")
    expected_reference_sha256 = sha256_file(Path(config.reference.artifact_path))
    if dataset_metadata.get("reference_sha256") != expected_reference_sha256:
        raise ValueError("initial checkpoint was trained against a different fixed reference")
    project = configuration.get("project")
    if not isinstance(project, dict):
        raise ValueError("initial checkpoint has no project configuration")
    model_configuration = project.get("model")
    if not isinstance(model_configuration, dict):
        raise ValueError("initial checkpoint has no model configuration")
    bc_configuration = project.get("behavioral_cloning")
    if not isinstance(bc_configuration, dict):
        raise ValueError("initial checkpoint has no behavioral-cloning configuration")
    checkpoint_action_loss = bc_configuration.get(
        "action_loss_method_variant",
        "uniform_action_mse_v1",
    )
    if checkpoint_action_loss != config.behavioral_cloning.action_loss_method_variant:
        raise ValueError("initial checkpoint BC action-loss variant does not match")
    checkpoint_steering_weight = float(bc_configuration.get("steering_loss_weight", 1.0))
    if checkpoint_steering_weight != config.behavioral_cloning.steering_loss_weight:
        raise ValueError("initial checkpoint BC steering-loss weight does not match")
    _validate_initial_checkpoint_steering_robustness(bc_configuration, config)
    checkpoint_predictor_normalization = bc_configuration.get(
        "predictor_target_normalization_method_variant",
        "global_coordinate_scale_v1",
    )
    if (
        checkpoint_predictor_normalization
        != config.behavioral_cloning.predictor_target_normalization_method_variant
    ):
        raise ValueError("initial checkpoint predictor target normalization does not match")
    trajectory_codec = model_configuration.get("trajectory_codec")
    if trajectory_codec != BEZIER_TRAJECTORY_CODEC:
        raise ValueError(
            "initial checkpoint trajectory codec does not match: "
            f"expected {BEZIER_TRAJECTORY_CODEC!r}, "
            f"got {trajectory_codec!r}"
        )


def _validate_initial_checkpoint_steering_robustness(
    bc_configuration: dict[str, Any],
    config: ProjectConfig,
) -> None:
    checkpoint_method = bc_configuration.get(
        "steering_robustness_method_variant",
        "none_v1",
    )
    if checkpoint_method != config.behavioral_cloning.steering_robustness_method_variant:
        raise ValueError("initial checkpoint BC steering-robustness variant does not match")
    checkpoint_weight = float(bc_configuration.get("steering_robustness_loss_weight", 1.0))
    if checkpoint_weight != config.behavioral_cloning.steering_robustness_loss_weight:
        raise ValueError("initial checkpoint BC steering-robustness weight does not match")
