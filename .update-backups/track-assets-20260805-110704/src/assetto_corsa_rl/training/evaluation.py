"""Deterministic checkpoint evaluation and reproducible artefact export."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Independent, Normal

from assetto_corsa_rl.ac_integration.track_assets import (
    TrackAssetBundle,
    boundary_preview_world,
    ensure_track_assets,
    load_npz_arrays,
    local_xz_to_world_xy,
    reference_preview_world,
)
from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.envs.environment import AssettoCorsaEnv
from assetto_corsa_rl.models.bezier import decode_bezier_trajectory
from assetto_corsa_rl.models.driver_network import DriverNetwork
from assetto_corsa_rl.training.tracking import ExperimentTracker, MetricValue
from assetto_corsa_rl.types import REFERENCE_TIME_OFFSETS_S, FloatArray

EnvironmentFactory = Callable[[], AssettoCorsaEnv]
TRACK_BOUNDARY_PREVIEW_DISTANCES_M = np.asarray(
    [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0],
    dtype=np.float32,
)


@dataclass(frozen=True, slots=True)
class EvaluationEpisode:
    """Aggregate outcome of one evaluation episode."""

    episode: int
    seed: int
    reward: float
    steps: int
    terminated: bool
    truncated: bool
    final_progress_m: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Completed evaluation plus its local reproducible artefacts."""

    output_directory: Path
    summary_path: Path
    episodes_path: Path
    trajectories_path: Path | None
    episodes: tuple[EvaluationEpisode, ...]
    metrics: dict[str, MetricValue]


@dataclass(slots=True)
class _TrajectoryBuffer:
    """Concatenated step-level data across variable-length episodes."""

    observations: list[FloatArray]
    actions: list[FloatArray]
    rewards: list[float]
    positions_xy_m: list[FloatArray]
    vehicle_heading_rad: list[float]
    simulation_time_s: list[float]
    reference_progress_m: list[float]
    bezier_prediction_local_xz_m: list[FloatArray]
    bezier_prediction_world_xy_m: list[FloatArray]
    reference_preview_world_xy_m: list[FloatArray]
    left_boundary_preview_world_xy_m: list[FloatArray]
    right_boundary_preview_world_xy_m: list[FloatArray]
    episode_ids: list[int]
    episode_steps: list[int]

    @classmethod
    def create(cls) -> _TrajectoryBuffer:
        """Create an empty evaluation trajectory buffer."""

        return cls(
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )


@dataclass(frozen=True, slots=True)
class _StaticEvaluationGeometry:
    """Unnormalised geometry used to create step-level previews."""

    track_assets: TrackAssetBundle | None
    geometry_arrays: dict[str, np.ndarray]
    reference_arrays: dict[str, np.ndarray]


def evaluate_policy(
    *,
    config: ProjectConfig,
    model: DriverNetwork,
    device: torch.device,
    training_run_directory: Path,
    checkpoint_path: Path,
    checkpoint_update: int,
    environment_steps: int,
    backend_name: str,
    tracker: ExperimentTracker | None,
    environment_factory: EnvironmentFactory | None = None,
    environment: AssettoCorsaEnv | None = None,
) -> EvaluationResult:
    """Evaluate one checkpoint and log a nested MLflow run.

    Mock evaluation normally owns a fresh environment. The live Assetto Corsa
    backend reuses the connected training environment sequentially because only
    one simulator session and controller stream exist.
    """

    if (environment is None) == (environment_factory is None):
        raise ValueError("provide exactly one of environment or environment_factory")

    output_directory = (
        training_run_directory / "evaluations" / f"update_{checkpoint_update:06d}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    parent_run_id = tracker.run_id if tracker is not None else None
    child_run_started = _start_evaluation_run(
        tracker,
        config=config,
        checkpoint_path=checkpoint_path,
        checkpoint_update=checkpoint_update,
        environment_steps=environment_steps,
        backend_name=backend_name,
        parent_run_id=parent_run_id,
    )
    run_status = "FAILED"
    owns_environment = environment is None
    evaluation_environment = (
        environment_factory() if environment_factory is not None else environment
    )
    if evaluation_environment is None:
        raise RuntimeError("evaluation environment could not be created")

    static_geometry = _load_static_evaluation_geometry(
        config,
        backend_name=backend_name,
    )

    try:
        fork_devices = (
            [device.index]
            if device.type == "cuda" and device.index is not None
            else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(config.experiment.seed + config.evaluation.seed_offset)
            episodes, trajectories = _run_episodes(
                config=config,
                model=model,
                environment=evaluation_environment,
                device=device,
                static_geometry=static_geometry,
            )
        metrics = _evaluation_metrics(episodes, checkpoint_update)
        result = _write_evaluation_artifacts(
            config=config,
            output_directory=output_directory,
            checkpoint_path=checkpoint_path,
            checkpoint_update=checkpoint_update,
            environment_steps=environment_steps,
            backend_name=backend_name,
            episodes=episodes,
            trajectories=trajectories,
            metrics=metrics,
            static_geometry=static_geometry,
        )
        _log_evaluation_result(
            tracker,
            result=result,
            checkpoint_update=checkpoint_update,
            child_run_started=child_run_started,
        )
        run_status = "FINISHED"
        return result
    except KeyboardInterrupt:
        run_status = "KILLED"
        raise
    finally:
        if owns_environment:
            evaluation_environment.close()
        if child_run_started and tracker is not None:
            tracker.end_run(status=run_status)


def _load_static_evaluation_geometry(
    config: ProjectConfig,
    *,
    backend_name: str,
) -> _StaticEvaluationGeometry:
    if backend_name != "assetto-corsa":
        return _StaticEvaluationGeometry(None, {}, {})
    bundle = ensure_track_assets(config)
    return _StaticEvaluationGeometry(
        track_assets=bundle,
        geometry_arrays=load_npz_arrays(bundle.paths.geometry_path),
        reference_arrays=load_npz_arrays(bundle.paths.reference_path),
    )


def _start_evaluation_run(
    tracker: ExperimentTracker | None,
    *,
    config: ProjectConfig,
    checkpoint_path: Path,
    checkpoint_update: int,
    environment_steps: int,
    backend_name: str,
    parent_run_id: str | None,
) -> bool:
    """Start a nested evaluation run when a parent tracker run is active."""

    if tracker is None or not tracker.active:
        return False
    run_id = tracker.start_run(
        run_name=f"eval_update_{checkpoint_update:06d}",
        parameters={
            "evaluation.checkpoint_update": checkpoint_update,
            "evaluation.environment_steps": environment_steps,
            "evaluation.episodes": config.evaluation.episodes,
            "evaluation.deterministic": config.evaluation.deterministic,
            "evaluation.seed_offset": config.evaluation.seed_offset,
            "evaluation.record_trajectories": config.evaluation.record_trajectories,
            "evaluation.checkpoint_local_path": str(checkpoint_path),
            "evaluation.checkpoint_artifact_path": (
                f"checkpoints/update_{checkpoint_update:06d}/checkpoint.pt"
            ),
        },
        tags={
            "run_type": "evaluation",
            "algorithm": "PPO",
            "method_variant": config.ppo.method_variant,
            "backend": backend_name,
            "checkpoint_update": str(checkpoint_update),
            "parent_training_run_id": parent_run_id or "",
        },
        nested=True,
    )
    return run_id is not None and tracker.run_id == run_id


def _run_episodes(
    *,
    config: ProjectConfig,
    model: DriverNetwork,
    environment: AssettoCorsaEnv,
    device: torch.device,
    static_geometry: _StaticEvaluationGeometry,
) -> tuple[tuple[EvaluationEpisode, ...], _TrajectoryBuffer]:
    """Run configured evaluation episodes without updating the model."""

    model.eval()
    episodes: list[EvaluationEpisode] = []
    trajectories = _TrajectoryBuffer.create()
    for episode_index in range(config.evaluation.episodes):
        seed = config.experiment.seed + config.evaluation.seed_offset + episode_index
        episodes.append(
            _run_episode(
                config=config,
                model=model,
                environment=environment,
                device=device,
                episode_index=episode_index,
                seed=seed,
                trajectories=trajectories,
                static_geometry=static_geometry,
            )
        )
    return tuple(episodes), trajectories


def _run_episode(
    *,
    config: ProjectConfig,
    model: DriverNetwork,
    environment: AssettoCorsaEnv,
    device: torch.device,
    episode_index: int,
    seed: int,
    trajectories: _TrajectoryBuffer,
    static_geometry: _StaticEvaluationGeometry,
) -> EvaluationEpisode:
    """Execute and optionally record one bounded evaluation episode."""

    observation, info = environment.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    truncated = False
    final_progress_m = _progress_from_info(info)
    steps = 0

    for step_index in range(config.experiment.max_episode_steps):
        observation_tensor = torch.as_tensor(observation, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(observation_tensor)
            action_tensor = _evaluation_action(
                output.action_mean,
                output.action_log_std,
                deterministic=config.evaluation.deterministic,
            )
            prediction = decode_bezier_trajectory(
                output.trajectory_parameters,
                sample_count=REFERENCE_TIME_OFFSETS_S.size,
            )

        action = np.asarray(action_tensor.squeeze(0).cpu(), dtype=np.float32)
        if config.evaluation.record_trajectories:
            _append_trajectory_step(
                trajectories,
                observation=observation,
                action=action,
                info=info,
                prediction_local_xz_m=np.asarray(
                    prediction.squeeze(0).cpu(),
                    dtype=np.float32,
                ),
                episode_index=episode_index,
                step_index=step_index,
                static_geometry=static_geometry,
            )

        next_observation, reward, terminated, truncated, next_info = environment.step(
            action
        )
        total_reward += float(reward)
        steps = step_index + 1
        final_progress_m = _progress_from_info(next_info)
        if config.evaluation.record_trajectories:
            trajectories.rewards[-1] = float(reward)
        observation, info = next_observation, next_info
        if terminated or truncated:
            break
    else:
        truncated = True

    return EvaluationEpisode(
        episode=episode_index,
        seed=seed,
        reward=total_reward,
        steps=steps,
        terminated=bool(terminated),
        truncated=bool(truncated),
        final_progress_m=final_progress_m,
    )


def _evaluation_action(
    action_mean: torch.Tensor,
    action_log_std: torch.Tensor,
    *,
    deterministic: bool,
) -> torch.Tensor:
    """Choose the policy mean or sample from its Gaussian distribution."""

    if deterministic:
        return action_mean
    distribution: Independent[Normal] = Independent(
        Normal(action_mean, action_log_std.exp()),
        1,
    )
    return distribution.sample()


def _append_trajectory_step(
    trajectories: _TrajectoryBuffer,
    *,
    observation: FloatArray,
    action: FloatArray,
    info: dict[str, Any],
    prediction_local_xz_m: FloatArray,
    episode_index: int,
    step_index: int,
    static_geometry: _StaticEvaluationGeometry,
) -> None:
    """Append one policy decision and unnormalised simulator geometry."""

    telemetry = info["telemetry"]
    position = np.asarray(info["position_xy_m"], dtype=np.float32)
    heading = float(info["vehicle_heading_rad"])
    progress = float(telemetry["progress_m"])

    prediction_world = local_xz_to_world_xy(
        prediction_local_xz_m,
        position,
        heading,
    )
    reference_preview = _empty_points(REFERENCE_TIME_OFFSETS_S.size)
    left_preview = _empty_points(TRACK_BOUNDARY_PREVIEW_DISTANCES_M.size)
    right_preview = _empty_points(TRACK_BOUNDARY_PREVIEW_DISTANCES_M.size)
    if static_geometry.reference_arrays:
        reference_preview = reference_preview_world(
            static_geometry.reference_arrays,
            progress,
            np.asarray(REFERENCE_TIME_OFFSETS_S, dtype=np.float32),
        )
    if static_geometry.geometry_arrays:
        left_preview, right_preview = boundary_preview_world(
            static_geometry.geometry_arrays,
            progress,
            TRACK_BOUNDARY_PREVIEW_DISTANCES_M,
        )

    trajectories.observations.append(observation.copy())
    trajectories.actions.append(action.copy())
    trajectories.rewards.append(0.0)
    trajectories.positions_xy_m.append(position)
    trajectories.vehicle_heading_rad.append(heading)
    trajectories.simulation_time_s.append(float(telemetry["simulation_time_s"]))
    trajectories.reference_progress_m.append(progress)
    trajectories.bezier_prediction_local_xz_m.append(
        prediction_local_xz_m.copy()
    )
    trajectories.bezier_prediction_world_xy_m.append(prediction_world)
    trajectories.reference_preview_world_xy_m.append(reference_preview)
    trajectories.left_boundary_preview_world_xy_m.append(left_preview)
    trajectories.right_boundary_preview_world_xy_m.append(right_preview)
    trajectories.episode_ids.append(episode_index)
    trajectories.episode_steps.append(step_index)


def _empty_points(count: int) -> np.ndarray:
    return np.full((count, 2), np.nan, dtype=np.float32)


def _progress_from_info(info: dict[str, Any]) -> float:
    """Read current reference progress from environment info."""

    return float(info["telemetry"]["progress_m"])


def _evaluation_metrics(
    episodes: tuple[EvaluationEpisode, ...],
    checkpoint_update: int,
) -> dict[str, MetricValue]:
    """Aggregate episode outcomes into consistently named metrics."""

    rewards = np.asarray([episode.reward for episode in episodes], dtype=np.float64)
    lengths = np.asarray([episode.steps for episode in episodes], dtype=np.float64)
    progress = np.asarray(
        [episode.final_progress_m for episode in episodes],
        dtype=np.float64,
    )
    return {
        "eval/reward/mean": float(rewards.mean()),
        "eval/reward/std": float(rewards.std()),
        "eval/episode_length/mean": float(lengths.mean()),
        "eval/terminated/count": sum(episode.terminated for episode in episodes),
        "eval/truncated/count": sum(episode.truncated for episode in episodes),
        "eval/progress/final_mean_m": float(progress.mean()),
        "eval/checkpoint/update": checkpoint_update,
    }


def _write_evaluation_artifacts(
    *,
    config: ProjectConfig,
    output_directory: Path,
    checkpoint_path: Path,
    checkpoint_update: int,
    environment_steps: int,
    backend_name: str,
    episodes: tuple[EvaluationEpisode, ...],
    trajectories: _TrajectoryBuffer,
    metrics: dict[str, MetricValue],
    static_geometry: _StaticEvaluationGeometry,
) -> EvaluationResult:
    """Write JSON, CSV and optional NPZ artefacts for one evaluation."""

    episodes_path = output_directory / "episodes.csv"
    with episodes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(episodes[0])))
        writer.writeheader()
        writer.writerows(asdict(episode) for episode in episodes)

    trajectories_path: Path | None = None
    if config.evaluation.record_trajectories:
        trajectories_path = output_directory / "trajectories.npz"
        asset_bundle = static_geometry.track_assets
        np.savez_compressed(
            trajectories_path,
            observations=np.asarray(trajectories.observations, dtype=np.float32),
            actions=np.asarray(trajectories.actions, dtype=np.float32),
            rewards=np.asarray(trajectories.rewards, dtype=np.float32),
            positions_xy_m=np.asarray(trajectories.positions_xy_m, dtype=np.float32),
            vehicle_heading_rad=np.asarray(
                trajectories.vehicle_heading_rad,
                dtype=np.float32,
            ),
            simulation_time_s=np.asarray(
                trajectories.simulation_time_s,
                dtype=np.float32,
            ),
            reference_progress_m=np.asarray(
                trajectories.reference_progress_m,
                dtype=np.float32,
            ),
            bezier_prediction_points=np.asarray(
                trajectories.bezier_prediction_local_xz_m,
                dtype=np.float32,
            ),
            bezier_prediction_local_xz_m=np.asarray(
                trajectories.bezier_prediction_local_xz_m,
                dtype=np.float32,
            ),
            bezier_prediction_world_xy_m=np.asarray(
                trajectories.bezier_prediction_world_xy_m,
                dtype=np.float32,
            ),
            reference_preview_world_xy_m=np.asarray(
                trajectories.reference_preview_world_xy_m,
                dtype=np.float32,
            ),
            left_boundary_preview_world_xy_m=np.asarray(
                trajectories.left_boundary_preview_world_xy_m,
                dtype=np.float32,
            ),
            right_boundary_preview_world_xy_m=np.asarray(
                trajectories.right_boundary_preview_world_xy_m,
                dtype=np.float32,
            ),
            reference_time_offsets_s=np.asarray(
                REFERENCE_TIME_OFFSETS_S,
                dtype=np.float32,
            ),
            boundary_preview_distances_m=TRACK_BOUNDARY_PREVIEW_DISTANCES_M,
            episode_ids=np.asarray(trajectories.episode_ids, dtype=np.int64),
            episode_steps=np.asarray(trajectories.episode_steps, dtype=np.int64),
            coordinate_frame=np.asarray("assetto_corsa_world_xz_stored_as_xy"),
            local_trajectory_axes=np.asarray("x=lateral_left,z=forward"),
            observations_are_normalized=np.asarray(True),
            track_geometry_sha256=np.asarray(
                asset_bundle.geometry_sha256 if asset_bundle is not None else ""
            ),
            reference_sha256=np.asarray(
                asset_bundle.reference_sha256 if asset_bundle is not None else ""
            ),
            source_fast_lane_sha256=np.asarray(
                asset_bundle.source_sha256 if asset_bundle is not None else ""
            ),
        )

    asset_bundle = static_geometry.track_assets
    summary_path = output_directory / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "backend": backend_name,
                "checkpoint_local_path": str(checkpoint_path),
                "checkpoint_update": checkpoint_update,
                "deterministic": config.evaluation.deterministic,
                "environment_steps": environment_steps,
                "episodes": config.evaluation.episodes,
                "metrics": metrics,
                "record_trajectories": config.evaluation.record_trajectories,
                "seed_offset": config.evaluation.seed_offset,
                "track_geometry_sha256": (
                    asset_bundle.geometry_sha256 if asset_bundle is not None else None
                ),
                "reference_sha256": (
                    asset_bundle.reference_sha256 if asset_bundle is not None else None
                ),
                "source_fast_lane_sha256": (
                    asset_bundle.source_sha256 if asset_bundle is not None else None
                ),
                "coordinate_frame": "assetto_corsa_world_xz_stored_as_xy",
                "observations_are_normalized": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return EvaluationResult(
        output_directory=output_directory,
        summary_path=summary_path,
        episodes_path=episodes_path,
        trajectories_path=trajectories_path,
        episodes=episodes,
        metrics=metrics,
    )


def _log_evaluation_result(
    tracker: ExperimentTracker | None,
    *,
    result: EvaluationResult,
    checkpoint_update: int,
    child_run_started: bool,
) -> None:
    """Log evaluation metrics and artefacts to the active child run."""

    if tracker is None or not child_run_started or not tracker.active:
        return
    tracker.log_metrics(result.metrics, step=checkpoint_update)
    tracker.log_artifact(result.summary_path)
    tracker.log_artifact(result.episodes_path)
    if result.trajectories_path is not None:
        tracker.log_artifact(result.trajectories_path)
