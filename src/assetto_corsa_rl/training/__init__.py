"""Behavioural cloning, rollout contracts and synchronous PPO training."""

from assetto_corsa_rl.training.behavioral_cloning import (
    BCEvaluationMetrics,
    BCTrainingResult,
    evaluate_behavioral_cloning,
    train_behavioral_cloning,
)
from assetto_corsa_rl.training.gae import AdvantageBatch, compute_gae
from assetto_corsa_rl.training.ppo import (
    PPOIterationMetrics,
    PPOTrainingResult,
    train_ppo,
)
from assetto_corsa_rl.training.rollout import (
    OnPolicyCollector,
    RolloutBatch,
    TrajectoryContext,
    TrajectoryFragment,
)
from assetto_corsa_rl.training.targets import (
    FutureTrajectoryTargets,
    ReferenceProjectionPaths,
    build_observed_trajectory_targets,
    build_reference_projection_paths,
    build_reference_trajectory_targets,
)

__all__ = [
    "AdvantageBatch",
    "BCEvaluationMetrics",
    "BCTrainingResult",
    "FutureTrajectoryTargets",
    "OnPolicyCollector",
    "PPOIterationMetrics",
    "PPOTrainingResult",
    "ReferenceProjectionPaths",
    "RolloutBatch",
    "TrajectoryContext",
    "TrajectoryFragment",
    "build_observed_trajectory_targets",
    "build_reference_projection_paths",
    "build_reference_trajectory_targets",
    "compute_gae",
    "evaluate_behavioral_cloning",
    "train_behavioral_cloning",
    "train_ppo",
]
