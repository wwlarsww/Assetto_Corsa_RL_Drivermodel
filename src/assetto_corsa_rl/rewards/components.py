"""Reward decomposition from the thesis, split at the rollout boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.references.projection import project_points_to_polyline_progress
from assetto_corsa_rl.types import FloatArray, TerminationReason


@dataclass(frozen=True, slots=True)
class EnvironmentRewardBreakdown:
    """Reward terms available immediately after a simulator step."""

    progress: float
    termination: float

    @property
    def total(self) -> float:
        """Return the sum of immediate environment reward terms."""

        return self.progress + self.termination

    def as_dict(self) -> dict[str, float]:
        """Return stable metric names for every immediate reward term."""

        return {
            "reward_progress": self.progress,
            "reward_termination": self.termination,
            "reward_environment_total": self.total,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentReward:
    """Progress plus a penalty for safety-related termination."""

    progress_scale: float
    termination_penalty: float

    def __post_init__(self) -> None:
        if self.progress_scale <= 0:
            raise ValueError("progress_scale must be positive")
        if self.termination_penalty > 0:
            raise ValueError("termination_penalty must be non-positive")

    def compute(
        self,
        reference_progress_delta_m: float,
        termination_reason: TerminationReason | None,
    ) -> EnvironmentRewardBreakdown:
        """Compute progress reward and an optional safety penalty."""

        if not np.isfinite(reference_progress_delta_m):
            raise ValueError("reference_progress_delta_m must be finite")
        penalty = (
            self.termination_penalty
            if termination_reason is not None and termination_reason.is_safety_violation
            else 0.0
        )
        return EnvironmentRewardBreakdown(
            progress=self.progress_scale * reference_progress_delta_m,
            termination=penalty,
        )


@dataclass(frozen=True, slots=True)
class PredictedProgressRewardBreakdown:
    """Prediction-only reward data for the fixed-reference variant."""

    predicted_progress: FloatArray
    valid_mask: npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class PredictedProgressReward:
    """Project the predicted Bézier path onto the prescribed reference line."""

    def compute(
        self,
        predicted_xz_m: npt.ArrayLike,
        reference_path_xz_m: npt.ArrayLike,
        reference_progress_m: npt.ArrayLike,
        valid_mask: npt.ArrayLike | None = None,
    ) -> PredictedProgressRewardBreakdown:
        """Measure predicted path progress along the prescribed reference."""

        predicted = np.asarray(predicted_xz_m, dtype=np.float32)
        reference = np.asarray(reference_path_xz_m, dtype=np.float32)
        reference_progress = np.asarray(reference_progress_m, dtype=np.float32)
        if predicted.ndim != 3 or predicted.shape[-1] != 2 or predicted.shape[1] < 2:
            raise ValueError("predicted_xz_m must have shape (steps, horizon>=2, 2)")
        if (
            reference.ndim != 3
            or reference.shape[0] != predicted.shape[0]
            or reference.shape[-1] != 2
            or reference.shape[1] < 2
        ):
            raise ValueError("reference_path_xz_m must have shape (steps, reference_points>=2, 2)")
        if reference_progress.shape != reference.shape[:2]:
            raise ValueError("reference_progress_m must match reference_path_xz_m")
        if (
            not np.all(np.isfinite(predicted))
            or not np.all(np.isfinite(reference))
            or not np.all(np.isfinite(reference_progress))
        ):
            raise ValueError("predicted/reference paths contain non-finite values")
        mask = (
            np.ones(predicted.shape[0], dtype=np.bool_)
            if valid_mask is None
            else np.asarray(valid_mask, dtype=np.bool_)
        )
        if mask.shape != (predicted.shape[0],):
            raise ValueError("valid_mask must have shape (steps,)")
        progress = np.zeros(predicted.shape[0], dtype=np.float32)
        for index in np.flatnonzero(mask):
            projected_progress = project_points_to_polyline_progress(
                predicted[index],
                reference[index],
                reference_progress[index],
            )
            progress[index] = projected_progress[-1]
        progress = np.where(mask, progress, 0.0)
        return PredictedProgressRewardBreakdown(
            predicted_progress=np.asarray(progress, dtype=np.float32),
            valid_mask=mask,
        )

    @staticmethod
    def shape_rollout_rewards(
        environment_rewards: npt.ArrayLike,
        prediction: PredictedProgressRewardBreakdown,
        *,
        predicted_progress_scale: float,
    ) -> FloatArray:
        """Add prediction-only rewards after collection and before GAE."""

        base = np.asarray(environment_rewards, dtype=np.float32)
        if base.shape != prediction.predicted_progress.shape:
            raise ValueError("environment and prediction rewards must have equal shape")
        if predicted_progress_scale < 0:
            raise ValueError("predicted_progress_scale must be non-negative")
        return np.asarray(
            base + predicted_progress_scale * prediction.predicted_progress,
            dtype=np.float32,
        )
