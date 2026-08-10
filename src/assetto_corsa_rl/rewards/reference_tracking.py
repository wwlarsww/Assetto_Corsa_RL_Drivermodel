"""Reference-line distance shaping for PPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.types import FloatArray


@dataclass(frozen=True, slots=True)
class ReferenceTrackingRewardBreakdown:
    """Per-step reference distance and its non-positive shaping reward."""

    distance_m: FloatArray
    reward: FloatArray


@dataclass(frozen=True, slots=True)
class ReferenceTrackingReward:
    """Penalise distance outside a free corridor around the reference line.

    The reward is zero inside ``free_distance_m`` and decreases quadratically
    outside it.  It never gives a positive reward for merely standing still on
    the line, so forward progress remains the task's positive driving signal.
    """

    penalty_scale: float
    free_distance_m: float
    distance_scale_m: float

    def __post_init__(self) -> None:
        if self.penalty_scale < 0.0:
            raise ValueError("penalty_scale must be non-negative")
        if self.free_distance_m < 0.0:
            raise ValueError("free_distance_m must be non-negative")
        if self.distance_scale_m <= 0.0:
            raise ValueError("distance_scale_m must be positive")

    def compute(
        self,
        reference_path_xz_m: npt.ArrayLike,
        valid_mask: npt.ArrayLike | None = None,
    ) -> ReferenceTrackingRewardBreakdown:
        """Compute distance from the vehicle to the current reference projection."""

        reference = np.asarray(reference_path_xz_m, dtype=np.float32)
        if reference.ndim != 3 or reference.shape[1] < 1 or reference.shape[-1] != 2:
            raise ValueError("reference_path_xz_m must have shape (steps, reference_points>=1, 2)")
        if not np.all(np.isfinite(reference)):
            raise ValueError("reference_path_xz_m contains non-finite values")

        mask = (
            np.ones(reference.shape[0], dtype=np.bool_)
            if valid_mask is None
            else np.asarray(valid_mask, dtype=np.bool_)
        )
        if mask.shape != (reference.shape[0],):
            raise ValueError("valid_mask must have shape (steps,)")

        # build_reference_projection_paths() stores the current reference
        # projection as the first ego-frame point.  Its norm is therefore the
        # current vehicle-to-reference distance, independent of vehicle heading.
        distance = np.linalg.norm(reference[:, 0, :], axis=-1).astype(np.float32)
        excess = np.maximum(distance - self.free_distance_m, 0.0)
        reward = -self.penalty_scale * np.square(excess / self.distance_scale_m)
        distance = np.where(mask, distance, 0.0).astype(np.float32)
        reward = np.where(mask, reward, 0.0).astype(np.float32)
        return ReferenceTrackingRewardBreakdown(
            distance_m=distance,
            reward=reward,
        )


@dataclass(frozen=True, slots=True)
class SuccessorReferenceRewardBreakdown:
    """Successor-state line, heading and progress-alignment diagnostics."""

    distance_m: FloatArray
    absolute_heading_error_rad: FloatArray
    progress_alignment: FloatArray
    reward: FloatArray


@dataclass(frozen=True, slots=True)
class SuccessorReferenceTrackingReward:
    """Gate positive progress and penalise successor line/heading error.

    Unlike the legacy current-state distance term, this reward is evaluated at
    the state reached by the sampled action.  Positive progress is retained
    only while the successor stays close and aligned with the fixed reference.
    """

    lateral_penalty_scale: float
    free_distance_m: float
    distance_scale_m: float
    heading_penalty_scale: float
    heading_scale_rad: float

    def __post_init__(self) -> None:
        for name, value in {
            "lateral_penalty_scale": self.lateral_penalty_scale,
            "free_distance_m": self.free_distance_m,
            "heading_penalty_scale": self.heading_penalty_scale,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.distance_scale_m <= 0.0 or self.heading_scale_rad <= 0.0:
            raise ValueError("distance and heading scales must be positive")

    def compute(
        self,
        *,
        successor_lateral_offset_m: npt.ArrayLike,
        successor_heading_error_rad: npt.ArrayLike,
        progress_reward: npt.ArrayLike,
    ) -> SuccessorReferenceRewardBreakdown:
        """Return shaping attached to the action that reached each successor."""

        lateral = np.asarray(successor_lateral_offset_m, dtype=np.float32)
        heading = np.asarray(successor_heading_error_rad, dtype=np.float32)
        progress = np.asarray(progress_reward, dtype=np.float32)
        if lateral.ndim != 1 or heading.shape != lateral.shape or progress.shape != lateral.shape:
            raise ValueError("successor lateral, heading and progress arrays must have equal shape")
        if not (
            np.all(np.isfinite(lateral))
            and np.all(np.isfinite(heading))
            and np.all(np.isfinite(progress))
        ):
            raise ValueError("successor reference reward inputs contain non-finite values")

        distance = np.abs(lateral)
        absolute_heading = np.abs(heading)
        excess = np.maximum(distance - self.free_distance_m, 0.0)
        scaled_distance = excess / self.distance_scale_m
        distance_alignment = np.exp(-0.5 * np.square(scaled_distance))
        heading_alignment = np.clip(np.cos(absolute_heading), 0.0, 1.0)
        progress_alignment = distance_alignment * heading_alignment
        positive_progress = np.maximum(progress, 0.0)
        alignment_penalty = positive_progress * (progress_alignment - 1.0)
        lateral_penalty = -self.lateral_penalty_scale * np.square(scaled_distance)
        heading_penalty = -self.heading_penalty_scale * np.square(
            absolute_heading / self.heading_scale_rad
        )
        reward = alignment_penalty + lateral_penalty + heading_penalty
        return SuccessorReferenceRewardBreakdown(
            distance_m=np.asarray(distance, dtype=np.float32),
            absolute_heading_error_rad=np.asarray(absolute_heading, dtype=np.float32),
            progress_alignment=np.asarray(progress_alignment, dtype=np.float32),
            reward=np.asarray(reward, dtype=np.float32),
        )
