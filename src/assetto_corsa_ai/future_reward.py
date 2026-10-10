"""Detached look-ahead reward on the same reference line as the actual step."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from assetto_corsa_ai.predictor import trajectory_to_world
from assetto_corsa_ai.reward import RewardCalculator, calculate_progress_delta
from assetto_corsa_ai.track_projection import TrackProjector


@dataclass(frozen=True)
class FuturePredictionRewardConfig:
    enabled: bool
    performance_weight: float
    style_weight: float
    horizon_discount: float
    warmup_steps: int
    ramp_steps: int

    def validate(self) -> None:
        for name in ("performance_weight", "style_weight"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"Future prediction {name} must be finite and non-negative.")
        if not isfinite(self.horizon_discount) or not 0 < self.horizon_discount <= 1:
            raise ValueError("horizon_discount must be in (0, 1].")
        if any(type(value) is not int or value < 0 for value in (self.warmup_steps, self.ramp_steps)):
            raise ValueError("warmup_steps and ramp_steps must be non-negative integers.")

    def weight_scale(self, step: int) -> float:
        """Global training timestep schedule, also continued after resume."""
        if type(step) is not int or step < 0:
            raise ValueError("Training step must be a non-negative integer.")
        if not self.enabled or step < self.warmup_steps:
            return 0.0
        if not self.ramp_steps:
            return 1.0
        return min(1.0, (step - self.warmup_steps) / self.ramp_steps)


@dataclass(frozen=True)
class FutureReward:
    projected_s: NDArray[np.float64]
    lateral_offsets: NDArray[np.float64]
    progress_deltas: NDArray[np.float64]
    horizon_weights: NDArray[np.float64]
    progress_contributions: NDArray[np.float64]
    style_contributions: NDArray[np.float64]
    style_score: float
    performance_prediction: float
    style_prediction: float
    weight_scale: float


class FuturePredictorRewardCalculator:
    def __init__(self, config: FuturePredictionRewardConfig, step_calculator: RewardCalculator) -> None:
        config.validate()
        self.config = config
        self.step_calculator = step_calculator

    def calculate(
        self, prediction: NDArray[np.float32], *, position: NDArray[np.float64],
        forward: NDArray[np.float64], current_s: float, projector: TrackProjector, step: int,
    ) -> FutureReward:
        """Use predicted points only; actual future rollout positions are absent."""
        if not isfinite(current_s):
            raise ValueError("Current reference progress must be finite.")
        world = trajectory_to_world(prediction, position, forward)
        projections = [projector.project((float(p[0]), float(p[1]), float(p[2]))) for p in world]
        progress = np.asarray([p.distance_along_track for p in projections], dtype=np.float64)
        lateral = np.asarray([p.lateral_offset for p in projections], dtype=np.float64)
        previous = np.concatenate(([current_s], progress[:-1]))
        deltas = np.asarray([
            calculate_progress_delta(float(current), float(prior), projector.track_length)
            for current, prior in zip(progress, previous)
        ], dtype=np.float64)
        weights = self.config.horizon_discount ** np.arange(len(progress), dtype=np.float64)
        scores = np.asarray([self.step_calculator.style_score(float(offset)) for offset in lateral])
        style_score = float(np.dot(weights, scores) / weights.sum())
        scale = self.config.weight_scale(step)
        progress_terms = scale * self.config.performance_weight * weights * deltas
        style_terms = scale * self.config.style_weight * weights * scores / weights.sum()
        return FutureReward(
            projected_s=progress, lateral_offsets=lateral, progress_deltas=deltas,
            horizon_weights=weights, progress_contributions=progress_terms,
            style_contributions=style_terms, style_score=style_score,
            performance_prediction=float(progress_terms.sum()), style_prediction=scale * self.config.style_weight * style_score,
            weight_scale=scale,
        )
