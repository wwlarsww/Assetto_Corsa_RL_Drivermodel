"""Reward calculation for the RL driving agent."""

from dataclasses import dataclass
import math

from assetto_corsa_ai.termination import (
    TerminationReason,
    TerminationResult,
)


@dataclass(frozen=True)
class Reward:
    """Reward breakdown for one agent step.

    Prediction-based reward terms are reserved for the future auxiliary
    trajectory predictor and remain zero until that component is added.
    """

    total: float

    performance_step: float
    style_step: float

    performance_prediction: float
    style_prediction: float

    termination_penalty: float


def calculate_progress_delta(
    current_s: float,
    previous_s: float,
    track_length: float,
) -> float:
    """Calculate signed progress along a closed reference line.

    Positive values indicate forward progress.
    Negative values indicate backwards progress.

    Start/finish wraparound is handled automatically.
    """

    if track_length <= 0.0:
        raise ValueError("track_length must be positive.")

    delta_s = current_s - previous_s
    half_track_length = track_length / 2.0

    # Crossing start/finish backwards.
    if delta_s > half_track_length:
        delta_s -= track_length

    # Crossing start/finish forwards.
    elif delta_s < -half_track_length:
        delta_s += track_length

    return delta_s


class RewardCalculator:
    """Calculate performance, imitation and termination rewards.

    Current formulation:

        r_t =
            r_performance_step
            + lambda_style * r_style_step
            + termination_penalty

    Future formulation:

        r_t =
            r_performance_step
            + r_performance_prediction
            + lambda_style * (
                r_style_step
                + r_style_prediction
            )
            + termination_penalty
    """

    def __init__(
        self,
        performance_weight: float = 1.0,
        style_weight: float = 1.0,
        style_sigma: float = 1.0,
        off_track_penalty: float = -100.0,
        stuck_penalty: float = -100.0,
        collision_penalty: float = -100.0,
    ) -> None:
        if performance_weight < 0.0:
            raise ValueError("performance_weight must not be negative.")

        if style_weight < 0.0:
            raise ValueError("style_weight must not be negative.")

        if style_sigma <= 0.0:
            raise ValueError("style_sigma must be positive.")

        if off_track_penalty > 0.0:
            raise ValueError("off_track_penalty must not be positive.")

        if stuck_penalty > 0.0:
            raise ValueError("stuck_penalty must not be positive.")

        if collision_penalty > 0.0:
            raise ValueError("collision_penalty must not be positive.")

        self._performance_weight = performance_weight

        # Corresponds to lambda_s in the reward formulation.
        self._style_weight = style_weight
        self._style_sigma = style_sigma

        self._off_track_penalty = off_track_penalty
        self._stuck_penalty = stuck_penalty
        self._collision_penalty = collision_penalty

    def calculate(
        self,
        delta_s: float,
        lateral_offset: float,
        termination: TerminationResult,
    ) -> Reward:
        """Calculate the reward for one agent step.

        Args:
            delta_s:
                Signed progress along the reference line since the
                previous agent step [m].

            lateral_offset:
                Signed lateral distance from the reference line [m].
                Only the magnitude affects the imitation reward.

            termination:
                Result of the episode termination check.
        """

        performance_step = self._performance_step_reward(
            delta_s=delta_s,
        )

        style_step = self._style_step_reward(
            lateral_offset=lateral_offset,
        )

        # Reserved for the future auxiliary trajectory predictor.
        performance_prediction = 0.0
        style_prediction = 0.0

        termination_penalty = self._termination_penalty(
            reason=termination.reason,
        )

        total = (
            performance_step
            + performance_prediction
            + self._style_weight * (style_step + style_prediction)
            + termination_penalty
        )

        return Reward(
            total=total,
            performance_step=performance_step,
            style_step=style_step,
            performance_prediction=performance_prediction,
            style_prediction=style_prediction,
            termination_penalty=termination_penalty,
        )

    def _performance_step_reward(
        self,
        delta_s: float,
    ) -> float:
        """Reward signed progress along the reference line."""

        return self._performance_weight * delta_s

    def _style_step_reward(
        self,
        lateral_offset: float,
    ) -> float:
        """Reward proximity to the reference line.

        A Gaussian function gives the maximum reward directly on the
        reference line and smoothly decreases with lateral distance.

        The unweighted result lies in the interval:

            0 < reward <= 1
        """

        return math.exp(-(lateral_offset**2) / (2.0 * self._style_sigma**2))

    def _termination_penalty(
        self,
        reason: TerminationReason | None,
    ) -> float:
        """Return the penalty associated with a termination reason."""

        if reason is TerminationReason.OFF_TRACK:
            return self._off_track_penalty

        if reason is TerminationReason.STUCK_OR_WRONG_WAY:
            return self._stuck_penalty

        if reason is TerminationReason.STRONG_COLLISION:
            return self._collision_penalty

        return 0.0
