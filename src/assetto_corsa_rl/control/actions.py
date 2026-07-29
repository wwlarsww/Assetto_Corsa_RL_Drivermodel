"""Cumulative delta-action integration for pedal and steering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.types import FloatArray


@dataclass(frozen=True, slots=True)
class IntegratedAction:
    """Applied command delta and resulting normalised absolute command."""

    delta: FloatArray
    absolute: FloatArray


@dataclass(slots=True)
class ActionIntegrator:
    """Scale policy deltas and integrate them into bounded absolute controls.

    Index 0 is the combined pedal command (positive throttle, negative brake);
    index 1 is steering. The Gaussian policy delta is dimensionless and may be
    outside ``[-1, 1]`` when sampled. It is scaled by ``delta_limits`` before
    Equation 2.7 from the thesis clips the resulting absolute actuator command.
    This preserves the correspondence between a sampled PPO action and its log
    probability.
    """

    delta_limits: FloatArray
    lower_bounds: FloatArray = field(
        default_factory=lambda: np.asarray([-1.0, -1.0], dtype=np.float32)
    )
    upper_bounds: FloatArray = field(
        default_factory=lambda: np.asarray([1.0, 1.0], dtype=np.float32)
    )
    _current: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.delta_limits = self._validate_vector(self.delta_limits, "delta_limits")
        self.lower_bounds = self._validate_vector(self.lower_bounds, "lower_bounds")
        self.upper_bounds = self._validate_vector(self.upper_bounds, "upper_bounds")
        if np.any(self.delta_limits <= 0):
            raise ValueError("delta limits must be positive")
        if np.any(self.lower_bounds >= self.upper_bounds):
            raise ValueError("each lower bound must be below its upper bound")
        self._current = np.zeros(2, dtype=np.float32)
        self.reset()

    @staticmethod
    def _validate_vector(value: npt.ArrayLike, name: str) -> FloatArray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (2,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite vector with shape (2,)")
        return array

    @property
    def current(self) -> FloatArray:
        """Return a copy so callers cannot mutate integrator state."""

        return self._current.copy()

    def reset(self, initial_action: npt.ArrayLike | None = None) -> FloatArray:
        """Reset the integrated command at the beginning of an episode."""

        initial = (
            np.zeros(2, dtype=np.float32)
            if initial_action is None
            else self._validate_vector(initial_action, "initial_action")
        )
        self._current = np.clip(initial, self.lower_bounds, self.upper_bounds).astype(np.float32)
        return self.current

    def integrate(self, policy_delta: npt.ArrayLike) -> IntegratedAction:
        """Scale, integrate and clip one policy action."""

        policy_action = self._validate_vector(policy_delta, "policy_delta")
        command_delta = (policy_action * self.delta_limits).astype(np.float32)
        previous = self._current.copy()
        self._current = np.clip(
            previous + command_delta,
            self.lower_bounds,
            self.upper_bounds,
        ).astype(np.float32)
        applied_delta = (self._current - previous).astype(np.float32)
        return IntegratedAction(delta=applied_delta, absolute=self.current)
