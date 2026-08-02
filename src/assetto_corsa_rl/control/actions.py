"""Bounded absolute-action adaptation for pedal and steering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.types import FloatArray


@dataclass(frozen=True, slots=True)
class AppliedAction:
    """Applied normalised command and its change from the previous command."""

    change: FloatArray
    absolute: FloatArray


@dataclass(slots=True)
class BoundedActionAdapter:
    """Apply policy outputs as bounded absolute actuator commands.

    Index 0 is the combined pedal command (positive throttle, negative brake);
    index 1 is steering. There is deliberately no rate or delta limit: either
    command can move from ``-1`` to ``+1`` in one environment step. The policy
    distribution itself is responsible for producing values in ``[-1, 1]``;
    this adapter clips once more at the simulator boundary as a safety guard.
    """

    lower_bounds: FloatArray = field(
        default_factory=lambda: np.asarray([-1.0, -1.0], dtype=np.float32)
    )
    upper_bounds: FloatArray = field(
        default_factory=lambda: np.asarray([1.0, 1.0], dtype=np.float32)
    )
    _current: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.lower_bounds = self._validate_vector(self.lower_bounds, "lower_bounds")
        self.upper_bounds = self._validate_vector(self.upper_bounds, "upper_bounds")
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
        """Return a copy so callers cannot mutate the latest command."""

        return self._current.copy()

    def reset(self, initial_action: npt.ArrayLike | None = None) -> FloatArray:
        """Reset the remembered command at the beginning of an episode."""

        initial = (
            np.zeros(2, dtype=np.float32)
            if initial_action is None
            else self._validate_vector(initial_action, "initial_action")
        )
        self._current = np.clip(initial, self.lower_bounds, self.upper_bounds).astype(np.float32)
        return self.current

    def apply(self, policy_action: npt.ArrayLike) -> AppliedAction:
        """Clip and apply one absolute policy command."""

        requested = self._validate_vector(policy_action, "policy_action")
        previous = self._current.copy()
        self._current = np.clip(requested, self.lower_bounds, self.upper_bounds).astype(np.float32)
        applied_change = (self._current - previous).astype(np.float32)
        return AppliedAction(change=applied_change, absolute=self.current)
