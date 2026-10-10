"""Limit executed controls while keeping PPO's original samples intact."""

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike

from assetto_corsa_ai.action import ActionConfig, AgentAction
from assetto_corsa_ai.RacingEnvironment import FloatArray, RacingEnvironment


class ActionFilterWrapper(gym.Wrapper[FloatArray, FloatArray, FloatArray, FloatArray]):
    def __init__(self, environment: RacingEnvironment, config: ActionConfig) -> None:
        config.validate()
        super().__init__(environment)
        self.config = config
        self._controls: AgentAction | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[FloatArray, dict[str, Any]]:
        self._controls = None
        observation, info = self.env.reset(seed=seed, options=options)
        self._controls = self._observation_controls(observation)
        return observation, info

    def step(self, action: ArrayLike) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        if self._controls is None:
            raise RuntimeError("Call reset() before applying filtered controls.")
        requested = np.asarray(action, dtype=np.float32)
        if requested.shape != (2,) or not np.isfinite(requested).all() or not self.action_space.contains(requested):
            raise ValueError("Action must have shape (2,) with finite values in [-1, 1].")
        constrained = self.config.constrain(AgentAction(float(requested[0]), float(requested[1])), self._controls)
        applied = np.asarray([constrained.steering, constrained.longitudinal], dtype=np.float32)
        try:
            observation, reward, terminated, truncated, info = self.env.step(applied)
            self._controls = None if terminated or truncated else self._observation_controls(observation)
        except BaseException:
            self._controls = None
            raise
        info["requested_action"], info["applied_action"] = requested.copy(), applied.copy()
        info["observed_action"] = observation[-2:].copy()
        return observation, float(reward), terminated, truncated, info

    @staticmethod
    def _observation_controls(observation: FloatArray) -> AgentAction:
        return AgentAction(float(observation[-2]), float(observation[-1]))
