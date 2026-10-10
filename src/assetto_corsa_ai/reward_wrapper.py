"""Add frozen prediction rewards before Monitor and VecNormalize see a step."""

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike

from assetto_corsa_ai.future_reward import FuturePredictionRewardConfig, FuturePredictorRewardCalculator
from assetto_corsa_ai.RacingEnvironment import FloatArray, RacingEnvironment
from assetto_corsa_ai.reward import Reward
from assetto_corsa_ai.track_projection import TrackProjection


PredictionSnapshot = Callable[[], tuple[FloatArray, int]]


class FutureRewardWrapper(gym.Wrapper[FloatArray, FloatArray, FloatArray, FloatArray]):
    def __init__(
        self, environment: gym.Env[FloatArray, FloatArray], config: FuturePredictionRewardConfig,
        snapshot: PredictionSnapshot,
    ) -> None:
        super().__init__(environment)
        base = environment.unwrapped
        if not isinstance(base, RacingEnvironment):
            raise ValueError("Future reward requires a RacingEnvironment.")
        self.racing_environment = base
        self.config = config
        self.snapshot = snapshot
        self._pose_info: dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[FloatArray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._pose_info = info
        return observation, info

    def step(self, action: ArrayLike) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        if self._pose_info is None:
            raise RuntimeError("Call reset() before collecting prediction rewards.")
        # Consume exactly the detached prediction made with this action's
        # feature tensor, before stepping or a possible automatic reset.
        prediction, step = self.snapshot()
        scale = self.config.weight_scale(step)
        future = None
        if scale > 0:
            environment = self.racing_environment
            calculator = FuturePredictorRewardCalculator(self.config, environment.reward_calculator)
            projection = self._pose_info.get("projection")
            if not isinstance(projection, TrackProjection):
                raise ValueError("Future reward requires the current reference projection in pose info.")
            future = calculator.calculate(
                prediction, position=np.asarray(self._pose_info["vehicle_position"], dtype=np.float64),
                forward=np.asarray(self._pose_info["vehicle_forward"], dtype=np.float64),
                current_s=projection.distance_along_track, projector=environment.reference_projector, step=step,
            )
        observation, reward, terminated, truncated, info = self.env.step(np.asarray(action, dtype=np.float32))
        self._pose_info = None if terminated or truncated else info
        info["future_weight_scale"] = scale
        info["prediction_mean_predicted_progress"] = 0.0
        info["prediction_mean_abs_lateral_offset"] = 0.0
        if future is not None:
            base = info.get("reward")
            if not isinstance(base, Reward):
                raise ValueError("Future reward requires the actual step's Reward breakdown.")
            combined = self.racing_environment.reward_calculator.with_prediction(
                base, performance_prediction=future.performance_prediction, style_prediction=future.style_prediction,
            )
            info["reward"] = combined
            reward = combined.total
            info["prediction_mean_predicted_progress"] = float(np.cumsum(future.progress_deltas).mean())
            info["prediction_mean_abs_lateral_offset"] = float(np.abs(future.lateral_offsets).mean())
        return observation, float(reward), terminated, truncated, info
