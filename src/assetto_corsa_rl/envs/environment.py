"""Gymnasium environment around a simulator-independent vehicle backend."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from assetto_corsa_rl.control.actions import ActionIntegrator
from assetto_corsa_rl.envs.backend import SimulatorBackend
from assetto_corsa_rl.features.observation import ObservationBuilder
from assetto_corsa_rl.references.projection import wrapped_progress_delta
from assetto_corsa_rl.references.promp import ReferenceSampler, ReferenceTrajectory
from assetto_corsa_rl.rewards.components import EnvironmentReward
from assetto_corsa_rl.types import FloatArray, TelemetryFrame


class AssettoCorsaEnv(gym.Env[FloatArray, FloatArray]):
    """Single-agent racing environment with cumulative control increments."""

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        *,
        backend: SimulatorBackend,
        reference_sampler: ReferenceSampler,
        observation_builder: ObservationBuilder,
        action_integrator: ActionIntegrator,
        environment_reward: EnvironmentReward,
        max_episode_steps: int,
    ) -> None:
        super().__init__()
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.backend = backend
        self.reference_sampler = reference_sampler
        self.observation_builder = observation_builder
        self.action_integrator = action_integrator
        self.environment_reward = environment_reward
        self.max_episode_steps = max_episode_steps
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32,
        )
        observation_limit = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-observation_limit,
            high=observation_limit,
            shape=(self.observation_builder.layout.size,),
            dtype=np.float32,
        )
        self._step_count = 0
        self._active_reference: ReferenceTrajectory | None = None
        self._last_telemetry: TelemetryFrame | None = None
        self._last_reference_progress_m: float | None = None

    @property
    def active_reference(self) -> ReferenceTrajectory:
        """Reference context sampled at reset and fixed for this episode."""

        if self._active_reference is None:
            raise RuntimeError("environment has not been reset")
        return self._active_reference

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            unknown = ", ".join(sorted(options))
            raise ValueError(f"unsupported reset option(s): {unknown}")

        self._step_count = 0
        self.action_integrator.reset()
        self._active_reference = self.reference_sampler.sample(self.np_random)
        backend_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
        self._last_telemetry = self.backend.reset(
            seed=backend_seed,
            reference=self._active_reference,
        )
        self._last_reference_progress_m = self._last_telemetry.progress_m
        observation = self.observation_builder.build(
            self._last_telemetry,
            self.action_integrator.current,
        )
        return observation, self._build_info(
            self._last_telemetry,
            reward_terms=None,
            termination_reason=None,
        )

    def step(
        self,
        action: npt.NDArray[np.float32],
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        if self._last_telemetry is None or self._last_reference_progress_m is None:
            raise RuntimeError("reset must be called before step")
        integrated = self.action_integrator.integrate(action)
        transition = self.backend.step(integrated.absolute)
        self._last_telemetry = transition.telemetry
        self._step_count += 1

        reference_progress_delta_m = wrapped_progress_delta(
            self._last_reference_progress_m,
            transition.telemetry.progress_m,
            self.active_reference.track_length_m,
        )
        self._last_reference_progress_m = transition.telemetry.progress_m
        reward_terms = self.environment_reward.compute(
            reference_progress_delta_m,
            transition.termination_reason,
        )
        terminated = transition.termination_reason is not None
        truncated = self._step_count >= self.max_episode_steps and not terminated
        observation = self.observation_builder.build(
            transition.telemetry,
            integrated.absolute,
        )
        info = self._build_info(
            transition.telemetry,
            reward_terms=reward_terms.as_dict(),
            termination_reason=(
                transition.termination_reason.value
                if transition.termination_reason is not None
                else None
            ),
        )
        info["action_delta"] = integrated.delta.copy()
        info["action_absolute"] = integrated.absolute.copy()
        info["reference_progress_delta_m"] = reference_progress_delta_m
        info["backend"] = transition.diagnostics or {}
        return observation, float(reward_terms.total), terminated, truncated, info

    def close(self) -> None:
        self.backend.close()
        self._last_telemetry = None
        self._last_reference_progress_m = None
        self._active_reference = None

    @staticmethod
    def _build_info(
        telemetry: TelemetryFrame,
        *,
        reward_terms: dict[str, float] | None,
        termination_reason: str | None,
    ) -> dict[str, Any]:
        return {
            "telemetry": telemetry.metrics(),
            "position_xy_m": telemetry.position_xy_m.copy(),
            "vehicle_heading_rad": telemetry.vehicle_heading_rad,
            "reward_terms": reward_terms or {},
            "termination_reason": termination_reason,
        }
