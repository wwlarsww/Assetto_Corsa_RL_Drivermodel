"""Composition root for ready-to-run environments."""

from __future__ import annotations

import numpy as np

from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.control.actions import ActionIntegrator
from assetto_corsa_rl.envs.environment import AssettoCorsaEnv
from assetto_corsa_rl.envs.mock_backend import MockAssettoCorsaBackend
from assetto_corsa_rl.features.observation import ObservationBuilder
from assetto_corsa_rl.references.promp import FixedReferenceSampler, ReferenceTrajectory
from assetto_corsa_rl.rewards.components import EnvironmentReward


def create_mock_environment(config: ProjectConfig) -> AssettoCorsaEnv:
    """Wire all baseline components to the deterministic mock backend."""

    environment = config.environment
    reference = ReferenceTrajectory.constant(
        track_length_m=environment.track_length_m,
        speed_mps=max(environment.initial_speed_mps, 25.0),
    )
    backend = MockAssettoCorsaBackend(
        control_frequency_hz=environment.control_frequency_hz,
        track_length_m=environment.track_length_m,
        track_half_width_m=environment.track_half_width_m,
        initial_speed_mps=environment.initial_speed_mps,
        minimum_speed_mps=environment.minimum_speed_mps,
        minimum_speed_grace_steps=environment.minimum_speed_grace_steps,
        critical_slip_ratio=environment.critical_slip_ratio,
    )
    return AssettoCorsaEnv(
        backend=backend,
        reference_sampler=FixedReferenceSampler(reference),
        observation_builder=ObservationBuilder(),
        action_integrator=ActionIntegrator(
            delta_limits=np.asarray(
                [
                    config.actions.pedal_delta_limit,
                    config.actions.steering_delta_limit,
                ],
                dtype=np.float32,
            )
        ),
        environment_reward=EnvironmentReward(
            progress_scale=config.rewards.progress_scale,
            termination_penalty=config.rewards.termination_penalty,
        ),
        max_episode_steps=config.experiment.max_episode_steps,
    )
