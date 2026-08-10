"""Composition root for ready-to-run environments."""

from __future__ import annotations

from assetto_corsa_rl.ac_integration.ai_spline import AssettoCorsaAISpline
from assetto_corsa_rl.ac_integration.launcher import configured_fast_lane_path
from assetto_corsa_rl.ac_integration.live_backend import AssettoCorsaLiveBackend
from assetto_corsa_rl.ac_integration.reset_bridge import (
    HOTLAP_REFERENCE_POSE_SPEED_RESET_METHOD_VARIANT,
    HOTLAP_REFERENCE_SPEED_RESET_METHOD_VARIANT,
)
from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.control.actions import BoundedActionAdapter
from assetto_corsa_rl.control.vgamepad import VGamepadControlWriter
from assetto_corsa_rl.data.bc_dataset import load_bc_normalizer
from assetto_corsa_rl.envs.environment import AssettoCorsaEnv
from assetto_corsa_rl.envs.mock_backend import MockAssettoCorsaBackend
from assetto_corsa_rl.features.observation import ObservationBuilder
from assetto_corsa_rl.features.track_geometry import TrackGeometry
from assetto_corsa_rl.references.factory import load_reference_sampler
from assetto_corsa_rl.references.fixed_reference import FixedReferenceSampler, ReferenceTrajectory
from assetto_corsa_rl.rewards.components import EnvironmentReward


def create_mock_environment(config: ProjectConfig) -> AssettoCorsaEnv:
    """Wire all baseline components to the deterministic mock backend."""

    environment = config.environment
    reference = ReferenceTrajectory.constant(
        track_length_m=environment.mock_track_length_m,
        speed_mps=max(environment.initial_speed_mps, 25.0),
    )
    backend = MockAssettoCorsaBackend(
        control_frequency_hz=environment.control_frequency_hz,
        track_length_m=environment.mock_track_length_m,
        track_half_width_m=environment.mock_track_half_width_m,
        initial_speed_mps=environment.initial_speed_mps,
        minimum_speed_mps=environment.minimum_speed_mps,
        stall_startup_grace_steps=environment.stall_startup_grace_steps,
        stall_debounce_steps=environment.stall_debounce_steps,
        termination_method_variant=environment.termination_method_variant,
        progress_stall_window_s=environment.progress_stall_window_s,
        progress_stall_minimum_progress_m=(environment.progress_stall_minimum_progress_m),
        critical_body_sideslip_angle_deg=environment.critical_body_sideslip_angle_deg,
        minimum_body_sideslip_termination_speed_mps=(
            environment.minimum_body_sideslip_termination_speed_mps
        ),
    )
    return AssettoCorsaEnv(
        backend=backend,
        reference_sampler=FixedReferenceSampler(reference),
        observation_builder=ObservationBuilder(),
        action_adapter=BoundedActionAdapter(),
        environment_reward=EnvironmentReward(
            progress_scale=config.rewards.progress_scale,
            termination_penalty=config.rewards.termination_penalty,
        ),
        max_episode_steps=config.experiment.max_episode_steps,
        initial_action=[environment.initial_combined_pedal, 0.0],
        truncate_on_max_episode_steps=config.experiment.truncate_on_max_episode_steps,
    )


def create_assetto_corsa_environment(
    config: ProjectConfig,
    *,
    control_writer: VGamepadControlWriter | None = None,
) -> AssettoCorsaEnv:
    """Connect the validated fixed-reference environment to a running AC session."""

    environment = config.environment
    runtime = config.assetto_corsa
    reference_sampler = load_reference_sampler(config)
    reference = reference_sampler.mean()
    if reference.world_position_xy_m is None:
        raise ValueError("the live fixed-reference artefact must contain world_position_xy_m")
    if (
        runtime.reset_backend
        in {
            HOTLAP_REFERENCE_SPEED_RESET_METHOD_VARIANT,
            HOTLAP_REFERENCE_POSE_SPEED_RESET_METHOD_VARIANT,
        }
        and reference.human_mean_speed_mps is None
    ):
        raise ValueError(
            f"{runtime.reset_backend} requires a reference artifact with "
            "human_mean_speed_mps; rebuild the reference from human recordings"
        )
    spline = AssettoCorsaAISpline.from_file(configured_fast_lane_path(config))
    track_geometry = TrackGeometry.from_ai_spline(spline)
    observation_builder = ObservationBuilder(
        load_bc_normalizer(config.behavioral_cloning.dataset_artifact_path)
    )
    action_adapter = BoundedActionAdapter()
    environment_reward = EnvironmentReward(
        progress_scale=config.rewards.progress_scale,
        termination_penalty=config.rewards.termination_penalty,
    )
    backend = AssettoCorsaLiveBackend.connect(
        track_geometry=track_geometry,
        control_frequency_hz=environment.control_frequency_hz,
        expected_car_id=config.dataset.car_id,
        expected_track_id=config.dataset.track_id,
        minimum_speed_mps=environment.minimum_speed_mps,
        stall_startup_grace_steps=environment.stall_startup_grace_steps,
        stall_debounce_steps=environment.stall_debounce_steps,
        termination_method_variant=environment.termination_method_variant,
        progress_stall_window_s=environment.progress_stall_window_s,
        progress_stall_minimum_progress_m=(environment.progress_stall_minimum_progress_m),
        critical_body_sideslip_angle_deg=environment.critical_body_sideslip_angle_deg,
        minimum_body_sideslip_termination_speed_mps=(
            environment.minimum_body_sideslip_termination_speed_mps
        ),
        frame_timeout_s=runtime.frame_timeout_s,
        reset_timeout_s=runtime.reset_timeout_s,
        reset_position_tolerance=runtime.reset_position_tolerance,
        reset_reference_lateral_offset_m=runtime.reset_reference_lateral_offset_m,
        reset_reference_heading_tolerance_rad=(runtime.reset_reference_heading_tolerance_rad),
        reset_speed_threshold_mps=runtime.reset_speed_threshold_mps,
        reset_reference_speed_tolerance_mps=(runtime.reset_reference_speed_tolerance_mps),
        reset_stable_frames=runtime.reset_stable_frames,
        reset_max_abs_slip_ratio=runtime.reset_max_abs_slip_ratio,
        reset_slip_stable_frames=runtime.reset_slip_stable_frames,
        reset_reference_speed_max_attempts=(runtime.reset_reference_speed_max_attempts),
        reset_collision_quiescence_s=runtime.reset_collision_quiescence_s,
        poll_interval_s=runtime.poll_interval_s,
        steering_output_sign=runtime.steering_output_sign,
        steering_output_gain=runtime.steering_output_gain,
        reset_backend=runtime.reset_backend,
        step_timing_method_variant=runtime.step_timing_method_variant,
        control_writer=control_writer,
    )
    try:
        return AssettoCorsaEnv(
            backend=backend,
            reference_sampler=reference_sampler,
            observation_builder=observation_builder,
            action_adapter=action_adapter,
            environment_reward=environment_reward,
            max_episode_steps=config.experiment.max_episode_steps,
            initial_action=[environment.initial_combined_pedal, 0.0],
            truncate_on_max_episode_steps=config.experiment.truncate_on_max_episode_steps,
        )
    except BaseException:
        backend.close()
        raise
