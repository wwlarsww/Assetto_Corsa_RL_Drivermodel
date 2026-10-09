"""RL environment orchestration for Assetto Corsa.

The environment coordinates:

- Assetto Corsa session lifecycle
- CSP runtime control (control, reset)
- telemetry acquisition
- track projection
- observation construction
- reward calculation
- episode termination
- fixed-rate control timing

It deliberately does not contain Assetto Corsa file-format parsing or CSP
shared-memory details. Those remain in their dedicated adapters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import cos, dist, hypot, isfinite, radians
import time
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import ArrayLike, NDArray

from assetto_corsa_ai.action import AgentAction
from assetto_corsa_ai.assetto_corsa.session_manager import (
    AssettoCorsaSessionManager,
    SimulatorScenario,
)
from assetto_corsa_ai.assetto_corsa.simulator_controller import (
    AssettoCorsaSimulatorController,
)
from assetto_corsa_ai.assetto_corsa.static_info_reader import StaticInfoReader
from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.observation_builder import ObservationBuilder
from assetto_corsa_ai.reward import Reward, RewardCalculator, calculate_progress_delta
from assetto_corsa_ai.static_info import SimulatorStaticInfo
from assetto_corsa_ai.telemetry import TelemetryFrame, Vector3
from assetto_corsa_ai.termination import (
    EpisodeTerminationChecker,
    TerminationResult,
)
from assetto_corsa_ai.track_geometry import TrackGeometry
from assetto_corsa_ai.track_projection import TrackProjection, TrackProjector


@dataclass(frozen=True)
class EpisodeStart:
    """Vehicle pose used to start a new episode."""

    position: Vector3
    forward: Vector3


TelemetryReaderFactory = Callable[[], TelemetryReader]
StaticInfoReaderFactory = Callable[[], StaticInfoReader]
TrackGeometryLoader = Callable[[SimulatorStaticInfo], TrackGeometry]
ObservationBuilderFactory = Callable[
    [SimulatorStaticInfo, TrackGeometry],
    ObservationBuilder,
]
TerminationCheckerFactory = Callable[
    [TrackProjector],
    EpisodeTerminationChecker,
]
StartSampler = Callable[
    [TrackGeometry, np.random.Generator],
    EpisodeStart,
]
ScenarioSampler = Callable[
    [np.random.Generator],
    SimulatorScenario,
]
StrongCollisionCheck = Callable[
    [TelemetryFrame],
    bool,
]


FloatArray = NDArray[np.float32]


class RacingEnvironment(gym.Env[FloatArray, FloatArray]):
    """Coordinate complete RL episodes in Assetto Corsa.

    Actions are float32 arrays ``[steering, longitudinal]`` in [-1, 1].
    Observations are flat float32 vectors from ``Observation.to_array()``.
    Their size is fixed at construction, including across scenario changes.

    ``observation, info = env.reset()``

    ``observation, reward, terminated, truncated, info = env.step(action)``

    ``terminated`` represents domain terminal states such as off-track or
    stuck/wrong-way driving.

    ``truncated`` represents external episode limits such as maximum episode
    time or maximum step count.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        *,
        session_manager: AssettoCorsaSessionManager,
        simulator_controller: AssettoCorsaSimulatorController,
        telemetry_reader_factory: TelemetryReaderFactory,
        static_info_reader_factory: StaticInfoReaderFactory,
        track_geometry_loader: TrackGeometryLoader,
        observation_builder_factory: ObservationBuilderFactory,
        termination_checker_factory: TerminationCheckerFactory,
        reward_calculator: RewardCalculator,
        observation_size: int,
        default_scenario: SimulatorScenario | None = None,
        scenario_sampler: ScenarioSampler | None = None,
        start_sampler: StartSampler | None = None,
        strong_collision_check: StrongCollisionCheck | None = None,
        control_interval: float = 0.05,
        max_episode_time: float | None = 120.0,
        max_episode_steps: int | None = None,
        reset_position_tolerance: float = 1.0,
        reset_heading_tolerance_degrees: float = 5.0,
        reset_speed_tolerance_kmh: float = 1.0,
        reset_timeout: float = 5.0,
        reset_poll_interval: float = 0.05,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if type(observation_size) is not int or observation_size <= 0:
            raise ValueError("observation_size must be a positive integer.")

        if default_scenario is None and scenario_sampler is None:
            raise ValueError(
                "Provide either default_scenario or scenario_sampler."
            )

        if not isfinite(control_interval) or control_interval <= 0.0:
            raise ValueError(
                "control_interval must be positive."
            )

        if max_episode_time is not None and (not isfinite(max_episode_time) or max_episode_time <= 0.0):
            raise ValueError(
                "max_episode_time must be positive or None."
            )

        if max_episode_steps is not None and (type(max_episode_steps) is not int or max_episode_steps <= 0):
            raise ValueError(
                "max_episode_steps must be positive or None."
            )

        if not isfinite(reset_position_tolerance) or reset_position_tolerance <= 0.0:
            raise ValueError(
                "reset_position_tolerance must be positive."
            )

        if not 0.0 < reset_heading_tolerance_degrees <= 180.0:
            raise ValueError(
                "reset_heading_tolerance_degrees must be in (0, 180]."
            )

        if not isfinite(reset_speed_tolerance_kmh) or reset_speed_tolerance_kmh < 0.0:
            raise ValueError(
                "reset_speed_tolerance_kmh must be >= 0."
            )

        if not isfinite(reset_timeout) or reset_timeout <= 0.0:
            raise ValueError(
                "reset_timeout must be positive."
            )

        if not isfinite(reset_poll_interval) or reset_poll_interval <= 0.0:
            raise ValueError(
                "reset_poll_interval must be positive."
            )

        self._session_manager = session_manager
        self._simulator = simulator_controller

        self._telemetry_reader_factory = telemetry_reader_factory
        self._static_info_reader_factory = static_info_reader_factory
        self._track_geometry_loader = track_geometry_loader
        self._observation_builder_factory = observation_builder_factory
        self._termination_checker_factory = termination_checker_factory

        self._reward_calculator = reward_calculator

        self._default_scenario = default_scenario
        self._scenario_sampler = scenario_sampler

        self._start_sampler = (
            start_sampler
            if start_sampler is not None
            else self.uniform_track_point_start
        )

        self._strong_collision_check = strong_collision_check

        self._control_interval = control_interval
        self._max_episode_time = max_episode_time
        self._max_episode_steps = max_episode_steps

        self._reset_position_tolerance = reset_position_tolerance
        self._reset_heading_tolerance_degrees = (
            reset_heading_tolerance_degrees
        )
        self._reset_speed_tolerance_kmh = reset_speed_tolerance_kmh
        self._reset_timeout = reset_timeout
        self._reset_poll_interval = reset_poll_interval

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # Dynamics and track-relative coordinates have no universal bounds.
        # Validate finiteness on return; normalization belongs in a wrapper.
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(observation_size,), dtype=np.float32,
        )
        if seed is not None:
            super().reset(seed=seed)

        self._telemetry_reader: TelemetryReader | None = None
        self._static_info: SimulatorStaticInfo | None = None
        self._track_geometry: TrackGeometry | None = None
        self._track_projector: TrackProjector | None = None
        self._observation_builder: ObservationBuilder | None = None
        self._termination_checker: EpisodeTerminationChecker | None = None

        self._scenario: SimulatorScenario | None = None

        self._previous_s: float | None = None
        self._step_count = 0
        self._episode_time = 0.0
        self._last_step_time: float | None = None

        self._episode_active = False

    @staticmethod
    def uniform_track_point_start(
        track_geometry: TrackGeometry,
        rng: np.random.Generator,
    ) -> EpisodeStart:
        """Choose a uniformly random existing reference-line point.

        This intentionally samples existing track points rather than performing
        interpolation. Interpolated/random-offset spawning can be added later
        after the basic CSP teleport path is validated with track geometry.
        """

        points = track_geometry.points

        if not points:
            raise RuntimeError(
                "Cannot sample an episode start from an empty TrackGeometry."
            )

        point = points[
            rng.integers(
                len(points)
            )
        ]

        return EpisodeStart(
            position=point.position,
            forward=point.forward,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Reset; options may override the sampled ``scenario`` and ``start``."""

        self._episode_active = False
        try:
            super().reset(seed=seed)
            options = options if options is not None else {}
            if options.keys() - {"scenario", "start"}:
                raise ValueError("Reset options only support 'scenario' and 'start'.")
            scenario = options.get("scenario")
            start = options.get("start")
            if scenario is not None and not isinstance(scenario, SimulatorScenario):
                raise ValueError("Reset option 'scenario' must be a SimulatorScenario.")
            if start is not None and not isinstance(start, EpisodeStart):
                raise ValueError("Reset option 'start' must be an EpisodeStart.")
            return self._reset_episode(scenario=scenario, start=start)
        except BaseException as error:
            try:
                self._disconnect_runtime()
            except Exception as cleanup_error:
                error.add_note(f"Runtime cleanup also failed: {cleanup_error}")
            raise

    def _reset_episode(
        self,
        *,
        scenario: SimulatorScenario | None,
        start: EpisodeStart | None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Start a new episode.

        A session restart is only performed when car/track/layout changed.
        Ordinary episodes on the same scenario reuse the current AC session
        and only teleport the vehicle to the next episode start.
        """

        requested_scenario = self._resolve_scenario(
            scenario
        )

        session_change_required = not (
            self._session_manager.is_running
            and self._session_manager.current_scenario == requested_scenario
        )

        if session_change_required:
            self._disconnect_runtime()

        session_changed = (
            self._session_manager.ensure_session(
                requested_scenario
            )
        )

        if session_changed:
            self._connect_runtime()
            self._reload_session_resources()

        else:
            if not self._simulator.is_connected:
                self._simulator.connect()

            if self._telemetry_reader is None:
                self._telemetry_reader = (
                    self._telemetry_reader_factory()
                )

            if not self._resources_ready():
                self._reload_session_resources()

        self._scenario = requested_scenario

        assert self._track_geometry is not None
        assert self._termination_checker is not None

        episode_start = (
            start
            if start is not None
            else self._start_sampler(
                self._track_geometry,
                self.np_random,
            )
        )

        self._termination_checker.reset()

        self._simulator.reset_episode(
            episode_start.position,
            episode_start.forward,
            is_ready=self._reset_readiness_check(
                episode_start
            ),
            timeout=self._reset_timeout,
            poll_interval=self._reset_poll_interval,
        )

        telemetry = self._read_telemetry()

        projection = self._project(
            telemetry
        )

        self._previous_s = (
            projection.distance_along_track
        )

        self._step_count = 0
        self._episode_time = 0.0

        observation = self._build_observation(
            telemetry=telemetry,
            projection=projection,
        )

        self._last_step_time = time.monotonic()
        self._episode_active = True

        info = {
            "scenario": self._scenario,
            "episode_start": episode_start,
            "projection": projection,
            "step_count": self._step_count,
            "episode_time": self._episode_time,
            "session_changed": session_changed,
        }

        return observation, info

    def step(
        self,
        action: ArrayLike,
    ) -> tuple[
        FloatArray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Apply one RL action and advance the environment by one control step."""

        if not self._episode_active:
            raise RuntimeError(
                "No active episode. Call reset() before step()."
            )

        try:
            return self._step(action)
        except BaseException as error:
            self._episode_active = False
            try:
                if self._simulator.is_connected:
                    self._simulator.set_controls(steering=0.0, throttle=0.0, brake=1.0)
            except Exception as cleanup_error:
                error.add_note(f"Applying safe controls also failed: {cleanup_error}")
            raise

    def _step(
        self, action: ArrayLike,
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:

        assert self._previous_s is not None
        assert self._last_step_time is not None
        assert self._static_info is not None
        assert self._track_geometry is not None
        assert self._track_projector is not None
        assert self._termination_checker is not None

        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError(
                "Action must have shape (2,) with finite "
                "steering and longitudinal values in [-1, 1]."
            )
        controls = AgentAction.map_agent_action(
            AgentAction(steering=float(action[0]), longitudinal=float(action[1])),
        )

        self._simulator.set_controls(
            steering=controls.steering,
            throttle=controls.throttle,
            brake=controls.brake,
        )

        self._wait_for_next_control_step()

        now = time.monotonic() # Current time

        dt = (
            now
            - self._last_step_time
        )

        if not isfinite(dt) or dt <= 0.0:
            raise RuntimeError("Control step duration must be finite and positive.")

        self._last_step_time = now

        telemetry = self._read_telemetry()

        projection = self._project(
            telemetry
        )

        current_s = (
            projection.distance_along_track
        )

        delta_s = calculate_progress_delta(
            current_s=current_s,
            previous_s=self._previous_s,
            # Projection progress uses geometric metres, not AC spline metadata.
            track_length=self._track_projector.track_length,
        )

        strong_collision = (
            self._strong_collision_check(
                telemetry
            )
            if self._strong_collision_check is not None
            else False
        )

        termination = (
            self._termination_checker.check(
                telemetry=telemetry,
                delta_s=delta_s,
                dt=dt,
                strong_collision=strong_collision,
            )
        )

        reward = self._reward_calculator.calculate(
            delta_s=delta_s,
            lateral_offset=projection.lateral_offset,
            termination=termination,
        )

        observation = self._build_observation(
            telemetry=telemetry,
            projection=projection,
        )

        self._previous_s = current_s
        self._step_count += 1
        self._episode_time += dt

        terminated = (
            termination.terminated
        )

        truncated = self._is_truncated()

        if terminated or truncated:
            self._episode_active = False
            # Leave the simulator in a safe state while the caller processes
            # the terminal transition and requests the next reset.
            self._simulator.set_controls(
                steering=0.0,
                throttle=0.0,
                brake=1.0,
            )

        info = self._build_step_info(
            projection=projection,
            delta_s=delta_s,
            reward=reward,
            termination=termination,
            dt=dt,
        )

        return (
            observation,
            float(reward.total),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(
        self,
        *,
        stop_session: bool = True,
    ) -> None:
        """Release runtime resources and optionally stop Assetto Corsa."""

        self._episode_active = False

        try:
            self._disconnect_runtime()
        finally:
            if stop_session:
                self._session_manager.stop_session()

    def _resolve_scenario(
        self,
        explicit_scenario: SimulatorScenario | None,
    ) -> SimulatorScenario:
        if explicit_scenario is not None:
            return explicit_scenario

        if self._scenario_sampler is not None:
            return self._scenario_sampler(
                self.np_random
            )

        assert self._default_scenario is not None
        return self._default_scenario

    def _connect_runtime(
        self,
    ) -> None:
        """Open runtime interfaces after AC loaded the requested session."""

        self._simulator.connect()

        self._telemetry_reader = (
            self._telemetry_reader_factory()
        )

    def _disconnect_runtime(
        self,
    ) -> None:
        """Release mappings before an AC process/session replacement."""

        self._episode_active = False
        reader = self._telemetry_reader
        self._telemetry_reader = None
        try:
            if self._simulator.is_connected:
                self._simulator.close()
        finally:
            try:
                if reader is not None:
                    reader.close()
            finally:
                self._clear_session_resources()

    def _reload_session_resources(
        self,
    ) -> None:
        """Reload all resources that depend on the current car or track."""

        static_reader = (
            self._static_info_reader_factory()
        )

        try:
            static_info = (
                static_reader.read()
            )
        finally:
            static_reader.close()

        if not isfinite(static_info.track_spline_length) or static_info.track_spline_length <= 0.0:
            raise ValueError("Static track spline length must be finite and positive.")

        track_geometry = (
            self._track_geometry_loader(
                static_info
            )
        )

        projector = TrackProjector(
            track_geometry
        )

        observation_builder = (
            self._observation_builder_factory(
                static_info,
                track_geometry,
            )
        )

        termination_checker = (
            self._termination_checker_factory(
                projector
            )
        )

        self._static_info = static_info
        self._track_geometry = track_geometry
        self._track_projector = projector
        self._observation_builder = observation_builder
        self._termination_checker = termination_checker

    def _clear_session_resources(
        self,
    ) -> None:
        self._static_info = None
        self._track_geometry = None
        self._track_projector = None
        self._observation_builder = None
        self._termination_checker = None
        self._previous_s = None
        self._last_step_time = None

    def _resources_ready(
        self,
    ) -> bool:
        return all(
            resource is not None
            for resource in (
                self._static_info,
                self._track_geometry,
                self._track_projector,
                self._observation_builder,
                self._termination_checker,
            )
        )

    def _wait_for_next_control_step(
        self,
    ) -> None:
        """Hold a fixed control period without accumulating sleep drift."""

        assert self._last_step_time is not None

        target_time = (
            self._last_step_time
            + self._control_interval
        )

        remaining = (
            target_time
            - time.monotonic()
        )

        if remaining > 0.0:
            time.sleep(
                remaining
            )

    def _read_telemetry(
        self,
    ) -> TelemetryFrame:
        if self._telemetry_reader is None:
            raise RuntimeError(
                "TelemetryReader is not connected."
            )

        return self._telemetry_reader.read()

    def _project(
        self,
        telemetry: TelemetryFrame,
    ) -> TrackProjection:
        if self._track_projector is None:
            raise RuntimeError(
                "TrackProjector is not initialized."
            )

        return self._track_projector.project(
            telemetry.car_coordinates
        )

    def _build_observation(
        self,
        *,
        telemetry: TelemetryFrame,
        projection: TrackProjection,
    ) -> FloatArray:
        if (
            self._observation_builder is None
            or self._static_info is None
            or self._track_geometry is None
        ):
            raise RuntimeError(
                "Observation resources are not initialized."
            )

        # Perception interpolates TrackPoint.length, which uses AC's spline
        # coordinate. Convert through the projected segment, keeping reward
        # progress in the projector's geometric coordinate.
        points = self._track_geometry.points
        index = projection.segment_index
        start_s = points[index].length
        end_s = points[index + 1].length if index + 1 < len(points) else self._static_info.track_spline_length
        current_s = start_s + projection.segment_fraction * (end_s - start_s)

        observation = self._observation_builder.build(
            telemetry=telemetry,
            static_info=self._static_info,
            track_geometry=self._track_geometry,
            current_s=current_s,
        ).to_array()
        if not self.observation_space.contains(observation) or not np.isfinite(observation).all():
            raise ValueError(
                "Observation must be finite and match the configured observation_size "
                f"{self.observation_space.shape}; got {observation.shape}."
            )
        return observation

    def _reset_readiness_check(
        self,
        start: EpisodeStart,
    ) -> Callable[[], bool]:
        """Build a telemetry acknowledgement for CSP episode reset."""

        target_forward = self._normalize_vector(
            start.forward
        )

        minimum_alignment = cos(
            radians(
                self._reset_heading_tolerance_degrees
            )
        )

        def is_ready() -> bool:
            telemetry = self._read_telemetry()

            if (
                not isfinite(
                    telemetry.speed_kmh
                )
                or abs(
                    telemetry.speed_kmh
                )
                > self._reset_speed_tolerance_kmh
            ):
                return False

            position_error = dist(
                telemetry.car_coordinates,
                start.position,
            )

            if (
                not isfinite(
                    position_error
                )
                or position_error
                > self._reset_position_tolerance
            ):
                return False

            try:
                actual_forward = (
                    self._vehicle_forward(
                        telemetry
                    )
                )
            except ValueError:
                return False

            alignment = sum(
                actual_forward[index]
                * target_forward[index]
                for index in range(3)
            )

            return (
                alignment
                >= minimum_alignment
            )

        return is_ready

    @staticmethod
    def _vehicle_forward(
        telemetry: TelemetryFrame,
    ) -> Vector3:
        """Derive vehicle forward from tyre contact positions."""

        fl, fr, rl, rr = (
            telemetry.tyre_contact_points
        )

        vector: Vector3 = (
            (fl[0] + fr[0] - rl[0] - rr[0]) / 2.0,
            (fl[1] + fr[1] - rl[1] - rr[1]) / 2.0,
            (fl[2] + fr[2] - rl[2] - rr[2]) / 2.0,
        )

        return RacingEnvironment._normalize_vector(
            vector
        )

    @staticmethod
    def _normalize_vector(
        vector: Vector3,
    ) -> Vector3:
        length = hypot(
            *vector
        )

        if (
            not isfinite(length)
            or length <= 1e-9
        ):
            raise ValueError(
                "Cannot normalize zero or invalid vector."
            )

        return (
            vector[0] / length,
            vector[1] / length,
            vector[2] / length,
        )

    def _is_truncated(
        self,
    ) -> bool:
        if (
            self._max_episode_time is not None
            and self._episode_time
            >= self._max_episode_time
        ):
            return True

        if (
            self._max_episode_steps is not None
            and self._step_count
            >= self._max_episode_steps
        ):
            return True

        return False

    def _build_step_info(
        self,
        *,
        projection: TrackProjection,
        delta_s: float,
        reward: Reward,
        termination: TerminationResult,
        dt: float,
    ) -> dict[str, Any]:
        return {
            "scenario": self._scenario,
            "projection": projection,
            "delta_s": delta_s,
            "reward": reward,
            "termination": termination,
            "dt": dt,
            "step_count": self._step_count,
            "episode_time": self._episode_time,
        }
