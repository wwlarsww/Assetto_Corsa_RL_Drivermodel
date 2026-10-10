"""Exercise environment orchestration without launching AC or opening game memory."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any
from unittest.mock import Mock, create_autospec

import gymnasium as gym
from gymnasium.utils.env_checker import check_env as check_gymnasium_env
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env as check_sb3_env
from stable_baselines3 import PPO

from assetto_corsa_ai import RacingEnvironment as module
from assetto_corsa_ai.RacingEnvironment import EpisodeStart, FloatArray, RacingEnvironment
from assetto_corsa_ai.assetto_corsa.session_manager import AssettoCorsaSessionManager, SimulatorScenario
from assetto_corsa_ai.assetto_corsa.simulator_controller import AssettoCorsaSimulatorController
from assetto_corsa_ai.assetto_corsa.static_info_reader import StaticInfoReader
from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.observation import Observation
from assetto_corsa_ai.observation_builder import ObservationBuilder
from assetto_corsa_ai.reward import RewardCalculator
from assetto_corsa_ai.static_info import SimulatorStaticInfo
from assetto_corsa_ai.telemetry import TelemetryFrame, Vector3
from assetto_corsa_ai.termination import EpisodeTerminationChecker, TerminationReason, TerminationResult
from assetto_corsa_ai.track_geometry import TrackGeometry, TrackPoint
from assetto_corsa_ai.track_perception import TrackPerceptionCalculator
from assetto_corsa_ai.tyre_dynamics import TyreDynamicsCalculator


SCENARIO = SimulatorScenario(car="test_car", track="test_track")
START = EpisodeStart((0.0, 0.0, -2.0), (0.0, 0.0, -1.0))


def action(steering: float, longitudinal: float) -> FloatArray:
    return np.array([steering, longitudinal], dtype=np.float32)


def telemetry(position: Vector3 = START.position) -> TelemetryFrame:
    x, y, z = position
    return TelemetryFrame(
        speed_kmh=0.0, gas=0.0, brake=0.0, fuel=30.0, gear=2, rpm=1000,
        steering_angle=0.0, acceleration_g=(0.0, 0.0, 0.0),
        wheel_slip=(0.0, 0.0, 0.0, 0.0), wheel_angular_speed=(0.0, 0.0, 0.0, 0.0),
        tyre_contact_points=((x - 0.5, y, z - 1.0), (x + 0.5, y, z - 1.0),
                             (x - 0.5, y, z + 1.0), (x + 0.5, y, z + 1.0)),
        local_velocity=(0.0, 0.0, 0.0), local_angular_velocity=(0.0, 0.0, 0.0),
        car_coordinates=position, normalized_track_position=0.0,
        car_damage=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


@dataclass
class Clock:
    now: float = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class Harness:
    manager: Mock
    simulator: Mock
    reader: Mock
    static_reader: Mock
    builder: Mock
    checker: Mock
    reader_factory: Mock
    static_factory: Mock
    track_loader: Mock
    builder_factory: Mock
    checker_factory: Mock
    track: TrackGeometry
    static: SimulatorStaticInfo
    clock: Clock
    events: Mock

    def create(self, **options: Any) -> RacingEnvironment:
        configuration: dict[str, Any] = {
            "default_scenario": SCENARIO, "max_episode_time": None, "observation_size": 10,
        }
        configuration.update(options)
        return RacingEnvironment(
            session_manager=self.manager, simulator_controller=self.simulator,
            telemetry_reader_factory=self.reader_factory,
            static_info_reader_factory=self.static_factory,
            track_geometry_loader=self.track_loader,
            observation_builder_factory=self.builder_factory,
            termination_checker_factory=self.checker_factory,
            reward_calculator=RewardCalculator(style_weight=0.5),
            **configuration,
        )


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    manager = create_autospec(AssettoCorsaSessionManager, instance=True)
    manager.is_running = False
    manager.current_scenario = None

    def ensure(scenario: SimulatorScenario) -> bool:
        changed = not (manager.is_running and manager.current_scenario == scenario)
        manager.is_running = True
        manager.current_scenario = scenario
        return changed

    manager.ensure_session.side_effect = ensure
    simulator = create_autospec(AssettoCorsaSimulatorController, instance=True)
    simulator.is_connected = False

    def connect() -> None:
        simulator.is_connected = True

    def close() -> None:
        simulator.is_connected = False

    simulator.connect.side_effect = connect
    simulator.close.side_effect = close
    reader = create_autospec(TelemetryReader, instance=True)
    reader.read.return_value = telemetry()
    static_reader = create_autospec(StaticInfoReader, instance=True)
    # Deliberately different AC and geometric lengths: the square is 40 m.
    static = SimulatorStaticInfo("test_car", "test_track", "", 8000, 60.0, (0.3, 0.3, 0.3, 0.3), 400.0)
    static_reader.read.return_value = static
    positions: tuple[Vector3, ...] = ((0.0, 0.0, 0.0), (0.0, 0.0, -10.0),
                                    (10.0, 0.0, -10.0), (10.0, 0.0, 0.0))
    forwards: tuple[Vector3, ...] = ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0),
                                   (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0))
    track = TrackGeometry([TrackPoint(p, float(i * 100), 5.0, 5.0, (0.0, 1.0, 0.0), forwards[i])
                           for i, p in enumerate(positions)])
    builder = create_autospec(ObservationBuilder, instance=True)
    builder.build.return_value = Observation(
        speed=0.0, longitudinal_acceleration=0.0, lateral_acceleration=0.0,
        yaw_rate=0.0, front_slip_ratio=0.0, rear_slip_ratio=0.0,
        front_slip_angle=0.0, rear_slip_angle=0.0, left_boundary=(),
        right_boundary=(), reference_waypoints=(), steering=0.0, longitudinal_control=0.0,
    )
    checker = create_autospec(EpisodeTerminationChecker, instance=True)
    checker.check.return_value = TerminationResult(False, None)
    reader_factory, static_factory = Mock(return_value=reader), Mock(return_value=static_reader)
    track_loader = Mock(return_value=track)
    builder_factory, checker_factory = Mock(return_value=builder), Mock(return_value=checker)
    events = Mock()
    for name, item in (("manager", manager), ("simulator", simulator), ("reader", reader),
                       ("static", static_reader), ("open_telemetry", reader_factory),
                       ("load_track", track_loader), ("builder", builder), ("checker", checker)):
        events.attach_mock(item, name)
    clock = Clock()
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)
    return Harness(manager, simulator, reader, static_reader, builder, checker,
                   reader_factory, static_factory, track_loader, builder_factory,
                   checker_factory, track, static, clock, events)


def test_reset_connects_after_launch_and_reuses_same_session(harness: Harness) -> None:
    env = harness.create()
    observation, info = env.reset(options={"start": START})
    assert isinstance(observation, np.ndarray)
    assert info["session_changed"] is True
    assert info["step_count"] == 0 and info["episode_time"] == 0.0
    names = [call[0] for call in harness.events.mock_calls]
    assert names.index("manager.ensure_session") < names.index("simulator.connect") < names.index("open_telemetry")
    harness.static_reader.close.assert_called_once()
    _, second = env.reset(options={"start": START})
    assert second["session_changed"] is False
    harness.simulator.connect.assert_called_once()
    harness.reader_factory.assert_called_once()
    harness.track_loader.assert_called_once()
    assert harness.checker.reset.call_count == 2


@pytest.mark.parametrize("field", ["brake", "gas", "steering_angle"])
def test_reset_waits_for_neutral_controls_before_building_observation(harness: Harness, field: str) -> None:
    stale = replace(telemetry(), **{field: 1.0})
    fresh = telemetry()
    harness.reader.read.side_effect = [stale, stale, fresh]
    env = harness.create()
    env.reset(options={"start": START})
    assert harness.clock.now == pytest.approx(0.1)
    assert harness.builder.build.call_args.kwargs["telemetry"] == fresh
    assert harness.reader.read.call_count == 3
    harness.reader.read.side_effect = None
    env.step(action(0.0, 0.1))
    assert harness.checker.check.call_args.kwargs["dt"] == pytest.approx(0.05)


def test_reset_rejects_permanently_stale_brake_telemetry(harness: Harness) -> None:
    harness.reader.read.return_value = replace(telemetry(), brake=1.0)
    env = harness.create(reset_timeout=0.1)
    with pytest.raises(RuntimeError, match="controls did not become neutral"):
        env.reset(options={"start": START})
    harness.builder.build.assert_not_called()
    harness.simulator.close.assert_called_once()
    harness.reader.close.assert_called_once()
    with pytest.raises(RuntimeError, match="No active episode"):
        env.step(action(0.0, 0.1))


def test_switch_releases_old_runtime_before_reloading_track(harness: Harness) -> None:
    env = harness.create()
    env.reset(options={"start": START})
    harness.events.reset_mock()
    new_scenario = SimulatorScenario("other_car", "other_track", track_layout="short")
    _, info = env.reset(options={"scenario": new_scenario, "start": START})
    assert info["scenario"] == new_scenario and info["session_changed"] is True
    names = [call[0] for call in harness.events.mock_calls]
    assert names.index("simulator.close") < names.index("reader.close") < names.index("manager.ensure_session")
    assert names.index("manager.ensure_session") < names.index("simulator.connect") < names.index("load_track")
    assert harness.builder_factory.call_count == harness.checker_factory.call_count == 2


def test_reset_attaches_to_already_running_scenario(harness: Harness) -> None:
    harness.manager.is_running = True
    harness.manager.current_scenario = SCENARIO
    env = harness.create()
    _, info = env.reset(options={"start": START})
    assert info["session_changed"] is False
    harness.simulator.connect.assert_called_once()
    harness.track_loader.assert_called_once()


@pytest.mark.parametrize("longitudinal, throttle, brake", [(0.7, 0.7, 0.0), (-0.4, 0.0, 0.4), (0.0, 0.0, 0.0)])
def test_step_maps_actions_and_combines_progress_reward_and_observation(
    harness: Harness, longitudinal: float, throttle: float, brake: float,
) -> None:
    env = harness.create(control_interval=0.05)
    env.reset(options={"start": START})
    harness.reader.read.return_value = telemetry((0.0, 0.0, -5.0))
    observation, reward, terminated, truncated, info = env.step(action(0.2, longitudinal))
    harness.simulator.set_controls.assert_called_once_with(
        steering=pytest.approx(0.2), throttle=pytest.approx(throttle), brake=pytest.approx(brake),
    )
    assert isinstance(observation, np.ndarray)
    assert reward == pytest.approx(3.5)
    assert not terminated and not truncated
    assert info["delta_s"] == pytest.approx(3.0)
    assert info["dt"] == pytest.approx(0.05)
    assert info["step_count"] == 1 and info["episode_time"] == pytest.approx(0.05)
    assert harness.builder.build.call_args.kwargs["current_s"] == pytest.approx(50.0)
    assert harness.checker.check.call_args.kwargs["strong_collision"] is False


@pytest.mark.parametrize("forward", [True, False])
def test_wraparound_uses_geometric_length_and_perception_uses_spline_coordinate(harness: Harness, forward: bool) -> None:
    env = harness.create()
    before, after = ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    if not forward:
        before, after = after, before
    harness.reader.read.return_value = telemetry(before)
    env.reset(options={"start": START})
    harness.reader.read.return_value = telemetry(after)
    _, _, _, _, info = env.step(action(0.0, 0.0))
    assert info["delta_s"] == pytest.approx(2.0 if forward else -2.0)
    expected_s = 10.0 if forward else 390.0
    assert harness.builder.build.call_args.kwargs["current_s"] == pytest.approx(expected_s)


@pytest.mark.parametrize("limit", ["steps", "time"])
def test_limits_truncate_without_terminal_penalty_and_require_reset(harness: Harness, limit: str) -> None:
    env = harness.create(**({"max_episode_steps": 2} if limit == "steps" else {"max_episode_time": 0.1}))
    env.reset(options={"start": START})
    assert env.step(action(0.0, 0.0))[3] is False
    _, reward, terminated, truncated, info = env.step(action(0.0, 0.0))
    assert truncated and not terminated
    assert reward == pytest.approx(0.5)
    assert info["step_count"] == 2
    harness.simulator.set_controls.assert_called_with(steering=0.0, throttle=0.0, brake=1.0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(action(0.0, 0.5))
    env.reset(options={"start": START})
    assert env.step(action(0.0, 0.0))[3] is False


def test_disabled_time_limit_keeps_episode_active_until_domain_failure(harness: Harness) -> None:
    env = harness.create(max_episode_time=None)
    env.reset(options={"start": START})
    for delay in (121.0, 3600.0):
        harness.clock.now += delay
        _, _, terminated, truncated, info = env.step(action(0.0, 0.0))
        assert not terminated and not truncated
        assert info["episode_time"] > 120.0

    harness.checker.check.return_value = TerminationResult(True, TerminationReason.OFF_TRACK)
    _, _, terminated, truncated, _ = env.step(action(0.0, 0.0))
    assert terminated and not truncated
    harness.simulator.set_controls.assert_called_with(steering=0.0, throttle=0.0, brake=1.0)


@pytest.mark.parametrize("reason", list(TerminationReason))
def test_domain_termination_brakes_and_applies_terminal_reward(harness: Harness, reason: TerminationReason) -> None:
    def collision_check(frame: TelemetryFrame) -> bool:
        return True

    env = harness.create(strong_collision_check=collision_check)
    env.reset(options={"start": START})
    harness.checker.check.return_value = TerminationResult(True, reason)
    _, reward, terminated, truncated, info = env.step(action(0.0, 0.7))
    assert terminated and not truncated
    assert reward == pytest.approx(-99.5)
    assert info["termination"].reason is reason
    assert harness.checker.check.call_args.kwargs["strong_collision"] is True
    harness.simulator.set_controls.assert_called_with(steering=0.0, throttle=0.0, brake=1.0)
    with pytest.raises(RuntimeError):
        env.step(action(0.0, 0.0))


def test_step_before_reset_does_not_send_controls(harness: Harness) -> None:
    with pytest.raises(RuntimeError, match="reset"):
        harness.create().step(action(0.0, 0.5))
    harness.simulator.set_controls.assert_not_called()


@pytest.mark.parametrize("stage", ["reset", "observation", "track", "static", "connect", "telemetry"])
def test_failed_reset_invalidates_episode_and_cleans_partial_resources(harness: Harness, stage: str) -> None:
    env = harness.create()
    if stage in ("reset", "observation"):
        env.reset(options={"start": START})
    failing = {
        "reset": harness.simulator.reset_episode,
        "observation": harness.builder.build,
        "track": harness.track_loader,
        "static": harness.static_reader.read,
        "connect": harness.simulator.connect,
        "telemetry": harness.reader_factory,
    }[stage]
    failing.side_effect = RuntimeError("Reset failed")
    with pytest.raises(RuntimeError, match="Reset failed"):
        env.reset(options={"start": START})
    with pytest.raises(RuntimeError, match="No active episode"):
        env.step(action(0.0, 0.5))
    if stage != "connect":
        harness.simulator.close.assert_called()
    if stage not in ("connect", "telemetry"):
        harness.reader.close.assert_called()
    if stage == "static":
        harness.static_reader.close.assert_called_once()


@pytest.mark.parametrize("failure", ["telemetry", "observation", "collision", "interrupted"])
def test_failed_step_cancels_driving_and_requires_new_reset(harness: Harness, failure: str) -> None:
    collision = Mock(return_value=False)
    env = harness.create(strong_collision_check=collision)
    env.reset(options={"start": START})
    error = KeyboardInterrupt if failure == "interrupted" else RuntimeError
    if failure in ("telemetry", "interrupted"):
        harness.reader.read.side_effect = error("Step failed")
    elif failure == "observation":
        harness.builder.build.side_effect = error("Step failed")
    else:
        collision.side_effect = error("Step failed")
    with pytest.raises(error):
        env.step(action(0.0, 0.8))
    harness.simulator.set_controls.assert_called_with(steering=0.0, throttle=0.0, brake=1.0)
    with pytest.raises(RuntimeError, match="No active episode"):
        env.step(action(0.0, 0.8))


@pytest.mark.parametrize("stop_session", [True, False])
@pytest.mark.parametrize("failure", [None, "controller", "telemetry"])
def test_close_cleans_all_resources_even_if_one_close_fails(harness: Harness, stop_session: bool, failure: str | None) -> None:
    env = harness.create()
    env.reset(options={"start": START})
    if failure:
        failing = harness.simulator.close if failure == "controller" else harness.reader.close
        failing.side_effect = OSError("Close failed")
        with pytest.raises(OSError):
            env.close(stop_session=stop_session)
    else:
        env.close(stop_session=stop_session)
    harness.simulator.close.assert_called_once()
    harness.reader.close.assert_called_once()
    if stop_session:
        harness.manager.stop_session.assert_called_once()
    else:
        harness.manager.stop_session.assert_not_called()
    with pytest.raises(RuntimeError):
        env.step(action(0.0, 0.5))


def test_context_manager_closes_environment_on_exception(harness: Harness) -> None:
    with pytest.raises(RuntimeError, match="Caller failed"):
        with harness.create() as env:
            env.reset(options={"start": START})
            raise RuntimeError("Caller failed")
    harness.reader.close.assert_called_once()
    harness.manager.stop_session.assert_called_once()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0])
@pytest.mark.parametrize("parameter", ["control_interval", "max_episode_time", "reset_position_tolerance",
                                      "reset_speed_tolerance_kmh", "reset_timeout", "reset_poll_interval"])
def test_nonfinite_or_negative_configuration_is_rejected(harness: Harness, parameter: str, invalid: float) -> None:
    with pytest.raises(ValueError, match=parameter):
        harness.create(**{parameter: invalid})


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_step_limit_requires_a_positive_integer(harness: Harness, value: int | float) -> None:
    with pytest.raises(ValueError, match="max_episode_steps"):
        harness.create(max_episode_steps=value)


@pytest.mark.parametrize("parameter", ["steering", "longitudinal"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.1, 1.1])
def test_invalid_actions_never_reach_simulator(harness: Harness, parameter: str, value: float) -> None:
    env = harness.create()
    env.reset(options={"start": START})
    command = action(**{"steering": 0.0, "longitudinal": 0.0, parameter: value})
    with pytest.raises(ValueError, match=parameter):
        env.step(command)
    harness.simulator.set_controls.assert_called_once_with(steering=0.0, throttle=0.0, brake=1.0)


def test_fixed_period_uses_actual_elapsed_time_when_agent_is_late(harness: Harness) -> None:
    env = harness.create(control_interval=0.05)
    env.reset(options={"start": START})
    assert env.step(action(0.0, 0.0))[4]["dt"] == pytest.approx(0.05)
    harness.clock.now += 0.2
    assert env.step(action(0.0, 0.0))[4]["dt"] == pytest.approx(0.2)
    assert env.step(action(0.0, 0.0))[4]["episode_time"] == pytest.approx(0.3)


@pytest.mark.parametrize("condition", ["ready", "speed", "position", "reverse", "invalid_heading", "nan"])
def test_reset_readiness_requires_pose_heading_and_low_speed(harness: Harness, condition: str) -> None:
    env = harness.create()
    env.reset(options={"start": START})
    check: Callable[[], bool] = harness.simulator.reset_episode.call_args.kwargs["is_ready"]
    current = telemetry()
    if condition == "speed":
        current = replace(current, speed_kmh=2.0)
    elif condition == "position":
        current = telemetry((0.0, 0.0, -4.0))
    elif condition == "reverse":
        fl, fr, rl, rr = current.tyre_contact_points
        current = replace(current, tyre_contact_points=(rl, rr, fl, fr))
    elif condition == "invalid_heading":
        current = replace(current, tyre_contact_points=((0.0, 0.0, 0.0),) * 4)
    elif condition == "nan":
        current = replace(current, speed_kmh=float("nan"))
    harness.reader.read.return_value = current
    assert check() is (condition == "ready")


def test_seed_reproduces_scenario_and_start_sampling(harness: Harness) -> None:
    def sample(rng: np.random.Generator) -> SimulatorScenario:
        return SimulatorScenario("test_car", str(rng.integers(100)))

    env = harness.create(scenario_sampler=sample)
    _, first = env.reset(seed=42)
    _, second = env.reset(seed=42)
    assert first["scenario"] == second["scenario"]
    assert first["episode_start"] == second["episode_start"]
    assert first["episode_start"].position in [p.position for p in harness.track.points]
    _, explicit = env.reset(options={"scenario": SCENARIO, "start": START})
    assert explicit["scenario"] == SCENARIO and explicit["episode_start"] == START


def test_real_observation_builder_and_termination_checker_work_in_step(harness: Harness) -> None:
    harness.builder_factory.return_value = ObservationBuilder(
        TyreDynamicsCalculator(1.0, 1.0), TrackPerceptionCalculator((5.0,), (5.0,)),
    )
    harness.checker_factory.side_effect = EpisodeTerminationChecker
    env = harness.create(observation_size=16)
    initial, _ = env.reset(options={"start": START})
    assert initial.shape == (16,)
    harness.reader.read.return_value = telemetry((-6.0, 0.0, -5.0))
    observation, _, terminated, _, info = env.step(action(0.0, 0.0))
    assert isinstance(observation, np.ndarray)
    assert terminated
    assert info["termination"].reason is TerminationReason.OFF_TRACK


def test_spaces_are_available_before_any_simulator_connection(harness: Harness) -> None:
    env = harness.create()
    assert isinstance(env, gym.Env)
    assert env.action_space.shape == (2,)
    assert env.action_space.dtype == env.observation_space.dtype == np.dtype(np.float32)
    assert env.action_space.contains(action(-1.0, -1.0))
    assert env.action_space.contains(action(1.0, 1.0))
    assert not env.action_space.contains(action(-1.1, 0.0))
    assert not env.action_space.contains(action(0.0, 1.1))
    assert env.observation_space.shape == (10,)
    assert env.render_mode is None
    harness.manager.ensure_session.assert_not_called()
    harness.simulator.connect.assert_not_called()
    harness.reader_factory.assert_not_called()


@pytest.mark.parametrize("size", [0, -1, 10.5, True])
def test_observation_size_requires_positive_integer(harness: Harness, size: Any) -> None:
    with pytest.raises(ValueError, match="observation_size"):
        harness.create(observation_size=size)


@pytest.mark.parametrize("command", [
    np.zeros(1, dtype=np.float32), np.zeros(3, dtype=np.float32),
    np.zeros((1, 2), dtype=np.float32), np.array(0.0, dtype=np.float32),
])
def test_action_shape_is_checked_before_driving(harness: Harness, command: Any) -> None:
    env = harness.create()
    env.reset()
    with pytest.raises(ValueError, match="Action must"):
        env.step(command)
    harness.simulator.set_controls.assert_called_once_with(steering=0.0, throttle=0.0, brake=1.0)


@pytest.mark.parametrize("command", [
    [0, 1], (0.0, 1.0), np.array([0, 1]),
    np.array([0.0, 1.0], dtype=np.float64),
    np.array([0.0, 1.0], dtype=np.float32),
])
def test_array_like_actions_are_converted_without_changing_controls(harness: Harness, command: Any) -> None:
    env = harness.create()
    env.reset()
    _, reward, terminated, truncated, _ = env.step(command)
    harness.simulator.set_controls.assert_called_once_with(steering=0.0, throttle=1.0, brake=0.0)
    assert type(reward) is float
    assert type(terminated) is bool
    assert type(truncated) is bool


@pytest.mark.parametrize("command", [[-1.1, 0.0], np.array([0.0, 1.1], dtype=np.float64)])
def test_action_conversion_does_not_clip_out_of_range_values(harness: Harness, command: Any) -> None:
    env = harness.create()
    env.reset()
    with pytest.raises(ValueError, match="values in"):
        env.step(command)
    harness.simulator.set_controls.assert_called_once_with(steering=0.0, throttle=0.0, brake=1.0)


@pytest.mark.parametrize("stage", ["reset", "step"])
@pytest.mark.parametrize("invalid", ["shape", "nan", "inf"])
def test_invalid_observation_cancels_episode(harness: Harness, stage: str, invalid: str) -> None:
    env = harness.create()
    if stage == "step":
        env.reset()
    current = harness.builder.build.return_value
    harness.builder.build.return_value = (
        replace(current, left_boundary=((1.0, 2.0),)) if invalid == "shape"
        else replace(current, speed=float(invalid))
    )
    with pytest.raises(ValueError, match="Observation must be finite"):
        if stage == "reset":
            env.reset()
        else:
            env.step(action(0.0, 0.5))
    if stage == "reset":
        harness.simulator.close.assert_called()
        harness.reader.close.assert_called()
    else:
        harness.simulator.set_controls.assert_called_with(steering=0.0, throttle=0.0, brake=1.0)
    with pytest.raises(RuntimeError, match="No active episode"):
        env.step(action(0.0, 0.5))


def test_observation_layout_and_space_stay_fixed_across_scenarios(harness: Harness) -> None:
    harness.builder.build.return_value = Observation(
        speed=1.0, longitudinal_acceleration=2.0, lateral_acceleration=3.0,
        yaw_rate=4.0, front_slip_ratio=5.0, rear_slip_ratio=6.0,
        front_slip_angle=7.0, rear_slip_angle=8.0, left_boundary=((9.0, 10.0),),
        right_boundary=((11.0, 12.0),), reference_waypoints=((13.0, 14.0),),
        steering=15.0, longitudinal_control=16.0,
    )
    env = harness.create(observation_size=16)
    space = env.observation_space
    observation, _ = env.reset()
    np.testing.assert_array_equal(observation, np.arange(1, 17, dtype=np.float32))
    assert space.contains(observation)
    next_observation = env.step(action(0.0, 0.0))[0]
    assert space.contains(next_observation)
    assert not np.shares_memory(observation, next_observation)
    observation[0] = -100.0
    assert next_observation[0] == 1.0
    env.reset(options={"scenario": SimulatorScenario("other_car", "other_track")})
    assert env.observation_space is space
    harness.builder.build.return_value = replace(harness.builder.build.return_value, left_boundary=())
    with pytest.raises(ValueError, match="observation_size"):
        env.reset()
    assert env.observation_space is space


@pytest.mark.parametrize("options", [
    {"unknown": 1}, {"scenario": "monza"}, {"start": (0.0, 0.0, 0.0)},
])
def test_invalid_reset_options_are_rejected_before_launch(harness: Harness, options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="[Oo]ption"):
        harness.create().reset(options=options)
    harness.manager.ensure_session.assert_not_called()
    harness.simulator.reset_episode.assert_not_called()


def test_seed_none_continues_gymnasium_rng_instead_of_reseeding(harness: Harness) -> None:
    starts: list[int] = []

    def sample(track: TrackGeometry, rng: np.random.Generator) -> EpisodeStart:
        starts.append(int(rng.integers(1_000_000)))
        return START

    env = harness.create(start_sampler=sample)
    env.reset(seed=123)
    rng = env.np_random
    env.reset()
    assert env.np_random is rng
    env.reset(seed=123)
    assert starts[0] == starts[2] and starts[0] != starts[1]


def test_gymnasium_and_sb3_environment_checkers(harness: Harness) -> None:
    # The clock restarts with each simulated reset, giving deterministic timing.
    def reset_clock(*args: Any, **kwargs: Any) -> None:
        harness.clock.now = 0.0

    harness.simulator.reset_episode.side_effect = reset_clock
    env = harness.create()
    with pytest.warns(UserWarning, match="infinity"):
        check_gymnasium_env(env, skip_render_check=True)
    check_sb3_env(env, warn=True)
    env.close()


def test_ppo_can_train_and_predict_with_mocked_simulator(harness: Harness) -> None:
    env = harness.create(max_episode_steps=3)
    model = PPO(
        "MlpPolicy", env, n_steps=4, batch_size=4, n_epochs=1,
        policy_kwargs={"net_arch": [8]}, device="cpu", seed=123,
    )
    try:
        # SB3's callback alias contains an unparameterized Callable.
        model.learn(total_timesteps=8)  # pyright: ignore[reportUnknownMemberType]
        assert harness.checker.check.call_count == 8
        observation, _ = env.reset()
        command, _ = model.predict(observation, deterministic=True)
        assert env.action_space.contains(command)
        assert env.observation_space.contains(env.step(command)[0])
    finally:
        env.close()
