"""Exercise validator failures and lifecycle without launching Assetto Corsa."""

from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock, create_autospec
from typing import Any

from gymnasium import spaces
import numpy as np
import pytest

from assetto_corsa_ai import main as wiring
from assetto_corsa_ai.RacingEnvironment import EpisodeStart, RacingEnvironment
from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.telemetry import TelemetryFrame
from assetto_corsa_ai.track_geometry import TrackGeometry, TrackPoint
from tools import validate_racing_environment as validation


@pytest.fixture
def environment() -> Mock:
    env = create_autospec(RacingEnvironment, instance=True)
    env.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    env.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    env.reset.return_value = (np.zeros(3, dtype=np.float32), {})
    env.step.return_value = (np.zeros(3, dtype=np.float32), 1.0, False, False, {"delta_s": 0.5, "dt": 0.05})
    return env


@pytest.mark.parametrize("ended", [None, "terminated", "truncated"])
def test_smoke_test_validates_steps_and_stops_at_episode_end(environment: Mock, ended: str | None) -> None:
    if ended:
        environment.step.return_value = (np.zeros(3, dtype=np.float32), 1.0,
                                         ended == "terminated", ended == "truncated", {})
    validation.run_smoke_test(environment, seed=42, steps=5, action=np.zeros(2, dtype=np.float32))
    environment.reset.assert_called_once_with(seed=42)
    assert environment.step.call_count == (1 if ended else 5)


def test_invalid_action_is_rejected_before_reset(environment: Mock) -> None:
    with pytest.raises(ValueError, match="action_space"):
        validation.run_smoke_test(environment, seed=42, steps=5, action=np.array([0.0, 1.5], dtype=np.float32))
    environment.reset.assert_not_called()


@pytest.mark.parametrize("observation", [
    np.zeros(3, dtype=np.float64), np.zeros(2, dtype=np.float32),
    np.array([float("nan"), 0.0, 0.0], dtype=np.float32),
    np.array([float("inf"), 0.0, 0.0], dtype=np.float32),
])
def test_invalid_reset_observation_stops_before_driving(environment: Mock, observation: Any) -> None:
    environment.reset.return_value = observation, {}
    with pytest.raises((TypeError, ValueError)):
        validation.run_smoke_test(environment, seed=42, steps=5, action=np.zeros(2, dtype=np.float32))
    environment.step.assert_not_called()


@pytest.mark.parametrize("invalid", ["reward_nan", "reward_type", "terminated_type", "truncated_type", "info"])
def test_invalid_step_return_is_rejected(environment: Mock, invalid: str) -> None:
    results: dict[str, tuple[Any, ...]] = {
        "reward_nan": (float("nan"), False, False, {}),
        "reward_type": (1, False, False, {}),
        "terminated_type": (1.0, np.bool_(False), False, {}),
        "truncated_type": (1.0, False, np.bool_(False), {}),
        "info": (1.0, False, False, []),
    }
    environment.step.return_value = (np.zeros(3, dtype=np.float32), *results[invalid])
    with pytest.raises((TypeError, ValueError)):
        validation.run_smoke_test(environment, seed=42, steps=5, action=np.zeros(2, dtype=np.float32))
    assert environment.step.call_count == 1


@pytest.mark.parametrize("keep_session", [False, True])
@pytest.mark.parametrize("failure", [None, "reset", "step", "checker", "interrupted"])
def test_main_runs_official_checker_and_always_closes_environment(
    environment: Mock, monkeypatch: pytest.MonkeyPatch, keep_session: bool, failure: str | None,
) -> None:
    factory = Mock(return_value=environment)
    monkeypatch.setattr(validation, "load_factory", Mock(return_value=factory))
    checker = Mock()
    monkeypatch.setattr(validation, "check_env", checker)
    arguments = ["validate_racing_environment.py", "--factory", "test:create", "--check-env", "--steps", "2"]
    if keep_session:
        arguments.append("--keep-session")
    monkeypatch.setattr(sys, "argv", arguments)
    error = KeyboardInterrupt if failure == "interrupted" else RuntimeError
    if failure in ("reset", "interrupted"):
        environment.reset.side_effect = error("validation failed")
    elif failure == "step":
        environment.step.side_effect = error("validation failed")
    elif failure == "checker":
        checker.side_effect = error("validation failed")
    if failure:
        with pytest.raises(error):
            validation.main()
    else:
        validation.main()
        checker.assert_called_once_with(environment, warn=True, skip_render_check=True)
        assert environment.step.call_count == 2
    if failure not in (None, "checker"):
        checker.assert_not_called()
    factory.assert_called_once_with()
    environment.close.assert_called_once_with(stop_session=not keep_session)


@pytest.mark.parametrize("options", [
    ["--steps", "-1"], ["--seed", "-1"], ["--action", "0", "nan"],
    ["--action", "inf", "0"], ["--action", "0", "1.5"],
])
def test_invalid_cli_arguments_do_not_create_environment(monkeypatch: pytest.MonkeyPatch, options: list[str]) -> None:
    factory_loader = Mock()
    monkeypatch.setattr(validation, "load_factory", factory_loader)
    monkeypatch.setattr(sys, "argv", ["validate_racing_environment.py", *options])
    with pytest.raises(SystemExit):
        validation.main()
    factory_loader.assert_not_called()


def test_default_factory_receives_driving_gate_and_fixed_start_sampler(
    environment: Mock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = Mock(return_value=environment)
    monkeypatch.setattr(validation, "load_factory", Mock(return_value=factory))
    monkeypatch.setattr(sys, "argv", ["validate_racing_environment.py", "--steps", "0", "--car", "test_car"])
    validation.main()
    assert factory.call_args.kwargs["car"] == "test_car"
    assert factory.call_args.kwargs["drive_ready_check"] is validation.confirm_driving
    assert callable(factory.call_args.kwargs["start_sampler"])
    environment.step.assert_not_called()


@pytest.mark.parametrize("path", ["", "module", ":factory", "module:"])
def test_invalid_factory_path_is_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="module:function"):
        validation.load_factory(path)


def test_factory_loader_checks_attribute_and_callability(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("validation_factory")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match="no attribute"):
        validation.load_factory("validation_factory:create")
    monkeypatch.setattr(module, "create", 5, raising=False)
    with pytest.raises(TypeError, match="not callable"):
        validation.load_factory("validation_factory:create")
    factory = Mock()
    monkeypatch.setattr(module, "create", factory)
    assert validation.load_factory("validation_factory:create") is factory


def test_captured_pose_is_reused_and_temporary_reader_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = Mock(spec=TelemetryFrame)
    frame.car_coordinates = (0.0, 1.0, 0.0)
    frame.speed_kmh = 0.0
    frame.tyre_contact_points = ((-0.5, 1.0, -1.0), (0.5, 1.0, -1.0),
                                 (-0.5, 1.0, 1.0), (0.5, 1.0, 1.0))
    reader = create_autospec(TelemetryReader, instance=True)
    reader.read.return_value = frame
    reader_factory = Mock(return_value=reader)
    monkeypatch.setattr(validation, "TelemetryReader", reader_factory)
    sampler = validation.captured_start_sampler()
    track = TrackGeometry([TrackPoint((0.0, 0.0, 0.0), 0.0, 5.0, 5.0, (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))])
    pose = sampler(track, np.random.default_rng(1))
    frame.car_coordinates = (5.0, 1.0, 10.0)
    assert sampler(track, np.random.default_rng(2)) is pose
    assert pose == EpisodeStart((0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
    reader_factory.assert_called_once()
    reader.close.assert_called_once()


def test_driving_confirmation_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = Mock(return_value="")
    monkeypatch.setattr("builtins.input", prompt)
    assert validation.confirm_driving()
    assert "Drive/Start" in prompt.call_args.args[0]


def test_live_factory_builds_observation_with_telemetry_axle_distances(
    environment: Mock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor = Mock(return_value=environment)
    monkeypatch.setattr(wiring, "RacingEnvironment", constructor)
    reader = create_autospec(TelemetryReader, instance=True)
    frame = Mock(spec=TelemetryFrame)
    frame.car_coordinates = (10.0, 2.0, 20.0)
    frame.tyre_contact_points = ((11.2, 2.0, 19.5), (11.2, 2.0, 20.5),
                                 (8.2, 2.0, 19.5), (8.2, 2.0, 20.5))
    reader.read.return_value = frame
    monkeypatch.setattr(wiring, "TelemetryReader", Mock(return_value=reader))
    drive_ready = Mock(return_value=True)
    wiring.create_environment(assetto_corsa_directory=tmp_path, drive_ready_check=drive_ready)
    build = constructor.call_args.kwargs["observation_builder_factory"]
    geometry = Mock(spec=TrackGeometry)
    builder = build(Mock(), geometry)
    assert isinstance(builder, wiring.ObservationBuilder)
    reader.close.assert_called_once()
