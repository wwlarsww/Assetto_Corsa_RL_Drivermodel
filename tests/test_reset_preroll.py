"""Check seeded starts, track-distance approach and isolated reset handoff."""

from dataclasses import replace
from pathlib import Path
import struct
from unittest.mock import Mock
from typing import Any

import numpy as np
import pytest

from assetto_corsa_ai.assetto_corsa.track_reader import AssettoCorsaTrackReader
from assetto_corsa_ai.episode_start import EpisodeStart, StartConfig
from assetto_corsa_ai.reset_preroll import PreRollError, ResetPreRollController
from assetto_corsa_ai.track_geometry import TrackGeometry, TrackPoint
from assetto_corsa_ai.training_config import TrainingConfig
from assetto_corsa_ai.telemetry import TelemetryFrame, Vector3
from test_racing_environment import Harness, START, Clock, telemetry
from test_racing_environment import harness as harness


def track(speed: float | None = 100.0) -> TrackGeometry:
    return TrackGeometry([
        TrackPoint((0, 0, 0), 0, 10, 10, (0, 1, 0), (0, 0, -1), speed),
        TrackPoint((0, 0, -1000), 1000, 10, 10, (0, 1, 0), (1, 0, 0), speed),
        TrackPoint((1000, 0, -1000), 2000, 10, 10, (0, 1, 0), (0, 0, 1), speed),
        TrackPoint((1000, 0, 0), 3000, 10, 10, (0, 1, 0), (-1, 0, 0), speed),
    ])


def test_reader_preserves_raw_reference_speed_and_geometry(tmp_path: Path) -> None:
    values = [0.0] * 18
    values[0], values[5], values[6], values[10], values[15] = 47.0, 4.0, 5.0, 1.0, -1.0
    path = tmp_path / "fast_lane.ai"
    path.write_bytes(struct.pack("<4i", 7, 1, 0, 0) + struct.pack("<ffffi", 1, 2, 3, 4, 0)
                     + struct.pack("<i", 1) + struct.pack("<18f", *values))
    point = AssettoCorsaTrackReader(path).read().points[0]
    assert point.reference_speed == 47.0
    assert point.position == (1, 2, 3) and point.side_left == 4 and point.side_right == 5
    assert point.forward == (0, 0, -1)


def test_sampling_is_seeded_and_clamped_in_explicit_units() -> None:
    config = StartConfig(enabled=True, reference_speed_unit="mps")
    first = config.sample(track(50), np.random.default_rng(42))
    assert first == config.sample(track(50), np.random.default_rng(42))
    assert first.reference_speed_kmh == 180
    assert first.speed_factor is not None and 0.3 <= first.speed_factor <= 0.6
    assert first.target_speed_kmh == pytest.approx(180 * first.speed_factor)
    assert config.sample(track(1000), np.random.default_rng()).target_speed_kmh == 120
    assert config.sample(track(1), np.random.default_rng()).target_speed_kmh == 20


@pytest.mark.parametrize("speed", [None, 0, -1, float("nan"), float("inf")])
def test_invalid_reference_speeds_have_no_fallback(speed: float | None) -> None:
    config = StartConfig(enabled=True, reference_speed_unit="kmh")
    with pytest.raises(ValueError, match="no valid"):
        config.sample(track(speed), np.random.default_rng())
    mixed = track(speed)
    points = list(mixed.points)
    points[2] = replace(points[2], reference_speed=100)
    assert config.sample(TrackGeometry(points), np.random.default_rng()).position == points[2].position


def test_unvalidated_units_cannot_enable_rolling_starts() -> None:
    with pytest.raises(ValueError, match="Validate reference_speed_unit"):
        StartConfig(enabled=True).validate()
    assert StartConfig().sample(track(None), np.random.default_rng()).target_speed_kmh == 0


def test_approach_wraps_start_finish_and_scales_with_speed() -> None:
    controller = ResetPreRollController(track(), StartConfig())
    slow, distance = controller.approach(EpisodeStart((0, 0, -2), (0, 0, -1), 20))
    assert slow.position == pytest.approx((distance - 2, 0, 0))
    assert slow.forward == (-1, 0, 0)
    _, fast_distance = controller.approach(EpisodeStart((0, 0, -2), (0, 0, -1), 60))
    assert fast_distance > distance
    with pytest.raises(PreRollError, match="Insufficient"):
        controller.approach(EpisodeStart((0, 0, -2), (0, 0, -1), 500))


@pytest.mark.parametrize("outcome", ["success", "timeout", "collision", "offtrack", "overshoot", "bad_heading", "bad_speed"])
def test_preroll_handoff_and_failure_braking(monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    clock = Clock()
    monkeypatch.setattr("assetto_corsa_ai.reset_preroll.time.monotonic", clock.monotonic)
    monkeypatch.setattr("assetto_corsa_ai.reset_preroll.time.sleep", clock.sleep)
    controller = ResetPreRollController(track(), replace(StartConfig(), timeout=0.15))
    start = EpisodeStart((0, 0, -100), (0, 0, -1), 36)
    initial = telemetry((0, 0, -80))
    frame = replace(telemetry(start.position), speed_kmh=36, gas=0.3)
    heading = (0.0, 0.0, -1.0)
    if outcome == "timeout": frame = initial
    if outcome == "collision": frame = replace(frame, car_damage=(1, 0, 0, 0, 0))
    if outcome == "offtrack": frame = replace(frame, tyre_contact_points=((30, 0, -100),) * 4)
    if outcome == "overshoot": frame = replace(telemetry((0, 0, -105)), speed_kmh=36)
    if outcome == "bad_heading": heading = (0.0, 0.0, 1.0)
    if outcome == "bad_speed": frame = replace(frame, speed_kmh=0)
    frames = iter([initial])
    read = lambda: next(frames, frame)
    simulator = Mock()
    def get_forward(_: TelemetryFrame) -> Vector3:
        return heading
    kwargs: dict[str, Any] = dict(simulator=simulator, read=read, forward=get_forward, interval=0.05,
                  position_tolerance=1.0, heading_tolerance_degrees=5.0, speed_tolerance_kmh=1.0)
    if outcome == "success":
        assert controller.run(start, 20, **kwargs) == frame
        simulator.set_controls.assert_not_called()
    else:
        with pytest.raises(PreRollError): controller.run(start, 20, **kwargs)
        simulator.set_controls.assert_called_with(0.0, 0.0, 1.0)


@pytest.mark.parametrize("speed,gas,brake", [(18.0, 1.0, 0.0), (45.0, 0.0, 0.9)])
def test_preroll_drives_and_then_preserves_controls_at_handoff(
    monkeypatch: pytest.MonkeyPatch, speed: float, gas: float, brake: float,
) -> None:
    clock = Clock()
    monkeypatch.setattr("assetto_corsa_ai.reset_preroll.time.monotonic", clock.monotonic)
    monkeypatch.setattr("assetto_corsa_ai.reset_preroll.time.sleep", clock.sleep)
    controller = ResetPreRollController(track(), StartConfig())
    start = EpisodeStart((0, 0, -100), (0, 0, -1), 36)
    initial = telemetry((0, 0, -80))
    moving = replace(telemetry((0, 0, -90)), speed_kmh=speed)
    ready = replace(telemetry(start.position), speed_kmh=36, gas=gas, brake=brake)
    read = Mock(side_effect=[initial, moving, ready])
    simulator = Mock()
    result = controller.run(start, 20, simulator=simulator, read=read,
        forward=lambda _: (0.0, 0.0, -1.0), interval=0.05, position_tolerance=1,
        heading_tolerance_degrees=5, speed_tolerance_kmh=1)
    assert result == ready
    simulator.set_controls.assert_called_once_with(0.0, pytest.approx(gas), pytest.approx(brake))
    assert clock.now == pytest.approx(0.05)


def test_moving_reset_initializes_rl_only_at_handoff(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    fresh = replace(telemetry(), speed_kmh=20, gas=0.4)
    def run(*args: object, **kwargs: object):
        harness.checker.reset.assert_not_called()
        harness.clock.now += 4
        return fresh
    monkeypatch.setattr(ResetPreRollController, "run", run)
    base = harness.builder.build.return_value
    def build(**kwargs: Any):
        return replace(base, longitudinal_control=kwargs["telemetry"].gas)
    harness.builder.build.side_effect = build
    env = harness.create()
    observation, info = env.reset(options={"start": replace(START, target_speed_kmh=20)})
    assert observation[-1] == pytest.approx(0.4)
    assert info["episode_time"] == 0 and info["step_count"] == 0
    assert harness.simulator.reset_episode.call_args.args[0] != START.position
    harness.checker.reset.assert_called_once()
    harness.checker.check.assert_not_called()
    harness.reader.read.return_value = fresh
    _, _, _, _, step = env.step(np.asarray([0, 0.4], dtype=np.float32))
    assert step["dt"] == pytest.approx(0.05) and step["delta_s"] == pytest.approx(0)


def test_retry_count_is_bounded_and_explicit_starts_are_not_resampled(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    failing = Mock(side_effect=PreRollError("failed"))
    monkeypatch.setattr(ResetPreRollController, "run", failing)
    start = replace(START, target_speed_kmh=20)
    sampler = Mock(return_value=start)
    env = harness.create(start_sampler=sampler, start_config=replace(StartConfig(), max_attempts=2))
    with pytest.raises(PreRollError): env.reset()
    assert failing.call_count == 2 and sampler.call_count == 2
    failing.reset_mock(); sampler.reset_mock()
    with pytest.raises(PreRollError): env.reset(options={"start": start})
    assert failing.call_count == 1 and sampler.call_count == 0


def test_legacy_config_stays_stationary_and_resume_requires_same_start_settings() -> None:
    content = Path("configs/training.toml").read_text()
    config = TrainingConfig.from_toml(content)
    begin, end = content.index("[start]"), content.index("[simulator]")
    legacy = TrainingConfig.from_toml(content[:begin] + content[end:])
    assert legacy.start == StartConfig()
    assert legacy.toml_text == content[:begin] + content[end:]
    with pytest.raises(ValueError):
        replace(config, start=replace(config.start, speed_gain=0.2)).require_compatible_resume(config)
