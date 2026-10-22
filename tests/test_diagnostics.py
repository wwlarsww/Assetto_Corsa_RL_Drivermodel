"""Read-only diagnostics core, independent of Qt and a running simulator."""

from dataclasses import replace
from pathlib import Path
import json
from typing import Any

import numpy as np
import pytest

from tools.diagnostics.buffer import FrameBuffer
from tools.diagnostics.checks import DiagnosticsChecks, reward_is_consistent
from tools.diagnostics.frame import DiagnosticsFrame, observation_geometry, observation_names
from tools.diagnostics.recorder import DiagnosticsRecorder, LapDetector
from tools.diagnostics.recording import Recording, future_trajectory, lap_statistics, load_lap, prediction_errors, save_lap, with_actual_future
from tools.diagnostics.sources import ReplaySource
from assetto_corsa_ai.reward import Reward
from assetto_corsa_ai.static_info import SimulatorStaticInfo
from assetto_corsa_ai.telemetry import TelemetryFrame
from assetto_corsa_ai.training_config import TrainingConfig
from tools.diagnostics import sources
from assetto_corsa_ai.track_geometry import TrackGeometry, TrackPoint
from config_helpers import replace_setting


def frame(timestamp: float = 100.0, progress: float = 0.5) -> DiagnosticsFrame:
    telemetry = TelemetryFrame(72.0, 0.5, 0.0, 20.0, 3, 4000, 0.1,
                               (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 0.0), (40.0, 40.0, 40.0, 40.0),
                               ((1.0, 0.0, -1.0), (1.0, 0.0, 1.0), (-1.0, 0.0, -1.0), (-1.0, 0.0, 1.0)),
                               (0.0, 20.0, 0.0), (0.0, 0.1, 0.0), (timestamp, 0.0, 0.0), progress, (0.0, 0.0, 0.0, 0.0, 0.0))
    static = SimulatorStaticInfo("car", "track", "", 8000, 100.0, (0.3, 0.3, 0.3, 0.3), 1000.0)
    return DiagnosticsFrame(timestamp, telemetry_timestamp=timestamp,
                            telemetry=telemetry, static_info=static, position=(timestamp, 0.0, 0.0), forward=(1.0, 0.0, 0.0),
                            normalized_progress=progress, progress=1000 * progress, track_length=1000.0,
                            lateral_offset=0.2, left_clearance=4.8, right_clearance=5.2,
                            reward=Reward(3.0, 1.0, 1.0, 0.5, 0.5, 0.0),
                            observed_action=np.asarray([0.1, 0.5]), observed_action_delta=np.asarray([0.05, 0.1]),
                            predicted_trajectory=np.asarray([[0.1, 0.0], [0.2, 0.0]]), horizons_seconds=(0.1, 0.2))


def test_frame_roundtrip_keeps_existing_models_arrays_missing_and_nonfinite_values() -> None:
    original = replace(frame(), raw_observation=np.asarray([1.0, np.nan, np.inf]), left_boundary=np.empty((0, 2)))
    restored = DiagnosticsFrame.from_bytes(original.to_bytes())
    assert restored.telemetry == original.telemetry
    assert restored.static_info == original.static_info
    assert restored.reward == original.reward
    assert restored.policy_mean is None
    np.testing.assert_array_equal(restored.raw_observation, original.raw_observation)
    assert restored.left_boundary is not None and restored.left_boundary.shape == (0, 2)
    assert restored.position == original.position
    assert DiagnosticsFrame.from_bytes(DiagnosticsFrame(1.0).to_bytes()).telemetry is None


def test_older_recording_frames_receive_defaults_for_checkpoint_and_observer_fields() -> None:
    data = json.loads(frame().to_bytes())
    for name in ("model_version", "model_timestamp", "policy_updates", "reward_style_weight",
                 "reward_source", "termination_reason", "reset_reason"):
        data.pop(name)
    restored = DiagnosticsFrame.from_bytes(json.dumps(data).encode())
    assert restored.model_version is None and restored.policy_updates is None
    assert restored.reward_source == "" and restored.reward_style_weight == 1.0
    assert restored.reward == frame().reward


@pytest.mark.parametrize("changes", [{"schema_version": 999}, {"timestamp": float("nan")},
                                      {"observed_action": [0.1]}, {"left_boundary": [[1.0, 2.0, 3.0]]},
                                      {"telemetry": []}, {"position": [0.0]}, {"progress": "bad"},
                                      {"status": 123}, {"reset": "false"}, {"sequence": -1}])
def test_frame_rejects_invalid_schema_or_dimensions(changes: dict[str, Any]) -> None:
    data = json.loads(frame().to_bytes())
    data.update(changes)
    with pytest.raises(ValueError):
        DiagnosticsFrame.from_bytes(json.dumps(data).encode())


def test_observation_layout_names_and_track_coordinates_match_builder_order() -> None:
    raw = np.arange(22, dtype=np.float64)
    names = observation_names(2, 2)
    geometry = observation_geometry(raw, 2, 2)
    assert len(names) == len(raw)
    assert names[-2:] == ("steering", "longitudinal_control")
    np.testing.assert_array_equal(geometry["left_boundary"], [[8, 9], [10, 11]])
    np.testing.assert_array_equal(geometry["reference_trajectory"], [[16, 17], [18, 19]])
    with pytest.raises(ValueError):
        observation_geometry(raw, 1, 2)


def test_ring_buffer_caps_samples_trims_time_and_resets_on_seek() -> None:
    history = FrameBuffer(capacity=3, seconds=2.0)
    for value in (0.0, 0.5, 1.0, 1.5):
        history.append(DiagnosticsFrame(value))
    assert [f.timestamp for f in history.frames()] == [0.5, 1.0, 1.5]
    history.append(DiagnosticsFrame(5.0))
    assert len(history.frames()) == 1
    history.append(DiagnosticsFrame(1.0))
    history.append(DiagnosticsFrame(1.0, sequence=4))
    assert len(history.frames()) == 1 and history.frames()[0].sequence == 4


def lap_frames() -> tuple[DiagnosticsFrame, ...]:
    # First crossing starts recording; the second crossing completes the lap.
    return tuple(frame(100.0 + index * 0.05, float(value % 1.0))
                 for index, value in enumerate(np.arange(0.95, 2.06, 0.05)))


def test_lap_detection_waits_for_two_forward_crossings() -> None:
    detector = LapDetector()
    laps = [lap for f in lap_frames() if (lap := detector.append(f)) is not None]
    assert len(laps) == 1
    assert laps[0][0].normalized_progress == pytest.approx(0.0)
    assert laps[0][-1].normalized_progress == pytest.approx(0.0)
    assert detector.discarded_laps == 0


@pytest.mark.parametrize("change", [{"offtrack": True}, {"reset": True}, {"terminated": True},
                                   {"truncated": True}, {"episode_id": 9}, {"normalized_progress": 0.9}])
def test_offtrack_reset_end_and_teleport_do_not_create_complete_laps(change: dict[str, Any]) -> None:
    detector = LapDetector()
    result: list[tuple[DiagnosticsFrame, ...]] = []
    for index, f in enumerate(lap_frames()):
        lap = detector.append(replace(f, **change) if index == 8 else f)
        if lap:
            result.append(lap)
    assert result == []
    assert detector.discarded_laps >= 1


def test_reverse_partial_gap_and_duplicate_frames_are_handled() -> None:
    detector = LapDetector()
    for progress in (0.1, 0.9, 0.8, 0.7, 0.2):
        assert detector.append(frame(100.0 + progress, progress)) is None
    detector = LapDetector(max_frames=4)
    for f in lap_frames():
        assert detector.append(f) is None
        assert detector.append(f) is None
    assert detector.discarded_laps > 0
    assert detector.last_discard_reason == "lap exceeds recording buffer"


def test_small_progress_jump_with_impossible_world_displacement_is_a_teleport() -> None:
    detector = LapDetector()
    for f in lap_frames()[:5]:
        detector.append(f)
    assert detector.append(replace(frame(100.25, 0.2), position=(1000.0, 0.0, 0.0))) is None
    assert detector.discarded_laps == 1
    assert "teleport" in detector.last_discard_reason


def test_recording_save_load_keeps_missing_data_provenance_and_never_overwrites(tmp_path: Path) -> None:
    recorder = DiagnosticsRecorder(tmp_path, "track", "car", run_id="abcdef")
    saved: Path | None = None
    for f in lap_frames():
        path = recorder.append(replace(f, run_id="abcdef"))
        saved = path or saved
    recorder.close()
    assert saved is not None
    recording = Recording.open(recorder.directory)
    assert recording.metadata["run_id"] == "abcdef"
    assert recording.metadata["complete_laps"] == 1
    assert recording.metadata["discarded_laps"] == 1  # trailing incomplete lap
    restored = load_lap(saved)
    assert restored[0].telemetry is not None
    assert restored[0].normalized_observation is None
    with pytest.raises(FileExistsError):
        save_lap(saved, restored)
    second = DiagnosticsRecorder(tmp_path, "track", "car")
    assert second.directory != recorder.directory
    second.close()


def test_recorder_assigns_first_agent_run_and_stops_on_changed_provenance(tmp_path: Path) -> None:
    recorder = DiagnosticsRecorder(tmp_path, "track", "car")
    recorder.append(replace(frame(), run_id="abc"))
    assert Recording.open(recorder.directory).metadata["run_id"] == "abc"
    with pytest.raises(ValueError, match="Training run changed"):
        recorder.append(replace(frame(100.1), run_id="def"))
    assert Recording.open(recorder.directory).metadata["last_discard_reason"] == "training run changed"
    assert frame().static_info is not None
    changed = SimulatorStaticInfo("other_car", "track", "", 8000, 100.0, (0.3, 0.3, 0.3, 0.3), 1000.0)
    with pytest.raises(ValueError, match="Vehicle / track changed"):
        recorder.append(replace(frame(100.2), static_info=changed))


def test_replay_frame_indexing_playback_speeds_and_empty_data() -> None:
    replay = ReplaySource(tuple(frame(100.0 + i * 0.05) for i in range(6)))
    replay.seek(-2)
    assert replay.index == 0
    replay.playing, replay.speed = True, 2.0
    replay.advance(0.05)
    assert replay.index == 2
    replay.seek(999)
    assert replay.index == 5
    replay.advance(1.0)
    assert not replay.playing
    current = replay.current_frame()
    assert current is not None and current.timestamp == 100.25
    assert ReplaySource().current_frame() is None


def test_future_uses_same_left_axis_as_training_interpolates_and_does_not_cross_resets() -> None:
    frames = tuple(replace(frame(100.0 + i * 0.05), position=(i * 0.05, 0.0, -i * 0.05)) for i in range(7))
    actual = future_trajectory(frames, 0)
    assert actual is not None
    np.testing.assert_allclose(actual, [[0.1, 0.1], [0.2, 0.2]], atol=1e-8)
    errors = prediction_errors(with_actual_future(frames, 0))
    assert errors is not None
    np.testing.assert_allclose(errors[:, 1], [0.1, 0.2], atol=1e-8)
    changed = tuple(replace(f, reset=True) if i == 3 else f for i, f in enumerate(frames))
    invalid = future_trajectory(changed, 0)
    assert invalid is not None and np.isnan(invalid[1]).all()
    last = future_trajectory(frames, len(frames) - 1)
    assert last is not None and np.isnan(last).all()


def test_lap_statistics_use_recorded_controls_rewards_and_actual_future_errors() -> None:
    frames = tuple(replace(frame(100.0 + i * 0.05), observed_action=np.asarray([0.1, 0.5 * (-1) ** i])) for i in range(8))
    statistics = lap_statistics(frames)
    assert statistics["mean_speed_kmh"] == 72.0
    assert statistics["total_reward"] == 24.0
    assert statistics["throttle_brake_switches"] == 7
    assert statistics["offtrack_events"] == 0
    assert statistics["mean_predictor_forward_mae_m"] == pytest.approx(0.0, abs=1e-8)
    missing = lap_statistics((DiagnosticsFrame(1.0), DiagnosticsFrame(2.0)))
    assert missing["total_reward"] is None
    assert missing["mean_predictor_forward_mae_m"] is None


def test_health_reports_missing_stale_invalid_clipped_and_inconsistent_values() -> None:
    checks = DiagnosticsChecks()
    assert reward_is_consistent(frame()) is True
    assert reward_is_consistent(DiagnosticsFrame(1.0)) is None
    invalid = replace(frame(), raw_observation=np.asarray([np.nan]), normalized_observation=np.asarray([10.0, 0.0, 0.0]),
                      clip_observations=10.0, lateral_offset=np.nan, reward=Reward(10.0, 1.0, 1.0, 1.0, 1.0, 0.0))
    messages = [c.message for c in checks.evaluate(invalid, now=110.0)]
    assert any("stale" in message for message in messages)
    assert any("NaN/Inf" in message for message in messages)
    assert any("clipping" in message for message in messages)
    assert "Track projection invalid" in messages
    assert "Reward components do not match total" in messages
    assert any("No model checkpoint" in c.message for c in checks.evaluate(DiagnosticsFrame(111.0)))


def test_checks_detect_pedal_switches_steering_oscillation_and_rising_error() -> None:
    checks = DiagnosticsChecks()
    result = ()
    for i in range(12):
        result = checks.evaluate(replace(frame(100.0 + i * 0.05), observed_action=np.asarray([0.0, (-1.0) ** i]),
                                         observed_action_delta=np.asarray([0.05 * (-1) ** i, 0.1]),
                                         predictor_lateral_mae=i * 0.1))
    messages = [c.message for c in result]
    assert any("switches" in m for m in messages)
    assert "High steering oscillation" in messages
    assert any("error rising" in m for m in messages)


def test_legacy_config_has_optional_disabled_diagnostics_and_template_enables_it() -> None:
    config = TrainingConfig.load(Path("configs/training.toml"))
    assert config.diagnostics.enabled
    text = config.toml_text
    start = text.index("[diagnostics]")
    end = text.find("\n[", start + 1)
    legacy = text[:start] + (text[end:] if end >= 0 else "")
    loaded = TrainingConfig.from_toml(legacy)
    assert not loaded.diagnostics.enabled
    assert loaded.toml_text == legacy
    config.require_compatible_resume(loaded)  # passive diagnostics may change


def test_live_source_without_agent_reuses_readers_geometry_and_reports_stale_packet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixture = frame()
    assert fixture.telemetry is not None and fixture.static_info is not None
    vehicle = fixture.telemetry
    shifted = [(x + 100.0, y, z) for x, y, z in vehicle.tyre_contact_points]
    vehicle.tyre_contact_points = (shifted[0], shifted[1], shifted[2], shifted[3])
    static = replace(fixture.static_info, track_spline_length=2000.0)
    closed: list[str] = []
    class Reader:
        last_packet_id = 3
        completed_laps = 0
        def read(self) -> TelemetryFrame:
            return vehicle
        def close(self) -> None:
            closed.append("telemetry")
    class StaticReader:
        def read(self) -> SimulatorStaticInfo:
            return static
        def close(self) -> None:
            closed.append("static")
    positions = [(0.0, 0.0, 0.0), (500.0, 0.0, 0.0), (500.0, 0.0, 500.0), (0.0, 0.0, 500.0)]
    directions = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)]
    track = TrackGeometry([TrackPoint(p, index * 500.0, 20.0, 20.0, (0.0, 1.0, 0.0), d)
                           for index, (p, d) in enumerate(zip(positions, directions))])
    monkeypatch.setattr(sources, "TelemetryReader", Reader)
    monkeypatch.setattr(sources, "StaticInfoReader", StaticReader)
    def read_track(self: sources.AssettoCorsaTrackReader) -> TrackGeometry:
        return track
    monkeypatch.setattr(sources.AssettoCorsaTrackReader, "read", read_track)
    monkeypatch.setattr(sources.time, "time", lambda: 100.0)
    spline_path = tmp_path / "content/tracks/track/ai/fast_lane.ai"
    spline_path.parent.mkdir(parents=True)
    spline_path.touch()
    config_file = tmp_path / "monitor.toml"
    content = TrainingConfig.load(Path("configs/training.toml")).toml_text
    content = replace_setting(content, "lookahead_distances", "lookahead_distances = [5.0, 10.0]")
    content = replace_setting(content, "reference_distances", "reference_distances = [5.0]")
    config_file.write_text(content)
    source = sources.LiveSource(tmp_path, config_file=config_file, runs_directory=tmp_path / "runs")
    try:
        current = source.current_frame()
        assert current.telemetry is vehicle
        assert current.raw_observation is not None and current.raw_observation[0] == 20.0
        assert current.left_boundary is not None and current.left_boundary.shape == (2, 2)
        assert current.reference_trajectory is not None
        assert current.reference_trajectory[0, 0] == pytest.approx(5.0)
        assert current.normalized_observation is None and current.reward is None
        monkeypatch.setattr(sources.time, "time", lambda: 102.0)
        stale = source.current_frame()
        assert stale.telemetry_timestamp == 100.0
        messages = [check.message for check in DiagnosticsChecks().evaluate(stale, 102.0)]
        assert any("telemetry stale" in message for message in messages)
    finally:
        source.close()
    assert closed == ["telemetry", "static"]


def test_live_source_survives_absent_ac_without_training_or_checkpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def absent() -> None:
        raise FileNotFoundError("AC missing")
    monkeypatch.setattr(sources, "TelemetryReader", absent)
    source = sources.LiveSource(tmp_path, runs_directory=tmp_path / "runs")
    try:
        first = source.current_frame()
        assert first.telemetry is None
        assert first.status == "No Assetto Corsa telemetry available"
        assert first.reward is None and first.observed_action is None
        assert first.normalized_observation is None
        assert any(check.message == "No model checkpoint loaded" for check in DiagnosticsChecks().evaluate(first))
    finally:
        source.close()
