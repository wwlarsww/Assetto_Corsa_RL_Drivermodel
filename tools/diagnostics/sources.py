"""Interchangeable read-only live and offline replay sources."""

from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np

from assetto_corsa_ai.assetto_corsa.static_info_reader import StaticInfoReader
from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.assetto_corsa.track_reader import AssettoCorsaTrackReader, track_ai_path
from tools.diagnostics.frame import DiagnosticsFrame, observation_names
from tools.diagnostics.model import CheckpointFollower
from tools.diagnostics.state import DiagnosticsState
from tools.diagnostics.recording import Recording, load_lap, with_actual_future
from assetto_corsa_ai.observation_builder import ObservationBuilder
from assetto_corsa_ai.static_info import SimulatorStaticInfo
from assetto_corsa_ai.telemetry import TelemetryFrame
from assetto_corsa_ai.track_geometry import TrackGeometry
from assetto_corsa_ai.track_perception import TrackPerceptionCalculator
from assetto_corsa_ai.track_projection import TrackProjector
from assetto_corsa_ai.vehicle_geometry import vehicle_forward
from assetto_corsa_ai.training_config import TrainingConfig


class DiagnosticsSource(Protocol):
    def current_frame(self) -> DiagnosticsFrame | None: ...


class LiveSource:
    """Only the existing read-only readers; deliberately no controller imports."""
    def __init__(self, ac_directory: Path | None = None, config_file: Path = Path("configs/training.toml"),
                 run_directory: Path | None = None, runs_directory: Path = Path("runs")) -> None:
        self._fallback = TrainingConfig.load(config_file) if config_file.is_file() else None
        self.config = self._fallback
        self.ac_directory = ac_directory or Path(self.config.simulator.assetto_corsa_directory if self.config else
                                                "C:/Program Files (x86)/Steam/steamapps/common/assettocorsa")
        self._ac_directory_override = ac_directory
        self.models = CheckpointFollower(run_directory, runs_directory)
        self._state: DiagnosticsState | None = None
        self._builder: ObservationBuilder | None = None
        self._telemetry_reader: TelemetryReader | None = None
        self._static_reader: StaticInfoReader | None = None
        self._projector: TrackProjector | None = None
        self._track: TrackGeometry | None = None
        self._track_key: tuple[str, str, str] | None = None
        self._packet_id: int | None = None
        self._telemetry_timestamp: float | None = None
        self._retry_at = 0.0
        self._frame: DiagnosticsFrame | None = None

    def _disconnect(self) -> None:
        for reader in (self._telemetry_reader, self._static_reader):
            if reader is not None:
                reader.close()
        self._telemetry_reader = None
        self._static_reader = None
        self._telemetry_timestamp = None
        self._packet_id = None
        self._state = None
        self._track_key, self._builder = None, None

    def _read_vehicle(self, now: float) -> tuple[TelemetryFrame | None, SimulatorStaticInfo | None, str]:
        try:
            if self._telemetry_reader is None or self._static_reader is None:
                if now < self._retry_at:
                    return None, None, "No Assetto Corsa telemetry available"
                self._retry_at = now + 1.0
                self._telemetry_reader = TelemetryReader()
                self._static_reader = StaticInfoReader()
            telemetry, static = self._telemetry_reader.read(), self._static_reader.read()
            if not static.car_model or not static.track_name or static.track_spline_length <= 0:
                return None, None, "No Assetto Corsa telemetry available"
            packet = self._telemetry_reader.last_packet_id
            if packet != self._packet_id:
                self._packet_id, self._telemetry_timestamp = packet, now
            return telemetry, static, ""
        except (OSError, ValueError):
            self._disconnect()
            self._retry_at = now + 1.0
            return None, None, "No Assetto Corsa telemetry available"

    def _track_frame(self, telemetry: TelemetryFrame, static: SimulatorStaticInfo) -> dict[str, Any]:
        key = (static.track_name, static.track_configuration, static.car_model)
        if key != self._track_key:
            track = AssettoCorsaTrackReader(track_ai_path(static, self.ac_directory)).read()
            self._track, self._projector, self._track_key = track, TrackProjector(track), key
            self._builder = None
            self._state = DiagnosticsState(self.config, self._projector) if self.config else None
        if self._track is None or self._projector is None:
            return {}
        projection = self._projector.project(telemetry.car_coordinates)
        spline_s = projection.spline_distance(self._track, static.track_spline_length)
        if self._builder is None:
            env = self.config.environment if self.config else None
            self._builder = ObservationBuilder.create(
                front_axle_distance=env.front_axle_distance if env else 0.0,
                rear_axle_distance=env.rear_axle_distance if env else 0.0,
                telemetry=telemetry, minimum_speed=env.minimum_speed if env else 1.0,
                lookahead_distances=env.lookahead_distances if env else TrackPerceptionCalculator.DEFAULT_LOOKAHEAD_DISTANCES,
                reference_distances=env.reference_distances if env else TrackPerceptionCalculator.DEFAULT_REFERENCE_DISTANCES,
            )
        observation = self._builder.build(telemetry, static, self._track, spline_s)
        return dict(position=telemetry.car_coordinates,
                    forward=vehicle_forward(telemetry.tyre_contact_points), progress=projection.distance_along_track,
                    normalized_progress=projection.normalized_track_position,
                    track_length=self._projector.track_length, lateral_offset=projection.lateral_offset,
                    left_clearance=projection.left_clearance, right_clearance=projection.right_clearance,
                    offtrack=self._projector.is_off_track(telemetry.tyre_contact_points),
                    raw_observation=np.asarray(observation.to_array(), dtype=np.float64),
                    observation_names=observation_names(len(observation.left_boundary), len(observation.reference_waypoints)),
                    left_boundary=np.asarray(observation.left_boundary, dtype=np.float64),
                    right_boundary=np.asarray(observation.right_boundary, dtype=np.float64),
                    reference_trajectory=np.asarray(observation.reference_waypoints, dtype=np.float64))

    def current_frame(self) -> DiagnosticsFrame:
        now = time.time()
        model = self.models.poll(now)
        config = self.models.config or self._fallback
        if config != self.config:
            self.config = config
            self._track_key, self._builder, self._state = None, None, None
            if config:
                self.ac_directory = self._ac_directory_override or Path(config.simulator.assetto_corsa_directory)
        telemetry, static, status = self._read_vehicle(now)
        direct: dict[str, Any] = {}
        if telemetry is not None and static is not None:
            try:
                direct = self._track_frame(telemetry, static)
            except (OSError, ValueError, IndexError):
                status = "Track geometry unavailable"
        frame = DiagnosticsFrame(timestamp=self._telemetry_timestamp or now, telemetry=telemetry, static_info=static,
                                 telemetry_timestamp=self._telemetry_timestamp, run_id=self.models.run_id,
                                 status=status or self.models.status,
                                 observed_action=np.asarray([telemetry.steering_angle, telemetry.gas - telemetry.brake]) if telemetry else None,
                                 **direct)
        if model is not None and static is not None and frame.raw_observation is not None:
            expected = model.config.environment
            if (static.car_model, static.track_name, static.track_configuration) != (expected.car, expected.track, expected.track_layout):
                frame = replace(frame, status="Selected run does not match the live car/track/layout")
            else:
                try:
                    inference = model.infer(frame.raw_observation)
                    frame = replace(frame, normalized_observation=inference.normalized, policy_mean=inference.policy_mean,
                                    predicted_trajectory=inference.trajectory, clip_observations=model.normalizer.clip_obs,
                                    horizons_seconds=model.config.predictor.horizons_seconds,
                                    model_version=model.directory.name, model_timestamp=model.created_at,
                                    step=model.step, policy_updates=model.updates)
                except (ValueError, RuntimeError) as error:
                    frame = replace(frame, status=f"Checkpoint inference unavailable: {error}")
        if self._state is not None:
            reader = self._telemetry_reader
            try:
                frame = self._state.evaluate(frame, self._packet_id, reader.completed_laps if reader is not None else None)
            except ValueError as error:
                # Invalid telemetry remains visible to health checks; discard
                # partial observer history before trying the next valid frame.
                if self.config is not None and self._projector is not None:
                    self._state = DiagnosticsState(self.config, self._projector)
                frame = replace(frame, reward=None, status=f"Local reward unavailable: {error}")
        self._frame = frame
        return self._frame

    def select_run(self, directory: Path) -> None:
        self.models.select_run(directory)

    def close(self) -> None:
        self._disconnect()


class ReplaySource:
    def __init__(self, frames: tuple[DiagnosticsFrame, ...] = ()) -> None:
        self.frames = frames
        self._times = np.asarray([frame.timestamp for frame in frames])
        self.recording: Recording | None = None
        self.index = 0
        self.playing = False
        self.speed = 1.0
        self._playback_time = frames[0].timestamp if frames else 0.0
        self._cached_index = -1
        self._cached_frame: DiagnosticsFrame | None = None

    def open(self, directory: Path) -> None:
        self.recording = Recording.open(directory)
        self.frames = ()
        if self.recording.laps:
            self.select_lap(0)

    def select_lap(self, index: int) -> None:
        if self.recording is None or not 0 <= index < len(self.recording.laps):
            raise IndexError("Replay lap index is outside the recording.")
        self.frames = load_lap(self.recording.laps[index])
        self._times = np.asarray([frame.timestamp for frame in self.frames])
        self.playing = False
        self._cached_index = -1
        self.seek(0)

    def seek(self, index: int) -> None:
        if not self.frames:
            self.index = 0
            return
        self.index = min(max(0, index), len(self.frames) - 1)
        self._playback_time = self.frames[self.index].timestamp

    def advance(self, elapsed: float) -> None:
        if not self.playing or not self.frames:
            return
        self._playback_time += max(0.0, elapsed) * self.speed
        while self.index + 1 < len(self.frames) and self.frames[self.index + 1].timestamp <= self._playback_time:
            self.index += 1
        if self.index == len(self.frames) - 1:
            self.playing = False

    def current_frame(self) -> DiagnosticsFrame | None:
        if not self.frames:
            return None
        if self._cached_index != self.index:
            self._cached_frame = with_actual_future(self.frames, self.index, self._times)
            self._cached_index = self.index
        return self._cached_frame
