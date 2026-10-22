"""All shared-memory, checkpoint and recording I/O lives in this worker."""

from dataclasses import dataclass
from pathlib import Path
from collections import deque
from threading import Event, Thread
import time
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Signal

from tools.diagnostics.buffer import FrameBuffer
from tools.diagnostics.checks import DiagnosticsChecks, HealthCheck
from tools.diagnostics.frame import Array, DiagnosticsFrame
from tools.diagnostics.recorder import DiagnosticsRecorder
from tools.diagnostics.recording import lap_statistics, prediction_errors, with_actual_future
from tools.diagnostics.sources import LiveSource, ReplaySource


@dataclass(frozen=True)
class MonitorUpdate:
    frame: DiagnosticsFrame
    history: tuple[DiagnosticsFrame, ...]
    checks: tuple[HealthCheck, ...]
    index: int = 0
    count: int = 0
    playing: bool = False


class MonitorWorker(QObject):
    # Only a wake-up is queued. The latest payload replaces older updates so a
    # slow UI cannot build an unbounded queue of 400-frame histories.
    updated = Signal()
    message = Signal(str)
    recording_opened = Signal(object)
    analysis_ready = Signal(object)

    def __init__(self, live_settings: dict[str, Any], recordings_root: Path) -> None:
        super().__init__()
        self.live_settings, self.recordings_root = live_settings, recordings_root
        self._commands: deque[tuple[str, Any]] = deque(maxlen=64)
        self._stop = Event()
        self._thread = Thread(target=self._run, name="diagnostics-source", daemon=True)
        self.latest: MonitorUpdate | None = None
        self._notification_pending = False

    def start(self) -> None:
        self._thread.start()

    def command(self, name: str, value: Any = None) -> None:
        self._commands.append((name, value))

    def consume_update(self) -> MonitorUpdate | None:
        self._notification_pending = False
        return self.latest

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _analyse(self, source: ReplaySource) -> None:
        frames = source.frames
        statistics = lap_statistics(frames)
        times = np.asarray([f.timestamp for f in frames])
        errors = [prediction_errors(with_actual_future(frames, i, times)) for i in range(len(frames))]
        progress = np.asarray([f.normalized_progress if f.normalized_progress is not None else np.nan for f in frames])
        length = next((f.track_length for f in frames if f.track_length is not None), None)
        if length is not None and np.all(np.isfinite(progress)):
            x = np.unwrap(progress * 2 * np.pi) * length / (2 * np.pi)
        else:
            x = np.asarray([f.progress if f.progress is not None else np.nan for f in frames])
        def telemetry(field: str) -> Array:
            return np.asarray([getattr(f.telemetry, field) if f.telemetry else np.nan for f in frames], dtype=np.float64)
        def mae(axis: int) -> Array:
            return np.asarray([float(np.mean(e[:, axis][np.isfinite(e[:, axis])])) if e is not None and np.any(np.isfinite(e[:, axis]))
                               else np.nan for e in errors])
        series = ((telemetry("speed_kmh"),), (np.asarray([f.reward.total if f.reward else np.nan for f in frames]),),
                  (np.asarray([f.lateral_offset if f.lateral_offset is not None else np.nan for f in frames]),),
                  (mae(0), mae(1)), (telemetry("steering_angle"),), (telemetry("gas"), telemetry("brake")))
        self.analysis_ready.emit((statistics, x, series))

    def _run(self) -> None:
        live: LiveSource | None = None
        replay = ReplaySource()
        mode = "LIVE"
        history, checks = FrameBuffer(), DiagnosticsChecks()
        recorder: DiagnosticsRecorder | None = None
        last = time.monotonic()
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                elapsed, last = started - last, started
                while True:
                    try:
                        name, value = self._commands.popleft()
                    except IndexError:
                        break
                    try:
                        if name == "mode":
                            mode = str(value)
                            history.clear()
                            checks = DiagnosticsChecks()
                            if mode == "REPLAY" and recorder:
                                recorder.close()
                                recorder = None
                                self.message.emit("Recording stopped on switching to Replay")
                        elif name == "open":
                            if recorder:
                                recorder.close()
                                recorder = None
                            replay.open(Path(value))
                            mode = "REPLAY"
                            history.clear()
                            checks = DiagnosticsChecks()
                            self.recording_opened.emit(tuple(p.stem for p in replay.recording.laps) if replay.recording else ())
                            self._analyse(replay)
                        elif name == "lap":
                            replay.select_lap(int(value))
                            history.clear()
                            checks = DiagnosticsChecks()
                            self._analyse(replay)
                        elif name == "seek":
                            replay.playing = False
                            replay.seek(int(value))
                            history.clear()
                            checks = DiagnosticsChecks()
                        elif name == "step":
                            replay.playing = False
                            replay.seek(replay.index + int(value))
                        elif name == "play":
                            replay.playing = bool(value)
                        elif name == "speed":
                            replay.speed = float(value)
                        elif name == "run":
                            self.live_settings["run_directory"] = Path(value)
                            if live:
                                live.select_run(Path(value))
                            self.message.emit(f"Following run: {value}")
                        elif name == "record":
                            if recorder:
                                recorder.close()
                                recorder = None
                                self.message.emit("Recording stopped")
                            elif mode == "LIVE" and self.latest and self.latest.frame.static_info:
                                frame = self.latest.frame
                                assert frame.static_info is not None
                                recorder = DiagnosticsRecorder(self.recordings_root, frame.static_info.track_name,
                                                               frame.static_info.car_model, frame.run_id)
                                self.message.emit(f"Recording {recorder.directory}; waiting for a full valid lap")
                            else:
                                self.message.emit("Recording requires live car/track telemetry")
                    except (OSError, ValueError, IndexError, KeyError) as error:
                        self.message.emit(f"Diagnostics: {error}")
                if mode == "LIVE":
                    if live is None:
                        try:
                            live = LiveSource(**self.live_settings)
                        except (OSError, ValueError) as error:
                            self.message.emit(f"Live source unavailable: {error}")
                            self._stop.wait(1.0)
                            continue
                    frame = live.current_frame()
                else:
                    replay.advance(elapsed)
                    frame = replay.current_frame()
                    if frame is None:
                        frame = DiagnosticsFrame(timestamp=time.time(), status="Open a recording with complete laps")
                if recorder:
                    try:
                        saved = recorder.append(frame)
                        if saved:
                            self.message.emit(f"Saved complete lap: {saved}")
                    except (OSError, ValueError) as error:
                        self.message.emit(f"Recording stopped: {error}")
                        recorder = None
                history.append(frame)
                visible_history = history.frames()
                if mode == "REPLAY" and replay.frames:
                    visible_history = tuple(f for f in replay.frames[max(0, replay.index - 399):replay.index + 1]
                                            if frame.timestamp - f.timestamp <= 20.0)
                    checks.history.clear()
                    for previous in visible_history[-100:]:
                        checks.history.append(previous)
                self.latest = MonitorUpdate(frame, visible_history, checks.evaluate(frame, time.time(), mode == "REPLAY"),
                                            replay.index, len(replay.frames), replay.playing)
                if not self._notification_pending:
                    self._notification_pending = True
                    self.updated.emit()
                self._stop.wait(max(0.0, 0.05 - (time.monotonic() - started)))
        except Exception as error:
            self.message.emit(f"Diagnostics source stopped: {error}")
        finally:
            if live:
                live.close()
            if recorder:
                recorder.close()
