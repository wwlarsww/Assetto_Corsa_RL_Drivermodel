"""Record complete forward laps, never partial episodes or reset jumps."""

from datetime import datetime
import json
from pathlib import Path
import re
import uuid

import numpy as np

from tools.diagnostics.frame import DiagnosticsFrame, SCHEMA_VERSION
from tools.diagnostics.recording import MAX_LAP_FRAMES, save_lap


class LapDetector:
    def __init__(self, max_frames: int = MAX_LAP_FRAMES) -> None:
        self.max_frames = max_frames
        self._previous: DiagnosticsFrame | None = None
        self._frames: list[DiagnosticsFrame] = []
        self._distance = 0.0
        self.discarded_laps = 0
        self.last_discard_reason = ""

    def _discard(self, reason: str) -> None:
        if self._frames:
            self.discarded_laps += 1
            self.last_discard_reason = reason
        self._frames.clear()
        self._distance = 0.0

    def append(self, frame: DiagnosticsFrame) -> tuple[DiagnosticsFrame, ...] | None:
        if self._previous is not None and frame.timestamp == self._previous.timestamp:
            return None
        previous, self._previous = self._previous, frame
        progress = frame.normalized_progress
        if progress is None or not np.isfinite(progress) or not 0 <= progress <= 1:
            self._discard("invalid projection")
            return None
        if frame.offtrack or frame.terminated or frame.truncated or frame.reset:
            self._discard("offtrack / episode ended / reset")
            return None
        if previous is None or previous.normalized_progress is None:
            return None
        delta = progress - previous.normalized_progress
        wrap = previous.normalized_progress > 0.8 and progress < 0.2
        movement = delta + 1 if wrap else delta
        teleport = False
        if frame.position is not None and previous.position is not None:
            speed = max(frame.telemetry.speed_kmh if frame.telemetry else 0.0,
                        previous.telemetry.speed_kmh if previous.telemetry else 0.0)
            displacement = float(np.linalg.norm(np.asarray(frame.position) - np.asarray(previous.position)))
            plausible_distance = 10.0 + (speed / 3.6 + 10.0) * max(0.0, frame.timestamp - previous.timestamp) * 2
            teleport = displacement > plausible_distance
        if (frame.timestamp <= previous.timestamp or frame.timestamp - previous.timestamp > 0.5
                or frame.episode_id != previous.episode_id or frame.run_id != previous.run_id
                or movement < -0.02 or movement > 0.2 or teleport):
            self._discard("gap / reverse / teleport / episode changed")
            return None
        completed: tuple[DiagnosticsFrame, ...] | None = None
        if wrap:
            if self._frames and self._distance + movement >= 0.9:
                completed = tuple(self._frames + [frame])
            elif self._frames:
                self._discard("incomplete lap")
            self._frames = [frame]
            self._distance = 0.0
        elif self._frames:
            self._distance += max(0.0, movement)
            self._frames.append(frame)
            if len(self._frames) >= self.max_frames:
                self._discard("lap exceeds recording buffer")
        return completed

    def close(self) -> None:
        self._discard("recording stopped before lap completion")


class DiagnosticsRecorder:
    def __init__(self, root: Path, track: str, car: str, run_id: str | None = None,
                 sample_rate_hz: float = 20.0) -> None:
        def identifier(value: str) -> str:
            return re.sub(r"[^A-Za-z0-9_-]", "_", value) or "unknown"
        now = datetime.now().astimezone()
        self.directory = root / f"{now:%Y%m%d_%H%M%S}_{identifier(track)}_{identifier(car)}_{uuid.uuid4().hex[:6]}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = {"schema_version": SCHEMA_VERSION, "track": track, "car": car,
                         "created_at": now.isoformat(), "sample_rate_hz": sample_rate_hz,
                         "run_id": run_id, "complete_laps": 0, "discarded_laps": 0,
                         "last_discard_reason": ""}
        self.detector = LapDetector()
        self.laps_saved = 0
        self._save_metadata()

    def _save_metadata(self) -> None:
        self.metadata["complete_laps"] = self.laps_saved
        self.metadata["discarded_laps"] = self.detector.discarded_laps
        self.metadata["last_discard_reason"] = self.detector.last_discard_reason
        (self.directory / "metadata.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def append(self, frame: DiagnosticsFrame) -> Path | None:
        if frame.static_info is not None and (frame.static_info.track_name != self.metadata["track"]
                                             or frame.static_info.car_model != self.metadata["car"]):
            self.detector.close()
            self.metadata["last_discard_reason"] = "vehicle / track changed"
            self.detector.last_discard_reason = "vehicle / track changed"
            self._save_metadata()
            raise ValueError("Vehicle / track changed; start a new recording.")
        if self.metadata["run_id"] is None and frame.run_id is not None:
            self.metadata["run_id"] = frame.run_id
            self._save_metadata()
        elif frame.run_id is not None and frame.run_id != self.metadata["run_id"]:
            self.detector.close()
            self.detector.last_discard_reason = "training run changed"
            self._save_metadata()
            raise ValueError("Training run changed; start a new recording.")
        discarded = self.detector.discarded_laps
        lap = self.detector.append(frame)
        if lap is None:
            if self.detector.discarded_laps != discarded:
                self._save_metadata()
            return None
        path = self.directory / f"lap_{self.laps_saved + 1:03d}.npz"
        save_lap(path, lap)
        self.laps_saved += 1
        self._save_metadata()
        return path

    def close(self) -> None:
        self.detector.close()
        self._save_metadata()
