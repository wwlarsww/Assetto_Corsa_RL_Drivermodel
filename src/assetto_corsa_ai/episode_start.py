"""Seeded episode starts derived from local AI-line speed samples."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from assetto_corsa_ai.track_geometry import TrackGeometry, Vector3


@dataclass(frozen=True)
class EpisodeStart:
    position: Vector3
    forward: Vector3
    target_speed_kmh: float = 0.0
    reference_speed_kmh: float | None = None
    speed_factor: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.target_speed_kmh) or self.target_speed_kmh < 0:
            raise ValueError("target_speed_kmh must be finite and non-negative.")


@dataclass(frozen=True)
class StartConfig:
    enabled: bool = False
    reference_speed_unit: str = "unvalidated"
    reference_speed_factor_min: float = 0.30
    reference_speed_factor_max: float = 0.60
    min_speed_kmh: float = 20.0
    max_speed_kmh: float = 120.0
    max_attempts: int = 3
    timeout: float = 20.0
    acceleration_estimate_mps2: float = 3.0
    settling_time: float = 1.0
    min_preroll_distance: float = 10.0
    max_preroll_distance: float = 250.0
    lookahead_distance: float = 8.0
    lookahead_time: float = 0.4
    steering_gain: float = 1.0
    speed_gain: float = 0.1

    def validate(self) -> None:
        if self.reference_speed_unit not in ("unvalidated", "kmh", "mps"):
            raise ValueError("reference_speed_unit must be unvalidated, kmh or mps.")
        if self.enabled and self.reference_speed_unit == "unvalidated":
            raise ValueError("Validate reference_speed_unit before enabling pre-roll.")
        if not 0 < self.reference_speed_factor_min <= self.reference_speed_factor_max <= 1:
            raise ValueError("Reference speed factors must satisfy 0 < min <= max <= 1.")
        if not 0 < self.min_speed_kmh <= self.max_speed_kmh:
            raise ValueError("Start speed limits must satisfy 0 < min <= max.")
        if type(self.max_attempts) is not int or self.max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer.")
        for name in ("timeout", "acceleration_estimate_mps2", "settling_time",
                     "min_preroll_distance", "max_preroll_distance", "lookahead_distance",
                     "lookahead_time", "steering_gain", "speed_gain", "min_speed_kmh",
                     "max_speed_kmh", "reference_speed_factor_min", "reference_speed_factor_max"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.min_preroll_distance > self.max_preroll_distance:
            raise ValueError("Pre-roll distance limits are reversed.")

    def sample(self, track: TrackGeometry, rng: np.random.Generator) -> EpisodeStart:
        self.validate()
        if not self.enabled:
            point = track.points[int(rng.integers(len(track.points)))]
            return EpisodeStart(point.position, point.forward)
        eligible = [p for p in track.points if p.reference_speed is not None
                    and isfinite(p.reference_speed) and p.reference_speed > 0]
        if not eligible:
            raise ValueError("Track contains no valid positive reference speeds.")
        point = eligible[int(rng.integers(len(eligible)))]
        assert point.reference_speed is not None
        reference = point.reference_speed * (3.6 if self.reference_speed_unit == "mps" else 1.0)
        factor = float(rng.uniform(self.reference_speed_factor_min, self.reference_speed_factor_max))
        speed = float(np.clip(reference * factor, self.min_speed_kmh, self.max_speed_kmh))
        return EpisodeStart(point.position, point.forward, speed, reference, factor)
