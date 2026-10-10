"""Drive to an episode start without producing RL transitions."""

from collections.abc import Callable
from math import atan2, cos, dist, radians
import time

import numpy as np

from assetto_corsa_ai.assetto_corsa.simulator_controller import AssettoCorsaSimulatorController
from assetto_corsa_ai.episode_start import EpisodeStart, StartConfig
from assetto_corsa_ai.reward import calculate_progress_delta
from assetto_corsa_ai.telemetry import TelemetryFrame
from assetto_corsa_ai.track_geometry import TrackGeometry, Vector3
from assetto_corsa_ai.track_projection import TrackProjector


class PreRollError(RuntimeError):
    """The requested rolling start could not be reached safely."""


class ResetPreRollController:
    def __init__(self, track: TrackGeometry, config: StartConfig) -> None:
        config.validate()
        self.config = config
        self.projector = TrackProjector(track)
        self.positions = np.asarray([p.position for p in track.points], dtype=np.float64)
        self.vectors = np.roll(self.positions, -1, axis=0) - self.positions
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.offsets = np.concatenate(([0.0], np.cumsum(self.lengths)))

    def pose_at(self, distance: float) -> EpisodeStart:
        distance %= self.projector.track_length
        index = min(int(np.searchsorted(self.offsets, distance, side="right") - 1), len(self.lengths) - 1)
        fraction = (distance - self.offsets[index]) / self.lengths[index]
        position = self.positions[index] + fraction * self.vectors[index]
        forward = self.vectors[index] / self.lengths[index]
        return EpisodeStart((float(position[0]), float(position[1]), float(position[2])),
                            (float(forward[0]), float(forward[1]), float(forward[2])))

    def approach(self, start: EpisodeStart) -> tuple[EpisodeStart, float]:
        speed = start.target_speed_kmh / 3.6
        distance = max(self.config.min_preroll_distance,
                       speed * speed / (2 * self.config.acceleration_estimate_mps2)
                       + speed * self.config.settling_time)
        if distance > min(self.config.max_preroll_distance, self.projector.track_length / 2):
            raise PreRollError("Insufficient track distance for the requested rolling start.")
        target_s = self.projector.project(start.position).distance_along_track
        return self.pose_at(target_s - distance), distance

    def run(self, start: EpisodeStart, approach_distance: float, *,
            simulator: AssettoCorsaSimulatorController, read: Callable[[], TelemetryFrame],
            forward: Callable[[TelemetryFrame], Vector3], interval: float,
            position_tolerance: float, heading_tolerance_degrees: float,
            speed_tolerance_kmh: float) -> TelemetryFrame:
        deadline = time.monotonic() + self.config.timeout
        initial = read()
        damage = np.asarray(initial.car_damage)
        previous_s = self.projector.project(initial.car_coordinates).distance_along_track
        progress = 0.0
        alignment_limit = cos(radians(heading_tolerance_degrees))
        target_forward = np.asarray(start.forward, dtype=float)
        target_forward /= np.linalg.norm(target_forward)
        try:
            while time.monotonic() < deadline:
                tick = time.monotonic()
                telemetry = read()
                if not np.isfinite(telemetry.speed_kmh) or np.any(np.asarray(telemetry.car_damage) > damage):
                    raise PreRollError("Invalid telemetry or collision during pre-roll.")
                tyre_projections = [self.projector.project(p) for p in telemetry.tyre_contact_points]
                if any(min(p.left_clearance, p.right_clearance) < 0 for p in tyre_projections):
                    raise PreRollError("Vehicle left the track during pre-roll.")
                projection = self.projector.project(telemetry.car_coordinates)
                progress += calculate_progress_delta(current_s=projection.distance_along_track,
                    previous_s=previous_s, track_length=self.projector.track_length)
                previous_s = projection.distance_along_track
                heading = np.asarray(forward(telemetry), dtype=float)
                if (dist(telemetry.car_coordinates, start.position) <= position_tolerance
                        and float(heading @ target_forward) >= alignment_limit
                        and abs(telemetry.speed_kmh - start.target_speed_kmh) <= speed_tolerance_kmh):
                    # Leave the applied controls in place for the residual policy.
                    return telemetry
                if progress > approach_distance + position_tolerance:
                    raise PreRollError("Pre-roll passed the start without meeting pose/speed tolerances.")
                lookahead = max(self.config.lookahead_distance,
                                telemetry.speed_kmh / 3.6 * self.config.lookahead_time)
                aim = self.pose_at(projection.distance_along_track + lookahead)
                direction = np.asarray(aim.position) - np.asarray(telemetry.car_coordinates)
                right = np.cross(heading, np.array([0.0, 1.0, 0.0]))
                steering = float(np.clip(self.config.steering_gain * atan2(float(direction @ right),
                    float(direction @ heading)), -1.0, 1.0))
                longitudinal = float(np.clip(self.config.speed_gain *
                    (start.target_speed_kmh - telemetry.speed_kmh), -1.0, 1.0))
                simulator.set_controls(steering, max(longitudinal, 0.0), max(-longitudinal, 0.0))
                time.sleep(max(0.0, min(interval - (time.monotonic() - tick), deadline - time.monotonic())))
            raise PreRollError("Pre-roll timed out before reaching the episode start.")
        except BaseException:
            simulator.set_controls(0.0, 0.0, 1.0)
            raise
