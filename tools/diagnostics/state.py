"""Independent reward/termination state, driven only by observed AC frames."""

from dataclasses import replace

import numpy as np

from tools.diagnostics.frame import DiagnosticsFrame
from assetto_corsa_ai.future_reward import FuturePredictorRewardCalculator
from assetto_corsa_ai.reward import RewardCalculator, calculate_progress_delta
from assetto_corsa_ai.termination import EpisodeTerminationChecker, TerminationReason
from assetto_corsa_ai.track_projection import TrackProjector
from assetto_corsa_ai.training_config import TrainingConfig


class DiagnosticsState:
    """Same core classes, fresh instances; episode boundaries are inferred.

    Attaching halfway through an episode cannot recover its preceding stuck
    history or exact reset time. Results are explicitly local reconstructions.
    """
    def __init__(self, config: TrainingConfig, projector: TrackProjector) -> None:
        self.config, self.projector = config, projector
        r = config.reward
        self.reward = RewardCalculator(r.performance_weight, r.style_weight, r.style_sigma,
                                       r.off_track_penalty, r.stuck_penalty, r.collision_penalty)
        self.termination = EpisodeTerminationChecker(projector, config.termination.stuck_time,
                                                    config.termination.minimum_progress)
        self.future = FuturePredictorRewardCalculator(r.future_prediction, self.reward)
        self.previous: DiagnosticsFrame | None = None
        self.previous_packet: int | None = None
        self.previous_laps: int | None = None
        self.elapsed = 0.0
        self.samples = 0
        self.episode_id = 0
        self.ended = False

    def _reset_reason(self, frame: DiagnosticsFrame, packet: int | None, laps: int | None) -> str:
        previous = self.previous
        if previous is None:
            return "Observer attached / configuration changed"
        if packet is not None and self.previous_packet is not None and packet < self.previous_packet:
            return "Telemetry packet counter restarted"
        if laps is not None and self.previous_laps is not None and laps < self.previous_laps:
            return "Session lap counter restarted"
        dt = frame.timestamp - previous.timestamp
        if dt <= 0 or dt > max(0.5, self.config.environment.control_interval * 3):
            return "Telemetry gap / clock discontinuity"
        if frame.position is not None and previous.position is not None:
            speed = max(frame.telemetry.speed_kmh if frame.telemetry else 0.0,
                        previous.telemetry.speed_kmh if previous.telemetry else 0.0)
            displacement = float(np.linalg.norm(np.asarray(frame.position) - np.asarray(previous.position)))
            if displacement > 10.0 + (speed / 3.6 + 10.0) * dt * 2:
                return "Position jump / teleport"
        if frame.progress is not None and previous.progress is not None:
            delta = calculate_progress_delta(frame.progress, previous.progress, self.projector.track_length)
            if abs(delta) > self.projector.track_length * 0.2:
                return "Track progress discontinuity"
        if frame.telemetry and previous.telemetry:
            if any(a + 1e-4 < b for a, b in zip(frame.telemetry.car_damage, previous.telemetry.car_damage)):
                return "Damage cleared / reset or repair"
        if self.ended and not frame.offtrack and frame.telemetry and frame.telemetry.speed_kmh > self.config.environment.minimum_speed * 3.6:
            return "Driving resumed after observed termination"
        return ""

    def evaluate(self, frame: DiagnosticsFrame, packet: int | None = None, laps: int | None = None) -> DiagnosticsFrame:
        if frame.telemetry is None or frame.progress is None or frame.lateral_offset is None:
            return frame
        if self.previous is not None and packet == self.previous_packet and frame.timestamp == self.previous.timestamp:
            return replace(frame, episode_id=self.episode_id, reward=self.previous.reward,
                           reward_source=self.previous.reward_source, reward_style_weight=self.config.reward.style_weight,
                           terminated=self.previous.terminated, truncated=self.previous.truncated,
                           termination_reason=self.previous.termination_reason)
        reason = self._reset_reason(frame, packet, laps)
        previous = self.previous
        if reason:
            if previous is not None:
                self.episode_id += 1
            self.termination.reset()
            self.elapsed, self.samples, self.ended = 0.0, 0, False
            result = replace(frame, reset=True, reset_reason=reason, episode_id=self.episode_id,
                             reward=None, reward_source="Local reconstruction; observer state", reward_style_weight=self.config.reward.style_weight)
        elif self.ended:
            result = replace(frame, episode_id=self.episode_id,
                             terminated=previous.terminated if previous else False,
                             truncated=previous.truncated if previous else False, reward=None,
                             termination_reason=previous.termination_reason if previous else "",
                             reward_source="Observed episode ended; waiting for reset/resumed driving",
                             reward_style_weight=self.config.reward.style_weight)
        else:
            assert previous is not None and previous.progress is not None
            dt = frame.timestamp - previous.timestamp
            self.elapsed += dt
            self.samples += 1
            delta = calculate_progress_delta(frame.progress, previous.progress, self.projector.track_length)
            # Match the configured environment: no collision classifier is
            # currently installed by the training composition root.
            termination = self.termination.check(frame.telemetry, delta, dt)
            reward = self.reward.calculate(delta, frame.lateral_offset, termination)
            predicted_progress, lateral = None, None
            if (previous.predicted_trajectory is not None and previous.position is not None
                    and previous.forward is not None and previous.step is not None):
                future = self.future.calculate(np.asarray(previous.predicted_trajectory, dtype=np.float32),
                                               position=np.asarray(previous.position, dtype=np.float64),
                                               forward=np.asarray(previous.forward, dtype=np.float64),
                                               current_s=previous.progress, projector=self.projector, step=previous.step)
                reward = self.reward.with_prediction(reward, performance_prediction=future.performance_prediction,
                                                     style_prediction=future.style_prediction)
                predicted_progress = float(np.cumsum(future.progress_deltas).mean())
                lateral = float(np.abs(future.lateral_offsets).mean())
            env = self.config.environment
            truncated = bool((env.max_episode_time > 0 and self.elapsed >= env.max_episode_time)
                             or (env.max_episode_steps > 0 and self.samples >= env.max_episode_steps))
            self.ended = termination.terminated or truncated
            observed = np.asarray([frame.telemetry.steering_angle, frame.telemetry.gas - frame.telemetry.brake])
            old = (np.asarray([previous.telemetry.steering_angle, previous.telemetry.gas - previous.telemetry.brake])
                   if previous.telemetry else None)
            result = replace(frame, episode_id=self.episode_id, reward=reward,
                             reward_source="Local reconstruction; sampled telemetry + checkpoint",
                             reward_style_weight=self.config.reward.style_weight,
                             terminated=termination.terminated, truncated=truncated,
                             offtrack=frame.offtrack or termination.reason == TerminationReason.OFF_TRACK,
                             termination_reason=termination.reason.name if termination.reason else "",
                             previous_observed_action=old, observed_action_delta=observed - old if old is not None else None,
                             mean_predicted_progress=predicted_progress, mean_predicted_lateral_offset=lateral)
        self.previous, self.previous_packet, self.previous_laps = result, packet, laps
        return result
