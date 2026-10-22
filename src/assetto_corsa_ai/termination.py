"""Episode termination logic for the RL driving environment."""

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

from assetto_corsa_ai.telemetry import TelemetryFrame
from assetto_corsa_ai.track_projection import TrackProjector


class TerminationReason(Enum):
    """Reason why an RL episode was terminated."""

    OFF_TRACK = auto()
    STUCK_OR_WRONG_WAY = auto()
    STRONG_COLLISION = auto()


@dataclass(frozen=True)
class TerminationResult:
    """Result of an episode termination check."""

    terminated: bool
    reason: TerminationReason | None


class EpisodeTerminationChecker:
    """Detect conditions that require termination of an episode.

    Current termination conditions:

    - at least one tyre contact point is outside the track
    - insufficient forward progress over a configurable time window
    - externally detected strong collision
    """

    def __init__(
        self,
        track_projector: TrackProjector,
        stuck_time: float = 4.0,
        minimum_progress: float = 2.0,
    ) -> None:
        if stuck_time <= 0.0:
            raise ValueError("stuck_time must be positive.")

        if minimum_progress < 0.0:
            raise ValueError("minimum_progress must not be negative.")

        self._track_projector = track_projector

        self._stuck_time = stuck_time
        self._minimum_progress = minimum_progress

        self._elapsed_time = 0.0

        self._progress_history: deque[tuple[float, float]] = deque()

    def reset(self) -> None:
        """Reset internal state at the beginning of an episode."""

        self._elapsed_time = 0.0
        self._progress_history.clear()

    def check(
        self,
        telemetry: TelemetryFrame,
        delta_s: float,
        dt: float,
        strong_collision: bool = False,
    ) -> TerminationResult:
        """Check whether the current episode should terminate.

        Args:
            telemetry:
                Current simulator telemetry.

            delta_s:
                Signed progress since the previous agent step [m].

            dt:
                Duration of the current agent step [s].

            strong_collision:
                Whether an external collision detector classified the
                current event as a strong collision.
        """

        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        if self._is_off_track(
            telemetry=telemetry,
        ):
            return TerminationResult(
                terminated=True,
                reason=TerminationReason.OFF_TRACK,
            )

        if strong_collision:
            return TerminationResult(
                terminated=True,
                reason=TerminationReason.STRONG_COLLISION,
            )

        if self._is_stuck_or_wrong_way(
            delta_s=delta_s,
            dt=dt,
        ):
            return TerminationResult(
                terminated=True,
                reason=TerminationReason.STUCK_OR_WRONG_WAY,
            )

        return TerminationResult(
            terminated=False,
            reason=None,
        )

    def _is_off_track(
        self,
        telemetry: TelemetryFrame,
    ) -> bool:
        """Return True if at least one tyre is outside the track.
        This is determined by projecting the tyre contact points onto the track and checking the left and right clearances."""

        return self._track_projector.is_off_track(telemetry.tyre_contact_points)

    def _is_stuck_or_wrong_way(
        self,
        delta_s: float,
        dt: float,
    ) -> bool:
        """Detect insufficient forward progress over a time window.

        Both standing still and sustained backwards driving produce
        insufficient net forward progress and therefore trigger the
        same termination condition.
        """

        self._elapsed_time += dt

        self._progress_history.append(
            (
                self._elapsed_time,
                delta_s,
            )
        )

        window_start = self._elapsed_time - self._stuck_time

        # Entries contain progress accumulated up to their timestamp. A step
        # ending at the window start belongs entirely to the expired interval.
        while self._progress_history and self._progress_history[0][0] <= window_start:
            self._progress_history.popleft()

        # Do not terminate during the initial observation window.
        if self._elapsed_time < self._stuck_time:
            return False

        total_progress = sum(delta for _, delta in self._progress_history)

        return total_progress < self._minimum_progress
