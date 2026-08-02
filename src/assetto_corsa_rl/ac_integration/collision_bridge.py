"""Latched local CSP collision-event transport."""

from __future__ import annotations

import math
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

CSP_STATE_MAGIC = "ACRL_CSP_STATE_V1"
CSP_STATE_FILE_NAME = "assetto_corsa_rl_csp_state.txt"


class CollisionBridgeError(RuntimeError):
    """Base error for missing, stale, or invalid CSP collision state."""


class CollisionBridgeTimeoutError(CollisionBridgeError):
    """Raised when no fresh CSP state arrives before a deadline."""


class CollisionBridgeProtocolError(CollisionBridgeError):
    """Raised when a state file does not match the fixed local protocol."""


@dataclass(frozen=True, slots=True)
class CollisionState:
    """Latest collision sequence latched by the CSP Lua callback."""

    track_id: str
    collision_sequence: int
    simulator_frame: int
    received_monotonic_s: float


def default_csp_state_path() -> Path:
    """Return the per-user temporary file shared with the CSP Lua app."""

    return Path(tempfile.gettempdir()) / CSP_STATE_FILE_NAME


def decode_collision_state(payload: str, *, received_monotonic_s: float) -> CollisionState:
    """Decode the exact newline-delimited CSP state format."""

    lines = payload.splitlines()
    if len(lines) != 4 or lines[0] != CSP_STATE_MAGIC:
        raise CollisionBridgeProtocolError("invalid CSP collision-state header or field count")
    track_id = lines[1].strip()
    if not track_id:
        raise CollisionBridgeProtocolError("CSP collision state has an empty track ID")
    try:
        collision_sequence = int(lines[2], 10)
        simulator_frame = int(lines[3], 10)
    except ValueError as error:
        raise CollisionBridgeProtocolError("CSP collision counters must be integers") from error
    if not 0 <= collision_sequence <= 0xFFFFFFFF:
        raise CollisionBridgeProtocolError("CSP collision sequence is outside uint32")
    if simulator_frame < 0:
        raise CollisionBridgeProtocolError("CSP simulator frame must be non-negative")
    if not math.isfinite(received_monotonic_s):
        raise ValueError("received_monotonic_s must be finite")
    return CollisionState(
        track_id=track_id,
        collision_sequence=collision_sequence,
        simulator_frame=simulator_frame,
        received_monotonic_s=received_monotonic_s,
    )


class AssettoCorsaCollisionBridge:
    """Read fresh CSP state and preserve collisions across agent-rate steps."""

    def __init__(
        self,
        path: Path,
        *,
        maximum_age_s: float = 1.0,
        poll_interval_s: float = 0.001,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if maximum_age_s <= 0 or poll_interval_s <= 0:
            raise ValueError("collision bridge age and poll interval must be positive")
        self.path = path
        self.maximum_age_s = maximum_age_s
        self.poll_interval_s = poll_interval_s
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._sleep = sleep
        self._baseline_sequence: int | None = None
        self._expected_track_id: str | None = None
        self._closed = False

    @classmethod
    def connect(cls, path: Path | None = None) -> Self:
        """Open the default per-user CSP state reader."""

        return cls(default_csp_state_path() if path is None else path)

    def receive(self, *, timeout_s: float) -> CollisionState:
        """Wait for a complete, fresh state file."""

        if self._closed:
            raise CollisionBridgeError("collision bridge is closed")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = self._monotonic_clock() + timeout_s
        last_error = "state file unavailable"
        while True:
            received = self._monotonic_clock()
            try:
                stat = self.path.stat()
                age_s = abs(self._wall_clock() - stat.st_mtime)
                if age_s > self.maximum_age_s:
                    last_error = f"state file is stale ({age_s:.3f} s)"
                else:
                    state = decode_collision_state(
                        self.path.read_text(encoding="ascii"),
                        received_monotonic_s=received,
                    )
                    return state
            except (OSError, UnicodeError, CollisionBridgeProtocolError) as error:
                last_error = str(error)
            remaining = deadline - self._monotonic_clock()
            if remaining <= 0:
                raise CollisionBridgeTimeoutError(
                    f"no fresh CSP collision state before timeout ({last_error})"
                )
            self._sleep(min(self.poll_interval_s, remaining))

    def synchronise(self, *, expected_track_id: str, timeout_s: float) -> CollisionState:
        """Ignore pre-reset events and establish the episode collision baseline."""

        state = self.receive(timeout_s=timeout_s)
        if state.track_id != expected_track_id:
            raise CollisionBridgeProtocolError(
                f"CSP collision track {state.track_id!r} != expected {expected_track_id!r}"
            )
        self._expected_track_id = expected_track_id
        self._baseline_sequence = state.collision_sequence
        return state

    def collision_since_synchronisation(self, *, timeout_s: float) -> tuple[bool, CollisionState]:
        """Return whether the Lua callback latched any post-reset collision."""

        if self._baseline_sequence is None or self._expected_track_id is None:
            raise CollisionBridgeError("collision bridge must be synchronised after reset")
        state = self.receive(timeout_s=timeout_s)
        if state.track_id != self._expected_track_id:
            raise CollisionBridgeProtocolError("CSP collision-state track changed during episode")
        return state.collision_sequence != self._baseline_sequence, state

    def close(self) -> None:
        """Close the logical reader idempotently."""

        self._closed = True
