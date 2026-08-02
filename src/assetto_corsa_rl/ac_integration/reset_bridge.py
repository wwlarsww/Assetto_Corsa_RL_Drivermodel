"""Bounded local command protocol for CSP-backed Assetto Corsa resets."""

from __future__ import annotations

import math
import struct
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

RESET_COMMAND_FILE_NAME = "assetto_corsa_rl_command.bin"
RESET_ACK_FILE_NAME = "assetto_corsa_rl_ack.bin"
RESET_COMMAND_MAGIC = b"ACRC"
RESET_ACK_MAGIC = b"ACRA"
RESET_BRIDGE_VERSION = 1
RESET_ACTION = 1
COMMAND_REPLACE_RETRY_COUNT = 20
COMMAND_REPLACE_RETRY_DELAY_S = 0.001

HOTLAP_RESET_COMMAND_FILE_NAME = "assetto_corsa_rl_hotlap_reset_command.txt"
HOTLAP_RESET_ACK_FILE_NAME = "assetto_corsa_rl_hotlap_reset_ack.txt"
HOTLAP_RESET_COMMAND_MAGIC = "ACRL_HOTLAP_RESET_COMMAND_V1"
HOTLAP_RESET_ACK_MAGIC = "ACRL_HOTLAP_RESET_ACK_V1"
HOTLAP_RESET_METHOD_VARIANT = "csp_hotlap_spawn_v1"

_COMMAND_PACKET = struct.Struct("<4sBIB")
_ACK_PACKET = struct.Struct("<4sBIBd")


class ResetBridgeStatus(IntEnum):
    """Status written by the in-game bridge after processing a command."""

    SUCCESS = 1
    CSP_UNAVAILABLE = 2
    FAILED = 3


class ResetBridgeError(RuntimeError):
    """Base error for reset command transport and execution failures."""


class ResetBridgeProtocolError(ResetBridgeError):
    """Raised when a command or acknowledgement has an invalid wire format."""


class ResetBridgeTimeoutError(ResetBridgeError):
    """Raised when the in-game bridge does not acknowledge a reset in time."""


class ResetBridgeUnavailableError(ResetBridgeError):
    """Raised when the bridge reports that CSP reset support is unavailable."""


@dataclass(frozen=True, slots=True)
class ResetAcknowledgement:
    """Validated acknowledgement emitted after one reset request."""

    sequence: int
    status: ResetBridgeStatus
    source_time_s: float
    simulator_frame: int | None = None
    method_variant: str = "python_ext_reset_car_v1"


def default_reset_command_path() -> Path:
    """Return the per-user command path shared with the in-game bridge."""

    return Path(tempfile.gettempdir()) / RESET_COMMAND_FILE_NAME


def default_reset_ack_path() -> Path:
    """Return the per-user acknowledgement path shared with the in-game bridge."""

    return Path(tempfile.gettempdir()) / RESET_ACK_FILE_NAME


def default_hotlap_reset_command_path() -> Path:
    """Return the command path consumed by the CSP Lua hotlap reset app."""

    return Path(tempfile.gettempdir()) / HOTLAP_RESET_COMMAND_FILE_NAME


def default_hotlap_reset_ack_path() -> Path:
    """Return the acknowledgement path published by the CSP Lua app."""

    return Path(tempfile.gettempdir()) / HOTLAP_RESET_ACK_FILE_NAME


def encode_reset_command(sequence: int) -> bytes:
    """Encode the only accepted command: reset the player car."""

    if not 0 <= sequence <= 0xFFFFFFFF:
        raise ValueError("sequence must fit an unsigned 32-bit integer")
    return _COMMAND_PACKET.pack(
        RESET_COMMAND_MAGIC,
        RESET_BRIDGE_VERSION,
        sequence,
        RESET_ACTION,
    )


def decode_reset_command(payload: bytes) -> int:
    """Validate a command packet and return its sequence."""

    if len(payload) != _COMMAND_PACKET.size:
        raise ResetBridgeProtocolError("invalid reset command packet size")
    magic, version, sequence, action = _COMMAND_PACKET.unpack(payload)
    if magic != RESET_COMMAND_MAGIC:
        raise ResetBridgeProtocolError("invalid reset command magic")
    if version != RESET_BRIDGE_VERSION:
        raise ResetBridgeProtocolError("unsupported reset command version")
    if action != RESET_ACTION:
        raise ResetBridgeProtocolError("unsupported reset command action")
    return int(sequence)


def encode_reset_acknowledgement(
    *,
    sequence: int,
    status: ResetBridgeStatus,
    source_time_s: float,
) -> bytes:
    """Encode an acknowledgement for protocol and in-game app tests."""

    if not 0 <= sequence <= 0xFFFFFFFF:
        raise ValueError("sequence must fit an unsigned 32-bit integer")
    if not math.isfinite(source_time_s):
        raise ValueError("source_time_s must be finite")
    return _ACK_PACKET.pack(
        RESET_ACK_MAGIC,
        RESET_BRIDGE_VERSION,
        sequence,
        int(status),
        source_time_s,
    )


def decode_reset_acknowledgement(payload: bytes) -> ResetAcknowledgement:
    """Decode and validate one reset acknowledgement."""

    if len(payload) != _ACK_PACKET.size:
        raise ResetBridgeProtocolError("invalid reset acknowledgement packet size")
    magic, version, sequence, raw_status, source_time_s = _ACK_PACKET.unpack(payload)
    if magic != RESET_ACK_MAGIC:
        raise ResetBridgeProtocolError("invalid reset acknowledgement magic")
    if version != RESET_BRIDGE_VERSION:
        raise ResetBridgeProtocolError("unsupported reset acknowledgement version")
    try:
        status = ResetBridgeStatus(raw_status)
    except ValueError as error:
        raise ResetBridgeProtocolError("unknown reset acknowledgement status") from error
    if not math.isfinite(source_time_s):
        raise ResetBridgeProtocolError("reset acknowledgement contains a non-finite timestamp")
    return ResetAcknowledgement(
        sequence=int(sequence),
        status=status,
        source_time_s=float(source_time_s),
    )


class AssettoCorsaResetBridge:
    """Issue local reset commands and wait for the CSP call acknowledgement."""

    def __init__(
        self,
        command_path: Path,
        acknowledgement_path: Path,
        *,
        poll_interval_s: float = 0.005,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        self.command_path = command_path
        self.acknowledgement_path = acknowledgement_path
        self.poll_interval_s = poll_interval_s
        self._monotonic_clock = monotonic_clock
        self._sleep = sleep
        self._closed = False

    @classmethod
    def connect(cls) -> AssettoCorsaResetBridge:
        """Use the per-user temporary files shared with the in-game app."""

        return cls(default_reset_command_path(), default_reset_ack_path())

    def request_reset(self, *, timeout_s: float) -> ResetAcknowledgement:
        """Atomically publish a reset and wait for its matching acknowledgement."""

        if self._closed:
            raise ResetBridgeError("reset bridge is closed")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        previous = self._read_acknowledgement()
        sequence = (
            int(time.time_ns() & 0xFFFFFFFF)
            if previous is None
            else (previous.sequence + 1) & 0xFFFFFFFF
        )
        self.command_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.command_path.with_suffix(self.command_path.suffix + ".tmp")
        temporary.write_bytes(encode_reset_command(sequence))
        for attempt in range(COMMAND_REPLACE_RETRY_COUNT):
            try:
                temporary.replace(self.command_path)
                break
            except PermissionError:
                if attempt + 1 >= COMMAND_REPLACE_RETRY_COUNT:
                    raise
                self._sleep(COMMAND_REPLACE_RETRY_DELAY_S)

        deadline = self._monotonic_clock() + timeout_s
        while True:
            acknowledgement = self._read_acknowledgement()
            if acknowledgement is not None and acknowledgement.sequence == sequence:
                if acknowledgement.status is ResetBridgeStatus.SUCCESS:
                    return acknowledgement
                if acknowledgement.status is ResetBridgeStatus.CSP_UNAVAILABLE:
                    raise ResetBridgeUnavailableError(
                        "the in-game bridge cannot call ac.ext_resetCar(); install and enable CSP"
                    )
                raise ResetBridgeError(
                    "the in-game bridge failed while executing ac.ext_resetCar(); check py_log.txt"
                )
            remaining = deadline - self._monotonic_clock()
            if remaining <= 0:
                raise ResetBridgeTimeoutError(
                    "the in-game bridge did not acknowledge the reset; "
                    "install/enable the updated app"
                )
            self._sleep(min(self.poll_interval_s, remaining))

    def _read_acknowledgement(self) -> ResetAcknowledgement | None:
        try:
            return decode_reset_acknowledgement(self.acknowledgement_path.read_bytes())
        except (OSError, ResetBridgeProtocolError):
            return None

    def close(self) -> None:
        """Mark the stateless file adapter closed."""

        self._closed = True


def encode_hotlap_reset_command(*, sequence: int, track_id: str) -> bytes:
    """Encode one deterministic reset-to-HOTLAP_START request."""

    if not 0 <= sequence <= 0xFFFFFFFF:
        raise ValueError("sequence must fit an unsigned 32-bit integer")
    if not track_id or any(character in track_id for character in "\r\n"):
        raise ValueError("track_id must be a non-empty single line")
    return f"{HOTLAP_RESET_COMMAND_MAGIC}\n{sequence}\n{track_id}\n".encode("ascii")


def decode_hotlap_reset_acknowledgement(
    payload: str,
    *,
    source_time_s: float,
) -> ResetAcknowledgement:
    """Decode the exact CSP Lua hotlap reset acknowledgement."""

    lines = payload.splitlines()
    if len(lines) != 5 or lines[0] != HOTLAP_RESET_ACK_MAGIC:
        raise ResetBridgeProtocolError("invalid hotlap reset acknowledgement")
    try:
        sequence = int(lines[1], 10)
        status = ResetBridgeStatus(int(lines[2], 10))
        simulator_frame = int(lines[3], 10)
    except ValueError as error:
        raise ResetBridgeProtocolError(
            "hotlap reset acknowledgement contains invalid numeric fields"
        ) from error
    if not 0 <= sequence <= 0xFFFFFFFF or simulator_frame < 0:
        raise ResetBridgeProtocolError("hotlap reset acknowledgement is out of range")
    if not math.isfinite(source_time_s):
        raise ValueError("source_time_s must be finite")
    return ResetAcknowledgement(
        sequence=sequence,
        status=status,
        source_time_s=source_time_s,
        simulator_frame=simulator_frame,
        method_variant=HOTLAP_RESET_METHOD_VARIANT,
    )


class AssettoCorsaHotlapResetBridge:
    """Request a deterministic CSP teleport to the configured hotlap spawn."""

    def __init__(
        self,
        command_path: Path,
        acknowledgement_path: Path,
        *,
        expected_track_id: str,
        poll_interval_s: float = 0.005,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not expected_track_id:
            raise ValueError("expected_track_id must not be empty")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        self.command_path = command_path
        self.acknowledgement_path = acknowledgement_path
        self.expected_track_id = expected_track_id
        self.poll_interval_s = poll_interval_s
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._sleep = sleep
        self._closed = False

    @classmethod
    def connect(cls, *, expected_track_id: str) -> AssettoCorsaHotlapResetBridge:
        """Connect to the per-user files shared with the CSP Lua app."""

        return cls(
            default_hotlap_reset_command_path(),
            default_hotlap_reset_ack_path(),
            expected_track_id=expected_track_id,
        )

    def request_reset(self, *, timeout_s: float) -> ResetAcknowledgement:
        """Publish one request and wait for the matching CSP acknowledgement."""

        if self._closed:
            raise ResetBridgeError("hotlap reset bridge is closed")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        previous = self._read_acknowledgement()
        sequence = (
            int(time.time_ns() & 0xFFFFFFFF)
            if previous is None
            else (previous.sequence + 1) & 0xFFFFFFFF
        )
        self.command_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.command_path.with_suffix(self.command_path.suffix + ".tmp")
        temporary.write_bytes(
            encode_hotlap_reset_command(
                sequence=sequence,
                track_id=self.expected_track_id,
            )
        )
        for attempt in range(COMMAND_REPLACE_RETRY_COUNT):
            try:
                temporary.replace(self.command_path)
                break
            except PermissionError:
                if attempt + 1 >= COMMAND_REPLACE_RETRY_COUNT:
                    raise
                self._sleep(COMMAND_REPLACE_RETRY_DELAY_S)

        deadline = self._monotonic_clock() + timeout_s
        while True:
            acknowledgement = self._read_acknowledgement()
            if acknowledgement is not None and acknowledgement.sequence == sequence:
                if acknowledgement.status is ResetBridgeStatus.SUCCESS:
                    return acknowledgement
                if acknowledgement.status is ResetBridgeStatus.CSP_UNAVAILABLE:
                    raise ResetBridgeUnavailableError(
                        "CSP physics access is unavailable for HOTLAP_START reset"
                    )
                raise ResetBridgeError(
                    "CSP failed to reset the car to HOTLAP_START; check CSP log.txt"
                )
            remaining = deadline - self._monotonic_clock()
            if remaining <= 0:
                raise ResetBridgeTimeoutError(
                    "the CSP Lua app did not acknowledge the HOTLAP_START reset"
                )
            self._sleep(min(self.poll_interval_s, remaining))

    def _read_acknowledgement(self) -> ResetAcknowledgement | None:
        try:
            stat = self.acknowledgement_path.stat()
            return decode_hotlap_reset_acknowledgement(
                self.acknowledgement_path.read_text(encoding="ascii"),
                source_time_s=stat.st_mtime if stat.st_mtime > 0 else self._wall_clock(),
            )
        except (OSError, UnicodeError, ResetBridgeProtocolError):
            return None

    def close(self) -> None:
        """Mark the stateless file adapter closed."""

        self._closed = True
