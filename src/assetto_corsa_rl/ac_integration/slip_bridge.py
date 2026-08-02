"""Versioned local protocol for tyre signals exposed only to AC Python apps."""

from __future__ import annotations

import ipaddress
import math
import socket
import struct
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

SLIP_BRIDGE_HOST = "127.0.0.1"
SLIP_BRIDGE_PORT = 27942
SLIP_BRIDGE_FILE_NAME = "assetto_corsa_rl_bridge.bin"
SLIP_BRIDGE_MAGIC = b"ACRL"
SLIP_BRIDGE_VERSION = 1
WHEEL_NAMES = ("front_left", "front_right", "rear_left", "rear_right")

# magic, protocol version, sequence, source wall-clock timestamp, four ratios,
# and four raw API slip angles. Standard little-endian layout has no padding.
_PACKET = struct.Struct("<4sBId8f")
SLIP_BRIDGE_PACKET_SIZE = _PACKET.size

WheelValues = tuple[float, float, float, float]


class SlipBridgeError(RuntimeError):
    """Base error for the local tyre-signal bridge."""


class SlipBridgeProtocolError(SlipBridgeError):
    """Raised for malformed or unsupported datagrams."""


class SlipBridgeTimeoutError(SlipBridgeError):
    """Raised when the in-game app does not send before a bounded timeout."""


@dataclass(frozen=True, slots=True)
class SlipBridgePacket:
    """One validated in-game telemetry datagram.

    ``slip_angle_raw`` deliberately preserves the Assetto Corsa Python API
    values on the wire. The named feature adapter owns the accepted degree to
    radian conversion. Wheel order is front-left, front-right, rear-left,
    rear-right.
    """

    sequence: int
    source_time_s: float
    received_monotonic_s: float
    slip_ratio: WheelValues
    slip_angle_raw: WheelValues

    def axle_arithmetic_means(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return front/rear means without changing raw signs or angle units."""

        ratio = (
            0.5 * (self.slip_ratio[0] + self.slip_ratio[1]),
            0.5 * (self.slip_ratio[2] + self.slip_ratio[3]),
        )
        angle = (
            0.5 * (self.slip_angle_raw[0] + self.slip_angle_raw[1]),
            0.5 * (self.slip_angle_raw[2] + self.slip_angle_raw[3]),
        )
        return ratio, angle


def _validate_wheel_values(name: str, values: WheelValues) -> None:
    if len(values) != 4:
        raise ValueError(f"{name} must contain FL, FR, RL, RR")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")


def encode_slip_bridge_packet(
    *,
    sequence: int,
    source_time_s: float,
    slip_ratio: WheelValues,
    slip_angle_raw: WheelValues,
) -> bytes:
    """Encode a packet for deterministic protocol tests and mock producers."""

    if not 0 <= sequence <= 0xFFFFFFFF:
        raise ValueError("sequence must fit an unsigned 32-bit integer")
    if not math.isfinite(source_time_s):
        raise ValueError("source_time_s must be finite")
    _validate_wheel_values("slip_ratio", slip_ratio)
    _validate_wheel_values("slip_angle_raw", slip_angle_raw)
    return _PACKET.pack(
        SLIP_BRIDGE_MAGIC,
        SLIP_BRIDGE_VERSION,
        sequence,
        source_time_s,
        *slip_ratio,
        *slip_angle_raw,
    )


def decode_slip_bridge_packet(
    payload: bytes,
    *,
    received_monotonic_s: float | None = None,
) -> SlipBridgePacket:
    """Decode and validate one datagram without trusting its producer."""

    if len(payload) != SLIP_BRIDGE_PACKET_SIZE:
        raise SlipBridgeProtocolError(
            f"invalid slip-bridge packet size {len(payload)}; expected {SLIP_BRIDGE_PACKET_SIZE}"
        )
    magic, version, sequence, source_time_s, *values = _PACKET.unpack(payload)
    if magic != SLIP_BRIDGE_MAGIC:
        raise SlipBridgeProtocolError("invalid slip-bridge packet magic")
    if version != SLIP_BRIDGE_VERSION:
        raise SlipBridgeProtocolError(
            f"unsupported slip-bridge protocol version {version}; expected {SLIP_BRIDGE_VERSION}"
        )
    ratio = tuple(values[:4])
    angle = tuple(values[4:])
    if not math.isfinite(source_time_s) or not all(math.isfinite(value) for value in values):
        raise SlipBridgeProtocolError("slip-bridge packet contains a non-finite value")
    received = time.monotonic() if received_monotonic_s is None else received_monotonic_s
    if not math.isfinite(received):
        raise ValueError("received_monotonic_s must be finite")
    return SlipBridgePacket(
        sequence=sequence,
        source_time_s=source_time_s,
        received_monotonic_s=received,
        slip_ratio=(ratio[0], ratio[1], ratio[2], ratio[3]),
        slip_angle_raw=(angle[0], angle[1], angle[2], angle[3]),
    )


class AssettoCorsaSlipBridgeReceiver:
    """Bounded UDP receiver restricted to the local loopback interface."""

    def __init__(self, datagram_socket: socket.socket) -> None:
        self._socket = datagram_socket
        self._closed = False

    @classmethod
    def bind(
        cls,
        *,
        host: str = SLIP_BRIDGE_HOST,
        port: int = SLIP_BRIDGE_PORT,
    ) -> AssettoCorsaSlipBridgeReceiver:
        """Bind a receiver; non-loopback addresses are rejected."""

        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError("the slip bridge may bind only to a numeric loopback address")
        if not 1 <= port <= 65535:
            raise ValueError("port must be in the range 1..65535")
        datagram_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            datagram_socket.bind((host, port))
        except BaseException:
            datagram_socket.close()
            raise
        return cls(datagram_socket)

    def receive(self, *, timeout_s: float) -> SlipBridgePacket:
        """Receive one valid datagram or fail after ``timeout_s``."""

        if self._closed:
            raise SlipBridgeError("slip-bridge receiver is closed")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._socket.settimeout(timeout_s)
        try:
            payload, _address = self._socket.recvfrom(SLIP_BRIDGE_PACKET_SIZE + 1)
        except TimeoutError as error:
            raise SlipBridgeTimeoutError(
                "no slip telemetry received; enable the Assetto Corsa RL Bridge app"
            ) from error
        return decode_slip_bridge_packet(payload)

    def close(self) -> None:
        """Release the UDP port idempotently."""

        if not self._closed:
            self._socket.close()
            self._closed = True

    def __enter__(self) -> AssettoCorsaSlipBridgeReceiver:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


def default_slip_bridge_path() -> Path:
    """Return the per-user path shared by AC and the external process."""

    return Path(tempfile.gettempdir()) / SLIP_BRIDGE_FILE_NAME


class AssettoCorsaSlipBridgeFileReader:
    """Poll fresh packets from the file transport supported by Kunos Python 3.3."""

    def __init__(
        self,
        path: Path,
        *,
        max_packet_age_s: float = 1.0,
        poll_interval_s: float = 0.002,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_packet_age_s <= 0:
            raise ValueError("max_packet_age_s must be positive")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        self.path = path
        self.max_packet_age_s = max_packet_age_s
        self.poll_interval_s = poll_interval_s
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._sleep = sleep
        self._last_sequence: int | None = None
        self._closed = False

    @classmethod
    def connect(
        cls,
        path: Path | None = None,
    ) -> AssettoCorsaSlipBridgeFileReader:
        """Create a reader for the default per-user bridge file."""

        return cls(default_slip_bridge_path() if path is None else path)

    def receive(self, *, timeout_s: float) -> SlipBridgePacket:
        """Return the next fresh complete packet or fail after ``timeout_s``."""

        if self._closed:
            raise SlipBridgeError("slip-bridge file reader is closed")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = self._monotonic_clock() + timeout_s
        while True:
            received_monotonic_s = self._monotonic_clock()
            try:
                payload = self.path.read_bytes()
                packet = decode_slip_bridge_packet(
                    payload,
                    received_monotonic_s=received_monotonic_s,
                )
                age_s = abs(self._wall_clock() - packet.source_time_s)
                if age_s <= self.max_packet_age_s and packet.sequence != self._last_sequence:
                    self._last_sequence = packet.sequence
                    return packet
            except (OSError, SlipBridgeProtocolError):
                # A writer can be between seek and write; retry only until the
                # public timeout rather than accepting a partial packet.
                pass
            remaining_s = deadline - self._monotonic_clock()
            if remaining_s <= 0:
                raise SlipBridgeTimeoutError(
                    "no fresh slip telemetry received; enable the Assetto Corsa RL Bridge app"
                )
            self._sleep(min(self.poll_interval_s, remaining_s))

    def close(self) -> None:
        """Mark the reader closed; no operating-system handle is retained."""

        self._closed = True

    def __enter__(self) -> AssettoCorsaSlipBridgeFileReader:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()
