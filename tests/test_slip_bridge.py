from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any, cast

import pytest

from assetto_corsa_rl.ac_integration import slip_bridge
from assetto_corsa_rl.ac_integration.slip_bridge import (
    SLIP_BRIDGE_PACKET_SIZE,
    SLIP_BRIDGE_VERSION,
    AssettoCorsaSlipBridgeFileReader,
    AssettoCorsaSlipBridgeReceiver,
    SlipBridgeError,
    SlipBridgeProtocolError,
    SlipBridgeTimeoutError,
    decode_slip_bridge_packet,
    default_slip_bridge_path,
    encode_slip_bridge_packet,
)


def _payload(sequence: int = 7) -> bytes:
    return encode_slip_bridge_packet(
        sequence=sequence,
        source_time_s=123.5,
        slip_ratio=(0.1, 0.2, -0.3, -0.4),
        slip_angle_raw=(0.01, 0.02, -0.03, -0.04),
    )


def test_slip_packet_round_trip_and_axle_means() -> None:
    payload = _payload()
    assert len(payload) == SLIP_BRIDGE_PACKET_SIZE
    packet = decode_slip_bridge_packet(payload, received_monotonic_s=456.0)
    assert packet.sequence == 7
    assert packet.source_time_s == 123.5
    assert packet.received_monotonic_s == 456.0
    assert packet.slip_ratio == pytest.approx((0.1, 0.2, -0.3, -0.4))
    ratios, angles = packet.axle_arithmetic_means()
    assert ratios == pytest.approx((0.15, -0.35))
    assert angles == pytest.approx((0.015, -0.035))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sequence": -1}, "unsigned 32-bit"),
        ({"sequence": 0x100000000}, "unsigned 32-bit"),
        ({"source_time_s": math.nan}, "finite"),
        ({"slip_ratio": (0.0, 0.0, 0.0)}, "FL, FR, RL, RR"),
        ({"slip_angle_raw": (0.0, 0.0, 0.0, math.inf)}, "finite"),
    ],
)
def test_slip_packet_encoder_rejects_invalid_values(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    valid: dict[str, Any] = {
        "sequence": 1,
        "source_time_s": 1.0,
        "slip_ratio": (0.0, 0.0, 0.0, 0.0),
        "slip_angle_raw": (0.0, 0.0, 0.0, 0.0),
    }
    valid.update(kwargs)
    with pytest.raises(ValueError, match=message):
        encode_slip_bridge_packet(**valid)


def test_slip_packet_decoder_rejects_invalid_protocol() -> None:
    payload = _payload()
    with pytest.raises(SlipBridgeProtocolError, match="size"):
        decode_slip_bridge_packet(payload[:-1])
    wrong_magic = b"NOPE" + payload[4:]
    with pytest.raises(SlipBridgeProtocolError, match="magic"):
        decode_slip_bridge_packet(wrong_magic)
    wrong_version = payload[:4] + bytes((SLIP_BRIDGE_VERSION + 1,)) + payload[5:]
    with pytest.raises(SlipBridgeProtocolError, match="version"):
        decode_slip_bridge_packet(wrong_version)
    non_finite = bytearray(payload)
    # The source timestamp starts after magic, version, and sequence.
    non_finite[9:17] = struct.pack("<d", math.nan)
    with pytest.raises(SlipBridgeProtocolError, match="non-finite"):
        decode_slip_bridge_packet(bytes(non_finite))
    with pytest.raises(ValueError, match="received_monotonic_s"):
        decode_slip_bridge_packet(payload, received_monotonic_s=math.inf)


class _FakeSocket:
    def __init__(
        self,
        *,
        payload: bytes | None = None,
        bind_error: OSError | None = None,
    ) -> None:
        self.payload = payload
        self.bind_error = bind_error
        self.bound: tuple[str, int] | None = None
        self.timeout: float | None = None
        self.closed = False

    def bind(self, address: tuple[str, int]) -> None:
        if self.bind_error is not None:
            raise self.bind_error
        self.bound = address

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
        if self.payload is None:
            raise TimeoutError
        return self.payload, ("127.0.0.1", 27942)

    def close(self) -> None:
        self.closed = True


def test_receiver_binds_loopback_receives_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSocket(payload=_payload())
    socket_module: Any = vars(slip_bridge)["socket"]
    monkeypatch.setattr(socket_module, "socket", lambda *_args: fake)
    with AssettoCorsaSlipBridgeReceiver.bind(port=27943) as receiver:
        packet = receiver.receive(timeout_s=0.25)
        assert packet.sequence == 7
        assert fake.bound == ("127.0.0.1", 27943)
        assert fake.timeout == 0.25
    assert fake.closed
    receiver.close()
    with pytest.raises(SlipBridgeError, match="closed"):
        receiver.receive(timeout_s=0.1)


def test_receiver_rejects_unsafe_bind_and_closes_after_bind_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="loopback"):
        AssettoCorsaSlipBridgeReceiver.bind(host="192.0.2.1")
    with pytest.raises(ValueError, match=r"1\.\.65535"):
        AssettoCorsaSlipBridgeReceiver.bind(port=0)
    fake = _FakeSocket(bind_error=OSError("occupied"))
    socket_module: Any = vars(slip_bridge)["socket"]
    monkeypatch.setattr(socket_module, "socket", lambda *_args: fake)
    with pytest.raises(OSError, match="occupied"):
        AssettoCorsaSlipBridgeReceiver.bind()
    assert fake.closed


def test_receiver_timeout_and_argument_validation() -> None:
    receiver = AssettoCorsaSlipBridgeReceiver(cast(Any, _FakeSocket()))
    with pytest.raises(ValueError, match="positive"):
        receiver.receive(timeout_s=0.0)
    with pytest.raises(SlipBridgeTimeoutError, match="enable"):
        receiver.receive(timeout_s=0.01)
    receiver.close()


class _FakeClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def sleep(self, duration_s: float) -> None:
        self.value += duration_s


def test_file_reader_returns_only_fresh_new_complete_packets(tmp_path: Path) -> None:
    path = tmp_path / "bridge.bin"
    path.write_bytes(_payload())
    monotonic = _FakeClock(10.0)
    reader = AssettoCorsaSlipBridgeFileReader(
        path,
        max_packet_age_s=1.0,
        poll_interval_s=0.01,
        monotonic_clock=monotonic,
        wall_clock=lambda: 123.5,
        sleep=monotonic.sleep,
    )
    packet = reader.receive(timeout_s=0.1)
    assert packet.sequence == 7
    assert packet.received_monotonic_s == 10.0
    with pytest.raises(SlipBridgeTimeoutError, match="fresh"):
        reader.receive(timeout_s=0.02)
    path.write_bytes(_payload(sequence=8))
    assert reader.receive(timeout_s=0.1).sequence == 8
    reader.close()
    reader.close()
    with pytest.raises(SlipBridgeError, match="closed"):
        reader.receive(timeout_s=0.1)


def test_file_reader_rejects_stale_partial_and_invalid_configuration(tmp_path: Path) -> None:
    path = tmp_path / "bridge.bin"
    monotonic = _FakeClock(1.0)
    path.write_bytes(_payload()[:-1])
    reader = AssettoCorsaSlipBridgeFileReader(
        path,
        monotonic_clock=monotonic,
        wall_clock=lambda: 1_000.0,
        sleep=monotonic.sleep,
    )
    with pytest.raises(SlipBridgeTimeoutError):
        reader.receive(timeout_s=0.01)
    path.write_bytes(_payload())
    with pytest.raises(SlipBridgeTimeoutError):
        reader.receive(timeout_s=0.01)
    with pytest.raises(ValueError, match="max_packet_age_s"):
        AssettoCorsaSlipBridgeFileReader(path, max_packet_age_s=0.0)
    with pytest.raises(ValueError, match="poll_interval_s"):
        AssettoCorsaSlipBridgeFileReader(path, poll_interval_s=0.0)
    with pytest.raises(ValueError, match="timeout_s"):
        reader.receive(timeout_s=0.0)


def test_default_file_reader_path_uses_system_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tempfile_module: Any = vars(slip_bridge)["tempfile"]
    monkeypatch.setattr(tempfile_module, "gettempdir", lambda: str(tmp_path))
    assert default_slip_bridge_path() == tmp_path / "assetto_corsa_rl_bridge.bin"
    reader = AssettoCorsaSlipBridgeFileReader.connect()
    assert reader.path == default_slip_bridge_path()
