from __future__ import annotations

import ctypes
import struct
from typing import Any

import pytest

from assetto_corsa_rl.ac_integration import shared_memory
from assetto_corsa_rl.ac_integration.shared_memory import (
    GRAPHICS_PAGE_SIZE,
    PHYSICS_PAGE_SIZE,
    STATIC_PAGE_SIZE,
    AssettoCorsaSharedMemoryReader,
    AssettoCorsaTelemetryUnavailableError,
    _WindowsNamedMapping,
    decode_graphics_page,
    decode_physics_page,
    decode_static_page,
)


def _pack_utf16(buffer: bytearray, offset: int, size: int, value: str) -> None:
    encoded = value.encode("utf-16-le")[: 2 * (size - 1)]
    buffer[offset : offset + len(encoded)] = encoded


def test_decode_assetto_corsa_shared_memory_layout() -> None:
    physics = bytearray(PHYSICS_PAGE_SIZE)
    struct.pack_into("<i", physics, 0, 42)
    struct.pack_into("<fff", physics, 4, 0.75, 0.1, 30.0)
    struct.pack_into("<ii", physics, 16, 4, 6500)
    struct.pack_into("<ff", physics, 24, -0.2, 123.4)
    struct.pack_into("<3f", physics, 32, 1.0, 2.0, 3.0)
    struct.pack_into("<3f", physics, 44, 0.1, 0.2, 0.3)
    struct.pack_into("<4f", physics, 56, 1.0, 2.0, 3.0, 4.0)
    struct.pack_into("<4f", physics, 72, 100.0, 101.0, 102.0, 103.0)
    struct.pack_into("<4f", physics, 104, 10.0, 11.0, 12.0, 13.0)
    struct.pack_into("<f", physics, 208, 1.25)
    struct.pack_into("<i", physics, 244, 2)
    struct.pack_into("<3f", physics, 296, 0.4, 0.5, 0.6)
    struct.pack_into("<f", physics, 364, 0.7)
    struct.pack_into("<12f", physics, 516, *range(12))
    struct.pack_into("<f", physics, 564, 0.63)
    struct.pack_into("<3f", physics, 568, -1.0, 0.0, 34.0)
    decoded_physics = decode_physics_page(bytes(physics))
    assert decoded_physics.packet_id == 42
    assert decoded_physics.speed_kmh == pytest.approx(123.4)
    assert decoded_physics.local_velocity_mps == pytest.approx((-1.0, 0.0, 34.0))
    assert decoded_physics.tyre_contact_heading[3] == pytest.approx((9.0, 10.0, 11.0))

    graphics = bytearray(GRAPHICS_PAGE_SIZE)
    struct.pack_into("<iii", graphics, 0, 7, 2, 3)
    struct.pack_into("<i", graphics, 132, 4)
    struct.pack_into("<ii", graphics, 140, 12345, 90000)
    struct.pack_into("<f", graphics, 156, 321.0)
    struct.pack_into("<i", graphics, 160, 1)
    struct.pack_into("<f", graphics, 248, 0.25)
    struct.pack_into("<3f", graphics, 252, 10.0, 20.0, 30.0)
    struct.pack_into("<i", graphics, 276, 1)
    struct.pack_into("<f", graphics, 280, 0.98)
    decoded_graphics = decode_graphics_page(bytes(graphics))
    assert decoded_graphics.status == 2
    assert decoded_graphics.current_lap_time_ms == 12345
    assert decoded_graphics.car_coordinates_m == pytest.approx((10.0, 20.0, 30.0))

    static = bytearray(STATIC_PAGE_SIZE)
    _pack_utf16(static, 0, 15, "1.1")
    _pack_utf16(static, 30, 15, "1.16.4")
    _pack_utf16(static, 68, 33, "ks_bmw_z4_gt3")
    _pack_utf16(static, 134, 33, "monza")
    struct.pack_into("<4f", static, 436, 0.31, 0.31, 0.33, 0.33)
    struct.pack_into("<f", static, 520, 5793.0)
    _pack_utf16(static, 524, 33, "default")
    decoded_static = decode_static_page(bytes(static))
    assert decoded_static.car_model == "ks_bmw_z4_gt3"
    assert decoded_static.track == "monza"
    assert decoded_static.tyre_radius_m == pytest.approx((0.31, 0.31, 0.33, 0.33))


class _FixedMapping:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.closed = False

    def read(self) -> bytes:
        return self.payload

    def close(self) -> None:
        self.closed = True


def test_reader_uses_consistent_pages_and_closes_idempotently() -> None:
    physics_payload = bytearray(PHYSICS_PAGE_SIZE)
    graphics_payload = bytearray(GRAPHICS_PAGE_SIZE)
    static_payload = bytearray(STATIC_PAGE_SIZE)
    struct.pack_into("<i", physics_payload, 0, 1)
    struct.pack_into("<i", graphics_payload, 0, 2)
    physics = _FixedMapping(bytes(physics_payload))
    graphics = _FixedMapping(bytes(graphics_payload))
    static = _FixedMapping(bytes(static_payload))
    reader = AssettoCorsaSharedMemoryReader(physics, graphics, static)
    assert reader.read().physics.packet_id == 1
    reader.close()
    reader.close()
    assert physics.closed
    assert graphics.closed
    assert static.closed
    with pytest.raises(RuntimeError, match="closed"):
        reader.read()


class _FakeFunction:
    def __init__(self, result: int) -> None:
        self.result = result
        self.argtypes: object = None
        self.restype: object = None
        self.calls = 0

    def __call__(self, *_args: object) -> int:
        self.calls += 1
        return self.result


class _FakeKernel:
    def __init__(self, view: int, *, handle: int = 123) -> None:
        self.OpenFileMappingW = _FakeFunction(handle)
        self.MapViewOfFile = _FakeFunction(view)
        self.UnmapViewOfFile = _FakeFunction(1)
        self.CloseHandle = _FakeFunction(1)


def test_windows_mapping_opens_existing_page_and_never_creates_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_ctypes: Any = vars(shared_memory)["ctypes"]
    buffer = ctypes.create_string_buffer(b"test")
    kernel = _FakeKernel(ctypes.addressof(buffer))
    monkeypatch.setattr(
        shared_ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: kernel,
        raising=False,
    )
    mapping = _WindowsNamedMapping("page", 4)
    assert mapping.read() == b"test"
    mapping.close()
    mapping.close()
    assert kernel.UnmapViewOfFile.calls == 1
    assert kernel.CloseHandle.calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        mapping.read()

    unavailable = _FakeKernel(0, handle=0)
    monkeypatch.setattr(
        shared_ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: unavailable,
        raising=False,
    )
    monkeypatch.setattr(shared_ctypes, "get_last_error", lambda: 2, raising=False)
    with pytest.raises(AssettoCorsaTelemetryUnavailableError, match="unavailable"):
        _WindowsNamedMapping("missing", 4)


def test_reader_connect_platform_and_partial_open_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_sys: Any = vars(shared_memory)["sys"]
    monkeypatch.setattr(shared_sys, "platform", "linux")
    with pytest.raises(AssettoCorsaTelemetryUnavailableError, match="only on Windows"):
        AssettoCorsaSharedMemoryReader.connect()

    monkeypatch.setattr(shared_sys, "platform", "win32")
    first = _FixedMapping(bytes(PHYSICS_PAGE_SIZE))
    calls = 0

    def mapping_factory(_name: str, _size: int) -> _FixedMapping:
        nonlocal calls
        calls += 1
        if calls == 1:
            return first
        raise AssettoCorsaTelemetryUnavailableError("second page failed")

    monkeypatch.setattr(shared_memory, "_WindowsNamedMapping", mapping_factory)
    with pytest.raises(AssettoCorsaTelemetryUnavailableError, match="second page"):
        AssettoCorsaSharedMemoryReader.connect()
    assert first.closed


@pytest.mark.parametrize(
    ("decoder", "expected"),
    [
        (decode_physics_page, PHYSICS_PAGE_SIZE),
        (decode_graphics_page, GRAPHICS_PAGE_SIZE),
        (decode_static_page, STATIC_PAGE_SIZE),
    ],
)
def test_decoders_reject_truncated_pages(
    decoder: object,
    expected: int,
) -> None:
    with pytest.raises(ValueError, match=str(expected)):
        decoder(b"")  # type: ignore[operator]
