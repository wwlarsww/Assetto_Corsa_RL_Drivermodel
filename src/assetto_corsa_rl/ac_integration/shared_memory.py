"""Read Assetto Corsa's documented Windows shared-memory pages.

The ABI uses four-byte packing and two-byte Windows ``wchar_t`` values. Parsing
is therefore based on explicit little-endian byte offsets instead of host
``ctypes`` structures, which keeps layout tests valid on Windows and Linux.
"""

from __future__ import annotations

import ctypes
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any, Protocol, Self

Vector3 = tuple[float, float, float]
WheelValues = tuple[float, float, float, float]

PHYSICS_MAP_NAME = "acpmf_physics"
GRAPHICS_MAP_NAME = "acpmf_graphics"
STATIC_MAP_NAME = "acpmf_static"

PHYSICS_PAGE_SIZE = 580
GRAPHICS_PAGE_SIZE = 296
STATIC_PAGE_SIZE = 684

_FILE_MAP_READ = 0x0004


class AssettoCorsaTelemetryUnavailableError(RuntimeError):
    """Raised when Assetto Corsa has not published readable telemetry pages."""


def _kernel32_library() -> Any:
    """Load the Win32 API without assuming its ctypes symbols exist on Linux."""

    win_dll: Any = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        raise AssettoCorsaTelemetryUnavailableError(
            "Assetto Corsa shared memory is available only on Windows"
        )
    return win_dll("kernel32", use_last_error=True)


def _last_windows_error() -> int:
    """Return the calling thread's Win32 error or zero on a non-Windows host."""

    get_last_error: Any = getattr(ctypes, "get_last_error", None)
    return int(get_last_error()) if get_last_error is not None else 0


def _running_on_windows() -> bool:
    """Keep the runtime platform check testable on every CI host."""

    return sys.platform == "win32"


class _ReadableMapping(Protocol):
    def read(self) -> bytes:
        """Copy the complete current page."""

    def close(self) -> None:
        """Release the mapped view and handle."""


@dataclass(frozen=True, slots=True)
class PhysicsPage:
    """Selected raw signals from ``SPageFilePhysics`` without reinterpretation."""

    packet_id: int
    gas: float
    brake: float
    clutch: float
    gear: int
    rpms: int
    steer_angle: float
    speed_kmh: float
    velocity: Vector3
    acceleration_g: Vector3
    wheel_slip: WheelValues
    wheel_load_n: WheelValues
    wheel_angular_speed_rad_s: WheelValues
    heading_rad: float
    local_angular_velocity_rad_s: Vector3
    local_velocity_mps: Vector3
    tyre_contact_heading: tuple[Vector3, Vector3, Vector3, Vector3]
    number_of_tyres_out: int
    brake_bias: float


@dataclass(frozen=True, slots=True)
class GraphicsPage:
    """Selected raw session signals from ``SPageFileGraphic``."""

    packet_id: int
    status: int
    session: int
    completed_laps: int
    current_lap_time_ms: int
    last_lap_time_ms: int
    distance_traveled_m: float
    is_in_pit: bool
    normalized_car_position: float
    car_coordinates_m: Vector3
    is_in_pit_lane: bool
    surface_grip: float


@dataclass(frozen=True, slots=True)
class StaticPage:
    """Session identity and calibration constants from ``SPageFileStatic``."""

    shared_memory_version: str
    assetto_corsa_version: str
    car_model: str
    track: str
    track_configuration: str
    track_spline_length_m: float
    tyre_radius_m: WheelValues


@dataclass(frozen=True, slots=True)
class AssettoCorsaTelemetry:
    """One externally timestamped snapshot of all three shared-memory pages."""

    monotonic_time_s: float
    physics: PhysicsPage
    graphics: GraphicsPage
    static: StaticPage


def _require_size(payload: bytes, expected: int, page: str) -> None:
    if len(payload) < expected:
        raise ValueError(f"{page} page must contain at least {expected} bytes, got {len(payload)}")


def _int32(payload: bytes, offset: int) -> int:
    return int(struct.unpack_from("<i", payload, offset)[0])


def _float32(payload: bytes, offset: int) -> float:
    return float(struct.unpack_from("<f", payload, offset)[0])


def _float_tuple(payload: bytes, offset: int, count: int) -> tuple[float, ...]:
    return tuple(float(value) for value in struct.unpack_from(f"<{count}f", payload, offset))


def _vector3(payload: bytes, offset: int) -> Vector3:
    values = _float_tuple(payload, offset, 3)
    return values[0], values[1], values[2]


def _wheel_values(payload: bytes, offset: int) -> WheelValues:
    values = _float_tuple(payload, offset, 4)
    return values[0], values[1], values[2], values[3]


def _utf16(payload: bytes, offset: int, character_count: int) -> str:
    raw = payload[offset : offset + 2 * character_count]
    return raw.decode("utf-16-le", errors="replace").split("\0", 1)[0]


def decode_physics_page(payload: bytes) -> PhysicsPage:
    """Decode the published AC 1.14 physics ABI from a copied byte page."""

    _require_size(payload, PHYSICS_PAGE_SIZE, "physics")
    contact_headings = tuple(_vector3(payload, 516 + 12 * index) for index in range(4))
    return PhysicsPage(
        packet_id=_int32(payload, 0),
        gas=_float32(payload, 4),
        brake=_float32(payload, 8),
        gear=_int32(payload, 16),
        rpms=_int32(payload, 20),
        steer_angle=_float32(payload, 24),
        speed_kmh=_float32(payload, 28),
        velocity=_vector3(payload, 32),
        acceleration_g=_vector3(payload, 44),
        wheel_slip=_wheel_values(payload, 56),
        wheel_load_n=_wheel_values(payload, 72),
        wheel_angular_speed_rad_s=_wheel_values(payload, 104),
        heading_rad=_float32(payload, 208),
        local_angular_velocity_rad_s=_vector3(payload, 296),
        clutch=_float32(payload, 364),
        tyre_contact_heading=(
            contact_headings[0],
            contact_headings[1],
            contact_headings[2],
            contact_headings[3],
        ),
        brake_bias=_float32(payload, 564),
        local_velocity_mps=_vector3(payload, 568),
        number_of_tyres_out=_int32(payload, 244),
    )


def decode_graphics_page(payload: bytes) -> GraphicsPage:
    """Decode the published AC 1.14 graphics ABI from a copied byte page."""

    _require_size(payload, GRAPHICS_PAGE_SIZE, "graphics")
    return GraphicsPage(
        packet_id=_int32(payload, 0),
        status=_int32(payload, 4),
        session=_int32(payload, 8),
        completed_laps=_int32(payload, 132),
        current_lap_time_ms=_int32(payload, 140),
        last_lap_time_ms=_int32(payload, 144),
        distance_traveled_m=_float32(payload, 156),
        is_in_pit=bool(_int32(payload, 160)),
        normalized_car_position=_float32(payload, 248),
        car_coordinates_m=_vector3(payload, 252),
        is_in_pit_lane=bool(_int32(payload, 276)),
        surface_grip=_float32(payload, 280),
    )


def decode_static_page(payload: bytes) -> StaticPage:
    """Decode the published AC 1.14 static ABI from a copied byte page."""

    _require_size(payload, STATIC_PAGE_SIZE, "static")
    return StaticPage(
        shared_memory_version=_utf16(payload, 0, 15),
        assetto_corsa_version=_utf16(payload, 30, 15),
        car_model=_utf16(payload, 68, 33),
        track=_utf16(payload, 134, 33),
        tyre_radius_m=_wheel_values(payload, 436),
        track_spline_length_m=_float32(payload, 520),
        track_configuration=_utf16(payload, 524, 33),
    )


class _WindowsNamedMapping:
    """Read-only view opened with Win32 ``OpenFileMappingW``.

    Using the Win32 open call is important: Python's named ``mmap`` constructor
    creates an empty mapping when AC is absent, which can look like valid
    all-zero telemetry.
    """

    def __init__(self, name: str, size: int) -> None:
        kernel32 = _kernel32_library()
        kernel32.OpenFileMappingW.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_wchar_p,
        ]
        kernel32.OpenFileMappingW.restype = ctypes.c_void_p
        kernel32.MapViewOfFile.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_size_t,
        ]
        kernel32.MapViewOfFile.restype = ctypes.c_void_p
        kernel32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
        kernel32.UnmapViewOfFile.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int

        handle: int | None = None
        opened_name = ""
        for candidate in (f"Local\\{name}", name):
            raw_handle = kernel32.OpenFileMappingW(_FILE_MAP_READ, False, candidate)
            if raw_handle:
                handle = int(raw_handle)
                opened_name = candidate
                break
        if handle is None:
            error = _last_windows_error()
            raise AssettoCorsaTelemetryUnavailableError(
                f"shared-memory page {name!r} is unavailable (Windows error {error}); "
                "start an on-track Assetto Corsa session"
            )

        view = kernel32.MapViewOfFile(handle, _FILE_MAP_READ, 0, 0, size)
        if not view:
            error = _last_windows_error()
            kernel32.CloseHandle(handle)
            raise AssettoCorsaTelemetryUnavailableError(
                f"shared-memory page {opened_name!r} cannot be mapped (Windows error {error})"
            )
        self._kernel32 = kernel32
        self._handle = handle
        self._view = int(view)
        self._size = size

    def read(self) -> bytes:
        if self._view == 0:
            raise RuntimeError("shared-memory mapping is closed")
        return ctypes.string_at(self._view, self._size)

    def close(self) -> None:
        if self._view:
            self._kernel32.UnmapViewOfFile(self._view)
            self._view = 0
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = 0


def _stable_packet(mapping: _ReadableMapping, *, page: str, attempts: int = 8) -> bytes:
    for _ in range(attempts):
        first = mapping.read()
        second = mapping.read()
        if _int32(first, 0) == _int32(second, 0):
            return second
    raise RuntimeError(f"{page} shared-memory page changed during {attempts} consecutive reads")


class AssettoCorsaSharedMemoryReader:
    """Read-only owner of Assetto Corsa physics, graphics, and static mappings."""

    def __init__(
        self,
        physics: _ReadableMapping,
        graphics: _ReadableMapping,
        static: _ReadableMapping,
    ) -> None:
        self._physics = physics
        self._graphics = graphics
        self._static = static
        self._closed = False

    @classmethod
    def connect(cls) -> Self:
        """Open existing Windows mappings without creating fake zero pages."""

        if not _running_on_windows():
            raise AssettoCorsaTelemetryUnavailableError(
                "Assetto Corsa shared memory is available only on Windows"
            )
        opened: list[_ReadableMapping] = []
        try:
            physics = _WindowsNamedMapping(PHYSICS_MAP_NAME, PHYSICS_PAGE_SIZE)
            opened.append(physics)
            graphics = _WindowsNamedMapping(GRAPHICS_MAP_NAME, GRAPHICS_PAGE_SIZE)
            opened.append(graphics)
            static = _WindowsNamedMapping(STATIC_MAP_NAME, STATIC_PAGE_SIZE)
            opened.append(static)
            return cls(physics, graphics, static)
        except Exception:
            for mapping in opened:
                mapping.close()
            raise

    def read(self) -> AssettoCorsaTelemetry:
        """Copy consistent dynamic pages and decode one telemetry snapshot."""

        if self._closed:
            raise RuntimeError("shared-memory reader is closed")
        physics = decode_physics_page(_stable_packet(self._physics, page="physics"))
        graphics = decode_graphics_page(_stable_packet(self._graphics, page="graphics"))
        static = decode_static_page(self._static.read())
        return AssettoCorsaTelemetry(
            monotonic_time_s=time.perf_counter(),
            physics=physics,
            graphics=graphics,
            static=static,
        )

    def close(self) -> None:
        """Release all three mappings idempotently."""

        if self._closed:
            return
        self._physics.close()
        self._graphics.close()
        self._static.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
