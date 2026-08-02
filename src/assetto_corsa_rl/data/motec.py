"""Small read-only parser for the MoTeC ``.ld`` subset used by ACTI.

The binary layout follows the reverse-engineered format published by
``gotzl/ldparser`` and used by AssettoCorsaGym. This implementation deliberately
supports reading selected numeric channels only; it cannot execute embedded
objects and it never writes back into a source recording.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from assetto_corsa_rl.types import FloatArray

_HEADER = struct.Struct(
    "<"
    "I4x"  # marker
    "II"  # channel metadata and data pointers
    "20x"
    "I"  # event pointer
    "24x"
    "HHH"
    "I"
    "8s"
    "H"
    "H"
    "I"  # channel count
    "4x"
    "16s"
    "16x"
    "16s"
    "16x"
    "64s"
    "64s"
    "64x"
    "64s"
    "64x"
    "1024x"
    "I"
    "66x"
    "64s"
    "126x"
)
_CHANNEL = struct.Struct(
    "<"
    "IIII"  # previous, next, data pointer, sample count
    "H"
    "HHH"  # storage family, storage width, frequency
    "hhhh"  # shift, multiplier, scale, decimal places
    "32s"
    "8s"
    "12s"
    "40x"
)
_DTYPES: dict[tuple[int, int], np.dtype[np.generic]] = {
    (0x07, 2): np.dtype("<f2"),
    (0x07, 4): np.dtype("<f4"),
    (0x00, 2): np.dtype("<i2"),
    (0x00, 4): np.dtype("<i4"),
    (0x03, 2): np.dtype("<i2"),
    (0x03, 4): np.dtype("<i4"),
    (0x05, 2): np.dtype("<i2"),
    (0x05, 4): np.dtype("<i4"),
}


@dataclass(frozen=True, slots=True)
class MotecHeader:
    """Non-sensitive recording metadata needed for provenance checks."""

    driver: str
    vehicle_id: str
    venue: str
    date: str
    time: str
    channel_count: int


@dataclass(frozen=True, slots=True)
class _Channel:
    name: str
    data_pointer: int
    sample_count: int
    frequency_hz: int
    dtype: np.dtype[np.generic]
    shift: int
    multiplier: int
    scale: int
    decimal_places: int


def _decode_ascii(value: bytes) -> str:
    return value.decode("ascii", errors="replace").rstrip("\0").strip()


def _read_header(handle: BinaryIO, *, file_size: int) -> tuple[MotecHeader, int]:
    if file_size < _HEADER.size:
        raise ValueError("file is too small to contain a MoTeC header")
    raw = handle.read(_HEADER.size)
    values = _HEADER.unpack(raw)
    metadata_pointer = int(values[1])
    channel_count = int(values[11])
    if channel_count <= 0 or channel_count > 10_000:
        raise ValueError(f"invalid MoTeC channel count: {channel_count}")
    if metadata_pointer <= 0 or metadata_pointer + _CHANNEL.size > file_size:
        raise ValueError("invalid first-channel metadata pointer")
    return (
        MotecHeader(
            driver=_decode_ascii(values[14]),
            vehicle_id=_decode_ascii(values[15]),
            venue=_decode_ascii(values[16]),
            date=_decode_ascii(values[12]),
            time=_decode_ascii(values[13]),
            channel_count=channel_count,
        ),
        metadata_pointer,
    )


def _read_channel_metadata(
    handle: object,
    *,
    metadata_pointer: int,
    file_size: int,
) -> tuple[_Channel, int]:
    if metadata_pointer <= 0 or metadata_pointer + _CHANNEL.size > file_size:
        raise ValueError(f"invalid channel metadata pointer: {metadata_pointer}")
    handle.seek(metadata_pointer)  # type: ignore[attr-defined]
    values = _CHANNEL.unpack(handle.read(_CHANNEL.size))  # type: ignore[attr-defined]
    next_pointer = int(values[1])
    data_pointer = int(values[2])
    sample_count = int(values[3])
    storage = (int(values[5]), int(values[6]))
    try:
        dtype = _DTYPES[storage]
    except KeyError as error:
        name = _decode_ascii(values[12])
        raise ValueError(f"unsupported storage type {storage} for channel {name!r}") from error
    data_end = data_pointer + sample_count * dtype.itemsize
    if sample_count <= 0 or data_pointer <= 0 or data_end > file_size:
        raise ValueError(f"invalid data range for channel {_decode_ascii(values[12])!r}")
    if int(values[10]) == 0:
        raise ValueError(f"zero scale for channel {_decode_ascii(values[12])!r}")
    return (
        _Channel(
            name=_decode_ascii(values[12]),
            data_pointer=data_pointer,
            sample_count=sample_count,
            frequency_hz=int(values[7]),
            dtype=dtype,
            shift=int(values[8]),
            multiplier=int(values[9]),
            scale=int(values[10]),
            decimal_places=int(values[11]),
        ),
        next_pointer,
    )


def read_motec_channels(
    path: str | Path,
    channel_names: tuple[str, ...],
) -> tuple[MotecHeader, dict[str, FloatArray]]:
    """Read selected channels and return them under the requested names.

    Channel lookup is case-insensitive. Every returned array is converted to
    ``float32`` after applying the scale/shift metadata stored in the file.
    """

    if not channel_names:
        raise ValueError("at least one channel name is required")
    requested = {name.casefold(): name for name in channel_names}
    if len(requested) != len(channel_names):
        raise ValueError("channel names must be unique ignoring case")

    source = Path(path)
    file_size = source.stat().st_size
    selected: dict[str, _Channel] = {}
    with source.open("rb") as handle:
        header, metadata_pointer = _read_header(handle, file_size=file_size)
        visited: set[int] = set()
        while metadata_pointer:
            if metadata_pointer in visited:
                raise ValueError("cycle in MoTeC channel metadata")
            visited.add(metadata_pointer)
            if len(visited) > header.channel_count:
                raise ValueError("channel metadata exceeds header channel count")
            channel, metadata_pointer = _read_channel_metadata(
                handle,
                metadata_pointer=metadata_pointer,
                file_size=file_size,
            )
            requested_name = requested.get(channel.name.casefold())
            if requested_name is not None:
                selected[requested_name] = channel

        missing = [name for name in channel_names if name not in selected]
        if missing:
            raise ValueError(f"missing MoTeC channel(s): {', '.join(missing)}")

        result: dict[str, FloatArray] = {}
        for requested_name in channel_names:
            channel = selected[requested_name]
            handle.seek(channel.data_pointer)
            raw = np.fromfile(handle, dtype=channel.dtype, count=channel.sample_count)
            if raw.size != channel.sample_count:
                raise ValueError(f"incomplete data for channel {requested_name!r}")
            scaled = (
                raw.astype(np.float64) / channel.scale * 10.0 ** (-channel.decimal_places)
                + channel.shift
            ) * channel.multiplier
            result[requested_name] = np.asarray(scaled, dtype=np.float32)
    return header, result
