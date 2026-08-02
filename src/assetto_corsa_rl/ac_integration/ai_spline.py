"""Strict reader for Assetto Corsa ``ai/fast_lane.ai`` geometry."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from assetto_corsa_rl.types import FloatArray

FAST_LANE_FORMAT_VERSION = 7
_HEADER = struct.Struct("<4i")
_POINT_SIZE = 20
_EXTRA_POINT_SIZE = 72
_EXTRA_FLOAT_COUNT = _EXTRA_POINT_SIZE // 4
_SIDE_LEFT_INDEX = 5
_SIDE_RIGHT_INDEX = 6


class AISplineFormatError(ValueError):
    """Raised when a fast-lane file is unsupported or structurally invalid."""


@dataclass(frozen=True, slots=True)
class AssettoCorsaAISpline:
    """Validated centreline and boundary distances from one v7 AI spline."""

    distance_m: FloatArray
    position_ac_xyz_m: FloatArray
    side_left_m: FloatArray
    side_right_m: FloatArray
    track_length_m: float
    source_sha256: str

    def __post_init__(self) -> None:
        distance = np.asarray(self.distance_m, dtype=np.float32)
        position = np.asarray(self.position_ac_xyz_m, dtype=np.float32)
        side_left = np.asarray(self.side_left_m, dtype=np.float32)
        side_right = np.asarray(self.side_right_m, dtype=np.float32)
        if distance.ndim != 1 or distance.size < 3:
            raise AISplineFormatError("AI spline must contain at least three points")
        if position.shape != (distance.size, 3):
            raise AISplineFormatError("AI spline positions must have shape (points, 3)")
        if side_left.shape != distance.shape or side_right.shape != distance.shape:
            raise AISplineFormatError("AI spline side distances must match the point count")
        if not all(
            np.all(np.isfinite(array)) for array in (distance, position, side_left, side_right)
        ):
            raise AISplineFormatError("AI spline contains non-finite values")
        if not np.isclose(distance[0], 0.0, atol=1e-5) or np.any(np.diff(distance) <= 0):
            raise AISplineFormatError("AI spline distance must start at zero and increase")
        if np.any(side_left < 0) or np.any(side_right < 0):
            raise AISplineFormatError("AI spline boundary distances must be non-negative")
        if np.any(side_left + side_right <= 0):
            raise AISplineFormatError("AI spline must have a positive width at every point")
        if not np.isfinite(self.track_length_m) or self.track_length_m <= float(distance[-1]):
            raise AISplineFormatError("AI spline closed-lap length is invalid")
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_sha256
        ):
            raise AISplineFormatError("AI spline SHA-256 must be lowercase hexadecimal")
        object.__setattr__(self, "distance_m", distance)
        object.__setattr__(self, "position_ac_xyz_m", position)
        object.__setattr__(self, "side_left_m", side_left)
        object.__setattr__(self, "side_right_m", side_right)

    @property
    def centre_xy_m(self) -> FloatArray:
        """Return the spline in the repository's world convention ``(AC z, AC x)``."""

        return np.asarray(self.position_ac_xyz_m[:, (2, 0)], dtype=np.float32)

    def boundaries_xy_m(self) -> tuple[FloatArray, FloatArray]:
        """Construct left/right boundary points from centreline side distances."""

        centre = self.centre_xy_m.astype(np.float64)
        tangent = np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0)
        tangent_norm = np.linalg.norm(tangent, axis=1)
        if np.any(tangent_norm <= 1e-9):
            raise AISplineFormatError("AI spline contains a degenerate centreline tangent")
        tangent /= tangent_norm[:, None]
        left_normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        left = centre + left_normal * self.side_left_m[:, None]
        right = centre - left_normal * self.side_right_m[:, None]
        return np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Decode a v7 fast-lane file and reject unknown layouts fail-closed."""

        source = Path(path)
        try:
            payload = source.read_bytes()
        except OSError as error:
            raise AISplineFormatError(f"cannot read AI spline {source}: {error}") from error
        if len(payload) < _HEADER.size:
            raise AISplineFormatError("AI spline is smaller than its header")
        version, point_count, _lap_time, _sample_count = _HEADER.unpack_from(payload)
        if version != FAST_LANE_FORMAT_VERSION:
            raise AISplineFormatError(
                f"unsupported AI spline version {version}; expected {FAST_LANE_FORMAT_VERSION}"
            )
        if point_count < 3:
            raise AISplineFormatError("AI spline point count must be at least three")

        points_offset = _HEADER.size
        extra_count_offset = points_offset + point_count * _POINT_SIZE
        extras_offset = extra_count_offset + 4
        required_size = extras_offset + point_count * _EXTRA_POINT_SIZE
        if required_size > len(payload):
            raise AISplineFormatError("AI spline ends before its point metadata")
        extra_count = struct.unpack_from("<i", payload, extra_count_offset)[0]
        if extra_count != point_count:
            raise AISplineFormatError(
                f"AI spline point/metadata count mismatch: {point_count} != {extra_count}"
            )

        point_dtype = np.dtype(
            [
                ("position", "<f4", (3,)),
                ("distance", "<f4"),
                ("identifier", "<i4"),
            ]
        )
        points = np.frombuffer(
            payload,
            dtype=point_dtype,
            count=point_count,
            offset=points_offset,
        )
        identifiers = np.asarray(points["identifier"], dtype=np.int32)
        if not np.array_equal(identifiers, np.arange(point_count, dtype=np.int32)):
            raise AISplineFormatError("AI spline point identifiers are not consecutive")
        position = np.asarray(points["position"], dtype=np.float32).copy()
        distance = np.asarray(points["distance"], dtype=np.float32).copy()
        extras = np.frombuffer(
            payload,
            dtype="<f4",
            count=point_count * _EXTRA_FLOAT_COUNT,
            offset=extras_offset,
        ).reshape(point_count, _EXTRA_FLOAT_COUNT)
        side_left = np.asarray(extras[:, _SIDE_LEFT_INDEX], dtype=np.float32).copy()
        side_right = np.asarray(extras[:, _SIDE_RIGHT_INDEX], dtype=np.float32).copy()
        closing_length = float(np.linalg.norm(position[-1].astype(np.float64) - position[0]))
        track_length = float(distance[-1]) + closing_length
        return cls(
            distance_m=distance,
            position_ac_xyz_m=position,
            side_left_m=side_left,
            side_right_m=side_right,
            track_length_m=track_length,
            source_sha256=hashlib.sha256(payload).hexdigest(),
        )
