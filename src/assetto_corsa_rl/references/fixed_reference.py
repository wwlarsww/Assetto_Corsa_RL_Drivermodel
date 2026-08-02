"""Reference-trajectory contract used by the initial ``baseline_v1`` method.

The baseline reuses one deterministic racing line in every episode. This module
owns interpolation, validation, persistence and sampling; it does not interact
with Assetto Corsa or perform learning.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.references.projection import (
    project_points_to_closed_reference_progress,
)
from assetto_corsa_rl.types import (
    REFERENCE_TIME_OFFSETS_S,
    FloatArray,
    ReferenceFeaturePreview,
)


def _validate_world_position(
    value: npt.ArrayLike | None,
    sample_count: int,
) -> FloatArray | None:
    if value is None:
        return None
    world_position = np.asarray(value, dtype=np.float32)
    if world_position.shape != (sample_count, 2):
        raise ValueError(
            f"world_position_xy_m must have shape ({sample_count}, 2), got {world_position.shape}"
        )
    if not np.all(np.isfinite(world_position)):
        raise ValueError("world_position_xy_m contains non-finite values")
    return world_position


@dataclass(frozen=True, slots=True)
class ReferenceTrajectory:
    """Reference features parameterised over distance around one closed lap.

    ``heading_rad`` is the world-frame tangent heading of the reference line.
    ``world_position_xy_m`` is optional for generic reference examples, but is
    required when prediction targets are expressed in the vehicle's current ego frame.
    """

    track_distance_m: FloatArray
    lateral_offset_m: FloatArray
    heading_rad: FloatArray
    speed_mps: FloatArray
    track_length_m: float
    world_position_xy_m: FloatArray | None = None
    source_track_sha256: str | None = None
    _lap_time_nodes_s: FloatArray = field(init=False, repr=False, compare=False)
    _lap_time_s: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        distance = np.asarray(self.track_distance_m, dtype=np.float32)
        arrays = {
            "lateral_offset_m": np.asarray(self.lateral_offset_m, dtype=np.float32),
            "heading_rad": np.asarray(self.heading_rad, dtype=np.float32),
            "speed_mps": np.asarray(self.speed_mps, dtype=np.float32),
        }
        if not np.isfinite(self.track_length_m) or self.track_length_m <= 0:
            raise ValueError("track_length_m must be positive and finite")
        if distance.ndim != 1 or distance.size < 2:
            raise ValueError("track_distance_m must contain at least two samples")
        if not np.all(np.diff(distance) > 0):
            raise ValueError("track distances must be strictly increasing")
        if not np.isclose(distance[0], 0.0, atol=1e-5):
            raise ValueError("track distances must start at zero")
        if distance[-1] >= self.track_length_m:
            raise ValueError("reference distances must lie in [0, track_length_m)")
        for name, array in arrays.items():
            if array.shape != distance.shape:
                raise ValueError(f"{name} must have shape {distance.shape}, got {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")
        if np.any(arrays["speed_mps"] <= 0):
            raise ValueError("reference speed_mps must be positive")

        world_position = _validate_world_position(self.world_position_xy_m, distance.size)
        if self.source_track_sha256 is not None and (
            len(self.source_track_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_track_sha256)
        ):
            raise ValueError("source_track_sha256 must be lowercase SHA-256 hexadecimal")

        object.__setattr__(self, "track_distance_m", distance)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "world_position_xy_m", world_position)

        distance_nodes = np.append(distance.astype(np.float64), self.track_length_m)
        speed_nodes = np.append(
            arrays["speed_mps"].astype(np.float64),
            float(arrays["speed_mps"][0]),
        )
        average_segment_speed = 0.5 * (speed_nodes[:-1] + speed_nodes[1:])
        segment_time = np.diff(distance_nodes) / np.maximum(average_segment_speed, 0.1)
        lap_time_nodes = np.concatenate(([0.0], np.cumsum(segment_time)))
        object.__setattr__(self, "_lap_time_nodes_s", lap_time_nodes.astype(np.float32))
        object.__setattr__(self, "_lap_time_s", float(lap_time_nodes[-1]))

    @classmethod
    def constant(
        cls,
        *,
        track_length_m: float,
        speed_mps: float,
        lateral_offset_m: float = 0.0,
        sample_count: int = 256,
        source_track_sha256: str | None = None,
    ) -> ReferenceTrajectory:
        """Create a simple closed reference used by the mock backend.

        Args:
            track_length_m: Closed-lap length in metres.
            speed_mps: Constant positive reference speed.
            lateral_offset_m: Constant line offset used in mock coordinates.
            sample_count: Number of evenly spaced interpolation nodes.
            source_track_sha256: Optional source AI-spline digest.

        Returns:
            Deterministic reference with explicit world positions.
        """

        if sample_count < 2:
            raise ValueError("sample_count must be at least two")
        distance = np.linspace(
            0.0,
            track_length_m,
            sample_count,
            endpoint=False,
            dtype=np.float32,
        )
        return cls(
            track_distance_m=distance,
            lateral_offset_m=np.full_like(distance, lateral_offset_m),
            heading_rad=np.zeros_like(distance),
            speed_mps=np.full_like(distance, speed_mps),
            track_length_m=track_length_m,
            world_position_xy_m=np.column_stack(
                (distance, np.full_like(distance, lateral_offset_m))
            ).astype(np.float32),
            source_track_sha256=source_track_sha256,
        )

    def preview(
        self,
        progress_m: float,
        time_offsets_s: npt.ArrayLike = REFERENCE_TIME_OFFSETS_S,
    ) -> ReferenceFeaturePreview:
        """Interpolate a time-spaced feature preview on the closed lap.

        Args:
            progress_m: Current reference arc-length coordinate.
            time_offsets_s: Positive future offsets, normally twenty values.

        Returns:
            Offset, heading, and speed at every requested future time.
        """

        times = np.asarray(time_offsets_s, dtype=np.float32)
        if times.shape != (20,) or np.any(times <= 0):
            raise ValueError("time_offsets_s must contain twenty positive values")

        query_distance = self.future_distances(progress_m, times)
        return ReferenceFeaturePreview(
            lateral_offset_m=self._periodic_interpolate(
                query_distance,
                self.lateral_offset_m,
            ),
            heading_rad=self._periodic_interpolate_heading(query_distance),
            speed_mps=self._periodic_interpolate(query_distance, self.speed_mps),
        )

    def future_distances(
        self,
        progress_m: float | npt.ArrayLike,
        time_offsets_s: npt.ArrayLike = REFERENCE_TIME_OFFSETS_S,
    ) -> FloatArray:
        """Map current progress values to time-parameterised future distances.

        Args:
            progress_m: Scalar or array of closed-lap arc-length coordinates.
            time_offsets_s: Positive future offsets in seconds.

        Returns:
            Future arc-length coordinates with offsets on the final axis.
        """

        times = np.asarray(time_offsets_s, dtype=np.float64)
        if times.ndim != 1 or times.size < 1:
            raise ValueError("time_offsets_s must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(times)) or np.any(times <= 0):
            raise ValueError("time_offsets_s must contain finite positive values")
        progress = np.asarray(progress_m, dtype=np.float64)
        if not np.all(np.isfinite(progress)):
            raise ValueError("progress_m must be finite")

        distance_nodes = np.append(
            self.track_distance_m.astype(np.float64),
            self.track_length_m,
        )
        current_distance = progress % self.track_length_m
        current_lap_time = np.interp(
            current_distance,
            distance_nodes,
            self._lap_time_nodes_s.astype(np.float64),
        )
        query_lap_time = np.mod(
            np.expand_dims(current_lap_time, axis=-1) + times,
            self._lap_time_s,
        )
        return np.asarray(
            np.interp(
                query_lap_time.reshape(-1),
                self._lap_time_nodes_s.astype(np.float64),
                distance_nodes,
            ).reshape(query_lap_time.shape),
            dtype=np.float32,
        )

    def ego_path_preview(
        self,
        progress_m: float,
        vehicle_position_xy_m: npt.ArrayLike,
        vehicle_heading_rad: float,
        time_offsets_s: npt.ArrayLike = REFERENCE_TIME_OFFSETS_S,
    ) -> FloatArray:
        """Return future reference points in the current vehicle frame.

        Args:
            progress_m: Current reference arc-length coordinate.
            vehicle_position_xy_m: Current world-space vehicle position.
            vehicle_heading_rad: Current world-space vehicle heading.
            time_offsets_s: Positive future offsets in seconds.

        Returns:
            Ego-frame ``(x=lateral-left, z=forward)`` points.
        """

        return self.ego_path_with_origin(
            progress_m,
            vehicle_position_xy_m,
            vehicle_heading_rad,
            time_offsets_s,
        )[1:]

    def ego_path_with_origin(
        self,
        progress_m: float,
        vehicle_position_xy_m: npt.ArrayLike,
        vehicle_heading_rad: float,
        time_offsets_s: npt.ArrayLike = REFERENCE_TIME_OFFSETS_S,
    ) -> FloatArray:
        """Return the current reference projection followed by future points.

        Args:
            progress_m: Current reference arc-length coordinate.
            vehicle_position_xy_m: Current world-space vehicle position.
            vehicle_heading_rad: Current world-space vehicle heading.
            time_offsets_s: Positive future offsets in seconds.

        Returns:
            Current-plus-future ego-frame reference polyline.
        """

        if self.world_position_xy_m is None:
            raise ValueError(
                "world_position_xy_m is required for ego-frame reference prediction targets"
            )
        origin = np.asarray(vehicle_position_xy_m, dtype=np.float32)
        if origin.shape != (2,) or not np.all(np.isfinite(origin)):
            raise ValueError("vehicle_position_xy_m must be a finite vector with shape (2,)")
        if not np.isfinite(vehicle_heading_rad):
            raise ValueError("vehicle_heading_rad must be finite")

        future_distance = self.future_distances(progress_m, time_offsets_s)
        query_distance = np.concatenate(
            (
                np.asarray([progress_m % self.track_length_m], dtype=np.float32),
                future_distance,
            )
        )
        world_x = self._periodic_interpolate(query_distance, self.world_position_xy_m[:, 0])
        world_y = self._periodic_interpolate(query_distance, self.world_position_xy_m[:, 1])
        displacement = np.column_stack((world_x, world_y)) - origin
        cosine = np.cos(vehicle_heading_rad)
        sine = np.sin(vehicle_heading_rad)
        world_to_ego = np.asarray(
            [[-sine, cosine], [cosine, sine]],
            dtype=np.float32,
        )
        return np.asarray(displacement @ world_to_ego.T, dtype=np.float32)

    def project_world_positions(self, positions_xy_m: npt.ArrayLike) -> FloatArray:
        """Project world positions onto this closed reference line.

        Args:
            positions_xy_m: World-space points with final dimension two.

        Returns:
            Wrap-aware arc-length coordinate for every input point.
        """

        if self.world_position_xy_m is None:
            raise ValueError("world_position_xy_m is required for progress projection")
        return project_points_to_closed_reference_progress(
            positions_xy_m,
            self.world_position_xy_m,
            self.track_distance_m,
            self.track_length_m,
        )

    def world_pose_at(self, progress_m: float) -> tuple[FloatArray, float]:
        """Interpolate the baseline line's world pose.

        Args:
            progress_m: Reference arc length, wrapped to the closed lap.

        Returns:
            World position and wrapped tangent heading in radians.
        """

        if self.world_position_xy_m is None:
            raise ValueError("world_position_xy_m is required for world-pose interpolation")
        if not np.isfinite(progress_m):
            raise ValueError("progress_m must be finite")
        query = np.asarray([progress_m], dtype=np.float32)
        position = np.asarray(
            [
                self._periodic_interpolate(query, self.world_position_xy_m[:, 0])[0],
                self._periodic_interpolate(query, self.world_position_xy_m[:, 1])[0],
            ],
            dtype=np.float32,
        )
        heading = float(self._periodic_interpolate_heading(query)[0])
        return position, heading

    def _periodic_interpolate(
        self,
        query_distance_m: float | FloatArray,
        values: FloatArray,
    ) -> FloatArray:
        query = np.mod(query_distance_m, self.track_length_m)
        periodic_distance = np.append(self.track_distance_m, self.track_length_m)
        periodic_values = np.append(values, values[0])
        return np.asarray(
            np.interp(query, periodic_distance, periodic_values),
            dtype=np.float32,
        )

    def _periodic_interpolate_heading(
        self,
        query_distance_m: float | FloatArray,
    ) -> FloatArray:
        unwrapped = np.unwrap(self.heading_rad.astype(np.float64))
        closing_delta = np.arctan2(
            np.sin(float(self.heading_rad[0]) - unwrapped[-1]),
            np.cos(float(self.heading_rad[0]) - unwrapped[-1]),
        )
        periodic_heading = np.append(unwrapped, unwrapped[-1] + closing_delta)
        periodic_distance = np.append(self.track_distance_m, self.track_length_m)
        query = np.mod(query_distance_m, self.track_length_m)
        interpolated = np.interp(query, periodic_distance, periodic_heading)
        wrapped = np.arctan2(np.sin(interpolated), np.cos(interpolated))
        return np.asarray(wrapped, dtype=np.float32)

    def as_feature_matrix(self) -> FloatArray:
        """Return shape ``(samples, 3)`` for baseline feature export."""

        return np.column_stack((self.lateral_offset_m, self.heading_rad, self.speed_mps)).astype(
            np.float32
        )

    def to_npz(
        self,
        path: str | Path,
        *,
        extra_arrays: Mapping[str, npt.ArrayLike] | None = None,
    ) -> None:
        """Persist a reference in a deterministic, non-executable NumPy archive.

        Args:
            path: Destination archive path.
            extra_arrays: Optional provenance arrays whose names do not replace
                trajectory fields.
        """

        reserved = {
            "track_distance_m",
            "lateral_offset_m",
            "heading_rad",
            "speed_mps",
            "track_length_m",
            "world_position_xy_m",
            "source_track_sha256",
        }
        extras = dict(extra_arrays or {})
        overlap = reserved & extras.keys()
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"extra arrays override reserved key(s): {names}")
        arrays: dict[str, npt.ArrayLike] = {
            "track_distance_m": self.track_distance_m,
            "lateral_offset_m": self.lateral_offset_m,
            "heading_rad": self.heading_rad,
            "speed_mps": self.speed_mps,
            "track_length_m": np.asarray(self.track_length_m, dtype=np.float64),
            **extras,
        }
        if self.world_position_xy_m is not None:
            arrays["world_position_xy_m"] = self.world_position_xy_m
        if self.source_track_sha256 is not None:
            arrays["source_track_sha256"] = np.asarray(self.source_track_sha256)
        for name, value in arrays.items():
            if np.asarray(value).dtype.hasobject:
                raise ValueError(f"array {name!r} has unsafe object dtype")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.tmp.npz")
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(  # type: ignore[no-untyped-call]
                    buffer,
                    np.asarray(arrays[name]),
                    allow_pickle=False,
                )
                member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.create_system = 3
                member.external_attr = 0o600 << 16
                archive.writestr(
                    member,
                    buffer.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        temporary.replace(destination)

    @classmethod
    def from_npz(cls, path: str | Path) -> ReferenceTrajectory:
        """Load a reference archive with pickle support explicitly disabled.

        Args:
            path: Archive produced by :meth:`to_npz` or the offline builder.

        Returns:
            Validated immutable reference trajectory.
        """

        required = {
            "track_distance_m",
            "lateral_offset_m",
            "heading_rad",
            "speed_mps",
            "track_length_m",
        }
        with np.load(Path(path), allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"reference archive misses key(s): {names}")
            track_length = np.asarray(archive["track_length_m"], dtype=np.float64)
            if track_length.shape != ():
                raise ValueError("track_length_m must be a scalar")
            heading = (
                archive["world_heading_rad"]
                if "world_heading_rad" in archive.files
                else archive["heading_rad"]
            )
            world_position = (
                archive["world_position_xy_m"] if "world_position_xy_m" in archive.files else None
            )
            source_track_sha256 = (
                str(archive["source_track_sha256"].item())
                if "source_track_sha256" in archive.files
                else None
            )
            return cls(
                track_distance_m=archive["track_distance_m"],
                lateral_offset_m=archive["lateral_offset_m"],
                heading_rad=heading,
                speed_mps=archive["speed_mps"],
                track_length_m=float(track_length),
                world_position_xy_m=world_position,
                source_track_sha256=source_track_sha256,
            )


class ReferenceSampler(Protocol):
    """Episode-level source of a reference held constant until termination."""

    def sample(self, rng: np.random.Generator) -> ReferenceTrajectory:
        """Sample one episode context."""

        ...

    def mean(self) -> ReferenceTrajectory:
        """Return the deterministic mean reference used for evaluation."""

        ...


@dataclass(frozen=True, slots=True)
class BaselineReferenceSampler:
    """Sampler that returns the baseline's immutable line in every episode."""

    reference: ReferenceTrajectory

    def sample(self, rng: np.random.Generator) -> ReferenceTrajectory:
        """Return the prescribed line independently of the episode RNG.

        Args:
            rng: Episode generator accepted to satisfy :class:`ReferenceSampler`.

        Returns:
            The immutable baseline reference.
        """

        del rng
        return self.reference

    def mean(self) -> ReferenceTrajectory:
        """Return the immutable baseline reference for evaluation."""

        return self.reference


# Backwards-compatibility alias: older code expected a ``FixedReferenceSampler``
# name. Keep the alias so imports elsewhere in the codebase remain stable
# after the earlier rename to `BaselineReferenceSampler`.
FixedReferenceSampler = BaselineReferenceSampler
