"""Probabilistic reference trajectories fitted to expert demonstrations."""

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
    ``world_position_xy_m`` is optional for generic ProMP examples, but is
    required when prediction targets and style rewards are expressed in the
    vehicle's current ego frame.
    """

    track_distance_m: FloatArray
    lateral_offset_m: FloatArray
    heading_rad: FloatArray
    speed_mps: FloatArray
    track_length_m: float
    world_position_xy_m: FloatArray | None = None
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
    ) -> ReferenceTrajectory:
        """Create a simple closed reference used by the mock backend."""

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
        )

    def preview(
        self,
        progress_m: float,
        time_offsets_s: npt.ArrayLike = REFERENCE_TIME_OFFSETS_S,
    ) -> ReferenceFeaturePreview:
        """Interpolate a time-spaced preview while respecting the closed lap."""

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
        """Map one or more current progress values to future reference distances."""

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
        """Return ego-frame ``(x=lateral-left, z=forward)`` reference points."""

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
        """Return the current reference projection followed by future points."""

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
        """Project world positions onto this closed reference line."""

        if self.world_position_xy_m is None:
            raise ValueError("world_position_xy_m is required for progress projection")
        return project_points_to_closed_reference_progress(
            positions_xy_m,
            self.world_position_xy_m,
            self.track_distance_m,
            self.track_length_m,
        )

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
        """Return shape ``(samples, 3)`` for ProMP fitting."""

        return np.column_stack((self.lateral_offset_m, self.heading_rad, self.speed_mps)).astype(
            np.float32
        )

    def to_npz(
        self,
        path: str | Path,
        *,
        extra_arrays: Mapping[str, npt.ArrayLike] | None = None,
    ) -> None:
        """Persist a reference in a non-executable NumPy archive."""

        reserved = {
            "track_distance_m",
            "lateral_offset_m",
            "heading_rad",
            "speed_mps",
            "track_length_m",
            "world_position_xy_m",
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
        """Load a reference archive with pickle support explicitly disabled."""

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
            return cls(
                track_distance_m=archive["track_distance_m"],
                lateral_offset_m=archive["lateral_offset_m"],
                heading_rad=heading,
                speed_mps=archive["speed_mps"],
                track_length_m=float(track_length),
                world_position_xy_m=world_position,
            )


class ReferenceSampler(Protocol):
    """Episode-level source of a reference held constant until termination."""

    def sample(self, rng: np.random.Generator) -> ReferenceTrajectory:
        """Sample one episode context."""

    def mean(self) -> ReferenceTrajectory:
        """Return the deterministic mean reference used for evaluation."""


@dataclass(frozen=True, slots=True)
class FixedReferenceSampler:
    """Sampler that always returns one immutable reference."""

    reference: ReferenceTrajectory

    def sample(self, rng: np.random.Generator) -> ReferenceTrajectory:
        del rng
        return self.reference

    def mean(self) -> ReferenceTrajectory:
        return self.reference


@dataclass(frozen=True, slots=True)
class ProbabilisticMovementPrimitive:
    """Gaussian distribution over RBF trajectory weights.

    Demonstrations must share the same strictly increasing track-distance grid.
    The fitted distribution is immutable during RL training, matching the
    contextual reference mechanism used in the thesis.
    """

    track_distance_m: FloatArray
    track_length_m: float
    basis_centres: FloatArray
    basis_width: float
    weight_mean: FloatArray
    weight_covariance: FloatArray
    output_dimension: int = 3

    @classmethod
    def fit(
        cls,
        demonstrations: npt.ArrayLike,
        *,
        track_distance_m: npt.ArrayLike,
        track_length_m: float,
        basis_count: int = 32,
        ridge: float = 1e-6,
        covariance_jitter: float = 1e-6,
    ) -> ProbabilisticMovementPrimitive:
        """Fit RBF weights and their Gaussian distribution offline."""

        demos = np.asarray(demonstrations, dtype=np.float64)
        distance = np.asarray(track_distance_m, dtype=np.float64)
        if demos.ndim != 3 or demos.shape[2] != 3:
            raise ValueError("demonstrations must have shape (laps, samples, 3)")
        if demos.shape[1] != distance.size:
            raise ValueError("demonstration sample count must match track distances")
        if demos.shape[0] < 1:
            raise ValueError("at least one demonstration is required")
        if basis_count < 2:
            raise ValueError("basis_count must be at least two")
        if ridge <= 0 or covariance_jitter <= 0:
            raise ValueError("regularisation values must be positive")
        if not np.all(np.isfinite(demos)):
            raise ValueError("demonstrations contain non-finite values")
        if (
            distance.ndim != 1
            or distance.size < 2
            or not np.all(np.diff(distance) > 0)
            or distance[0] < 0
            or distance[-1] >= track_length_m
        ):
            raise ValueError("invalid closed-lap track-distance grid")

        demos = demos.copy()
        demos[:, :, 1] = np.unwrap(demos[:, :, 1], axis=1)
        phase = distance / track_length_m
        centres = np.linspace(0.0, 1.0, basis_count, endpoint=False, dtype=np.float64)
        width = float(1.5 / basis_count)
        design = _normalised_rbf_design(phase, centres, width)
        regularised = design.T @ design + ridge * np.eye(basis_count)
        projection = np.linalg.solve(regularised, design.T)
        weights = np.stack([projection @ lap for lap in demos], axis=0)
        flattened = weights.reshape(demos.shape[0], basis_count * demos.shape[2])
        weight_mean = flattened.mean(axis=0)

        if demos.shape[0] == 1:
            covariance = np.zeros((flattened.shape[1], flattened.shape[1]))
        else:
            centred = flattened - weight_mean
            covariance = centred.T @ centred / demos.shape[0]
        covariance = np.atleast_2d(covariance)
        covariance += covariance_jitter * np.eye(covariance.shape[0])

        return cls(
            track_distance_m=distance.astype(np.float32),
            track_length_m=float(track_length_m),
            basis_centres=centres.astype(np.float32),
            basis_width=width,
            weight_mean=weight_mean.astype(np.float32),
            weight_covariance=covariance.astype(np.float32),
        )

    def sample(self, rng: np.random.Generator) -> ReferenceTrajectory:
        """Draw one reference from the Gaussian weight distribution."""

        weights = rng.multivariate_normal(
            self.weight_mean.astype(np.float64),
            self.weight_covariance.astype(np.float64),
        )
        return self._decode(weights)

    def mean(self) -> ReferenceTrajectory:
        """Decode the distribution mean for deterministic evaluation."""

        return self._decode(self.weight_mean)

    def _decode(self, flattened_weights: npt.ArrayLike) -> ReferenceTrajectory:
        phase = self.track_distance_m.astype(np.float64) / self.track_length_m
        design = _normalised_rbf_design(
            phase,
            self.basis_centres.astype(np.float64),
            self.basis_width,
        )
        weights = np.asarray(flattened_weights, dtype=np.float64).reshape(
            self.basis_centres.size,
            self.output_dimension,
        )
        features = (design @ weights).astype(np.float32)
        features[:, 1] = np.arctan2(
            np.sin(features[:, 1]),
            np.cos(features[:, 1]),
        )
        return ReferenceTrajectory(
            track_distance_m=self.track_distance_m,
            lateral_offset_m=features[:, 0],
            heading_rad=features[:, 1],
            speed_mps=np.maximum(features[:, 2], 0.1),
            track_length_m=self.track_length_m,
        )


def _normalised_rbf_design(
    phase: npt.NDArray[np.float64],
    centres: npt.NDArray[np.float64],
    width: float,
) -> npt.NDArray[np.float64]:
    phase_distance = np.abs(phase[:, None] - centres[None, :])
    circular_distance = np.minimum(phase_distance, 1.0 - phase_distance)
    unnormalised = np.exp(-0.5 * (circular_distance / width) ** 2)
    return np.asarray(
        unnormalised / np.maximum(unnormalised.sum(axis=1, keepdims=True), 1e-12),
        dtype=np.float64,
    )
