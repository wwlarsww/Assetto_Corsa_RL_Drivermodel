"""Safe export of fixed-reference behavioural-cloning tensors from MoTeC laps."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.ac_integration.ai_spline import AssettoCorsaAISpline
from assetto_corsa_rl.ac_integration.launcher import configured_fast_lane_path
from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.data.augmentation import (
    BC_MIRROR_AUGMENTATION_NAME,
    mirror_bc_samples,
)
from assetto_corsa_rl.data.motec import read_motec_channels
from assetto_corsa_rl.features.normalizer import StaticZScoreNormalizer
from assetto_corsa_rl.features.preprocessing import (
    LOW_SPEED_SLIP_PREPROCESSING_NAME,
    PREVIOUS_ACTION_IDENTITY_FEATURE_NAMES,
    PREVIOUS_ACTION_NORMALIZATION_NAME,
    stabilise_low_speed_slip,
)
from assetto_corsa_rl.features.schema import OBSERVATION_FEATURE_NAMES, OBSERVATION_SIZE
from assetto_corsa_rl.features.track_geometry import (
    TrackGeometry,
    periodic_interpolate,
    world_to_ego,
)
from assetto_corsa_rl.models.trajectory_contract import (
    BEZIER_TRAJECTORY_CODEC,
    PREDICTOR_COORDINATE_SCALE_M,
)
from assetto_corsa_rl.references.fixed_reference import ReferenceTrajectory
from assetto_corsa_rl.types import (
    PREDICTOR_TIME_OFFSETS_S,
    REFERENCE_TIME_OFFSETS_S,
    FloatArray,
)

STANDARD_GRAVITY_MPS2 = 9.80665
ACTI_MOTEC_CALIBRATION_NAME = "acti_motec_ai_spline_absolute_v1"

_CHANNEL_NAMES = (
    "Ground Speed",
    "CG Accel Lateral",
    "CG Accel Longitudinal",
    "Tire Slip Ratio FL",
    "Tire Slip Ratio FR",
    "Tire Slip Ratio RL",
    "Tire Slip Ratio RR",
    "Tire Slip Angle FL",
    "Tire Slip Angle FR",
    "Tire Slip Angle RL",
    "Tire Slip Angle RR",
    "Car Coord X",
    "Car Coord Y",
    "Car Pos Norm",
    "Lap Time",
    "Throttle Pos",
    "Brake Pos",
    "Steering Angle",
    "In Pit",
    "Num Tires Off Track",
    "Lap Invalidated",
)


@dataclass(frozen=True, slots=True)
class BCSplit:
    """One session-disjoint dataset split."""

    observations: FloatArray
    expert_action: FloatArray
    predictor_target_xz_m: FloatArray
    predictor_valid_mask: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        observations = np.asarray(self.observations, dtype=np.float32)
        actions = np.asarray(self.expert_action, dtype=np.float32)
        targets = np.asarray(self.predictor_target_xz_m, dtype=np.float32)
        valid = np.asarray(self.predictor_valid_mask, dtype=np.bool_)
        sample_count = observations.shape[0] if observations.ndim == 2 else -1
        if observations.shape != (sample_count, OBSERVATION_SIZE):
            raise ValueError(f"observations must have shape (samples, {OBSERVATION_SIZE})")
        if actions.shape != (sample_count, 2):
            raise ValueError("expert_action must have shape (samples, 2)")
        if targets.shape != (sample_count, PREDICTOR_TIME_OFFSETS_S.size, 2):
            raise ValueError(
                "predictor_target_xz_m must have shape "
                f"(samples, {PREDICTOR_TIME_OFFSETS_S.size}, 2)"
            )
        if valid.shape != (sample_count,):
            raise ValueError("predictor_valid_mask must have shape (samples,)")
        if not all(np.all(np.isfinite(array)) for array in (observations, actions, targets)):
            raise ValueError("BC tensors must contain only finite values")
        if np.any(np.abs(actions) > 1.0 + 1e-6):
            raise ValueError("expert actions must lie in [-1, 1]")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "expert_action", actions)
        object.__setattr__(self, "predictor_target_xz_m", targets)
        object.__setattr__(self, "predictor_valid_mask", valid)

    @property
    def sample_count(self) -> int:
        """Return the number of aligned demonstration samples."""

        return int(self.observations.shape[0])


@dataclass(frozen=True, slots=True)
class BehavioralCloningDataset:
    """Normalised train/validation/test tensors plus fixed training statistics."""

    train: BCSplit
    validation: BCSplit
    test: BCSplit
    normalizer: StaticZScoreNormalizer
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if self.normalizer.feature_names != OBSERVATION_FEATURE_NAMES:
            raise ValueError("BC normalizer does not match the observation schema")
        if (
            min(
                self.train.sample_count,
                self.validation.sample_count,
                self.test.sample_count,
            )
            < 1
        ):
            raise ValueError("every BC split must contain at least one sample")

    def to_npz(self, path: str | Path) -> None:
        """Atomically write numeric arrays with pickle support unnecessary."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.tmp.npz")
        arrays: dict[str, npt.ArrayLike] = {
            "feature_names": np.asarray(OBSERVATION_FEATURE_NAMES),
            "normalizer_mean": self.normalizer.mean,
            "normalizer_std": self.normalizer.standard_deviation,
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        for name, split in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            arrays[f"{name}_observations"] = split.observations
            arrays[f"{name}_expert_action"] = split.expert_action
            arrays[f"{name}_predictor_target_xz_m"] = split.predictor_target_xz_m
            arrays[f"{name}_predictor_valid_mask"] = split.predictor_valid_mask
        np.savez_compressed(temporary, **arrays)  # type: ignore[arg-type]
        temporary.replace(destination)

    @classmethod
    def from_npz(cls, path: str | Path) -> Self:
        """Load and validate a previously exported BC artefact."""

        with np.load(Path(path), allow_pickle=False) as archive:
            names = tuple(str(name) for name in archive["feature_names"].tolist())
            if names != OBSERVATION_FEATURE_NAMES:
                raise ValueError("BC artefact feature order does not match the code schema")
            normalizer = StaticZScoreNormalizer(
                mean=archive["normalizer_mean"],
                standard_deviation=archive["normalizer_std"],
                feature_names=names,
            )

            def load_split(name: str) -> BCSplit:
                return BCSplit(
                    observations=archive[f"{name}_observations"],
                    expert_action=archive[f"{name}_expert_action"],
                    predictor_target_xz_m=archive[f"{name}_predictor_target_xz_m"],
                    predictor_valid_mask=archive[f"{name}_predictor_valid_mask"],
                )

            metadata_object: object = json.loads(str(archive["metadata_json"].item()))
            if not isinstance(metadata_object, dict):
                raise ValueError("BC metadata must decode to an object")
            metadata_mapping = cast(dict[object, object], metadata_object)
            if not all(isinstance(key, str) for key in metadata_mapping):
                raise ValueError("BC metadata keys must be strings")
            metadata = cast(dict[str, object], metadata_mapping)
            return cls(
                train=load_split("train"),
                validation=load_split("validation"),
                test=load_split("test"),
                normalizer=normalizer,
                metadata=metadata,
            )


def load_bc_normalizer(path: str | Path) -> StaticZScoreNormalizer:
    """Load only the immutable 77-feature statistics from a BC artefact."""

    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"feature_names", "normalizer_mean", "normalizer_std"}
        missing = required - set(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"BC artefact misses normalizer key(s): {names}")
        feature_names = tuple(str(name) for name in archive["feature_names"].tolist())
        if feature_names != OBSERVATION_FEATURE_NAMES:
            raise ValueError("BC artefact feature order does not match the code schema")
        return StaticZScoreNormalizer(
            mean=archive["normalizer_mean"],
            standard_deviation=archive["normalizer_std"],
            feature_names=feature_names,
        )


@dataclass(frozen=True, slots=True)
class BCDatasetBuildReport:
    """Compact result returned by the exporter and CLI."""

    destination: Path
    recordings: int
    complete_laps: int
    sessions: int
    train_samples: int
    validation_samples: int
    test_samples: int
    predictor_valid_train_samples: int


@dataclass(frozen=True, slots=True)
class _RawSessionSamples:
    observations: FloatArray
    expert_action: FloatArray
    predictor_target_xz_m: FloatArray
    predictor_valid_mask: npt.NDArray[np.bool_]
    complete_laps: int


def split_session_ids(
    session_ids: tuple[str, ...],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    sample_counts: Mapping[str, int] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Create deterministic, sample-balanced, session-disjoint splits."""

    unique = tuple(sorted(set(session_ids)))
    if len(unique) < 3:
        raise ValueError("at least three sessions are required for train/validation/test")
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("split fractions must lie in (0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("split fractions must leave at least one training session")
    counts = (
        dict.fromkeys(unique, 1)
        if sample_counts is None
        else {session_id: int(sample_counts[session_id]) for session_id in unique}
    )
    if any(count <= 0 for count in counts.values()):
        raise ValueError("every session sample count must be positive")

    split_fractions = {
        "train": 1.0 - validation_fraction - test_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    split_names = tuple(split_fractions)
    random = np.random.default_rng(seed)
    session_tie_break = {session_id: float(random.random()) for session_id in unique}
    split_tie_break = {name: rank for rank, name in enumerate(random.permutation(split_names))}
    ordered_sessions = sorted(
        unique,
        key=lambda session_id: (
            -counts[session_id],
            session_tie_break[session_id],
            session_id,
        ),
    )
    total_samples = sum(counts.values())
    target_samples = {name: total_samples * fraction for name, fraction in split_fractions.items()}
    assigned: dict[str, list[str]] = {name: [] for name in split_names}
    assigned_samples = dict.fromkeys(split_names, 0)
    for index, session_id in enumerate(ordered_sessions):
        empty_splits = [name for name in split_names if not assigned[name]]
        remaining_sessions = len(ordered_sessions) - index
        candidates = empty_splits if remaining_sessions == len(empty_splits) else list(split_names)
        destination = min(
            candidates,
            key=lambda name: (
                (assigned_samples[name] + counts[session_id]) / target_samples[name],
                split_tie_break[name],
            ),
        )
        assigned[destination].append(session_id)
        assigned_samples[destination] += counts[session_id]
    return {name: tuple(sorted(ids)) for name, ids in assigned.items()}


def build_bc_dataset(
    config: ProjectConfig,
    *,
    output_path: str | Path | None = None,
    maximum_recordings: int | None = None,
) -> tuple[BehavioralCloningDataset, BCDatasetBuildReport]:
    """Build the 77-feature fixed-reference dataset from complete human laps."""

    if maximum_recordings is not None and maximum_recordings <= 0:
        raise ValueError("maximum_recordings must be positive")
    reference = ReferenceTrajectory.from_npz(config.reference.artifact_path)
    if reference.world_position_xy_m is None:
        raise ValueError("fixed reference must contain world_position_xy_m")
    spline = AssettoCorsaAISpline.from_file(configured_fast_lane_path(config))
    if reference.source_track_sha256 != spline.source_sha256:
        raise ValueError("reference was not built from the configured fast_lane.ai")
    geometry = TrackGeometry.from_ai_spline(spline)
    recordings = sorted(config.dataset.recordings_directory.rglob("*.ld"))
    if maximum_recordings is not None:
        recordings = recordings[:maximum_recordings]
    if not recordings:
        raise FileNotFoundError(f"no MoTeC recordings below {config.dataset.recordings_directory}")

    by_session: dict[str, list[_RawSessionSamples]] = {}
    processed_recordings = 0
    for recording in recordings:
        relative = recording.relative_to(config.dataset.recordings_directory)
        session_id = relative.parts[0]
        samples = _read_recording(
            recording,
            config=config,
            reference=reference,
            geometry=geometry,
        )
        if samples.observations.shape[0] == 0:
            continue
        by_session.setdefault(session_id, []).append(samples)
        processed_recordings += 1
    if len(by_session) < 3:
        raise ValueError("fewer than three sessions produced usable BC samples")

    session_samples = {
        session_id: _concatenate_raw(parts) for session_id, parts in by_session.items()
    }
    split_ids = split_session_ids(
        tuple(session_samples),
        seed=config.experiment.seed,
        validation_fraction=config.behavioral_cloning.validation_fraction,
        test_fraction=config.behavioral_cloning.test_fraction,
        sample_counts={
            session_id: samples.observations.shape[0]
            for session_id, samples in session_samples.items()
        },
    )
    raw_splits = {
        name: _concatenate_raw([session_samples[session_id] for session_id in ids])
        for name, ids in split_ids.items()
    }
    unaugmented_train_samples = int(raw_splits["train"].observations.shape[0])
    if config.behavioral_cloning.mirror_augmentation:
        mirrored_observations, mirrored_actions, mirrored_targets = mirror_bc_samples(
            raw_splits["train"].observations,
            raw_splits["train"].expert_action,
            raw_splits["train"].predictor_target_xz_m,
        )
        mirrored_train = _RawSessionSamples(
            observations=mirrored_observations,
            expert_action=mirrored_actions,
            predictor_target_xz_m=mirrored_targets,
            predictor_valid_mask=raw_splits["train"].predictor_valid_mask.copy(),
            complete_laps=0,
        )
        raw_splits["train"] = _concatenate_raw([raw_splits["train"], mirrored_train])
    normalizer = StaticZScoreNormalizer.fit(
        raw_splits["train"].observations,
        feature_names=OBSERVATION_FEATURE_NAMES,
        identity_feature_names=PREVIOUS_ACTION_IDENTITY_FEATURE_NAMES,
    )

    def normalise(raw: _RawSessionSamples) -> BCSplit:
        return BCSplit(
            observations=normalizer.transform_batch(raw.observations),
            expert_action=raw.expert_action,
            predictor_target_xz_m=raw.predictor_target_xz_m,
            predictor_valid_mask=raw.predictor_valid_mask,
        )

    complete_laps = sum(samples.complete_laps for samples in session_samples.values())
    dataset = BehavioralCloningDataset(
        train=normalise(raw_splits["train"]),
        validation=normalise(raw_splits["validation"]),
        test=normalise(raw_splits["test"]),
        normalizer=normalizer,
        metadata={
            "schema_version": 4,
            "method_variant": config.behavioral_cloning.method_variant,
            "action_method_variant": config.actions.method_variant,
            "fast_lane_sha256": spline.source_sha256,
            "calibration": ACTI_MOTEC_CALIBRATION_NAME,
            "slip_preprocessing": LOW_SPEED_SLIP_PREPROCESSING_NAME,
            "previous_action_normalization": PREVIOUS_ACTION_NORMALIZATION_NAME,
            "minimum_speed_mps": config.behavioral_cloning.minimum_speed_mps,
            "training_augmentation": (
                BC_MIRROR_AUGMENTATION_NAME
                if config.behavioral_cloning.mirror_augmentation
                else "none"
            ),
            "unaugmented_train_samples": unaugmented_train_samples,
            "trajectory_codec": BEZIER_TRAJECTORY_CODEC,
            "predictor_coordinate_scale_m": PREDICTOR_COORDINATE_SCALE_M,
            "track_id": config.dataset.track_id,
            "car_id": config.dataset.car_id,
            "control_frequency_hz": config.environment.control_frequency_hz,
            "feature_count": OBSERVATION_SIZE,
            "reference_point_count": int(REFERENCE_TIME_OFFSETS_S.size),
            "predictor_point_count": int(PREDICTOR_TIME_OFFSETS_S.size),
            "predictor_horizon_s": float(PREDICTOR_TIME_OFFSETS_S[-1]),
            "predictor_time_offsets_s": PREDICTOR_TIME_OFFSETS_S.tolist(),
            "split_sessions": {name: list(ids) for name, ids in split_ids.items()},
            "session_sample_counts": {
                session_id: samples.observations.shape[0]
                for session_id, samples in sorted(session_samples.items())
            },
            "source_recordings": processed_recordings,
            "complete_laps": complete_laps,
        },
    )
    destination = Path(output_path or config.behavioral_cloning.dataset_artifact_path)
    dataset.to_npz(destination)
    report = BCDatasetBuildReport(
        destination=destination,
        recordings=processed_recordings,
        complete_laps=complete_laps,
        sessions=len(session_samples),
        train_samples=dataset.train.sample_count,
        validation_samples=dataset.validation.sample_count,
        test_samples=dataset.test.sample_count,
        predictor_valid_train_samples=int(dataset.train.predictor_valid_mask.sum()),
    )
    return dataset, report


def _read_recording(
    path: Path,
    *,
    config: ProjectConfig,
    reference: ReferenceTrajectory,
    geometry: TrackGeometry,
) -> _RawSessionSamples:
    _, channels = read_motec_channels(path, _CHANNEL_NAMES)
    phase = channels["Car Pos Norm"].astype(np.float64)
    wraps = np.flatnonzero(np.diff(phase) < -0.5) + 1
    starts = np.concatenate(([0], wraps[:-1]))
    parts: list[_RawSessionSamples] = []
    for start, end in zip(starts, wraps, strict=True):
        if phase[start] > 0.05 or phase[end - 1] < 0.95:
            continue
        lap = _resample_lap(
            channels,
            start=int(start),
            end=int(end),
            config=config,
            reference=reference,
            geometry=geometry,
        )
        if lap.observations.shape[0]:
            parts.append(lap)
    return _concatenate_raw(parts)


def _resample_lap(
    channels: dict[str, FloatArray],
    *,
    start: int,
    end: int,
    config: ProjectConfig,
    reference: ReferenceTrajectory,
    geometry: TrackGeometry,
) -> _RawSessionSamples:
    raw_time = channels["Lap Time"][start:end].astype(np.float64)
    raw_time -= raw_time[0]
    if raw_time.size < 2 or np.any(np.diff(raw_time) < 0):
        return _empty_raw()
    # MoTeC lap time is quantised more coarsely than some ACTI recordings.
    # Consecutive telemetry rows can therefore carry an identical timestamp.
    # Keep the first row for every timestamp so np.interp receives a strictly
    # increasing time axis without changing the recorded lap order.
    raw_time, unique_indices = np.unique(raw_time, return_index=True)
    raw_indices = start + unique_indices
    if raw_time.size < 2:
        return _empty_raw()
    sample_period = 1.0 / config.environment.control_frequency_hz
    query_time = np.arange(0.0, raw_time[-1], sample_period, dtype=np.float64)
    if query_time.size < math.ceil(float(PREDICTOR_TIME_OFFSETS_S[-1]) / sample_period) + 2:
        return _empty_raw()

    def interpolate(name: str) -> FloatArray:
        return np.asarray(
            np.interp(query_time, raw_time, channels[name][raw_indices]),
            dtype=np.float32,
        )

    speed_mps = interpolate("Ground Speed") / 3.6
    phase = interpolate("Car Pos Norm")
    progress_m = np.asarray(phase * reference.track_length_m, dtype=np.float32)
    position_xy_m = np.column_stack(
        (
            interpolate("Car Coord Y"),
            -interpolate("Car Coord X"),
        )
    ).astype(np.float32)
    heading_rad = _derive_heading(position_xy_m)

    raw_valid = (
        (channels["In Pit"][raw_indices] < 0.5)
        & (channels["Num Tires Off Track"][raw_indices] < 0.5)
        & (channels["Lap Invalidated"][raw_indices] < 0.5)
    )
    valid = np.interp(query_time, raw_time, raw_valid.astype(np.float32)) > 0.999
    valid &= speed_mps >= config.behavioral_cloning.minimum_speed_mps
    valid &= (phase >= 0.0) & (phase <= 1.0)

    front_ratio = 0.005 * (interpolate("Tire Slip Ratio FL") + interpolate("Tire Slip Ratio FR"))
    rear_ratio = 0.005 * (interpolate("Tire Slip Ratio RL") + interpolate("Tire Slip Ratio RR"))
    degree_to_radian_axle_mean = math.pi / 360.0
    front_angle = degree_to_radian_axle_mean * (
        interpolate("Tire Slip Angle FL") + interpolate("Tire Slip Angle FR")
    )
    rear_angle = degree_to_radian_axle_mean * (
        interpolate("Tire Slip Angle RL") + interpolate("Tire Slip Angle RR")
    )
    slip_features = stabilise_low_speed_slip(
        speed_mps,
        np.column_stack(
            (
                front_ratio,
                rear_ratio,
                front_angle,
                rear_angle,
            )
        ),
    )
    vehicle_features = np.column_stack(
        (
            speed_mps,
            interpolate("CG Accel Lateral") * STANDARD_GRAVITY_MPS2,
            interpolate("CG Accel Longitudinal") * STANDARD_GRAVITY_MPS2,
            slip_features,
        )
    ).astype(np.float32)

    left_world, right_world = geometry.preview_world(progress_m)
    left_ego = world_to_ego(left_world, position_xy_m, heading_rad)
    right_ego = world_to_ego(right_world, position_xy_m, heading_rad)

    reference_distance = reference.future_distances(progress_m)
    if reference.world_position_xy_m is None:
        raise ValueError("reference world geometry is unavailable")
    reference_world = np.stack(
        (
            periodic_interpolate(
                reference_distance,
                reference.track_distance_m,
                reference.world_position_xy_m[:, 0],
                reference.track_length_m,
            ),
            periodic_interpolate(
                reference_distance,
                reference.track_distance_m,
                reference.world_position_xy_m[:, 1],
                reference.track_length_m,
            ),
        ),
        axis=-1,
    )
    reference_ego = world_to_ego(reference_world, position_xy_m, heading_rad)

    absolute_action = np.column_stack(
        (
            (interpolate("Throttle Pos") - interpolate("Brake Pos")) / 100.0,
            interpolate("Steering Angle")
            * config.dataset.steering_lock_normalized
            / config.dataset.steering_lock_degrees,
        )
    ).astype(np.float32)
    absolute_action = np.clip(absolute_action, -1.0, 1.0)
    expert_action = np.asarray(absolute_action[1:], dtype=np.float32)

    raw_observation = np.concatenate(
        (
            vehicle_features,
            left_ego.reshape(query_time.size, -1),
            right_ego.reshape(query_time.size, -1),
            reference_ego.reshape(query_time.size, -1),
            absolute_action,
        ),
        axis=1,
    ).astype(np.float32)
    if raw_observation.shape[1] != OBSERVATION_SIZE:
        raise RuntimeError(f"internal BC observation width is {raw_observation.shape[1]}")

    predictor_target, predictor_valid = _future_trajectory_targets(
        position_xy_m,
        heading_rad,
        query_time,
        valid,
    )
    policy_valid = valid[:-1] & valid[1:]
    return _RawSessionSamples(
        observations=raw_observation[:-1][policy_valid],
        expert_action=expert_action[policy_valid],
        predictor_target_xz_m=predictor_target[:-1][policy_valid],
        predictor_valid_mask=predictor_valid[:-1][policy_valid],
        complete_laps=1,
    )


def _future_trajectory_targets(
    positions_xy_m: FloatArray,
    heading_rad: FloatArray,
    sample_time_s: npt.NDArray[np.float64],
    valid_samples: npt.NDArray[np.bool_],
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    query = sample_time_s[:, None] + PREDICTOR_TIME_OFFSETS_S[None, :]
    future_world = np.stack(
        (
            np.interp(
                query.reshape(-1),
                sample_time_s,
                positions_xy_m[:, 0],
                left=np.nan,
                right=np.nan,
            ).reshape(query.shape),
            np.interp(
                query.reshape(-1),
                sample_time_s,
                positions_xy_m[:, 1],
                left=np.nan,
                right=np.nan,
            ).reshape(query.shape),
        ),
        axis=-1,
    )
    predictor_valid = np.all(np.isfinite(future_world), axis=(1, 2))
    window_size = (
        math.ceil(float(PREDICTOR_TIME_OFFSETS_S[-1]) / float(np.median(np.diff(sample_time_s))))
        + 1
    )
    full_valid_window = np.zeros(sample_time_s.size, dtype=np.bool_)
    if valid_samples.size >= window_size:
        counts = np.convolve(
            valid_samples.astype(np.int32),
            np.ones(window_size, dtype=np.int32),
            mode="valid",
        )
        full_valid_window[: counts.size] = counts == window_size
    predictor_valid &= full_valid_window
    future_world[~predictor_valid] = positions_xy_m[~predictor_valid, None, :]
    return (
        world_to_ego(
            np.asarray(future_world, dtype=np.float32),
            positions_xy_m,
            heading_rad,
        ),
        predictor_valid,
    )


def _derive_heading(position_xy_m: FloatArray, half_window: int = 2) -> FloatArray:
    if position_xy_m.ndim != 2 or position_xy_m.shape[1] != 2:
        raise ValueError("position_xy_m must have shape (samples, 2)")
    if position_xy_m.shape[0] <= 2 * half_window:
        raise ValueError("position series is too short for heading estimation")
    before = np.maximum(
        np.arange(position_xy_m.shape[0], dtype=np.intp) - half_window,
        0,
    )
    after = np.minimum(
        np.arange(position_xy_m.shape[0], dtype=np.intp) + half_window,
        position_xy_m.shape[0] - 1,
    )
    difference = position_xy_m[after] - position_xy_m[before]
    return np.asarray(np.arctan2(difference[:, 1], difference[:, 0]), dtype=np.float32)


def _concatenate_raw(parts: list[_RawSessionSamples]) -> _RawSessionSamples:
    non_empty = [part for part in parts if part.observations.shape[0] > 0]
    if not non_empty:
        return _empty_raw()
    return _RawSessionSamples(
        observations=np.concatenate([part.observations for part in non_empty]),
        expert_action=np.concatenate([part.expert_action for part in non_empty]),
        predictor_target_xz_m=np.concatenate([part.predictor_target_xz_m for part in non_empty]),
        predictor_valid_mask=np.concatenate([part.predictor_valid_mask for part in non_empty]),
        complete_laps=sum(part.complete_laps for part in non_empty),
    )


def _empty_raw() -> _RawSessionSamples:
    return _RawSessionSamples(
        observations=np.empty((0, OBSERVATION_SIZE), dtype=np.float32),
        expert_action=np.empty((0, 2), dtype=np.float32),
        predictor_target_xz_m=np.empty(
            (0, PREDICTOR_TIME_OFFSETS_S.size, 2),
            dtype=np.float32,
        ),
        predictor_valid_mask=np.empty(0, dtype=np.bool_),
        complete_laps=0,
    )
