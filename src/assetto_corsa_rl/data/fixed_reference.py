"""Build one deterministic Barcelona/F317 reference from human telemetry."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.data.motec import read_motec_channels
from assetto_corsa_rl.references.promp import ReferenceTrajectory

_SPEED_CHANNELS = (
    "Car Pos Norm",
    "Ground Speed",
    "In Pit",
    "Num Tires Off Track",
    "Lap Invalidated",
)


@dataclass(frozen=True, slots=True)
class FixedReferenceBuildReport:
    """Compact provenance and quality report for a generated reference."""

    recordings: int
    samples_seen: int
    samples_used: int
    distance_bins: int
    empty_bins_before_interpolation: int
    speed_min_mps: float
    speed_median_mps: float
    speed_max_mps: float


def _read_world_line(path: Path) -> npt.NDArray[np.float64]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"pos_x", "pos_y"} <= set(reader.fieldnames):
            raise ValueError("racing-line CSV must contain pos_x and pos_y")
        points = np.asarray(
            [(float(row["pos_x"]), float(row["pos_y"])) for row in reader],
            dtype=np.float64,
        )
    if points.ndim != 2 or points.shape[0] < 3 or not np.all(np.isfinite(points)):
        raise ValueError("racing-line CSV does not contain a finite closed line")
    return points


def _resample_closed_line(
    points: npt.NDArray[np.float64],
    reference_distance_m: npt.NDArray[np.float64],
    track_length_m: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    closed = np.vstack((points, points[0]))
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    geometric_distance = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if geometric_distance[-1] <= 0:
        raise ValueError("racing line has zero length")
    query = reference_distance_m / track_length_m * geometric_distance[-1]
    world_xy = np.column_stack(
        (
            np.interp(query, geometric_distance, closed[:, 0]),
            np.interp(query, geometric_distance, closed[:, 1]),
        )
    )
    tangent = np.roll(world_xy, -1, axis=0) - np.roll(world_xy, 1, axis=0)
    world_heading = np.arctan2(tangent[:, 1], tangent[:, 0])
    return world_xy.astype(np.float32), world_heading.astype(np.float32)


def _periodic_interpolate_missing(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    present = np.isfinite(values)
    if np.count_nonzero(present) < 2:
        raise ValueError("too few populated distance bins for a speed profile")
    indices = np.arange(values.size, dtype=np.float64)
    known = indices[present]
    extended_x = np.concatenate((known - values.size, known, known + values.size))
    extended_y = np.tile(values[present], 3)
    return np.asarray(np.interp(indices, extended_x, extended_y), dtype=np.float64)


def _circular_moving_average(
    values: npt.NDArray[np.float64],
    window: int,
) -> npt.NDArray[np.float64]:
    if window <= 0 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd number")
    if window == 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="wrap")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.asarray(np.convolve(padded, kernel, mode="valid"), dtype=np.float64)


def build_fixed_reference(
    *,
    recordings: list[Path],
    racing_line_csv: Path,
    output_path: Path,
    track_length_m: float,
    distance_bin_m: float = 1.0,
    speed_resolution_mps: float = 0.25,
    maximum_speed_mps: float = 100.0,
    smoothing_window_m: float = 15.0,
    provenance: dict[str, str] | None = None,
) -> FixedReferenceBuildReport:
    """Create a fixed line with a robust human median speed profile.

    Samples in the pits, outside the track, or marked as an invalid lap are
    discarded. A compact per-distance speed histogram is used instead of
    retaining millions of samples in memory.
    """

    if not recordings:
        raise ValueError("no MoTeC recordings were provided")
    if track_length_m <= 0 or distance_bin_m <= 0 or speed_resolution_mps <= 0:
        raise ValueError("track length and histogram resolutions must be positive")
    if maximum_speed_mps <= 0 or smoothing_window_m <= 0:
        raise ValueError("speed and smoothing limits must be positive")

    bin_count = int(np.ceil(track_length_m / distance_bin_m))
    speed_bin_count = int(np.ceil(maximum_speed_mps / speed_resolution_mps))
    histogram = np.zeros((bin_count, speed_bin_count), dtype=np.uint32)
    samples_seen = 0
    samples_used = 0

    for recording in sorted(recordings):
        _, channels = read_motec_channels(recording, _SPEED_CHANNELS)
        phase = channels["Car Pos Norm"].astype(np.float64)
        speed_mps = channels["Ground Speed"].astype(np.float64) / 3.6
        lengths = {array.size for array in channels.values()}
        if len(lengths) != 1:
            raise ValueError(f"channel length mismatch in {recording}")
        samples_seen += phase.size
        valid = (
            np.isfinite(phase)
            & np.isfinite(speed_mps)
            & (phase >= 0.0)
            & (phase < 1.0)
            & (speed_mps >= 2.0)
            & (speed_mps < maximum_speed_mps)
            & (channels["In Pit"] < 0.5)
            & (channels["Num Tires Off Track"] < 0.5)
            & (channels["Lap Invalidated"] < 0.5)
        )
        distance_bins = np.floor(phase[valid] * bin_count).astype(np.int64)
        speed_bins = np.floor(speed_mps[valid] / speed_resolution_mps).astype(np.int64)
        np.add.at(histogram, (distance_bins, speed_bins), 1)
        samples_used += int(np.count_nonzero(valid))

    counts = histogram.sum(axis=1)
    empty_bins = int(np.count_nonzero(counts == 0))
    targets = (counts + 1) // 2
    cumulative = np.cumsum(histogram, axis=1)
    median_bins = np.argmax(cumulative >= targets[:, None], axis=1)
    speed_profile = (median_bins.astype(np.float64) + 0.5) * speed_resolution_mps
    speed_profile[counts == 0] = np.nan
    speed_profile = _periodic_interpolate_missing(speed_profile)
    smoothing_bins = max(1, round(smoothing_window_m / distance_bin_m))
    if smoothing_bins % 2 == 0:
        smoothing_bins += 1
    speed_profile = _circular_moving_average(speed_profile, smoothing_bins)

    distance = np.linspace(0.0, track_length_m, bin_count, endpoint=False, dtype=np.float64)
    world_xy, world_heading = _resample_closed_line(
        _read_world_line(racing_line_csv),
        distance,
        track_length_m,
    )
    reference = ReferenceTrajectory(
        track_distance_m=distance.astype(np.float32),
        lateral_offset_m=np.zeros(bin_count, dtype=np.float32),
        heading_rad=world_heading,
        speed_mps=speed_profile.astype(np.float32),
        track_length_m=track_length_m,
        world_position_xy_m=world_xy,
    )
    report = FixedReferenceBuildReport(
        recordings=len(recordings),
        samples_seen=samples_seen,
        samples_used=samples_used,
        distance_bins=bin_count,
        empty_bins_before_interpolation=empty_bins,
        speed_min_mps=float(np.min(speed_profile)),
        speed_median_mps=float(np.median(speed_profile)),
        speed_max_mps=float(np.max(speed_profile)),
    )
    metadata = {
        "method": "fixed_official_line_with_valid_human_median_speed",
        "racing_line_file": racing_line_csv.name,
        **(provenance or {}),
        "build_report": json.dumps(asdict(report), sort_keys=True),
    }
    reference.to_npz(
        output_path,
        extra_arrays={
            "world_heading_rad": world_heading,
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        },
    )
    return report
