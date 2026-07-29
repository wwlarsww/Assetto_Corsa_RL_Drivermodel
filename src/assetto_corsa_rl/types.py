"""Domain data exchanged across simulator-independent module boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]

TRACK_PREVIEW_DISTANCES_M = np.asarray(
    [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0],
    dtype=np.float32,
)
REFERENCE_TIME_OFFSETS_S = np.arange(0.25, 5.01, 0.25, dtype=np.float32)


def _as_float_array(value: npt.ArrayLike, expected_shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True, slots=True)
class VehicleState:
    """Instantaneous dynamic quantities exposed to the agent."""

    speed_mps: float
    acceleration_x_mps2: float
    acceleration_z_mps2: float
    front_slip_ratio: float
    rear_slip_ratio: float
    front_sideslip_angle_rad: float
    rear_sideslip_angle_rad: float

    def as_array(self) -> FloatArray:
        """Return the thesis-defined seven scalar vehicle features."""

        values = np.asarray(
            [
                self.speed_mps,
                self.acceleration_x_mps2,
                self.acceleration_z_mps2,
                self.front_slip_ratio,
                self.rear_slip_ratio,
                self.front_sideslip_angle_rad,
                self.rear_sideslip_angle_rad,
            ],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("vehicle state contains non-finite values")
        return values


@dataclass(frozen=True, slots=True)
class TrackPreview:
    """Ego-centric left and right boundary coordinates at seven distances."""

    left_boundary_xz_m: FloatArray
    right_boundary_xz_m: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "left_boundary_xz_m",
            _as_float_array(self.left_boundary_xz_m, (7, 2), "left_boundary_xz_m"),
        )
        object.__setattr__(
            self,
            "right_boundary_xz_m",
            _as_float_array(self.right_boundary_xz_m, (7, 2), "right_boundary_xz_m"),
        )


@dataclass(frozen=True, slots=True)
class ReferenceFeaturePreview:
    """Legacy contextual reference features retained for the ProMP variant."""

    lateral_offset_m: FloatArray
    heading_rad: FloatArray
    speed_mps: FloatArray

    def __post_init__(self) -> None:
        expected = (20,)
        object.__setattr__(
            self,
            "lateral_offset_m",
            _as_float_array(self.lateral_offset_m, expected, "lateral_offset_m"),
        )
        object.__setattr__(
            self,
            "heading_rad",
            _as_float_array(self.heading_rad, expected, "heading_rad"),
        )
        object.__setattr__(
            self,
            "speed_mps",
            _as_float_array(self.speed_mps, expected, "speed_mps"),
        )

    def as_matrix(self) -> FloatArray:
        """Return the contextual ProMP triple representation."""

        return np.column_stack((self.lateral_offset_m, self.heading_rad, self.speed_mps)).astype(
            np.float32
        )


@dataclass(frozen=True, slots=True)
class ReferenceLinePreview:
    """Twenty prescribed reference-line points in the current vehicle frame."""

    points_xz_m: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "points_xz_m",
            _as_float_array(self.points_xz_m, (20, 2), "points_xz_m"),
        )


@dataclass(frozen=True, slots=True)
class TelemetryFrame:
    """One complete, already synchronised telemetry frame.

    ``progress_m`` is the vehicle position projected onto the active reference
    line and expressed as its closed-loop arc-length coordinate.
    """

    vehicle_state: VehicleState
    track_preview: TrackPreview
    reference_line_preview: ReferenceLinePreview
    progress_m: float
    lateral_offset_m: float
    heading_error_rad: float
    vehicle_heading_rad: float
    position_xy_m: FloatArray
    simulation_time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_xy_m",
            _as_float_array(self.position_xy_m, (2,), "position_xy_m"),
        )
        scalars = np.asarray(
            [
                self.progress_m,
                self.lateral_offset_m,
                self.heading_error_rad,
                self.vehicle_heading_rad,
                self.simulation_time_s,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("telemetry frame contains non-finite values")

    def metrics(self) -> dict[str, float]:
        """Expose standard scalar metrics for logging."""

        return {
            "simulation_time_s": self.simulation_time_s,
            "progress_m": self.progress_m,
            "speed_mps": self.vehicle_state.speed_mps,
            "lateral_offset_m": self.lateral_offset_m,
            "heading_error_rad": self.heading_error_rad,
            "vehicle_heading_rad": self.vehicle_heading_rad,
            "front_slip_ratio": self.vehicle_state.front_slip_ratio,
            "rear_slip_ratio": self.vehicle_state.rear_slip_ratio,
        }


class TerminationReason(StrEnum):
    """Task terminal states distinguished from external time limits."""

    LAP_COMPLETE = "lap_complete"
    OFF_TRACK = "off_track"
    STALLED = "stalled"
    EXCESSIVE_SLIP = "excessive_slip"

    @property
    def is_safety_violation(self) -> bool:
        return self is not TerminationReason.LAP_COMPLETE


@dataclass(frozen=True, slots=True)
class BackendTransition:
    """Result of advancing a simulator backend by one control step."""

    telemetry: TelemetryFrame
    termination_reason: TerminationReason | None = None
    diagnostics: dict[str, Any] | None = None
