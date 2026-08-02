from __future__ import annotations

import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from assetto_corsa_rl.ac_integration.ai_spline import AssettoCorsaAISpline
from assetto_corsa_rl.types import (
    ReferenceLinePreview,
    TelemetryFrame,
    TrackPreview,
    VehicleState,
)

FastLaneFactory = Callable[[Path, float, str], AssettoCorsaAISpline]


@pytest.fixture
def fast_lane_factory() -> FastLaneFactory:
    """Write a square v7 ``fast_lane.ai`` below a synthetic AC installation."""

    def write(
        installation: Path,
        track_length_m: float,
        track_id: str = "monza",
    ) -> AssettoCorsaAISpline:
        side = track_length_m / 4.0
        positions = (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, side),
            (side, 0.0, side),
            (side, 0.0, 0.0),
        )
        payload = bytearray(struct.pack("<4i", 7, len(positions), 0, 0))
        for identifier, position in enumerate(positions):
            payload.extend(struct.pack("<4fi", *position, identifier * side, identifier))
        payload.extend(struct.pack("<i", len(positions)))
        for _ in positions:
            metadata = [0.0] * 18
            metadata[5] = 5.0
            metadata[6] = 5.0
            payload.extend(struct.pack("<18f", *metadata))
        path = installation / "content" / "tracks" / track_id / "ai" / "fast_lane.ai"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return AssettoCorsaAISpline.from_file(path)

    return write


@pytest.fixture
def telemetry_frame() -> TelemetryFrame:
    distances = np.asarray([5, 10, 20, 40, 80, 160, 320], dtype=np.float32)
    left = np.column_stack((np.full(7, 5.0, dtype=np.float32), distances))
    right = np.column_stack((np.full(7, -5.0, dtype=np.float32), distances))
    return TelemetryFrame(
        vehicle_state=VehicleState(20.0, 1.0, 2.0, 0.1, 0.2, 0.01, 0.02),
        track_preview=TrackPreview(left, right),
        reference_line_preview=ReferenceLinePreview(
            points_xz_m=np.column_stack(
                (
                    np.zeros(20, dtype=np.float32),
                    np.arange(1, 21, dtype=np.float32),
                )
            ),
        ),
        progress_m=12.0,
        lateral_offset_m=0.0,
        heading_error_rad=0.0,
        vehicle_heading_rad=0.0,
        position_xy_m=np.asarray([12.0, 0.0], dtype=np.float32),
        simulation_time_s=0.5,
    )
