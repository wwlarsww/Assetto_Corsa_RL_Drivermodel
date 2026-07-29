from __future__ import annotations

import numpy as np
import pytest

from assetto_corsa_rl.types import (
    ReferenceLinePreview,
    TelemetryFrame,
    TrackPreview,
    VehicleState,
)


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
