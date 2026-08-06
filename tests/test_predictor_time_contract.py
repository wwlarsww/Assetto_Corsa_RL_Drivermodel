from __future__ import annotations

import numpy as np

from assetto_corsa_rl.training.targets import build_observed_trajectory_targets
from assetto_corsa_rl.types import (
    PREDICTOR_TIME_OFFSETS_S,
    REFERENCE_TIME_OFFSETS_S,
)


def test_predictor_and_reference_time_axes_are_independent() -> None:
    np.testing.assert_allclose(
        PREDICTOR_TIME_OFFSETS_S,
        np.arange(1, 11, dtype=np.float32) * 0.1,
        rtol=0.0,
        atol=1e-7,
    )
    assert PREDICTOR_TIME_OFFSETS_S.shape == (10,)
    assert PREDICTOR_TIME_OFFSETS_S[-1] == np.float32(1.0)

    assert REFERENCE_TIME_OFFSETS_S.shape == (20,)
    assert REFERENCE_TIME_OFFSETS_S[-1] == np.float32(5.0)


def test_constant_speed_targets_are_ten_equal_time_waypoints() -> None:
    simulation_time_s = np.arange(0.0, 2.01, 0.05, dtype=np.float64)
    positions_xy_m = np.column_stack(
        (
            10.0 * simulation_time_s,
            np.zeros_like(simulation_time_s),
        )
    ).astype(np.float32)
    heading_rad = np.zeros(simulation_time_s.size, dtype=np.float32)
    trajectory_id = np.zeros(simulation_time_s.size, dtype=np.int64)

    targets = build_observed_trajectory_targets(
        positions_xy_m=positions_xy_m,
        heading_rad=heading_rad,
        simulation_time_s=simulation_time_s,
        trajectory_id=trajectory_id,
    )

    assert targets.trajectory_xz_m.shape == (simulation_time_s.size, 10, 2)
    assert targets.valid_mask[0]
    np.testing.assert_allclose(
        targets.trajectory_xz_m[0, :, 0],
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        targets.trajectory_xz_m[0, :, 1],
        np.arange(1, 11, dtype=np.float32),
        atol=1e-5,
    )
    assert not targets.valid_mask[-1]
