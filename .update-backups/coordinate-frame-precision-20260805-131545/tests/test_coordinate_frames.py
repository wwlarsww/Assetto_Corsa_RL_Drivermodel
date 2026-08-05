"""Coordinate-frame regression tests for Assetto Corsa geometry."""

from __future__ import annotations

import math

import numpy as np

from assetto_corsa_rl.ac_integration.track_assets import (
    assetto_corsa_xyz_to_world_xz,
    local_xz_to_world_xz,
    predictor_xz_to_vehicle_fl,
    vehicle_fl_to_predictor_xz,
    world_xz_to_local_xz,
)


def test_native_assetto_corsa_axes_are_not_swapped() -> None:
    native_xyz = np.asarray(
        [
            [10.0, 2.0, 30.0],
            [-4.0, 1.0, 8.0],
        ]
    )

    world_xz = assetto_corsa_xyz_to_world_xz(native_xyz)

    np.testing.assert_allclose(
        world_xz,
        np.asarray([[10.0, 30.0], [-4.0, 8.0]]),
    )


def test_local_world_roundtrip() -> None:
    vehicle = np.asarray([659.4536, 41.7952])
    heading = -0.091
    local_xz = np.asarray(
        [
            [0.0, 5.0],
            [2.0, 20.0],
            [-3.0, 40.0],
        ]
    )

    world = local_xz_to_world_xz(local_xz, vehicle, heading)
    recovered = world_xz_to_local_xz(world, vehicle, heading)

    np.testing.assert_allclose(recovered, local_xz, atol=1e-5)


def test_model_xz_and_semantic_vehicle_fl_are_inverse() -> None:
    model_xz = np.asarray(
        [
            [2.0, 10.0],
            [-1.5, 25.0],
        ]
    )

    vehicle_fl = predictor_xz_to_vehicle_fl(model_xz)
    recovered = vehicle_fl_to_predictor_xz(vehicle_fl)

    np.testing.assert_allclose(
        vehicle_fl,
        np.asarray([[10.0, 2.0], [25.0, -1.5]]),
    )
    np.testing.assert_allclose(recovered, model_xz)


def test_heading_zero_points_along_positive_world_x() -> None:
    vehicle = np.asarray([100.0, 200.0])
    local_xz = np.asarray([[0.0, 10.0], [5.0, 0.0]])

    world = local_xz_to_world_xz(local_xz, vehicle, heading_rad=0.0)

    np.testing.assert_allclose(
        world,
        np.asarray([[110.0, 200.0], [100.0, 205.0]]),
    )


def test_positive_heading_rotates_forward_counterclockwise() -> None:
    vehicle = np.asarray([0.0, 0.0])
    local_xz = np.asarray([[0.0, 10.0]])

    world = local_xz_to_world_xz(
        local_xz,
        vehicle,
        heading_rad=0.5 * math.pi,
    )

    np.testing.assert_allclose(world, np.asarray([[0.0, 10.0]]), atol=1e-6)
