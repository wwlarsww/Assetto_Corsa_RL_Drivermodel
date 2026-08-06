from __future__ import annotations

import numpy as np

from assetto_corsa_rl.data.augmentation import mirror_bc_samples
from assetto_corsa_rl.features.schema import OBSERVATION_SIZE


def test_mirror_bc_samples_flips_lateral_values_and_swaps_boundaries() -> None:
    observations = np.zeros((1, OBSERVATION_SIZE), dtype=np.float32)
    observations[0, :7] = [
        20.0,
        3.0,
        4.0,
        0.1,
        0.2,
        0.3,
        -0.4,
    ]

    left = np.column_stack(
        (
            np.full(7, 2.0, dtype=np.float32),
            np.arange(7, dtype=np.float32) + 10.0,
        )
    )
    right = np.column_stack(
        (
            np.full(7, -3.0, dtype=np.float32),
            np.arange(7, dtype=np.float32) + 20.0,
        )
    )
    reference = np.column_stack(
        (
            np.linspace(-4.0, 5.0, 20, dtype=np.float32),
            np.linspace(5.0, 100.0, 20, dtype=np.float32),
        )
    )
    observations[0, 7:21] = left.reshape(-1)
    observations[0, 21:35] = right.reshape(-1)
    observations[0, 35:75] = reference.reshape(-1)
    observations[0, 75:] = [0.7, -0.8]

    actions = np.asarray([[0.6, -0.5]], dtype=np.float32)
    target = np.stack(
        (
            np.linspace(-2.0, 3.0, 10, dtype=np.float32),
            np.linspace(4.0, 40.0, 10, dtype=np.float32),
        ),
        axis=-1,
    )[None, ...]

    mirrored_observations, mirrored_actions, mirrored_target = mirror_bc_samples(
        observations, actions, target
    )

    np.testing.assert_allclose(
        mirrored_observations[0, [0, 2, 3, 4, 75]],
        observations[0, [0, 2, 3, 4, 75]],
    )
    np.testing.assert_allclose(
        mirrored_observations[0, [1, 5, 6, 76]],
        -observations[0, [1, 5, 6, 76]],
    )

    mirrored_left = mirrored_observations[0, 7:21].reshape(7, 2)
    mirrored_right = mirrored_observations[0, 21:35].reshape(7, 2)
    np.testing.assert_allclose(mirrored_left[:, 0], -right[:, 0])
    np.testing.assert_allclose(mirrored_left[:, 1], right[:, 1])
    np.testing.assert_allclose(mirrored_right[:, 0], -left[:, 0])
    np.testing.assert_allclose(mirrored_right[:, 1], left[:, 1])

    mirrored_reference = mirrored_observations[0, 35:75].reshape(20, 2)
    np.testing.assert_allclose(
        mirrored_reference[:, 0],
        -reference[:, 0],
    )
    np.testing.assert_allclose(
        mirrored_reference[:, 1],
        reference[:, 1],
    )
    np.testing.assert_allclose(mirrored_actions, [[0.6, 0.5]])
    np.testing.assert_allclose(mirrored_target[..., 0], -target[..., 0])
    np.testing.assert_allclose(mirrored_target[..., 1], target[..., 1])


def test_mirroring_twice_restores_original_samples() -> None:
    random = np.random.default_rng(7)
    observations = random.normal(size=(4, OBSERVATION_SIZE)).astype(np.float32)
    actions = random.uniform(-1.0, 1.0, size=(4, 2)).astype(np.float32)
    targets = random.normal(size=(4, 10, 2)).astype(np.float32)

    once = mirror_bc_samples(observations, actions, targets)
    twice = mirror_bc_samples(*once)

    np.testing.assert_allclose(twice[0], observations)
    np.testing.assert_allclose(twice[1], actions)
    np.testing.assert_allclose(twice[2], targets)
