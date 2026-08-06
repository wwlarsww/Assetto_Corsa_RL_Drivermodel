from __future__ import annotations

import numpy as np
import pytest

from assetto_corsa_rl.features.normalizer import StaticZScoreNormalizer
from assetto_corsa_rl.features.preprocessing import (
    PREVIOUS_ACTION_IDENTITY_FEATURE_NAMES,
    low_speed_slip_weight,
    stabilise_low_speed_slip,
)


def test_low_speed_slip_weight_is_zero_then_linear_then_full() -> None:
    speed_mps = np.asarray([0.0, 1.0, 3.0, 5.0, 10.0], dtype=np.float32)

    np.testing.assert_allclose(
        low_speed_slip_weight(speed_mps),
        [0.0, 0.0, 0.5, 1.0, 1.0],
    )


def test_low_speed_slip_preprocessing_scales_all_four_features_equally() -> None:
    speed_mps = np.asarray([1.0, 3.0, 5.0], dtype=np.float32)
    slip_features = np.asarray(
        [
            [2.0, 4.0, 0.2, 0.4],
            [2.0, 4.0, 0.2, 0.4],
            [2.0, 4.0, 0.2, 0.4],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        stabilise_low_speed_slip(speed_mps, slip_features),
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 0.1, 0.2],
            [2.0, 4.0, 0.2, 0.4],
        ],
    )


def test_normalizer_keeps_previous_actions_on_identity_scale() -> None:
    feature_names = (
        "vehicle.speed_mps",
        *PREVIOUS_ACTION_IDENTITY_FEATURE_NAMES,
    )
    demonstrations = np.asarray(
        [
            [10.0, 0.8, -0.1],
            [20.0, 1.0, 0.0],
            [30.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )

    normalizer = StaticZScoreNormalizer.fit(
        demonstrations,
        feature_names=feature_names,
        identity_feature_names=PREVIOUS_ACTION_IDENTITY_FEATURE_NAMES,
    )

    np.testing.assert_allclose(normalizer.mean[1:], [0.0, 0.0])
    np.testing.assert_allclose(normalizer.standard_deviation[1:], [1.0, 1.0])
    transformed = normalizer.transform(demonstrations[0])
    np.testing.assert_allclose(transformed[1:], demonstrations[0, 1:])


def test_low_speed_slip_preprocessing_rejects_negative_speed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        stabilise_low_speed_slip(
            -0.1,
            np.zeros(4, dtype=np.float32),
        )
