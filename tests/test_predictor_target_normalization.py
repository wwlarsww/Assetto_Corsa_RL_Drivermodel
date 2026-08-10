from __future__ import annotations

import numpy as np

from assetto_corsa_rl.models.predictor_target_normalization import (
    PREDICTOR_TARGET_SCALE_FLOOR_M,
    fit_predictor_target_scale_m,
    predictor_target_normalization_metadata,
    predictor_target_scale_m_from_metadata,
)
from assetto_corsa_rl.models.trajectory_contract import (
    TRAIN_SPLIT_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT,
)


def test_train_split_target_scales_are_per_horizon_axis_and_floored() -> None:
    target = np.zeros((4, 10, 2), dtype=np.float32)
    target[:, :, 0] = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)[:, None]
    target[:, :, 1] = np.asarray([-20.0, 0.0, 20.0, 40.0], dtype=np.float32)[:, None]
    valid = np.asarray([True, True, True, False])

    scale_m = fit_predictor_target_scale_m(
        target,
        valid,
        method_variant=TRAIN_SPLIT_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT,
    )

    np.testing.assert_allclose(scale_m[:, 0], PREDICTOR_TARGET_SCALE_FLOOR_M)
    assert np.all(scale_m[:, 1] > PREDICTOR_TARGET_SCALE_FLOOR_M)
    metadata = {
        "predictor_target_normalization": predictor_target_normalization_metadata(
            method_variant=TRAIN_SPLIT_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT,
            scale_m=scale_m,
        )
    }
    np.testing.assert_allclose(predictor_target_scale_m_from_metadata(metadata), scale_m)
