"""Behavioural-cloning augmentation in the local vehicle coordinate frame."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.features.schema import OBSERVATION_SIZE
from assetto_corsa_rl.types import (
    PREDICTOR_TIME_OFFSETS_S,
    REFERENCE_TIME_OFFSETS_S,
    TRACK_PREVIEW_DISTANCES_M,
    FloatArray,
)

BC_MIRROR_AUGMENTATION_NAME = "ego_left_right_mirror_v1"

_VEHICLE_FEATURE_COUNT = 7
_BOUNDARY_POINT_COUNT = int(TRACK_PREVIEW_DISTANCES_M.size)
_BOUNDARY_FEATURE_COUNT = 2 * _BOUNDARY_POINT_COUNT
_REFERENCE_POINT_COUNT = int(REFERENCE_TIME_OFFSETS_S.size)
_REFERENCE_FEATURE_COUNT = 2 * _REFERENCE_POINT_COUNT
_PREDICTOR_POINT_COUNT = int(PREDICTOR_TIME_OFFSETS_S.size)

_LEFT_START = _VEHICLE_FEATURE_COUNT
_LEFT_END = _LEFT_START + _BOUNDARY_FEATURE_COUNT
_RIGHT_START = _LEFT_END
_RIGHT_END = _RIGHT_START + _BOUNDARY_FEATURE_COUNT
_REFERENCE_START = _RIGHT_END
_REFERENCE_END = _REFERENCE_START + _REFERENCE_FEATURE_COUNT
_PREVIOUS_STEERING_INDEX = OBSERVATION_SIZE - 1


def mirror_bc_samples(
    observations: npt.ArrayLike,
    expert_action: npt.ArrayLike,
    predictor_target_xz_m: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Mirror BC samples at the vehicle longitudinal axis.

    Local ``x`` is positive to the left and local ``z`` is positive forward.
    Left and right boundaries therefore exchange semantic roles while all
    lateral coordinates, steering values, lateral acceleration and sideslip
    angles change sign.
    """

    source_observations = np.asarray(observations, dtype=np.float32)
    source_actions = np.asarray(expert_action, dtype=np.float32)
    source_targets = np.asarray(predictor_target_xz_m, dtype=np.float32)

    if source_observations.ndim != 2:
        raise ValueError("observations must have shape (samples, features)")
    sample_count = source_observations.shape[0]
    if source_observations.shape != (sample_count, OBSERVATION_SIZE):
        raise ValueError(f"observations must have shape (samples, {OBSERVATION_SIZE})")
    if source_actions.shape != (sample_count, 2):
        raise ValueError("expert_action must have shape (samples, 2)")
    if source_targets.shape != (
        sample_count,
        _PREDICTOR_POINT_COUNT,
        2,
    ):
        raise ValueError(
            f"predictor_target_xz_m must have shape (samples, {_PREDICTOR_POINT_COUNT}, 2)"
        )
    if not all(
        np.all(np.isfinite(array))
        for array in (
            source_observations,
            source_actions,
            source_targets,
        )
    ):
        raise ValueError("BC mirror inputs must contain only finite values")

    mirrored_observations = source_observations.copy()
    mirrored_actions = source_actions.copy()
    mirrored_targets = source_targets.copy()

    # Vehicle state: lateral acceleration and both signed sideslip angles.
    mirrored_observations[:, 1] *= -1.0
    mirrored_observations[:, 5:7] *= -1.0

    original_left = source_observations[:, _LEFT_START:_LEFT_END].reshape(
        sample_count,
        _BOUNDARY_POINT_COUNT,
        2,
    )
    original_right = source_observations[:, _RIGHT_START:_RIGHT_END].reshape(
        sample_count,
        _BOUNDARY_POINT_COUNT,
        2,
    )

    mirrored_left = original_right.copy()
    mirrored_left[..., 0] *= -1.0
    mirrored_right = original_left.copy()
    mirrored_right[..., 0] *= -1.0
    mirrored_observations[:, _LEFT_START:_LEFT_END] = mirrored_left.reshape(
        sample_count,
        -1,
    )
    mirrored_observations[:, _RIGHT_START:_RIGHT_END] = mirrored_right.reshape(
        sample_count,
        -1,
    )

    mirrored_reference = (
        source_observations[
            :,
            _REFERENCE_START:_REFERENCE_END,
        ]
        .reshape(sample_count, _REFERENCE_POINT_COUNT, 2)
        .copy()
    )
    mirrored_reference[..., 0] *= -1.0
    mirrored_observations[:, _REFERENCE_START:_REFERENCE_END] = mirrored_reference.reshape(
        sample_count, -1
    )

    mirrored_observations[:, _PREVIOUS_STEERING_INDEX] *= -1.0
    mirrored_actions[:, 1] *= -1.0
    mirrored_targets[..., 0] *= -1.0

    return mirrored_observations, mirrored_actions, mirrored_targets
