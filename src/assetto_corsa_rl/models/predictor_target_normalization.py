"""Deterministic target scaling for the auxiliary future-trajectory predictor."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.models.trajectory_contract import (
    LEGACY_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT,
    PREDICTOR_COORDINATE_SCALE_M,
    TRAIN_SPLIT_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT,
)
from assetto_corsa_rl.types import PREDICTOR_TIME_OFFSETS_S, FloatArray

PREDICTOR_TARGET_SCALE_FLOOR_M = 5.0
PREDICTOR_TARGET_NORMALIZATION_METADATA_KEY = "predictor_target_normalization"


def fit_predictor_target_scale_m(
    target_xz_m: npt.ArrayLike,
    valid_mask: npt.ArrayLike,
    *,
    method_variant: str,
) -> FloatArray:
    """Fit immutable scales using valid targets from the unaugmented train split."""

    targets = np.asarray(target_xz_m, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    expected_tail = (int(PREDICTOR_TIME_OFFSETS_S.size), 2)
    if targets.ndim != 3 or targets.shape[1:] != expected_tail:
        raise ValueError(f"predictor targets must have shape (samples, {expected_tail[0]}, 2)")
    if valid.shape != (targets.shape[0],):
        raise ValueError("predictor valid mask must have shape (samples,)")
    if not valid.any():
        raise ValueError("cannot fit predictor target scales without valid training targets")
    if not np.all(np.isfinite(targets[valid])):
        raise ValueError("valid predictor targets must be finite")

    if method_variant == LEGACY_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT:
        return np.full(expected_tail, PREDICTOR_COORDINATE_SCALE_M, dtype=np.float32)
    if method_variant == TRAIN_SPLIT_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT:
        standard_deviation = np.std(targets[valid], axis=0, dtype=np.float64)
        scale_m = np.maximum(standard_deviation, PREDICTOR_TARGET_SCALE_FLOOR_M)
        return validate_predictor_target_scale_m(scale_m)
    raise ValueError(f"unsupported predictor target normalization method: {method_variant!r}")


def predictor_target_normalization_metadata(
    *,
    method_variant: str,
    scale_m: npt.ArrayLike,
) -> dict[str, object]:
    """Return the JSON-safe normalization provenance stored with a BC dataset."""

    validated_scale = validate_predictor_target_scale_m(scale_m)
    return {
        "method_variant": method_variant,
        "fit_split": (
            "unaugmented_train"
            if method_variant == TRAIN_SPLIT_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT
            else "none"
        ),
        "scale_floor_m": (
            PREDICTOR_TARGET_SCALE_FLOOR_M
            if method_variant == TRAIN_SPLIT_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT
            else None
        ),
        "scale_m": validated_scale.tolist(),
    }


def predictor_target_normalization_method_from_metadata(
    metadata: Mapping[str, object],
) -> str:
    """Read the named method while treating pre-field datasets as the legacy contract."""

    payload = metadata.get(PREDICTOR_TARGET_NORMALIZATION_METADATA_KEY)
    if payload is None:
        return LEGACY_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT
    if not isinstance(payload, Mapping):
        raise ValueError("predictor target normalization metadata must be an object")
    method_variant = payload.get("method_variant")
    if not isinstance(method_variant, str):
        raise ValueError("predictor target normalization metadata misses method_variant")
    return method_variant


def predictor_target_scale_m_from_metadata(metadata: Mapping[str, object]) -> FloatArray:
    """Resolve and validate the target scales from dataset/checkpoint metadata."""

    method_variant = predictor_target_normalization_method_from_metadata(metadata)
    payload = metadata.get(PREDICTOR_TARGET_NORMALIZATION_METADATA_KEY)
    if payload is None:
        return fit_predictor_target_scale_m(
            np.zeros((1, PREDICTOR_TIME_OFFSETS_S.size, 2), dtype=np.float32),
            np.ones(1, dtype=np.bool_),
            method_variant=method_variant,
        )
    if not isinstance(payload, Mapping):
        raise ValueError("predictor target normalization metadata must be an object")
    if method_variant not in {
        LEGACY_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT,
        TRAIN_SPLIT_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT,
    }:
        raise ValueError(f"unsupported predictor target normalization method: {method_variant!r}")
    try:
        scale_m = payload["scale_m"]
    except KeyError as error:
        raise ValueError("predictor target normalization metadata misses scale_m") from error
    validated = validate_predictor_target_scale_m(scale_m)
    if method_variant == LEGACY_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT and not np.allclose(
        validated, PREDICTOR_COORDINATE_SCALE_M, rtol=0.0, atol=1e-6
    ):
        raise ValueError("legacy predictor target scales must equal the coordinate scale")
    return validated


def load_predictor_target_scale_m(
    dataset_path: str | Path,
    *,
    expected_method_variant: str,
) -> FloatArray:
    """Load only target-scaling metadata, avoiding the large BC tensors during PPO."""

    if expected_method_variant == LEGACY_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT:
        return np.full(
            (PREDICTOR_TIME_OFFSETS_S.size, 2),
            PREDICTOR_COORDINATE_SCALE_M,
            dtype=np.float32,
        )
    with np.load(Path(dataset_path), allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("BC artefact misses metadata_json")
        metadata = json.loads(str(archive["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise ValueError("BC metadata must decode to an object")
    actual_method = predictor_target_normalization_method_from_metadata(metadata)
    if actual_method != expected_method_variant:
        raise ValueError(
            "BC predictor target normalization does not match configuration: "
            f"expected {expected_method_variant!r}, got {actual_method!r}"
        )
    return predictor_target_scale_m_from_metadata(metadata)


def validate_predictor_target_scale_m(scale_m: npt.ArrayLike) -> FloatArray:
    """Validate one per-horizon, per-axis physical scale matrix."""

    scale = np.asarray(scale_m, dtype=np.float32)
    expected_shape = (int(PREDICTOR_TIME_OFFSETS_S.size), 2)
    if scale.shape != expected_shape:
        raise ValueError(f"predictor target scales must have shape {expected_shape}")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("predictor target scales must be finite and positive")
    return scale
