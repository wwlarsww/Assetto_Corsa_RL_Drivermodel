"""Deterministic assembly of the four thesis observation groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.features.normalizer import StaticZScoreNormalizer
from assetto_corsa_rl.features.schema import (
    OBSERVATION_FEATURE_NAMES,
    OBSERVATION_FEATURES,
    OBSERVATION_SIZE,
    FeatureSpec,
)
from assetto_corsa_rl.types import FloatArray, TelemetryFrame


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    """Stable slice layout of the 77-dimensional fixed-reference observation."""

    vehicle_state: slice = slice(0, 7)
    left_track_boundary: slice = slice(7, 21)
    right_track_boundary: slice = slice(21, 35)
    reference_line: slice = slice(35, 75)
    previous_action: slice = slice(75, 77)
    size: int = OBSERVATION_SIZE


class ObservationBuilder:
    """Concatenate telemetry features and apply fixed demonstration statistics."""

    layout = ObservationLayout()
    feature_schema: tuple[FeatureSpec, ...] = OBSERVATION_FEATURES

    def __init__(self, normalizer: StaticZScoreNormalizer | None = None) -> None:
        self.normalizer = normalizer or StaticZScoreNormalizer.identity(
            self.layout.size,
            feature_names=OBSERVATION_FEATURE_NAMES,
        )
        if self.normalizer.size != self.layout.size:
            raise ValueError(
                f"normalizer has {self.normalizer.size} features; expected {self.layout.size}"
            )
        if self.normalizer.feature_names != OBSERVATION_FEATURE_NAMES:
            raise ValueError(
                "normalizer feature order does not match the observation schema; "
                "refit or load schema-bound statistics"
            )

    def build_raw(
        self,
        telemetry: TelemetryFrame,
        previous_action: npt.ArrayLike,
    ) -> FloatArray:
        """Build a raw vector in the documented, immutable feature order."""

        action = np.asarray(previous_action, dtype=np.float32)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("previous_action must be a finite vector with shape (2,)")

        observation = np.concatenate(
            (
                telemetry.vehicle_state.as_array(),
                telemetry.track_preview.left_boundary_xz_m.reshape(-1),
                telemetry.track_preview.right_boundary_xz_m.reshape(-1),
                telemetry.reference_line_preview.points_xz_m.reshape(-1),
                action,
            )
        ).astype(np.float32)
        if observation.shape != (self.layout.size,):
            raise RuntimeError(f"internal observation layout error: {observation.shape}")
        return observation

    def build(
        self,
        telemetry: TelemetryFrame,
        previous_action: npt.ArrayLike,
    ) -> FloatArray:
        """Build and normalise one observation."""

        return self.normalizer.transform(self.build_raw(telemetry, previous_action))
