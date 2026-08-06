"""Static Z-score normalisation fitted to demonstration data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.types import FloatArray


@dataclass(frozen=True, slots=True)
class StaticZScoreNormalizer:
    """Immutable normaliser whose statistics never change during RL."""

    mean: FloatArray
    standard_deviation: FloatArray
    feature_names: tuple[str, ...] | None = None
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.standard_deviation, dtype=np.float32)
        if mean.ndim != 1 or mean.shape != std.shape:
            raise ValueError("normalisation arrays must be one-dimensional and equally shaped")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("normalisation arrays must be finite")
        if np.any(std < 0):
            raise ValueError("standard deviations may not be negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        names = self.feature_names
        if names is not None:
            names = tuple(names)
            if len(names) != mean.size:
                raise ValueError("feature_names must match the normalisation array size")
            if any(not name for name in names) or len(set(names)) != len(names):
                raise ValueError("feature_names must be non-empty and unique")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", std)
        object.__setattr__(self, "feature_names", names)

    @property
    def size(self) -> int:
        """Return the number of schema-bound features."""

        return int(self.mean.size)

    @classmethod
    def identity(
        cls,
        size: int,
        *,
        feature_names: tuple[str, ...] | None = None,
    ) -> Self:
        """Create identity statistics for mock-only development."""

        if size <= 0:
            raise ValueError("size must be positive")
        return cls(
            mean=np.zeros(size, dtype=np.float32),
            standard_deviation=np.ones(size, dtype=np.float32),
            feature_names=feature_names,
        )

    @classmethod
    def fit(
        cls,
        demonstrations: npt.ArrayLike,
        *,
        feature_names: tuple[str, ...] | None = None,
        identity_feature_names: tuple[str, ...] = (),
    ) -> Self:
        """Fit once to demonstrations with optional identity-scaled features."""

        values = np.asarray(demonstrations, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError("demonstrations must have shape (samples, features)")
        if not np.all(np.isfinite(values)):
            raise ValueError("demonstrations contain non-finite values")

        names = None if feature_names is None else tuple(feature_names)
        identity_names = tuple(identity_feature_names)
        if identity_names and names is None:
            raise ValueError("identity_feature_names require feature_names")
        if len(set(identity_names)) != len(identity_names):
            raise ValueError("identity_feature_names must be unique")
        if names is not None:
            unknown = set(identity_names) - set(names)
            if unknown:
                raise ValueError(
                    "identity feature names are absent from feature_names: "
                    + ", ".join(sorted(unknown))
                )

        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        standard_deviation = values.std(axis=0, dtype=np.float64).astype(np.float32)
        if names is not None:
            for feature_name in identity_names:
                feature_index = names.index(feature_name)
                mean[feature_index] = 0.0
                standard_deviation[feature_index] = 1.0

        return cls(
            mean=mean,
            standard_deviation=standard_deviation,
            feature_names=names,
        )

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        expected_feature_names: tuple[str, ...] | None = None,
    ) -> Self:
        """Load safe statistics and optionally enforce their feature order."""

        with np.load(Path(path), allow_pickle=False) as data:
            keys = set(data.files)
            if not {"mean", "std"} <= keys or keys - {"mean", "std", "feature_names"}:
                raise ValueError(
                    "normalisation artefact must contain 'mean' and 'std', "
                    "with optional 'feature_names'"
                )
            names = (
                tuple(str(name) for name in data["feature_names"].tolist())
                if "feature_names" in keys
                else None
            )
            if expected_feature_names is not None and names != expected_feature_names:
                raise ValueError("normalisation artefact does not match the expected feature order")
            return cls(
                mean=data["mean"],
                standard_deviation=data["std"],
                feature_names=names,
            )

    def transform(self, observation: npt.ArrayLike) -> FloatArray:
        """Apply the fixed Z-score transform."""

        values = np.asarray(observation, dtype=np.float32)
        if values.shape != self.mean.shape:
            raise ValueError(f"observation must have shape {self.mean.shape}, got {values.shape}")
        scale = np.maximum(self.standard_deviation, self.epsilon)
        return np.asarray((values - self.mean) / scale, dtype=np.float32)

    def transform_batch(self, observations: npt.ArrayLike) -> FloatArray:
        """Apply the same immutable statistics to a sample matrix."""

        values = np.asarray(observations, dtype=np.float32)
        if values.ndim != 2 or values.shape[1:] != self.mean.shape:
            raise ValueError(
                f"observations must have shape (samples, {self.size}), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("observations contain non-finite values")
        scale = np.maximum(self.standard_deviation, self.epsilon)
        return np.asarray((values - self.mean[None, :]) / scale[None, :], dtype=np.float32)

    def inverse_transform(self, normalised: npt.ArrayLike) -> FloatArray:
        """Recover the unnormalised values for diagnostics."""

        values = np.asarray(normalised, dtype=np.float32)
        if values.shape != self.mean.shape:
            raise ValueError(f"normalised input must have shape {self.mean.shape}")
        scale = np.maximum(self.standard_deviation, self.epsilon)
        return np.asarray(values * scale + self.mean, dtype=np.float32)
