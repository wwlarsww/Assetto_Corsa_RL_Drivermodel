from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from assetto_corsa_rl.features.normalizer import StaticZScoreNormalizer
from assetto_corsa_rl.features.observation import ObservationBuilder
from assetto_corsa_rl.features.schema import (
    OBSERVATION_FEATURE_NAMES,
    OBSERVATION_FEATURES,
)
from assetto_corsa_rl.types import TelemetryFrame


def test_observation_layout(telemetry_frame: TelemetryFrame) -> None:
    builder = ObservationBuilder()
    observation = builder.build(telemetry_frame, [0.25, -0.5])

    assert observation.shape == (77,)
    np.testing.assert_allclose(observation[:7], [20, 1, 2, 0.1, 0.2, 0.01, 0.02])
    np.testing.assert_allclose(observation[75:], [0.25, -0.5])
    np.testing.assert_allclose(observation[35:39], [0.0, 1.0, 0.0, 2.0])


def test_observation_schema_has_stable_names_units_and_groups() -> None:
    assert len(OBSERVATION_FEATURES) == 77
    assert OBSERVATION_FEATURES[0].name == "vehicle.speed_mps"
    assert OBSERVATION_FEATURES[0].unit == "m/s"
    assert OBSERVATION_FEATURES[35].name == "reference.t_plus_0_25s.x_m"
    assert OBSERVATION_FEATURES[-1].name == "action.previous_steering"
    assert len(set(OBSERVATION_FEATURE_NAMES)) == 77


def test_static_normalizer_round_trip_and_zero_variance() -> None:
    demonstrations = np.asarray([[1.0, 2.0], [3.0, 2.0]], dtype=np.float32)
    normalizer = StaticZScoreNormalizer.fit(demonstrations)
    transformed = normalizer.transform([3.0, 2.0])
    np.testing.assert_allclose(transformed, [1.0, 0.0])
    np.testing.assert_allclose(normalizer.inverse_transform(transformed), [3.0, 2.0])


def test_normalizer_npz_contract(tmp_path: Path) -> None:
    valid = tmp_path / "normalizer.npz"
    np.savez(
        valid,
        mean=np.zeros(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
    )
    assert StaticZScoreNormalizer.from_npz(valid).size == 2

    invalid = tmp_path / "invalid.npz"
    np.savez(invalid, mean=np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="must contain"):
        StaticZScoreNormalizer.from_npz(invalid)


def test_observation_builder_rejects_wrong_normalizer_size() -> None:
    with pytest.raises(ValueError, match="expected 77"):
        ObservationBuilder(StaticZScoreNormalizer.identity(5))
    with pytest.raises(ValueError, match="feature order"):
        ObservationBuilder(StaticZScoreNormalizer.identity(77))


def test_normalizer_rejects_invalid_statistics_and_input() -> None:
    with pytest.raises(ValueError, match="equally shaped"):
        StaticZScoreNormalizer(
            mean=np.zeros(2, dtype=np.float32),
            standard_deviation=np.ones(3, dtype=np.float32),
        )
    with pytest.raises(ValueError, match="negative"):
        StaticZScoreNormalizer(
            mean=np.zeros(2, dtype=np.float32),
            standard_deviation=np.asarray([1.0, -1.0], dtype=np.float32),
        )
    with pytest.raises(ValueError, match="positive"):
        StaticZScoreNormalizer.identity(0)

    normalizer = StaticZScoreNormalizer.identity(2)
    with pytest.raises(ValueError, match="shape"):
        normalizer.transform([1.0])
    with pytest.raises(ValueError, match="shape"):
        normalizer.inverse_transform([1.0])
