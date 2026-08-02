from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import FastLaneFactory

from assetto_corsa_rl.ac_integration.live_backend import AssettoCorsaLiveBackend
from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.envs.factory import create_assetto_corsa_environment
from assetto_corsa_rl.features.schema import OBSERVATION_FEATURE_NAMES, OBSERVATION_SIZE
from assetto_corsa_rl.references.fixed_reference import ReferenceTrajectory


class _Backend:
    control_frequency_hz = 25.0

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_live_factory_loads_fixed_artifacts_before_connecting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_lane_factory: FastLaneFactory,
) -> None:
    installation = tmp_path / "assettocorsa"
    spline = fast_lane_factory(installation, 1000.0, "monza")
    reference_path = tmp_path / "reference.npz"
    ReferenceTrajectory.constant(
        track_length_m=1000.0,
        speed_mps=25.0,
        source_track_sha256=spline.source_sha256,
    ).to_npz(reference_path)
    dataset_path = tmp_path / "bc_dataset.npz"
    np.savez(
        dataset_path,
        feature_names=np.asarray(OBSERVATION_FEATURE_NAMES),
        normalizer_mean=np.arange(OBSERVATION_SIZE, dtype=np.float32),
        normalizer_std=np.full(OBSERVATION_SIZE, 2.0, dtype=np.float32),
    )
    base = ProjectConfig()
    config = replace(
        base,
        dataset=replace(
            base.dataset,
            root_directory=str(tmp_path),
            car_id="bmw_z4_gt3",
            track_id="monza",
        ),
        assetto_corsa=replace(
            base.assetto_corsa,
            installation_directory=str(installation),
        ),
        behavioral_cloning=replace(
            base.behavioral_cloning,
            dataset_artifact_path=str(dataset_path),
        ),
        reference=replace(base.reference, artifact_path=str(reference_path)),
    )
    backend = _Backend()
    connect_arguments: dict[str, Any] = {}

    def connect(**kwargs: Any) -> _Backend:
        connect_arguments.update(kwargs)
        return backend

    monkeypatch.setattr(AssettoCorsaLiveBackend, "connect", connect)
    environment = create_assetto_corsa_environment(config)
    assert environment.observation_builder.normalizer.mean[5] == pytest.approx(5.0)
    assert connect_arguments["expected_car_id"] == "bmw_z4_gt3"
    assert connect_arguments["expected_track_id"] == "monza"
    assert connect_arguments["steering_output_sign"] == -1.0
    assert connect_arguments["reset_collision_quiescence_s"] == pytest.approx(0.5)
    assert connect_arguments["control_writer"] is None
    assert connect_arguments["track_geometry"].track_length_m == pytest.approx(1000.0)
    environment.close()
    assert backend.closed
