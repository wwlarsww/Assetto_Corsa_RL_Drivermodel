from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from assetto_corsa_rl.config import (
    BehavioralCloningConfig,
    DatasetConfig,
    EnvironmentConfig,
    ExperimentConfig,
    ModelConfig,
    ProjectConfig,
    ReferenceConfig,
)
from assetto_corsa_rl.data import bc_dataset
from assetto_corsa_rl.data.bc_dataset import (
    BehavioralCloningDataset,
    build_bc_dataset,
    split_session_ids,
)
from assetto_corsa_rl.data.motec import MotecHeader
from assetto_corsa_rl.references.promp import ReferenceTrajectory
from assetto_corsa_rl.training.behavioral_cloning import train_behavioral_cloning


def _fake_motec_channels() -> dict[str, np.ndarray]:
    lap_sample_count = 801
    phase = np.linspace(0.0, 1.0, lap_sample_count, dtype=np.float32)
    lap_time = np.floor(np.linspace(0.0, 8.0, lap_sample_count, dtype=np.float32) * 10.0) / 10.0

    def complete_lap(values: np.ndarray, reset_value: float | None = None) -> np.ndarray:
        reset = float(values[0]) if reset_value is None else reset_value
        return np.append(values, np.float32(reset)).astype(np.float32)

    zeros = np.zeros(lap_sample_count, dtype=np.float32)
    throttle = 50.0 + 5.0 * np.sin(2.0 * np.pi * phase)
    channels = {
        "Ground Speed": complete_lap(np.full_like(phase, 45.0)),
        "CG Accel Lateral": complete_lap(np.full_like(phase, 0.1)),
        "CG Accel Longitudinal": complete_lap(np.full_like(phase, 0.2)),
        "Tire Slip Ratio FL": complete_lap(np.full_like(phase, 10.0)),
        "Tire Slip Ratio FR": complete_lap(np.full_like(phase, 14.0)),
        "Tire Slip Ratio RL": complete_lap(np.full_like(phase, 16.0)),
        "Tire Slip Ratio RR": complete_lap(np.full_like(phase, 20.0)),
        "Tire Slip Angle FL": complete_lap(np.full_like(phase, 2.0)),
        "Tire Slip Angle FR": complete_lap(np.full_like(phase, 4.0)),
        "Tire Slip Angle RL": complete_lap(np.full_like(phase, 6.0)),
        "Tire Slip Angle RR": complete_lap(np.full_like(phase, 8.0)),
        "Car Coord X": complete_lap(zeros, 0.0),
        "Car Coord Y": complete_lap(100.0 * phase, 0.0),
        "Car Pos Norm": complete_lap(phase, 0.0),
        "Lap Time": complete_lap(lap_time, 0.0),
        "Throttle Pos": complete_lap(throttle),
        "Brake Pos": complete_lap(zeros),
        "Steering Angle": complete_lap(10.0 * np.sin(2.0 * np.pi * phase)),
        "In Pit": complete_lap(zeros),
        "Num Tires Off Track": complete_lap(zeros),
        "Lap Invalidated": complete_lap(zeros),
    }
    return {name: np.asarray(values, dtype=np.float32) for name, values in channels.items()}


def _write_track_geometry(path: Path) -> None:
    rows = (
        (0.0, 0.0),
        (25.0, 0.0),
        (25.0, 25.0),
        (0.0, 25.0),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "pos_x",
                "pos_y",
                "left_border_x",
                "left_border_y",
                "right_border_x",
                "right_border_y",
            ),
        )
        writer.writeheader()
        for x_m, y_m in rows:
            writer.writerow(
                {
                    "pos_x": x_m,
                    "pos_y": y_m,
                    "left_border_x": x_m,
                    "left_border_y": y_m + 5.0,
                    "right_border_x": x_m,
                    "right_border_y": y_m - 5.0,
                }
            )


def _project_config(tmp_path: Path) -> ProjectConfig:
    root = tmp_path / "external"
    for session in ("session_a", "session_b", "session_c"):
        directory = root / "data_sets" / "track" / "car" / session
        directory.mkdir(parents=True)
        (directory / "lap.ld").touch()
    _write_track_geometry(root / "track.csv")
    reference_path = tmp_path / "reference.npz"
    ReferenceTrajectory.constant(
        track_length_m=100.0,
        speed_mps=12.5,
        sample_count=400,
    ).to_npz(reference_path)
    return ProjectConfig(
        experiment=ExperimentConfig(
            seed=7,
            max_episode_steps=100,
            output_directory=str(tmp_path / "runs"),
        ),
        environment=EnvironmentConfig(
            control_frequency_hz=25.0,
            track_length_m=100.0,
            initial_speed_mps=20.0,
            minimum_speed_mps=2.0,
        ),
        model=ModelConfig(
            feature_hidden_sizes=(16,),
            policy_hidden_sizes=(8,),
            value_hidden_sizes=(8,),
            predictor_hidden_sizes=(8,),
        ),
        dataset=DatasetConfig(
            track_id="track",
            car_id="car",
            root_directory=str(root),
            racing_line_relative_path="unused.csv",
            track_geometry_relative_path="track.csv",
        ),
        behavioral_cloning=BehavioralCloningConfig(
            dataset_artifact_path=str(tmp_path / "bc_dataset.npz"),
            epochs=1,
            batch_size=64,
            validation_fraction=0.2,
            test_fraction=0.2,
            minimum_speed_mps=1.0,
        ),
        reference=ReferenceConfig(artifact_path=str(reference_path)),
    )


def test_session_split_keeps_the_dominant_session_in_training() -> None:
    split = split_session_ids(
        ("large", "medium", "small_a", "small_b", "small_c"),
        seed=7,
        validation_fraction=0.15,
        test_fraction=0.15,
        sample_counts={
            "large": 70_000,
            "medium": 20_000,
            "small_a": 10_000,
            "small_b": 10_000,
            "small_c": 10_000,
        },
    )
    assert "large" in split["train"]
    assert all(split.values())
    assert set().union(*map(set, split.values())) == {
        "large",
        "medium",
        "small_a",
        "small_b",
        "small_c",
    }


def test_fixed_reference_bc_export_and_joint_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channels = _fake_motec_channels()
    header = MotecHeader("driver", "car", "track", "date", "time", len(channels))

    def fake_read(
        _path: Path,
        names: tuple[str, ...],
    ) -> tuple[MotecHeader, dict[str, np.ndarray]]:
        return header, {name: channels[name] for name in names}

    monkeypatch.setattr(bc_dataset, "read_motec_channels", fake_read)
    config = _project_config(tmp_path)
    dataset, report = build_bc_dataset(config)

    assert report.recordings == 3
    assert report.complete_laps == 3
    assert report.sessions == 3
    assert (
        min(
            dataset.train.sample_count,
            dataset.validation.sample_count,
            dataset.test.sample_count,
        )
        > 0
    )
    assert dataset.train.observations.shape[1] == 77
    assert dataset.train.predictor_valid_mask.any()
    assert dataset.metadata["method_variant"] == "fixed_reference_no_promp_v1"
    assert dataset.metadata["reference_point_count"] == 20

    raw_observation = dataset.normalizer.inverse_transform(dataset.train.observations[0])
    assert raw_observation[0] == pytest.approx(12.5)
    assert raw_observation[1] == pytest.approx(0.1 * 9.80665)
    assert raw_observation[2] == pytest.approx(0.2 * 9.80665)
    assert raw_observation[3] == pytest.approx(0.12)
    reference_points_xz = raw_observation[35:75].reshape(20, 2)
    assert np.all(reference_points_xz[:, 1] > 0.0)

    loaded = BehavioralCloningDataset.from_npz(report.destination)
    np.testing.assert_allclose(loaded.train.observations, dataset.train.observations)
    assert loaded.normalizer.feature_names == dataset.normalizer.feature_names

    result = train_behavioral_cloning(
        config,
        loaded,
        command=["test", "bc-train"],
    )
    assert result.best_epoch == 1
    assert result.checkpoint_path.is_file()
    assert (result.run_directory / "metrics.csv").is_file()
    assert (result.run_directory / "summary.json").is_file()
    assert np.isfinite(result.test.action_mse)
    assert np.isfinite(result.test.predictor_ade_m)
