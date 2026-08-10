from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import FastLaneFactory

from assetto_corsa_rl.ac_integration.track_assets import sha256_file
from assetto_corsa_rl.config import (
    AssettoCorsaRuntimeConfig,
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
    ACTI_MOTEC_CALIBRATION_NAME,
    MOTEC_STEERING_CONVERSION_NAME,
    BCDatasetBuildReport,
    BehavioralCloningDataset,
    build_bc_dataset,
    split_session_ids,
)
from assetto_corsa_rl.data.motec import MotecHeader
from assetto_corsa_rl.features.schema import OBSERVATION_FEATURES, OBSERVATION_SIZE
from assetto_corsa_rl.models.trajectory_contract import (
    TRAIN_SPLIT_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT,
)
from assetto_corsa_rl.references.fixed_reference import ReferenceTrajectory
from assetto_corsa_rl.training.behavioral_cloning import (
    BCTrainingResult,
    _permute_causal_feedback_features,
    train_behavioral_cloning,
)


def test_causal_feedback_permutation_preserves_route_context() -> None:
    observation = torch.arange(3 * OBSERVATION_SIZE, dtype=torch.float32).reshape(
        3,
        OBSERVATION_SIZE,
    )
    donor = torch.tensor([2, 0, 1], dtype=torch.long)

    augmented = _permute_causal_feedback_features(observation, donor)

    feedback_indices = {
        feature.index
        for feature in OBSERVATION_FEATURES
        if (feature.group == "vehicle_state" and feature.name != "vehicle.speed_mps")
        or feature.group == "previous_action"
    }
    for feature in OBSERVATION_FEATURES:
        expected = (
            observation[donor, feature.index]
            if feature.index in feedback_indices
            else observation[:, feature.index]
        )
        torch.testing.assert_close(augmented[:, feature.index], expected)


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


def _project_config(tmp_path: Path, fast_lane_factory: FastLaneFactory) -> ProjectConfig:
    root = tmp_path / "external"
    for session in ("session_a", "session_b", "session_c"):
        directory = root / "data_sets" / "track" / "car" / session
        directory.mkdir(parents=True)
        (directory / "lap.ld").touch()
    installation = tmp_path / "assettocorsa"
    spline = fast_lane_factory(installation, 100.0, "track")
    world_xy = spline.centre_xy_m
    tangent = np.roll(world_xy, -1, axis=0) - np.roll(world_xy, 1, axis=0)
    reference_path = tmp_path / "reference.npz"
    ReferenceTrajectory(
        track_distance_m=spline.distance_m,
        lateral_offset_m=np.zeros_like(spline.distance_m),
        heading_rad=np.arctan2(tangent[:, 1], tangent[:, 0]).astype(np.float32),
        speed_mps=np.full_like(spline.distance_m, 12.5),
        track_length_m=100.0,
        world_position_xy_m=world_xy,
        source_track_sha256=spline.source_sha256,
    ).to_npz(reference_path)
    return ProjectConfig(
        experiment=ExperimentConfig(
            seed=7,
            max_episode_steps=100,
            output_directory=str(tmp_path / "runs"),
        ),
        environment=EnvironmentConfig(
            control_frequency_hz=25.0,
            mock_track_length_m=100.0,
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
        ),
        assetto_corsa=AssettoCorsaRuntimeConfig(
            installation_directory=str(installation),
        ),
        behavioral_cloning=BehavioralCloningConfig(
            dataset_artifact_path=str(tmp_path / "bc_dataset.npz"),
            epochs=1,
            batch_size=64,
            predictor_loss_weight=0.01,
            predictor_target_normalization_method_variant=(
                TRAIN_SPLIT_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT
            ),
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


def _assert_built_bc_dataset(
    config: ProjectConfig,
    dataset: BehavioralCloningDataset,
    report: BCDatasetBuildReport,
) -> None:
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
    assert dataset.metadata["schema_version"] == 6
    assert dataset.metadata["method_variant"] == "ai_spline_absolute_v1"
    assert dataset.metadata["action_method_variant"] == "absolute_bounded_v1"
    assert dataset.metadata["calibration"] == ACTI_MOTEC_CALIBRATION_NAME
    assert dataset.metadata["steering_conversion"] == MOTEC_STEERING_CONVERSION_NAME
    assert dataset.metadata["steering_convention"] == "policy_positive_left_v1"
    assert dataset.metadata["reference_sha256"] == sha256_file(Path(config.reference.artifact_path))
    assert dataset.metadata["reference_feature_contract"] == (
        "sha256_bound_time_parameterized_fixed_reference_v1"
    )
    assert dataset.metadata["reference_point_count"] == 20
    assert dataset.metadata["predictor_point_count"] == 10
    assert dataset.metadata["predictor_horizon_s"] == pytest.approx(1.0)
    predictor_time_offsets_s = np.asarray(
        dataset.metadata["predictor_time_offsets_s"], dtype=np.float32
    )
    np.testing.assert_allclose(predictor_time_offsets_s, np.arange(1, 11, dtype=np.float32) * 0.1)
    assert dataset.train.predictor_target_xz_m.shape[1:] == (10, 2)
    assert dataset.metadata["training_augmentation"] == ("ego_left_right_mirror_v1")
    unaugmented_train_samples = dataset.metadata["unaugmented_train_samples"]
    assert isinstance(unaugmented_train_samples, int)
    assert dataset.train.sample_count == (2 * unaugmented_train_samples)
    assert dataset.metadata["trajectory_codec"] == (
        "bezier-origin-5x2-bounded-monotonic-10x0p1s-v2"
    )
    assert dataset.metadata["predictor_coordinate_scale_m"] == pytest.approx(100.0)
    normalization = dataset.metadata["predictor_target_normalization"]
    assert isinstance(normalization, dict)
    assert normalization["method_variant"] == (
        TRAIN_SPLIT_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT
    )
    scale_m = np.asarray(normalization["scale_m"], dtype=np.float32)
    assert scale_m.shape == (10, 2)
    assert np.all(scale_m >= 5.0)
    np.testing.assert_allclose(dataset.predictor_target_scale_m, scale_m)
    assert dataset.metadata["predictor_lateral_limit_m"] == pytest.approx(20.0)
    assert dataset.metadata["predictor_forward_increment_max_m"] == pytest.approx(20.0)
    assert dataset.metadata["predictor_max_lateral_to_forward_ratio"] == pytest.approx(0.75)

    raw_observation = dataset.normalizer.inverse_transform(dataset.train.observations[0])
    assert raw_observation[0] == pytest.approx(12.5)
    assert raw_observation[1] == pytest.approx(0.1 * 9.80665)
    assert raw_observation[2] == pytest.approx(0.2 * 9.80665)
    assert raw_observation[3] == pytest.approx(0.12)
    reference_points_xz = raw_observation[35:75].reshape(20, 2)
    assert np.all(np.isfinite(reference_points_xz))
    assert np.linalg.norm(reference_points_xz[-1]) > 0.0
    assert 0.4 < float(dataset.train.expert_action[0, 0]) < 0.6
    # The synthetic raw MoTeC steering is initially positive-right. Dataset
    # targets must convert it to the policy's positive-left convention.
    assert dataset.validation.expert_action[0, 1] < 0.0


def _assert_epoch_metrics_csv(
    metrics_path: Path,
    config: ProjectConfig,
) -> None:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    nonnegative_fields = (
        "train_action_mse",
        "train_action_combined_pedal_mse",
        "train_action_steering_mse",
        "train_predictor_smooth_l1",
        "train_predictor_ade_m",
        "train_predictor_rmse_m",
        "train_predictor_lateral_mae_m",
        "train_predictor_longitudinal_mae_m",
        "train_predictor_error_at_0p1s_m",
        "train_predictor_error_at_0p5s_m",
        "train_predictor_error_at_1p0s_m",
        "train_joint_loss",
        "validation_action_mse",
        "validation_predictor_smooth_l1",
        "validation_predictor_ade_m",
        "validation_predictor_rmse_m",
        "validation_predictor_lateral_mae_m",
        "validation_predictor_longitudinal_mae_m",
        "validation_predictor_error_at_0p1s_m",
        "validation_predictor_error_at_0p5s_m",
        "validation_predictor_error_at_1p0s_m",
        "validation_joint_loss",
    )
    assert all(float(row[name]) >= 0.0 for name in nonnegative_fields)
    assert float(row["learning_rate"]) == pytest.approx(config.behavioral_cloning.learning_rate)


def _assert_bc_log_output(caplog: pytest.LogCaptureFixture) -> None:
    expected_fragments = (
        "BC training | epochs=1",
        "BC epoch 01/01 *",
        "losses     | action MSE T/V",
        "actions    | pedal MSE T/V",
        "trajectory | ADE T/V",
        "components | lateral MAE T/V",
        "horizons   | 0.1 s T/V",
        "BC test horizons",
    )
    assert all(fragment in caplog.text for fragment in expected_fragments)


def _assert_bc_summary(result: BCTrainingResult) -> None:
    assert np.isfinite(result.test.action_mse)
    assert np.isfinite(result.test.predictor_ade_m)
    assert np.isfinite(result.test.predictor_rmse_m)
    assert np.isfinite(result.test.predictor_lateral_mae_m)
    assert np.isfinite(result.test.predictor_longitudinal_mae_m)
    assert result.test.predictor_rmse_m >= result.test.predictor_ade_m
    assert result.test.predictor_error_at_1p0s_m == pytest.approx(result.test.predictor_fde_m)
    summary = json.loads((result.run_directory / "summary.json").read_text("utf-8"))
    assert summary["predictor_trained_with_bc"] is True
    assert summary["predictor_loss_weight"] == pytest.approx(0.01)
    assert summary["predictor_target_normalization_method_variant"] == (
        TRAIN_SPLIT_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT
    )
    assert len(summary["epoch_history"]) == 1
    assert summary["epoch_history"][0]["epoch"] == 1
    assert summary["epoch_history"][0]["train_joint_loss"] >= 0.0
    assert summary["epoch_history"][0]["validation_joint_loss"] >= 0.0
    assert result.best_validation_joint_loss >= 0.0


def _assert_train_behavioral_cloning_result(
    result: BCTrainingResult,
    config: ProjectConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert result.best_epoch == 1
    assert result.checkpoint_path.is_file()
    metrics_path = result.run_directory / "metrics.csv"
    assert metrics_path.is_file()
    assert (result.run_directory / "summary.json").is_file()
    _assert_epoch_metrics_csv(metrics_path, config)
    _assert_bc_log_output(caplog)
    _assert_bc_summary(result)


def test_fixed_reference_bc_export_and_joint_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_lane_factory: FastLaneFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    channels = _fake_motec_channels()
    header = MotecHeader("driver", "car", "track", "date", "time", len(channels))

    def fake_read(
        _path: Path,
        names: tuple[str, ...],
    ) -> tuple[MotecHeader, dict[str, np.ndarray]]:
        return header, {name: channels[name] for name in names}

    monkeypatch.setattr(bc_dataset, "read_motec_channels", fake_read)
    config = _project_config(tmp_path, fast_lane_factory)
    dataset, report = build_bc_dataset(config)

    _assert_built_bc_dataset(config, dataset, report)

    loaded = BehavioralCloningDataset.from_npz(report.destination)
    np.testing.assert_allclose(loaded.train.observations, dataset.train.observations)
    assert loaded.normalizer.feature_names == dataset.normalizer.feature_names

    caplog.set_level(
        logging.INFO,
        logger="assetto_corsa_rl.training.behavioral_cloning",
    )
    result = train_behavioral_cloning(
        config,
        loaded,
        command=["test", "bc-train"],
    )
    _assert_train_behavioral_cloning_result(result, config, caplog)
