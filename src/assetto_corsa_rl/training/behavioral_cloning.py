"""Joint behavioural-cloning warm start for policy and trajectory predictor."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.data.bc_dataset import BCSplit, BehavioralCloningDataset
from assetto_corsa_rl.experiment import ExperimentRecorder
from assetto_corsa_rl.features.schema import OBSERVATION_FEATURE_NAMES, OBSERVATION_SIZE
from assetto_corsa_rl.models.bezier import (
    decode_bezier_trajectory_normalized,
)
from assetto_corsa_rl.models.driver_network import DriverNetwork
from assetto_corsa_rl.models.trajectory_contract import (
    BEZIER_TRAJECTORY_CODEC,
    PREDICTOR_COORDINATE_SCALE_M,
)
from assetto_corsa_rl.training.checkpoints import (
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from assetto_corsa_rl.training.losses import behavioural_cloning_loss
from assetto_corsa_rl.types import PREDICTOR_TIME_OFFSETS_S

LOGGER = logging.getLogger(__name__)

_PREDICTOR_REPORT_HORIZONS_S = (0.1, 0.5, 1.0)
_PREDICTOR_REPORT_INDICES = tuple(
    int(np.argmin(np.abs(PREDICTOR_TIME_OFFSETS_S - horizon_s)))
    for horizon_s in _PREDICTOR_REPORT_HORIZONS_S
)


@dataclass(frozen=True, slots=True)
class BCEvaluationMetrics:
    """Policy and predictor errors for one complete split."""

    action_mse: float
    action_mae: float
    action_combined_pedal_mse: float
    action_steering_mse: float
    predictor_smooth_l1: float
    predictor_ade_m: float
    predictor_fde_m: float
    predictor_rmse_m: float
    predictor_lateral_mae_m: float
    predictor_longitudinal_mae_m: float
    predictor_error_at_0p1s_m: float
    predictor_error_at_0p5s_m: float
    predictor_error_at_1p0s_m: float


@dataclass(frozen=True, slots=True)
class BCTrainingResult:
    """Completed warm-start artefacts and final held-out metrics."""

    run_directory: Path
    checkpoint_path: Path
    epochs: int
    best_epoch: int
    best_validation_joint_loss: float
    best_validation_action_mse: float
    validation: BCEvaluationMetrics
    test: BCEvaluationMetrics


@dataclass(frozen=True, slots=True)
class _BCBestState:
    epoch: int = 0
    validation_joint_loss: float = float("inf")
    validation_action_mse: float = float("inf")


@dataclass(slots=True)
class _EvaluationAccumulator:
    action_squared_sum: float = 0.0
    action_absolute_sum: float = 0.0
    action_squared_by_output: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    action_elements: int = 0
    action_samples: int = 0
    predictor_smooth_sum: float = 0.0
    predictor_distance_sum: float = 0.0
    predictor_squared_distance_sum: float = 0.0
    predictor_final_distance_sum: float = 0.0
    predictor_lateral_absolute_sum: float = 0.0
    predictor_longitudinal_absolute_sum: float = 0.0
    predictor_horizon_distance_sum: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    predictor_samples: int = 0
    predictor_points: int = 0

    def add_action_error(self, action_error: Tensor) -> None:
        squared = action_error.square()
        self.action_squared_sum += float(squared.sum().item())
        self.action_absolute_sum += float(action_error.abs().sum().item())
        self.action_squared_by_output += squared.sum(dim=0).detach().cpu().numpy()
        self.action_elements += int(action_error.numel())
        self.action_samples += int(action_error.shape[0])

    def add_predictor_error(
        self,
        *,
        smooth_loss_per_sample: Tensor,
        prediction_m: Tensor,
        target_m: Tensor,
    ) -> None:
        difference = prediction_m - target_m
        distance = difference.square().sum(dim=-1).sqrt()
        sample_count, point_count = distance.shape
        if point_count != int(PREDICTOR_TIME_OFFSETS_S.size):
            raise ValueError("predictor evaluation point count does not match contract")

        self.predictor_smooth_sum += float(smooth_loss_per_sample.sum().item())
        self.predictor_distance_sum += float(distance.sum().item())
        self.predictor_squared_distance_sum += float(distance.square().sum().item())
        self.predictor_final_distance_sum += float(distance[:, -1].sum().item())
        self.predictor_lateral_absolute_sum += float(difference[..., 0].abs().sum().item())
        self.predictor_longitudinal_absolute_sum += float(difference[..., 1].abs().sum().item())
        self.predictor_horizon_distance_sum += (
            distance[:, _PREDICTOR_REPORT_INDICES].sum(dim=0).detach().cpu().numpy()
        )
        self.predictor_samples += sample_count
        self.predictor_points += sample_count * point_count

    def finalize(self) -> BCEvaluationMetrics:
        if (
            self.action_elements == 0
            or self.action_samples == 0
            or self.predictor_samples == 0
            or self.predictor_points == 0
        ):
            raise ValueError("evaluation split has insufficient valid samples")

        horizon_errors = self.predictor_horizon_distance_sum / self.predictor_samples
        return BCEvaluationMetrics(
            action_mse=self.action_squared_sum / self.action_elements,
            action_mae=self.action_absolute_sum / self.action_elements,
            action_combined_pedal_mse=(
                float(self.action_squared_by_output[0]) / self.action_samples
            ),
            action_steering_mse=(float(self.action_squared_by_output[1]) / self.action_samples),
            predictor_smooth_l1=self.predictor_smooth_sum / self.predictor_samples,
            predictor_ade_m=self.predictor_distance_sum / self.predictor_points,
            predictor_fde_m=(self.predictor_final_distance_sum / self.predictor_samples),
            predictor_rmse_m=float(
                np.sqrt(self.predictor_squared_distance_sum / self.predictor_points)
            ),
            predictor_lateral_mae_m=(self.predictor_lateral_absolute_sum / self.predictor_points),
            predictor_longitudinal_mae_m=(
                self.predictor_longitudinal_absolute_sum / self.predictor_points
            ),
            predictor_error_at_0p1s_m=float(horizon_errors[0]),
            predictor_error_at_0p5s_m=float(horizon_errors[1]),
            predictor_error_at_1p0s_m=float(horizon_errors[2]),
        )


def train_behavioral_cloning(
    config: ProjectConfig,
    dataset: BehavioralCloningDataset,
    *,
    command: list[str],
    device: str | torch.device = "cpu",
) -> BCTrainingResult:
    """Optimise the shared backbone, action head and predictor head."""

    _validate_dataset_contract(config, dataset)
    _validate_predictor_targets(dataset)
    model, optimizer, recorder, checkpoint_path, target_device = _initialize_bc_training(
        config, command, device
    )
    _log_training_start(config, dataset, target_device)
    best, epoch_history = _run_bc_epochs(
        config=config,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        recorder=recorder,
        checkpoint_path=checkpoint_path,
        device=target_device,
    )
    return _finalize_bc_training(
        config=config,
        dataset=dataset,
        model=model,
        recorder=recorder,
        checkpoint_path=checkpoint_path,
        target_device=target_device,
        best_epoch=best.epoch,
        best_validation_joint_loss=best.validation_joint_loss,
        best_validation_action_mse=best.validation_action_mse,
        epoch_history=epoch_history,
    )


def _validate_predictor_targets(dataset: BehavioralCloningDataset) -> None:
    if not dataset.train.predictor_valid_mask.any():
        raise ValueError("training split contains no valid predictor targets")
    if not dataset.validation.predictor_valid_mask.any():
        raise ValueError("validation split contains no valid predictor targets")


def _run_bc_epochs(
    *,
    config: ProjectConfig,
    dataset: BehavioralCloningDataset,
    model: DriverNetwork,
    optimizer: torch.optim.Optimizer,
    recorder: ExperimentRecorder,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[_BCBestState, list[dict[str, float | int]]]:
    best = _BCBestState()
    history: list[dict[str, float | int]] = []
    random = np.random.default_rng(config.experiment.seed)

    for epoch in range(1, config.behavioral_cloning.epochs + 1):
        _train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            config=config,
            random=random,
            device=device,
        )
        train_metrics, validation_metrics = _evaluate_epoch(
            config=config,
            dataset=dataset,
            model=model,
            device=device,
        )
        weight = config.behavioral_cloning.predictor_loss_weight
        train_joint_loss = _joint_loss(train_metrics, weight)
        validation_joint_loss = _joint_loss(validation_metrics, weight)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        is_best = validation_joint_loss < best.validation_joint_loss
        record = _build_epoch_record(
            epoch=epoch,
            learning_rate=learning_rate,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            train_joint_loss=train_joint_loss,
            validation_joint_loss=validation_joint_loss,
        )
        history.append(record)
        recorder.record_step(record)
        _log_epoch_metrics(
            epoch=epoch,
            total_epochs=config.behavioral_cloning.epochs,
            train=train_metrics,
            validation=validation_metrics,
            train_joint_loss=train_joint_loss,
            validation_joint_loss=validation_joint_loss,
            learning_rate=learning_rate,
            is_best=is_best,
        )
        best = _maybe_update_best_checkpoint(
            epoch=epoch,
            validation_metrics=validation_metrics,
            validation_joint_loss=validation_joint_loss,
            best=best,
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            config=config,
        )

    return best, history


def _evaluate_epoch(
    *,
    config: ProjectConfig,
    dataset: BehavioralCloningDataset,
    model: DriverNetwork,
    device: torch.device,
) -> tuple[BCEvaluationMetrics, BCEvaluationMetrics]:
    batch_size = config.behavioral_cloning.batch_size
    return (
        evaluate_behavioral_cloning(
            model,
            dataset.train,
            batch_size=batch_size,
            device=device,
        ),
        evaluate_behavioral_cloning(
            model,
            dataset.validation,
            batch_size=batch_size,
            device=device,
        ),
    )


def _finalize_bc_training(
    *,
    config: ProjectConfig,
    dataset: BehavioralCloningDataset,
    model: DriverNetwork,
    recorder: ExperimentRecorder,
    checkpoint_path: Path,
    target_device: torch.device,
    best_epoch: int,
    best_validation_joint_loss: float,
    best_validation_action_mse: float,
    epoch_history: list[dict[str, float | int]],
) -> BCTrainingResult:
    """Evaluate the best checkpoint and persist the final BC summary."""

    load_checkpoint(checkpoint_path, model=model, map_location=target_device)
    validation = evaluate_behavioral_cloning(
        model,
        dataset.validation,
        batch_size=config.behavioral_cloning.batch_size,
        device=target_device,
    )
    test = evaluate_behavioral_cloning(
        model,
        dataset.test,
        batch_size=config.behavioral_cloning.batch_size,
        device=target_device,
    )
    result = BCTrainingResult(
        run_directory=recorder.run_directory,
        checkpoint_path=checkpoint_path,
        epochs=config.behavioral_cloning.epochs,
        best_epoch=best_epoch,
        best_validation_joint_loss=best_validation_joint_loss,
        best_validation_action_mse=best_validation_action_mse,
        validation=validation,
        test=test,
    )
    _log_final_metrics(
        best_epoch=best_epoch,
        total_epochs=config.behavioral_cloning.epochs,
        validation=validation,
        test=test,
    )
    recorder.write_summary(
        {
            "method_variant": config.behavioral_cloning.method_variant,
            "action_method_variant": config.actions.method_variant,
            "dataset_metadata": dataset.metadata,
            "train_samples": dataset.train.sample_count,
            "validation_samples": dataset.validation.sample_count,
            "test_samples": dataset.test.sample_count,
            "best_epoch": best_epoch,
            "best_validation_joint_loss": best_validation_joint_loss,
            "best_validation_action_mse": best_validation_action_mse,
            "predictor_trained_with_bc": (config.behavioral_cloning.predictor_loss_weight > 0.0),
            "predictor_loss_weight": config.behavioral_cloning.predictor_loss_weight,
            "trajectory_codec": BEZIER_TRAJECTORY_CODEC,
            "predictor_coordinate_scale_m": PREDICTOR_COORDINATE_SCALE_M,
            "epoch_history": epoch_history,
            "final_validation": asdict(validation),
            "final_test": asdict(test),
            "checkpoint": checkpoint_path.name,
        }
    )
    recorder.close()
    return result


def _initialize_bc_training(
    config: ProjectConfig,
    command: list[str],
    device: str | torch.device,
) -> tuple[DriverNetwork, torch.optim.Optimizer, ExperimentRecorder, Path, torch.device]:
    torch.manual_seed(config.experiment.seed)  # pyright: ignore[reportUnknownMemberType]
    np.random.seed(config.experiment.seed)
    torch.use_deterministic_algorithms(True)
    target_device = torch.device(device)
    model = DriverNetwork(
        observation_size=OBSERVATION_SIZE,
        feature_hidden_sizes=config.model.feature_hidden_sizes,
        policy_hidden_sizes=config.model.policy_hidden_sizes,
        value_hidden_sizes=config.model.value_hidden_sizes,
        predictor_hidden_sizes=config.model.predictor_hidden_sizes,
        trajectory_output_size=config.model.trajectory_output_size,
        initial_log_std=config.model.initial_log_std,
    ).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.behavioral_cloning.learning_rate,
        weight_decay=config.behavioral_cloning.weight_decay,
    )
    recorder = ExperimentRecorder.create(
        output_directory=Path(config.experiment.output_directory) / "bc",
        config=config,
        command=command,
    )
    return model, optimizer, recorder, recorder.run_directory / "checkpoint.pt", target_device


def _train_one_epoch(
    *,
    model: DriverNetwork,
    optimizer: torch.optim.Optimizer,
    dataset: BehavioralCloningDataset,
    config: ProjectConfig,
    random: np.random.Generator,
    device: torch.device,
) -> None:
    model.train()
    permutation = random.permutation(dataset.train.sample_count)
    for start in range(0, permutation.size, config.behavioral_cloning.batch_size):
        indices = permutation[start : start + config.behavioral_cloning.batch_size]
        observation, action, target, valid = _batch_tensors(
            dataset.train,
            indices,
            device,
        )
        output = model(observation)
        action_loss = behavioural_cloning_loss(output.action_mean, action)
        predictor_loss = _masked_predictor_loss(
            output.trajectory_parameters,
            target,
            valid,
        )
        total_loss = action_loss + config.behavioral_cloning.predictor_loss_weight * predictor_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]


def _build_epoch_record(
    *,
    epoch: int,
    learning_rate: float,
    train_metrics: BCEvaluationMetrics,
    validation_metrics: BCEvaluationMetrics,
    train_joint_loss: float,
    validation_joint_loss: float,
) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_joint_loss": train_joint_loss,
        "validation_joint_loss": validation_joint_loss,
        **{f"train_{name}": value for name, value in asdict(train_metrics).items()},
        **{f"validation_{name}": value for name, value in asdict(validation_metrics).items()},
    }


def _maybe_update_best_checkpoint(
    *,
    epoch: int,
    validation_metrics: BCEvaluationMetrics,
    validation_joint_loss: float,
    best: _BCBestState,
    checkpoint_path: Path,
    model: DriverNetwork,
    optimizer: torch.optim.Optimizer,
    config: ProjectConfig,
) -> _BCBestState:
    if validation_joint_loss >= best.validation_joint_loss:
        return best

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        training_state=TrainingState(
            iteration=epoch,
            environment_steps=0,
            policy_version=0,
        ),
        configuration={
            "training_stage": "behavioral_cloning",
            "method_variant": config.behavioral_cloning.method_variant,
            "action_method_variant": config.actions.method_variant,
            "project": config.to_dict(),
            "observation_feature_names": list(OBSERVATION_FEATURE_NAMES),
            "predictor_trained_with_bc": (config.behavioral_cloning.predictor_loss_weight > 0.0),
            "predictor_loss_weight": config.behavioral_cloning.predictor_loss_weight,
            "trajectory_codec": BEZIER_TRAJECTORY_CODEC,
            "predictor_coordinate_scale_m": PREDICTOR_COORDINATE_SCALE_M,
        },
    )
    return _BCBestState(
        epoch=epoch,
        validation_joint_loss=validation_joint_loss,
        validation_action_mse=validation_metrics.action_mse,
    )


def _validate_dataset_contract(
    config: ProjectConfig,
    dataset: BehavioralCloningDataset,
) -> None:
    """Reject stale datasets built for an incompatible predictor contract."""

    if dataset.metadata.get("schema_version") != 4:
        raise ValueError("BC dataset schema is stale; rebuild the BC dataset")
    if dataset.metadata.get("trajectory_codec") != config.model.trajectory_codec:
        raise ValueError("BC dataset trajectory codec does not match the model")
    scale = dataset.metadata.get("predictor_coordinate_scale_m")
    if scale != PREDICTOR_COORDINATE_SCALE_M:
        raise ValueError("BC dataset predictor coordinate scale does not match")
    point_count = dataset.metadata.get("predictor_point_count")
    if point_count != int(PREDICTOR_TIME_OFFSETS_S.size):
        raise ValueError("BC dataset predictor point count does not match")
    offsets = np.asarray(
        dataset.metadata.get("predictor_time_offsets_s", []),
        dtype=np.float32,
    )
    if offsets.shape != PREDICTOR_TIME_OFFSETS_S.shape or not np.allclose(
        offsets,
        PREDICTOR_TIME_OFFSETS_S,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("BC dataset predictor time offsets do not match")


@torch.no_grad()
def evaluate_behavioral_cloning(
    model: DriverNetwork,
    split: BCSplit,
    *,
    batch_size: int,
    device: torch.device,
) -> BCEvaluationMetrics:
    """Evaluate action and decoded-trajectory errors without random sampling."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    accumulator = _EvaluationAccumulator()

    for start in range(0, split.sample_count, batch_size):
        indices = np.arange(start, min(start + batch_size, split.sample_count))
        observation, action, target, valid = _batch_tensors(split, indices, device)
        output = model(observation)
        accumulator.add_action_error(output.action_mean - action)
        if not valid.any():
            continue

        prediction_normalized = decode_bezier_trajectory_normalized(
            output.trajectory_parameters[valid],
            sample_count=target.shape[1],
        )
        valid_target = target[valid]
        target_normalized = valid_target / PREDICTOR_COORDINATE_SCALE_M
        smooth_loss = torch.nn.functional.smooth_l1_loss(
            prediction_normalized,
            target_normalized,
            reduction="none",
        ).mean(dim=(-2, -1))
        accumulator.add_predictor_error(
            smooth_loss_per_sample=smooth_loss,
            prediction_m=(prediction_normalized * PREDICTOR_COORDINATE_SCALE_M),
            target_m=valid_target,
        )

    return accumulator.finalize()


def _joint_loss(
    metrics: BCEvaluationMetrics,
    predictor_loss_weight: float,
) -> float:
    return metrics.action_mse + predictor_loss_weight * metrics.predictor_smooth_l1


def _log_training_start(
    config: ProjectConfig,
    dataset: BehavioralCloningDataset,
    device: torch.device,
) -> None:
    LOGGER.info(
        "BC training | epochs=%d | train=%d | validation=%d | batch=%d | device=%s",
        config.behavioral_cloning.epochs,
        dataset.train.sample_count,
        dataset.validation.sample_count,
        config.behavioral_cloning.batch_size,
        device,
    )
    LOGGER.info("BC legend   | T=train | V=validation | *=new best checkpoint")


def _log_epoch_metrics(
    *,
    epoch: int,
    total_epochs: int,
    train: BCEvaluationMetrics,
    validation: BCEvaluationMetrics,
    train_joint_loss: float,
    validation_joint_loss: float,
    learning_rate: float,
    is_best: bool,
) -> None:
    marker = "*" if is_best else " "
    LOGGER.info("BC epoch %02d/%02d %s", epoch, total_epochs, marker)
    LOGGER.info(
        "  losses     | action MSE T/V %.6f / %.6f | predictor L1 T/V %.6f / %.6f",
        train.action_mse,
        validation.action_mse,
        train.predictor_smooth_l1,
        validation.predictor_smooth_l1,
    )
    LOGGER.info(
        "  actions    | pedal MSE T/V %.6f / %.6f | steering MSE T/V %.6f / %.6f",
        train.action_combined_pedal_mse,
        validation.action_combined_pedal_mse,
        train.action_steering_mse,
        validation.action_steering_mse,
    )
    LOGGER.info(
        "  trajectory | ADE T/V %.3f / %.3f m | RMSE T/V %.3f / %.3f m",
        train.predictor_ade_m,
        validation.predictor_ade_m,
        train.predictor_rmse_m,
        validation.predictor_rmse_m,
    )
    LOGGER.info(
        "  components | lateral MAE T/V %.3f / %.3f m | longitudinal MAE T/V %.3f / %.3f m",
        train.predictor_lateral_mae_m,
        validation.predictor_lateral_mae_m,
        train.predictor_longitudinal_mae_m,
        validation.predictor_longitudinal_mae_m,
    )
    LOGGER.info(
        "  horizons   | 0.1 s T/V %.3f / %.3f m | 0.5 s T/V %.3f / %.3f m | "
        "1.0 s T/V %.3f / %.3f m",
        train.predictor_error_at_0p1s_m,
        validation.predictor_error_at_0p1s_m,
        train.predictor_error_at_0p5s_m,
        validation.predictor_error_at_0p5s_m,
        train.predictor_error_at_1p0s_m,
        validation.predictor_error_at_1p0s_m,
    )
    LOGGER.info(
        "  objective  | joint T/V %.6f / %.6f | lr %.3e",
        train_joint_loss,
        validation_joint_loss,
        learning_rate,
    )


def _log_final_metrics(
    *,
    best_epoch: int,
    total_epochs: int,
    validation: BCEvaluationMetrics,
    test: BCEvaluationMetrics,
) -> None:
    LOGGER.info(
        "BC best | epoch=%d/%d | validation action=%.6f | ADE=%.3f m | RMSE=%.3f m | FDE=%.3f m",
        best_epoch,
        total_epochs,
        validation.action_mse,
        validation.predictor_ade_m,
        validation.predictor_rmse_m,
        validation.predictor_fde_m,
    )
    LOGGER.info(
        "BC test | action=%.6f | ADE=%.3f m | RMSE=%.3f m | "
        "lateral MAE=%.3f m | longitudinal MAE=%.3f m",
        test.action_mse,
        test.predictor_ade_m,
        test.predictor_rmse_m,
        test.predictor_lateral_mae_m,
        test.predictor_longitudinal_mae_m,
    )
    LOGGER.info(
        "BC test horizons | 0.1 s %.3f m | 0.5 s %.3f m | 1.0 s %.3f m",
        test.predictor_error_at_0p1s_m,
        test.predictor_error_at_0p5s_m,
        test.predictor_error_at_1p0s_m,
    )


def _masked_predictor_loss(
    flat_prediction: Tensor,
    target_xz_m: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    if valid_mask.any():
        prediction_normalized = decode_bezier_trajectory_normalized(
            flat_prediction[valid_mask],
            sample_count=target_xz_m.shape[1],
        )
        return torch.nn.functional.smooth_l1_loss(
            prediction_normalized,
            target_xz_m[valid_mask] / PREDICTOR_COORDINATE_SCALE_M,
        )
    return flat_prediction.sum() * 0.0


def _batch_tensors(
    split: BCSplit,
    indices: npt.NDArray[np.integer],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.as_tensor(split.observations[indices], device=device),
        torch.as_tensor(split.expert_action[indices], device=device),
        torch.as_tensor(split.predictor_target_xz_m[indices], device=device),
        torch.as_tensor(split.predictor_valid_mask[indices], device=device),
    )
