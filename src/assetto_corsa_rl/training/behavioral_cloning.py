"""Joint behavioural-cloning warm start for policy and trajectory predictor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.data.bc_dataset import BCSplit, BehavioralCloningDataset
from assetto_corsa_rl.experiment import ExperimentRecorder
from assetto_corsa_rl.features.schema import OBSERVATION_FEATURE_NAMES, OBSERVATION_SIZE
from assetto_corsa_rl.models.bezier import decode_bezier_trajectory
from assetto_corsa_rl.models.driver_network import DriverNetwork
from assetto_corsa_rl.training.checkpoints import (
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from assetto_corsa_rl.training.losses import behavioural_cloning_loss


@dataclass(frozen=True, slots=True)
class BCEvaluationMetrics:
    """Policy and predictor errors for one complete split."""

    action_mse: float
    action_mae: float
    predictor_smooth_l1: float
    predictor_ade_m: float
    predictor_fde_m: float


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


def train_behavioral_cloning(
    config: ProjectConfig,
    dataset: BehavioralCloningDataset,
    *,
    command: list[str],
    device: str | torch.device = "cpu",
) -> BCTrainingResult:
    """Optimise the shared backbone, action head and predictor head."""

    if not dataset.train.predictor_valid_mask.any():
        raise ValueError("training split contains no valid predictor targets")
    if not dataset.validation.predictor_valid_mask.any():
        raise ValueError("validation split contains no valid predictor targets")
    torch.manual_seed(config.experiment.seed)
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
    checkpoint_path = recorder.run_directory / "checkpoint.pt"
    best_epoch = 0
    best_validation_joint_loss = float("inf")
    best_validation_action_mse = float("inf")
    random = np.random.default_rng(config.experiment.seed)
    for epoch in range(1, config.behavioral_cloning.epochs + 1):
        model.train()
        permutation = random.permutation(dataset.train.sample_count)
        for start in range(0, permutation.size, config.behavioral_cloning.batch_size):
            indices = permutation[start : start + config.behavioral_cloning.batch_size]
            observation, action, target, valid = _batch_tensors(
                dataset.train,
                indices,
                target_device,
            )
            output = model(observation)
            action_loss = behavioural_cloning_loss(output.action_mean, action)
            predictor_loss = _masked_predictor_loss(
                output.trajectory_parameters,
                target,
                valid,
            )
            total_loss = (
                action_loss + config.behavioral_cloning.predictor_loss_weight * predictor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        train_metrics = evaluate_behavioral_cloning(
            model,
            dataset.train,
            batch_size=config.behavioral_cloning.batch_size,
            device=target_device,
        )
        validation_metrics = evaluate_behavioral_cloning(
            model,
            dataset.validation,
            batch_size=config.behavioral_cloning.batch_size,
            device=target_device,
        )
        validation_joint_loss = (
            validation_metrics.action_mse
            + config.behavioral_cloning.predictor_loss_weight
            * validation_metrics.predictor_smooth_l1
        )
        recorder.record_step(
            {
                "epoch": epoch,
                "validation_joint_loss": validation_joint_loss,
                **{f"train_{name}": value for name, value in asdict(train_metrics).items()},
                **{
                    f"validation_{name}": value
                    for name, value in asdict(validation_metrics).items()
                },
            }
        )
        if validation_joint_loss < best_validation_joint_loss:
            best_epoch = epoch
            best_validation_joint_loss = validation_joint_loss
            best_validation_action_mse = validation_metrics.action_mse
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
                    "method_variant": config.behavioral_cloning.method_variant,
                    "project": config.to_dict(),
                    "observation_feature_names": list(OBSERVATION_FEATURE_NAMES),
                    "predictor_trained_with_bc": True,
                },
            )

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
    recorder.write_summary(
        {
            "method_variant": config.behavioral_cloning.method_variant,
            "dataset_metadata": dataset.metadata,
            "train_samples": dataset.train.sample_count,
            "validation_samples": dataset.validation.sample_count,
            "test_samples": dataset.test.sample_count,
            "predictor_loss_weight": config.behavioral_cloning.predictor_loss_weight,
            "best_epoch": best_epoch,
            "best_validation_joint_loss": best_validation_joint_loss,
            "best_validation_action_mse": best_validation_action_mse,
            "final_validation": asdict(validation),
            "final_test": asdict(test),
            "checkpoint": checkpoint_path.name,
        }
    )
    recorder.close()
    return result


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
    action_squared_sum = 0.0
    action_absolute_sum = 0.0
    action_elements = 0
    predictor_smooth_sum = 0.0
    predictor_ade_sum = 0.0
    predictor_fde_sum = 0.0
    predictor_samples = 0
    for start in range(0, split.sample_count, batch_size):
        indices = np.arange(start, min(start + batch_size, split.sample_count))
        observation, action, target, valid = _batch_tensors(split, indices, device)
        output = model(observation)
        action_error = output.action_mean - action
        action_squared_sum += float(action_error.square().sum().item())
        action_absolute_sum += float(action_error.abs().sum().item())
        action_elements += int(action_error.numel())
        if valid.any():
            prediction = decode_bezier_trajectory(
                output.trajectory_parameters[valid],
                sample_count=target.shape[1],
            )
            valid_target = target[valid]
            smooth = torch.nn.functional.smooth_l1_loss(
                prediction,
                valid_target,
                reduction="none",
            ).mean(dim=(-2, -1))
            distance = torch.linalg.vector_norm(prediction - valid_target, dim=-1)
            predictor_smooth_sum += float(smooth.sum().item())
            predictor_ade_sum += float(distance.mean(dim=-1).sum().item())
            predictor_fde_sum += float(distance[:, -1].sum().item())
            predictor_samples += int(valid.sum().item())
    if action_elements == 0 or predictor_samples == 0:
        raise ValueError("evaluation split has insufficient valid samples")
    return BCEvaluationMetrics(
        action_mse=action_squared_sum / action_elements,
        action_mae=action_absolute_sum / action_elements,
        predictor_smooth_l1=predictor_smooth_sum / predictor_samples,
        predictor_ade_m=predictor_ade_sum / predictor_samples,
        predictor_fde_m=predictor_fde_sum / predictor_samples,
    )


def _masked_predictor_loss(
    flat_prediction: Tensor,
    target_xz_m: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    if valid_mask.any():
        prediction = decode_bezier_trajectory(
            flat_prediction[valid_mask],
            sample_count=target_xz_m.shape[1],
        )
        return torch.nn.functional.smooth_l1_loss(
            prediction,
            target_xz_m[valid_mask],
        )
    return flat_prediction.sum() * 0.0


def _batch_tensors(
    split: BCSplit,
    indices: npt.NDArray[np.integer],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.as_tensor(split.observations[indices], device=device),
        torch.as_tensor(split.expert_action_delta[indices], device=device),
        torch.as_tensor(split.predictor_target_xz_m[indices], device=device),
        torch.as_tensor(split.predictor_valid_mask[indices], device=device),
    )
