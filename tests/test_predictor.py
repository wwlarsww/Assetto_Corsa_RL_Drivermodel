"""Geometry, horizons, partial supervision and gradient-safe masked loss."""

import numpy as np
import pytest
import torch

from assetto_corsa_ai.predictor import build_trajectory_targets, horizon_indices, masked_predictor_loss


def rollout(steps: int = 5, environments: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.zeros((steps, environments, 3), dtype=np.float64)
    positions[..., 0] = np.arange(steps)[:, None]
    forwards = np.zeros_like(positions)
    forwards[..., 0] = 1
    next_positions = positions.copy()
    next_positions[..., 0] += 1
    return positions, forwards, next_positions, np.zeros((steps, environments), dtype=np.bool_)


def test_horizons_use_configured_control_interval() -> None:
    horizons = tuple(index / 10 for index in range(1, 11))
    assert horizon_indices(horizons, 0.05) == tuple(range(2, 21, 2))
    assert horizon_indices(horizons, 0.1) == tuple(range(1, 11))


@pytest.mark.parametrize("horizons,interval", [
    ((), 0.05), ((0.0,), 0.05), ((0.1, 0.1), 0.05), ((0.2, 0.1), 0.05),
    ((0.075,), 0.05), ((float("nan"),), 0.05), ((0.1,), 0.0), ((0.1,), float("inf")),
])
def test_invalid_horizon_grid_is_rejected(horizons: tuple[float, ...], interval: float) -> None:
    with pytest.raises(ValueError):
        horizon_indices(horizons, interval)


def test_targets_are_relative_to_original_pose_and_ignore_world_height() -> None:
    positions, forwards, next_positions, dones = rollout()
    positions += (100.0, 50.0, -20.0)
    next_positions += (100.0, -100.0, -20.0)
    next_positions[..., 2] -= 2
    targets, mask = build_trajectory_targets(positions, forwards, next_positions, dones, (1, 2))
    np.testing.assert_allclose(targets[0, 0], [[1, 2], [2, 2]])
    assert mask[0, 0].all()
    assert targets.dtype == np.float32


def test_rotated_vehicle_facing_minus_z_has_left_along_minus_x() -> None:
    positions, forwards, next_positions, dones = rollout()
    positions[:] = (100, 0, 200)
    next_positions[:] = (97, 10, 195)
    forwards[:] = (0, 0, -3)
    targets, _ = build_trajectory_targets(positions, forwards, next_positions, dones, (1,))
    np.testing.assert_allclose(targets[:, 0, 0], np.tile([5, 3], (5, 1)))


def test_rollout_end_masks_only_unavailable_horizons_and_keeps_final_endpoint() -> None:
    targets, mask = build_trajectory_targets(*rollout(), (1, 2, 5, 6))
    assert mask[0, 0].tolist() == [True, True, True, False]
    assert mask[-1, 0].tolist() == [True, False, False, False]
    np.testing.assert_allclose(targets[0, 0, 2], [5, 0])
    assert (targets[~mask] == 0).all()


def test_episode_boundary_keeps_partial_targets_without_reset_leakage() -> None:
    positions, forwards, next_positions, dones = rollout(8)
    dones[2] = True
    positions[3:, :, 0] += 1000
    next_positions[3:, :, 0] += 1000
    targets, mask = build_trajectory_targets(positions, forwards, next_positions, dones, (1, 2, 3, 4, 5))
    assert mask[0, 0].tolist() == [True, True, True, False, False]
    assert mask[2, 0].tolist() == [True, False, False, False, False]
    assert mask[3, 0].all()
    np.testing.assert_allclose(targets[0, 0, :3, 0], [1, 2, 3])
    assert (targets[~mask] == 0).all()


def test_vector_env_masks_are_independent() -> None:
    data = rollout(environments=2)
    data[-1][1, 0] = True
    _, mask = build_trajectory_targets(*data, (1, 3))
    assert mask[0, 0].tolist() == [True, False]
    assert mask[0, 1].tolist() == [True, True]


def test_masked_smooth_l1_uses_only_valid_coordinates() -> None:
    prediction = torch.tensor([[[2.0, -0.5], [1000.0, -1000.0]]], requires_grad=True)
    target = torch.tensor([[[0.0, 0.0], [float("nan"), float("nan")]]])
    loss = masked_predictor_loss(prediction, target, torch.tensor([[True, False]]))
    assert loss.item() == pytest.approx((1.5 + 0.125) / 2)
    torch.autograd.backward(loss)
    assert prediction.grad is not None
    np.testing.assert_allclose(prediction.grad[0, 1].numpy(), [0, 0])
    np.testing.assert_allclose(prediction.grad[0, 0].numpy(), [0.5, -0.25])


def test_completely_masked_batch_has_zero_loss_and_zero_gradient() -> None:
    prediction = torch.ones((2, 3, 2), requires_grad=True)
    loss = masked_predictor_loss(prediction, torch.zeros_like(prediction), torch.zeros((2, 3), dtype=torch.bool))
    torch.autograd.backward(loss)
    assert loss.item() == 0
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0


def test_bad_mask_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        masked_predictor_loss(torch.zeros((2, 3, 2)), torch.zeros((2, 3, 2)), torch.ones((2, 3, 2), dtype=torch.bool))
