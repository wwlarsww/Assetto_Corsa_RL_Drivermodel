from __future__ import annotations

import pytest
import torch

from assetto_corsa_rl.features.schema import OBSERVATION_SIZE
from assetto_corsa_rl.models.bezier import (
    decode_bezier_trajectory,
    decode_control_points,
    sample_bezier,
)
from assetto_corsa_rl.models.driver_network import DriverNetwork
from assetto_corsa_rl.training.losses import (
    behavioural_cloning_loss,
    bezier_trajectory_prediction_loss,
    trajectory_prediction_loss,
)


def test_thesis_network_shapes_and_distribution() -> None:
    network = DriverNetwork()
    observation = torch.zeros((4, OBSERVATION_SIZE))
    output = network(observation)

    assert output.action_mean.shape == (4, 2)
    assert output.action_log_std.shape == (4, 2)
    assert output.value.shape == (4,)
    assert output.trajectory_parameters.shape == (4, 10)
    assert network.action_distribution(observation).batch_shape == (4,)
    assert network.act(observation, deterministic=True).shape == (4, 2)
    assert torch.isfinite(network.act(observation, deterministic=False)).all()


def test_bezier_decoder_preserves_endpoints() -> None:
    flat = torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0, 4.0]])
    control_points = decode_control_points(flat)
    curve = sample_bezier(control_points, sample_count=5)

    torch.testing.assert_close(curve[:, 0], control_points[:, 0])
    torch.testing.assert_close(curve[:, -1], control_points[:, -1])
    assert decode_bezier_trajectory(flat, sample_count=5).shape == (1, 5, 2)


def test_supervised_losses_and_shape_validation() -> None:
    predicted = torch.tensor([[0.0, 1.0]], requires_grad=True)
    target = torch.ones((1, 2))
    assert behavioural_cloning_loss(predicted, target).item() == 0.5
    assert trajectory_prediction_loss(predicted, target).item() == 0.25
    with pytest.raises(ValueError, match="identical"):
        behavioural_cloning_loss(predicted, torch.ones(1))

    flat_path = torch.zeros((2, 10), requires_grad=True)
    target_path = torch.zeros((2, 20, 2))
    valid = torch.tensor([True, False])
    assert bezier_trajectory_prediction_loss(flat_path, target_path, valid).item() == 0.0
    with pytest.raises(ValueError, match="boolean"):
        bezier_trajectory_prediction_loss(flat_path, target_path, torch.ones(2))


def test_network_and_bezier_inputs_are_validated() -> None:
    network = DriverNetwork(
        feature_hidden_sizes=(8,),
        policy_hidden_sizes=(4,),
        value_hidden_sizes=(4,),
        predictor_hidden_sizes=(4,),
    )
    with pytest.raises(ValueError, match="last observation dimension"):
        network(torch.zeros((1, 3)))
    with pytest.raises(ValueError, match="divisible"):
        decode_control_points(torch.zeros((1, 5)))
    with pytest.raises(ValueError, match="must end"):
        sample_bezier(torch.zeros(2))
    with pytest.raises(ValueError, match="at least two control"):
        sample_bezier(torch.zeros((1, 1, 2)))
    with pytest.raises(ValueError, match="sample_count"):
        sample_bezier(torch.zeros((1, 2, 2)), sample_count=1)
