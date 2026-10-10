"""One feature tensor, shared gradients, control units and residual actions."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from stable_baselines3.common.running_mean_std import RunningMeanStd
import torch
from torch import nn

from assetto_corsa_ai.main import create_agent
from assetto_corsa_ai.predictor import masked_predictor_loss
from assetto_corsa_ai.racing_policy import ActionAwareVecNormalize, RacingActorCriticPolicy
from assetto_corsa_ai.training_config import TrainingConfig
from tools.validate_predictor import ValidationEnvironment, synthetic_rollout


def small_config() -> TrainingConfig:
    config = TrainingConfig.load(Path(__file__).resolve().parents[1] / "configs/training.toml")
    return replace(
        config, network=replace(config.network, feature_layers=(7, 9), feature_dim=11, policy_layers=(6,), value_layers=(5,)),
        predictor=replace(config.predictor, layers=(3, 4)),
        reward=replace(config.reward, future_prediction=replace(config.reward.future_prediction, enabled=False)),
        ppo=replace(config.ppo, n_steps=32, batch_size=16, n_epochs=1),
        training=replace(config.training, device="cpu", verbose=0),
    )


def test_all_three_heads_receive_the_same_feature_tensor_and_predictor_gradient() -> None:
    config = small_config()
    agent = create_agent(ValidationEnvironment(synthetic_rollout(config)), config)
    try:
        policy = agent.model.policy
        assert isinstance(policy, RacingActorCriticPolicy)
        features: list[torch.Tensor] = []
        head_inputs: list[torch.Tensor] = []
        def capture_features(module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            features.append(output)
        def capture_inputs(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            head_inputs.append(inputs[0])
        hooks = [
            policy.features_extractor.register_forward_hook(capture_features),
            policy.mlp_extractor.register_forward_pre_hook(capture_inputs),
            policy.predictor_head.register_forward_pre_hook(capture_inputs),
        ]
        shape = agent.vec_env.observation_space.shape
        assert shape is not None
        observations = torch.ones((4, shape[0]))
        _, _, _, prediction = policy.evaluate_with_prediction(observations, torch.zeros((4, 2)))
        assert prediction.shape == (4, 10, 2)
        assert len(features) == 1
        assert len(head_inputs) == 2 and all(value is features[0] for value in head_inputs)
        for hook in hooks:
            hook.remove()
        policy.optimizer.zero_grad()
        torch.autograd.backward(masked_predictor_loss(prediction, torch.ones_like(prediction) * 2, torch.ones((4, 10), dtype=torch.bool)))
        assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0 for parameter in policy.features_extractor.parameters())
        assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0 for parameter in policy.predictor_head.parameters())
        assert all(parameter.grad is None for parameter in policy.action_net.parameters())
        optimizer_parameters = [parameter for group in policy.optimizer.param_groups for parameter in group["params"]]
        assert len(optimizer_parameters) == len({id(parameter) for parameter in optimizer_parameters})
        assert {id(parameter) for parameter in optimizer_parameters} == {id(parameter) for parameter in policy.parameters()}
    finally:
        agent.close()


def test_network_and_predictor_layer_sizes_come_from_config() -> None:
    config = small_config()
    agent = create_agent(ValidationEnvironment(synthetic_rollout(config)), config)
    try:
        policy = agent.model.policy
        assert isinstance(policy, RacingActorCriticPolicy)
        feature_sizes = [layer.out_features for layer in policy.features_extractor.modules() if isinstance(layer, nn.Linear)]
        predictor_sizes = [layer.out_features for layer in policy.predictor_head if isinstance(layer, nn.Linear)]
        assert feature_sizes == [7, 9, 11]
        assert predictor_sizes == [3, 4, 20]
        assert policy.mlp_extractor.latent_dim_pi == 6
        assert policy.mlp_extractor.latent_dim_vf == 5
        assert policy.pi_features_extractor is policy.vf_features_extractor is policy.features_extractor
    finally:
        agent.close()


def test_delta_actions_add_current_controls_before_final_clipping() -> None:
    config = small_config()
    data = synthetic_rollout(config)
    agent = create_agent(ValidationEnvironment(data), config)
    try:
        policy = agent.model.policy
        assert isinstance(policy, RacingActorCriticPolicy)
        assert isinstance(policy.action_net, nn.Linear)
        with torch.no_grad():
            policy.action_net.load_state_dict({
                "weight": torch.zeros_like(policy.action_net.weight),
                "bias": torch.tensor([0.5, -0.25]),
            })
        observation = data.observations[0, 0].copy()
        observation[-2:] = [0.3, 0.4]
        np.testing.assert_allclose(agent.predict(observation), [0.8, 0.15], atol=1e-6)
        observation[-2:] = [0.9, -0.9]
        np.testing.assert_allclose(agent.predict(observation), [1.0, -1.0])
        normalized = agent.vec_env.normalize_obs(observation)
        assert isinstance(normalized, np.ndarray)
        tensor = torch.as_tensor(normalized).unsqueeze(0)
        actions, _, original_log_prob = policy(tensor, deterministic=True)
        _, evaluated_log_prob, _ = policy.evaluate_actions(tensor, actions)
        torch.testing.assert_close(original_log_prob, evaluated_log_prob)
        torch.testing.assert_close(actions, torch.tensor([[1.4, -1.15]]))
    finally:
        agent.close()


def test_control_units_survive_normalization_and_inverse() -> None:
    config = small_config()
    data = synthetic_rollout(config)
    agent = create_agent(ValidationEnvironment(data), config)
    try:
        normalizer = agent.vec_env
        assert isinstance(normalizer, ActionAwareVecNormalize)
        assert isinstance(normalizer.obs_rms, RunningMeanStd)
        normalizer.obs_rms.mean[:] = 50
        normalizer.obs_rms.var[:] = 25
        observation = data.observations[0, 0].copy()
        observation[0] = 60
        observation[-2:] = [-0.3, 0.7]
        normalized = normalizer.normalize_obs(observation)
        assert isinstance(normalized, np.ndarray)
        assert normalized[0] == pytest.approx(2.0)
        np.testing.assert_array_equal(normalized[-2:], observation[-2:])
        restored = normalizer.unnormalize_obs(normalized)
        assert isinstance(restored, np.ndarray)
        np.testing.assert_array_equal(restored[-2:], observation[-2:])
    finally:
        agent.close()


def test_reward_prediction_is_detached_and_disabled_head_is_explicit() -> None:
    config = small_config()
    data = synthetic_rollout(config)
    agent = create_agent(ValidationEnvironment(data), config)
    try:
        policy = agent.model.policy
        assert isinstance(policy, RacingActorCriticPolicy)
        prediction = policy.prediction_for_reward(torch.ones((1, data.observations.shape[-1]), requires_grad=True))
        assert not prediction.requires_grad and prediction.grad_fn is None
        assert agent.predict_trajectory(data.observations[0, 0]).shape == (10, 2)
    finally:
        agent.close()
    disabled = create_agent(ValidationEnvironment(data), replace(config, predictor=replace(config.predictor, enabled=False)))
    try:
        with pytest.raises(RuntimeError, match="disabled"):
            disabled.predict_trajectory(data.observations[0, 0])
    finally:
        disabled.close()
