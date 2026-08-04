import math
from dataclasses import replace

import pytest
import torch

from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.training.ppo import (
    KLBandController,
    _apply_post_bc_exploration,
    build_driver_network,
)


def test_post_bc_exploration_override_changes_loaded_policy_sigma() -> None:
    config = ProjectConfig()
    model = build_driver_network(config)

    applied = _apply_post_bc_exploration(
        model,
        config,
        {"training_stage": "behavioral_cloning"},
    )

    assert applied
    expected_sigma = torch.full_like(
        model.action_log_std,
        math.exp(config.ppo.initial_log_std_after_bc),
    )
    torch.testing.assert_close(model.action_log_std.exp(), expected_sigma)


def test_post_bc_exploration_override_ignores_ppo_checkpoint() -> None:
    config = ProjectConfig()
    model = build_driver_network(config)
    original = model.action_log_std.detach().clone()

    applied = _apply_post_bc_exploration(model, config, {"training_stage": "ppo"})

    assert not applied
    torch.testing.assert_close(model.action_log_std, original)


def test_kl_band_controller_decreases_holds_and_recovers_learning_rate() -> None:
    base = ProjectConfig()
    config = replace(
        base,
        ppo=replace(
            base.ppo,
            learning_rate=1e-3,
            minimum_learning_rate=1e-4,
        ),
    )
    model = build_driver_network(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.learning_rate)
    controller = KLBandController(config)

    high = controller.update(controller.upper_bound + 0.01, optimizer)
    in_band = controller.update(controller.target, optimizer)
    low = controller.update(0.0, optimizer)

    assert high.status == "above_band"
    assert high.adjustment == "decrease"
    assert high.learning_rate == pytest.approx(5e-4)
    assert in_band.adjustment == "hold"
    assert in_band.learning_rate == pytest.approx(5e-4)
    assert low.status == "below_band"
    assert low.adjustment == "increase"
    assert low.learning_rate == pytest.approx(5.5e-4)
