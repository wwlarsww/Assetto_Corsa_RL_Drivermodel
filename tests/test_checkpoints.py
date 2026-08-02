from __future__ import annotations

from pathlib import Path

import torch

from assetto_corsa_rl.models.driver_network import DriverNetwork
from assetto_corsa_rl.training.checkpoints import (
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_round_trip_restores_training_state(tmp_path: Path) -> None:
    model = DriverNetwork(
        feature_hidden_sizes=(8,),
        policy_hidden_sizes=(4,),
        value_hidden_sizes=(4,),
        predictor_hidden_sizes=(4,),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    state = TrainingState(iteration=3, environment_steps=2048, policy_version=4)
    path = tmp_path / "checkpoint.pt"
    original = model.action_log_std.detach().clone()

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_state=state,
        configuration={"seed": 7},
    )
    with torch.no_grad():
        model.action_log_std.add_(10.0)

    restored_state, configuration = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
    )
    torch.testing.assert_close(model.action_log_std, original)
    assert restored_state == state
    assert configuration == {"seed": 7}
