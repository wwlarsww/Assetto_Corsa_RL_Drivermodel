from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from assetto_corsa_rl.ac_integration.track_assets import sha256_file
from assetto_corsa_rl.config import (
    ACTION_METHOD_VARIANT,
    BASELINE_METHOD_VARIANT,
    ProjectConfig,
)
from assetto_corsa_rl.features.schema import OBSERVATION_FEATURE_NAMES
from assetto_corsa_rl.models.driver_network import DriverNetwork
from assetto_corsa_rl.models.trajectory_contract import BEZIER_TRAJECTORY_CODEC
from assetto_corsa_rl.training import ppo


def test_initial_log_std_after_bc_overrides_checkpoint_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PPO must own exploration instead of inheriting BC checkpoint log std."""

    checkpoint = tmp_path / "bc_checkpoint.pt"
    checkpoint.write_bytes(b"stub")
    reference = tmp_path / "reference.npz"
    reference.write_bytes(b"reference")
    base = ProjectConfig()
    config = replace(
        base,
        ppo=replace(
            base.ppo,
            initial_checkpoint_path=str(checkpoint),
            initial_log_std_after_bc=-1.0,
        ),
        reference=replace(base.reference, artifact_path=str(reference)),
    )

    def fake_load_checkpoint(
        source: Path,
        *,
        model: DriverNetwork,
        map_location: torch.device,
    ) -> tuple[None, dict[str, object]]:
        assert source == checkpoint
        assert map_location == torch.device("cpu")
        with torch.no_grad():
            model.action_log_std.fill_(-3.0)
        return None, {
            "method_variant": BASELINE_METHOD_VARIANT,
            "action_method_variant": ACTION_METHOD_VARIANT,
            "observation_feature_names": list(OBSERVATION_FEATURE_NAMES),
            "dataset_metadata": {"reference_sha256": sha256_file(reference)},
            "project": {
                "model": {"trajectory_codec": BEZIER_TRAJECTORY_CODEC},
                "behavioral_cloning": {
                    "action_loss_method_variant": (
                        config.behavioral_cloning.action_loss_method_variant
                    ),
                    "steering_loss_weight": config.behavioral_cloning.steering_loss_weight,
                },
            },
        }

    monkeypatch.setattr(ppo, "load_checkpoint", fake_load_checkpoint)
    model, _, initial_checkpoint = ppo._initialise_training_model(
        config,
        torch.device("cpu"),
    )

    assert initial_checkpoint == str(checkpoint)
    torch.testing.assert_close(
        model.action_log_std,
        torch.full_like(model.action_log_std, -1.0),
    )
