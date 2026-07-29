from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from assetto_corsa_rl.config import AssettoCorsaRuntimeConfig, PPOConfig, ProjectConfig
from assetto_corsa_rl.training.readiness import assess_training_readiness


def test_mock_readiness_is_self_contained() -> None:
    config = ProjectConfig.from_toml("configs/mock.toml")
    report = assess_training_readiness(config, backend="mock")
    assert report.is_ready
    assert all(check.severity == "ok" for check in report.checks)


def test_live_readiness_reports_predictor_and_control_blockers(
    tmp_path: Path,
) -> None:
    installation = tmp_path / "assettocorsa"
    bridge = installation / "apps/python/assetto_corsa_rl_bridge"
    bridge.mkdir(parents=True)
    (installation / "acs.exe").write_bytes(b"test")
    (bridge / "assetto_corsa_rl_bridge.py").write_text("# test\n", encoding="utf-8")
    controls = tmp_path / "controls.ini"
    controls.write_text("[HEADER]\nINPUT_METHOD=KEYBOARD\n", encoding="utf-8")
    reference = tmp_path / "reference.npz"
    dataset = tmp_path / "dataset.npz"
    checkpoint = tmp_path / "checkpoint.pt"
    for path in (reference, dataset, checkpoint):
        path.write_bytes(b"test")
    metrics = tmp_path / "summary.json"
    metrics.write_text(
        json.dumps(
            {
                "final_validation": {
                    "predictor_ade_m": 30.0,
                    "predictor_fde_m": 50.0,
                }
            }
        ),
        encoding="utf-8",
    )
    base = ProjectConfig()
    config = replace(
        base,
        reference=replace(base.reference, artifact_path=str(reference)),
        behavioral_cloning=replace(
            base.behavioral_cloning,
            dataset_artifact_path=str(dataset),
        ),
        ppo=PPOConfig(
            rollout_steps=256,
            minibatch_size=128,
            use_predicted_progress_reward=True,
            initial_checkpoint_path=str(checkpoint),
            initial_checkpoint_metrics_path=str(metrics),
            maximum_initial_predictor_ade_m=25.0,
            maximum_initial_predictor_fde_m=40.0,
        ),
        assetto_corsa=AssettoCorsaRuntimeConfig(
            installation_directory=str(installation),
            controls_configuration_path=str(controls),
        ),
    )
    report = assess_training_readiness(config, backend="assetto-corsa")
    assert not report.is_ready
    errors = {check.name for check in report.checks if check.severity == "error"}
    assert "predictor_quality" in errors
    assert "analog_input_method" in errors
    assert "live_environment_backend" in errors
