from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from assetto_corsa_rl.ac_integration.track_assets import sha256_file
from assetto_corsa_rl.config import (
    BASELINE_METHOD_VARIANT,
    STABILIZED_PPO_METHOD_VARIANT,
    AssettoCorsaRuntimeConfig,
    PPOConfig,
    ProjectConfig,
)
from assetto_corsa_rl.training.readiness import (
    _bc_reference_compatibility_check,
    _checkpoint_variant_check,
    _collision_bridge_capability_check,
    _hotlap_reset_bridge_capability_check,
    assess_training_readiness,
)


def test_mock_readiness_is_self_contained() -> None:
    config = ProjectConfig.from_toml("configs/mock.toml")
    report = assess_training_readiness(config, backend="mock")
    assert report.is_ready
    assert all(check.severity == "ok" for check in report.checks)


def test_legacy_delta_action_checkpoint_is_rejected(tmp_path: Path) -> None:
    metrics = tmp_path / "summary.json"
    metrics.write_text(
        json.dumps({"method_variant": "fixed_reference_no_promp_v1"}),
        encoding="utf-8",
    )
    check = _checkpoint_variant_check(ProjectConfig(), metrics)
    assert check.severity == "error"
    assert "method/action variant" in check.message


def test_baseline_checkpoint_is_accepted_for_stabilized_ppo(tmp_path: Path) -> None:
    metrics = tmp_path / "summary.json"
    metrics.write_text(
        json.dumps(
            {
                "method_variant": BASELINE_METHOD_VARIANT,
                "action_method_variant": "absolute_bounded_v1",
                "final_validation": {
                    "predictor_ade_m": 1.0,
                    "predictor_fde_m": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    config = ProjectConfig()
    assert config.ppo.method_variant == STABILIZED_PPO_METHOD_VARIANT
    check = _checkpoint_variant_check(config, metrics)
    assert check.severity == "ok"
    assert "BC checkpoint variants match" in check.message


def test_bc_reference_compatibility_requires_matching_dataset_and_checkpoint(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npz"
    reference.write_bytes(b"active-reference")
    reference_sha256 = sha256_file(reference)
    metadata = {
        "schema_version": 6,
        "reference_feature_contract": "sha256_bound_time_parameterized_fixed_reference_v1",
        "reference_sha256": reference_sha256,
        "calibration": "acti_motec_ai_spline_absolute_steering_left_v2",
        "steering_conversion": "motec_positive_right_to_policy_positive_left_v1",
    }
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(dataset, metadata_json=np.asarray(json.dumps(metadata)))
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"dataset_metadata": metadata}), encoding="utf-8")

    matching = _bc_reference_compatibility_check(
        reference_path=reference,
        dataset_path=dataset,
        metrics_path=metrics,
    )
    assert matching.severity == "ok"

    stale_metadata = {**metadata, "reference_sha256": "0" * 64}
    metrics.write_text(
        json.dumps({"dataset_metadata": stale_metadata}),
        encoding="utf-8",
    )
    stale = _bc_reference_compatibility_check(
        reference_path=reference,
        dataset_path=dataset,
        metrics_path=metrics,
    )
    assert stale.severity == "error"
    assert "stale reference" in stale.message


def test_collision_bridge_capability_requires_latched_callback(tmp_path: Path) -> None:
    path = tmp_path / "bridge.lua"
    path.write_text("-- old bridge\n", encoding="utf-8")
    assert _collision_bridge_capability_check(path).severity == "error"
    path.write_text(
        "local magic = 'ACRL_CSP_STATE_V1'\nac.onCarCollision(0, function() end)\n",
        encoding="utf-8",
    )
    assert _collision_bridge_capability_check(path).severity == "ok"


def test_hotlap_reset_capability_requires_first_gear_engagement(tmp_path: Path) -> None:
    path = tmp_path / "bridge.lua"
    reset_source = (
        "ACRL_HOTLAP_RESET_COMMAND_V1\n"
        "physics.teleportCarTo(0, ac.SpawnSet.HotlapStart)\n"
        "physics.resetCarState(0)\n"
    )
    path.write_text(reset_source, encoding="utf-8")
    assert _hotlap_reset_bridge_capability_check(path).severity == "error"
    path.write_text(reset_source + "physics.engageGear(0, 1)\n", encoding="utf-8")
    assert _hotlap_reset_bridge_capability_check(path).severity == "ok"


def test_reference_speed_reset_capability_requires_velocity_and_wheel_sync(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bridge.lua"
    source = (
        "ACRL_HOTLAP_RESET_COMMAND_V2\n"
        "physics.teleportCarTo(0, ac.SpawnSet.HotlapStart)\n"
        "physics.resetCarState(0)\n"
    )
    path.write_text(source, encoding="utf-8")
    assert (
        _hotlap_reset_bridge_capability_check(
            path,
            method_variant="csp_hotlap_reference_speed_v1",
        ).severity
        == "error"
    )
    path.write_text(
        source
        + "physics.setWheelAngularVelocity(0, ac.Wheel.All, 0)\n"
        + "physics.setCarVelocity(0, car.look * startSpeedMps)\n",
        encoding="utf-8",
    )
    assert (
        _hotlap_reset_bridge_capability_check(
            path,
            method_variant="csp_hotlap_reference_speed_v1",
        ).severity
        == "error"
    )
    path.write_text(
        source
        + "ac.getCar(0)\n"
        + "physics.setWheelAngularVelocity(0, ac.Wheel.All, 0)\n"
        + "physics.setCarVelocity(0, car.look * startSpeedMps)\n",
        encoding="utf-8",
    )
    assert (
        _hotlap_reset_bridge_capability_check(
            path,
            method_variant="csp_hotlap_reference_speed_v1",
        ).severity
        == "error"
    )
    path.write_text(
        source
        + "ac.getCar(0)\n"
        + "physics.setWheelAngularVelocity(0, ac.Wheel.All, 0)\n"
        + "physics.setCarVelocity(0, car.look * startSpeedMps)\n"
        + "local referenceSpeedHoldDuration = 0.25\n"
        + "pendingReset.velocityHold = pendingReset.velocityHold + deltaTime\n",
        encoding="utf-8",
    )
    assert (
        _hotlap_reset_bridge_capability_check(
            path,
            method_variant="csp_hotlap_reference_speed_v1",
        ).severity
        == "ok"
    )


def test_reference_pose_reset_capability_requires_v3_position_and_heading(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bridge.lua"
    path.write_text(
        "ACRL_HOTLAP_RESET_COMMAND_V3\n"
        "physics.teleportCarTo(0, ac.SpawnSet.HotlapStart)\n"
        "physics.resetCarState(0)\n"
        "physics.setCarPosition(0, position, direction)\n"
        "pendingReset.targetHeadingRad\n"
        "local placementDirection = -velocityDirection\n"
        "pendingReset.posePlaced = true\n"
        "local resetCollisionSuppressionDuration = 1.0\n"
        "collisionSuppressionRemaining = 0\n"
        "physics.setWheelAngularVelocity(0, ac.Wheel.All, 0)\n"
        "physics.setCarVelocity(0, velocity)\n"
        "pendingReset.velocityHold = 1\n",
        encoding="utf-8",
    )

    check = _hotlap_reset_bridge_capability_check(
        path,
        method_variant="csp_reference_pose_speed_v2",
    )

    assert check.severity == "ok"


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
    assert "checkpoint_variant" in errors
    assert "analog_input_method" in errors
    assert "collision_bridge_app" in errors
    backend_check = next(
        check for check in report.checks if check.name == "live_environment_backend"
    )
    assert backend_check.severity == "ok"


def test_live_readiness_rejects_metadata_without_track_models(tmp_path: Path) -> None:
    installation = tmp_path / "assettocorsa"
    (installation / "content/tracks/monza/ui").mkdir(parents=True)
    base = ProjectConfig()
    config = replace(
        base,
        assetto_corsa=replace(
            base.assetto_corsa,
            installation_directory=str(installation),
        ),
    )

    report = assess_training_readiness(config, backend="assetto-corsa")

    ui_check = next(check for check in report.checks if check.name == "configured_track_ui")
    track_check = next(check for check in report.checks if check.name == "configured_track")
    assert ui_check.severity == "ok"
    assert track_check.severity == "error"
    assert "models.ini" in track_check.message


def test_live_readiness_distinguishes_package_from_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProjectConfig()
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    missing_package = assess_training_readiness(config, backend="assetto-corsa")
    package_check = next(
        check for check in missing_package.checks if check.name == "vgamepad_python_package"
    )
    assert package_check.severity == "error"
    assert all(check.name != "vigembus_driver" for check in missing_package.checks)

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())

    def missing_driver(_name: str) -> None:
        raise RuntimeError("VIGEM_ERROR_BUS_NOT_FOUND")

    monkeypatch.setattr(importlib, "import_module", missing_driver)
    missing_bus = assess_training_readiness(config, backend="assetto-corsa")
    import_check = next(check for check in missing_bus.checks if check.name == "vgamepad_import")
    assert import_check.severity == "error"
    assert "BUS_NOT_FOUND" in import_check.message
