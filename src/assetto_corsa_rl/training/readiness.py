"""Non-mutating preflight checks for mock and live RL training."""

from __future__ import annotations

import configparser
import importlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from assetto_corsa_rl.ac_integration.ai_spline import AISplineFormatError, AssettoCorsaAISpline
from assetto_corsa_rl.ac_integration.launcher import (
    configured_content_paths,
    configured_fast_lane_path,
    configured_track_model_path,
)
from assetto_corsa_rl.ac_integration.track_assets import sha256_file
from assetto_corsa_rl.config import (
    ACTION_METHOD_VARIANT,
    BASELINE_METHOD_VARIANT,
    LINEAR_BC_X360_CONTROL_PROFILE,
    POLICY_PROTECTED_PPO_METHOD_VARIANT,
    STABILIZED_PPO_METHOD_VARIANT,
    UNIFORM_BC_ACTION_LOSS_METHOD_VARIANT,
    UNRECOVERABLE_PROGRESS_STALL_TERMINATION_METHOD_VARIANT,
    ProjectConfig,
)
from assetto_corsa_rl.data.bc_dataset import (
    ACTI_MOTEC_CALIBRATION_NAME,
    BC_DATASET_SCHEMA_VERSION,
    MOTEC_STEERING_CONVERSION_NAME,
    REFERENCE_FEATURE_CONTRACT,
)
from assetto_corsa_rl.models.predictor_target_normalization import (
    predictor_target_normalization_method_from_metadata,
    predictor_target_scale_m_from_metadata,
)
from assetto_corsa_rl.models.trajectory_contract import (
    LEGACY_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT,
)


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    """One machine-readable preflight result."""

    name: str
    severity: str
    message: str

    def __post_init__(self) -> None:
        if self.severity not in {"ok", "warning", "error"}:
            raise ValueError("readiness severity must be ok, warning, or error")


@dataclass(frozen=True, slots=True)
class TrainingReadinessReport:
    """All preflight findings for one requested backend."""

    backend: str
    checks: tuple[ReadinessCheck, ...]

    @property
    def is_ready(self) -> bool:
        """Return whether no blocking preflight error was found."""

        return not any(check.severity == "error" for check in self.checks)


def assess_training_readiness(
    config: ProjectConfig,
    *,
    backend: str,
) -> TrainingReadinessReport:
    """Inspect required local state without starting or controlling the game."""

    if backend not in {"mock", "assetto-corsa"}:
        raise ValueError("backend must be mock or assetto-corsa")
    stall_speed_kmh = config.environment.minimum_speed_mps * 3.6
    stall_debounce_s = (
        config.environment.stall_debounce_steps / config.environment.control_frequency_hz
    )
    stall_startup_grace_s = (
        config.environment.stall_startup_grace_steps / config.environment.control_frequency_hz
    )
    progress_stall_description = (
        f"; additionally terminates after {config.environment.progress_stall_window_s:.2f} s "
        "with less than "
        f"{config.environment.progress_stall_minimum_progress_m:.3f} m net reference progress"
        if config.environment.termination_method_variant
        == UNRECOVERABLE_PROGRESS_STALL_TERMINATION_METHOD_VARIANT
        else ""
    )
    checks = [
        ReadinessCheck(
            "method_variant",
            "ok",
            f"named baseline: {config.ppo.method_variant}",
        )
    ]
    checks.append(
        ReadinessCheck(
            "rollout_horizon",
            "ok",
            f"{config.ppo.rollout_steps} steps are used for the PPO rollout",
        )
    )
    checks.append(
        ReadinessCheck(
            "episode_end",
            "ok",
            (
                f"{config.environment.termination_method_variant}: stall detection arms after "
                f"driving above {stall_speed_kmh:.3f} km/h and terminates after "
                f"{stall_debounce_s:.2f} s continuously below it; a car that never drives off "
                f"terminates after {stall_startup_grace_s:.2f} s"
                f"{progress_stall_description}; "
                + (
                    f"time-limit truncation after {config.experiment.max_episode_steps} steps"
                    if config.experiment.truncate_on_max_episode_steps
                    else "time-limit truncation disabled"
                )
            ),
        )
    )
    checks.append(
        ReadinessCheck(
            "reference_reward",
            "ok",
            (
                f"{config.rewards.reference_tracking_method_variant}: "
                f"lateral_scale={config.rewards.reference_tracking_penalty_scale:.3f}, "
                f"free_distance={config.rewards.reference_tracking_free_distance_m:.3f} m, "
                f"heading_scale={config.rewards.reference_heading_penalty_scale:.3f}"
            ),
        )
    )
    if backend == "mock":
        checks.append(
            ReadinessCheck(
                "mock_backend",
                "ok",
                "deterministic mock dynamics require no external simulator",
            )
        )
        return TrainingReadinessReport(backend=backend, checks=tuple(checks))

    checks.extend(_artefact_checks(config))
    checks.extend(_installation_checks(config))
    checks.extend(_control_checks(config))
    checks.append(
        ReadinessCheck(
            "automatic_shifting",
            "ok" if config.assetto_corsa.automatic_shifting else "warning",
            (
                "launched sessions temporarily enable AUTO_SHIFTER/AUTO_CLUTCH; "
                "live validation verifies a forward gear and forward motion"
                if config.assetto_corsa.automatic_shifting
                else "automatic shifting is disabled; this is not the automatic-gearbox baseline"
            ),
        )
    )
    checks.append(
        ReadinessCheck(
            "live_environment_backend",
            "ok",
            "frame-synchronised shared-memory, X360 control, slip telemetry, and "
            "acknowledged CSP reset adapters are implemented",
        )
    )
    checks.append(
        ReadinessCheck(
            "live_step_timing",
            "ok",
            config.assetto_corsa.step_timing_method_variant,
        )
    )
    return TrainingReadinessReport(backend=backend, checks=tuple(checks))


def _artefact_checks(config: ProjectConfig) -> list[ReadinessCheck]:
    reference_path = Path(config.reference.artifact_path)
    dataset_path = Path(config.behavioral_cloning.dataset_artifact_path)
    checks = [
        _file_check("fixed_reference", reference_path),
        _file_check(
            "bc_dataset",
            dataset_path,
        ),
    ]
    checkpoint_path = Path(config.ppo.initial_checkpoint_path)
    checks.append(_file_check("bc_checkpoint", checkpoint_path))
    metrics_path = Path(config.ppo.initial_checkpoint_metrics_path)
    checks.append(_file_check("bc_metrics", metrics_path))
    if metrics_path.is_file():
        checks.append(_checkpoint_variant_check(config, metrics_path))
        checks.append(_predictor_quality_check(config, metrics_path))
    if reference_path.is_file() and dataset_path.is_file() and metrics_path.is_file():
        checks.append(
            _bc_reference_compatibility_check(
                reference_path=reference_path,
                dataset_path=dataset_path,
                metrics_path=metrics_path,
            )
        )
    return checks


def _bc_reference_compatibility_check(
    *,
    reference_path: Path,
    dataset_path: Path,
    metrics_path: Path,
) -> ReadinessCheck:
    """Require dataset and BC checkpoint metrics to name the active reference hash."""

    try:
        active_sha256 = sha256_file(reference_path)
        with np.load(dataset_path, allow_pickle=False) as archive:
            dataset_metadata = json.loads(str(archive["metadata_json"].item()))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        checkpoint_metadata = metrics["dataset_metadata"]
        dataset_schema = int(dataset_metadata["schema_version"])
        dataset_contract = str(dataset_metadata["reference_feature_contract"])
        dataset_sha256 = str(dataset_metadata["reference_sha256"])
        checkpoint_schema = int(checkpoint_metadata["schema_version"])
        checkpoint_contract = str(checkpoint_metadata["reference_feature_contract"])
        checkpoint_sha256 = str(checkpoint_metadata["reference_sha256"])
        dataset_calibration = str(dataset_metadata["calibration"])
        checkpoint_calibration = str(checkpoint_metadata["calibration"])
        dataset_steering = str(dataset_metadata["steering_conversion"])
        checkpoint_steering = str(checkpoint_metadata["steering_conversion"])
        dataset_predictor_normalization = predictor_target_normalization_method_from_metadata(
            dataset_metadata
        )
        checkpoint_predictor_normalization = predictor_target_normalization_method_from_metadata(
            checkpoint_metadata
        )
        dataset_predictor_scale_m = predictor_target_scale_m_from_metadata(dataset_metadata)
        checkpoint_predictor_scale_m = predictor_target_scale_m_from_metadata(checkpoint_metadata)
    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        OSError,
        UnicodeError,
    ) as error:
        return ReadinessCheck(
            "bc_reference_compatibility",
            "error",
            f"cannot verify BC/reference provenance; rebuild dataset and checkpoint: {error}",
        )

    matches = (
        dataset_schema == BC_DATASET_SCHEMA_VERSION
        and checkpoint_schema == BC_DATASET_SCHEMA_VERSION
        and dataset_contract == REFERENCE_FEATURE_CONTRACT
        and checkpoint_contract == REFERENCE_FEATURE_CONTRACT
        and dataset_sha256 == active_sha256
        and checkpoint_sha256 == active_sha256
        and dataset_calibration == ACTI_MOTEC_CALIBRATION_NAME
        and checkpoint_calibration == ACTI_MOTEC_CALIBRATION_NAME
        and dataset_steering == MOTEC_STEERING_CONVERSION_NAME
        and checkpoint_steering == MOTEC_STEERING_CONVERSION_NAME
        and dataset_predictor_normalization == checkpoint_predictor_normalization
        and np.allclose(
            dataset_predictor_scale_m,
            checkpoint_predictor_scale_m,
            rtol=0.0,
            atol=1e-6,
        )
    )
    return ReadinessCheck(
        "bc_reference_compatibility",
        "ok" if matches else "error",
        (
            f"BC dataset/checkpoint match active reference sha256={active_sha256}"
            if matches
            else "BC dataset/checkpoint use a stale reference or steering calibration; "
            "rebuild both before PPO"
        ),
    )


def _file_check(name: str, path: Path) -> ReadinessCheck:
    if path.is_file():
        return ReadinessCheck(name, "ok", f"{path} ({path.stat().st_size} bytes)")
    return ReadinessCheck(name, "error", f"required file is missing: {path}")


def _directory_check(name: str, path: Path) -> ReadinessCheck:
    if path.is_dir():
        return ReadinessCheck(name, "ok", f"{path}")
    return ReadinessCheck(name, "error", f"required directory is missing: {path}")


def _predictor_quality_check(
    config: ProjectConfig,
    metrics_path: Path,
) -> ReadinessCheck:
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        validation = payload["final_validation"]
        ade_m = float(validation["predictor_ade_m"])
        fde_m = float(validation["predictor_fde_m"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError, UnicodeError) as error:
        return ReadinessCheck(
            "predictor_quality",
            "error",
            f"cannot read validation predictor ADE/FDE from {metrics_path}: {error}",
        )
    maximum_ade = config.ppo.maximum_initial_predictor_ade_m
    maximum_fde = config.ppo.maximum_initial_predictor_fde_m
    exceeds_limit = ade_m > maximum_ade or fde_m > maximum_fde
    severity = (
        "error"
        if config.ppo.use_predicted_progress_reward and exceeds_limit
        else "warning"
        if exceeds_limit
        else "ok"
    )
    return ReadinessCheck(
        "predictor_quality",
        severity,
        f"validation ADE/FDE={ade_m:.3f}/{fde_m:.3f} m; configured maxima="
        f"{maximum_ade:.3f}/{maximum_fde:.3f} m",
    )


def _checkpoint_variant_check(
    config: ProjectConfig,
    metrics_path: Path,
) -> ReadinessCheck:
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        method_variant = str(payload["method_variant"])
        action_method_variant = str(payload["action_method_variant"])
        action_loss_method_variant = str(
            payload.get(
                "action_loss_method_variant",
                UNIFORM_BC_ACTION_LOSS_METHOD_VARIANT,
            )
        )
        steering_loss_weight = float(payload.get("steering_loss_weight", 1.0))
        steering_robustness_method = str(
            payload.get("steering_robustness_method_variant", "none_v1")
        )
        steering_robustness_weight = float(payload.get("steering_robustness_loss_weight", 1.0))
        predictor_target_normalization = str(
            payload.get(
                "predictor_target_normalization_method_variant",
                LEGACY_PREDICTOR_TARGET_NORMALIZATION_METHOD_VARIANT,
            )
        )
    except (KeyError, TypeError, json.JSONDecodeError, OSError, UnicodeError) as error:
        return ReadinessCheck(
            "checkpoint_variant",
            "error",
            f"cannot read method/action variant from {metrics_path}: {error}",
        )
    expected = config.ppo.method_variant
    acceptable_method_variants = {
        expected,
    }
    if expected in {
        STABILIZED_PPO_METHOD_VARIANT,
        POLICY_PROTECTED_PPO_METHOD_VARIANT,
    }:
        acceptable_method_variants.add(BASELINE_METHOD_VARIANT)
    matches = (
        method_variant in acceptable_method_variants
        and action_method_variant == ACTION_METHOD_VARIANT
        and action_loss_method_variant == config.behavioral_cloning.action_loss_method_variant
        and steering_loss_weight == config.behavioral_cloning.steering_loss_weight
        and steering_robustness_method
        == config.behavioral_cloning.steering_robustness_method_variant
        and steering_robustness_weight == config.behavioral_cloning.steering_robustness_loss_weight
        and predictor_target_normalization
        == config.behavioral_cloning.predictor_target_normalization_method_variant
    )
    message = (
        f"BC checkpoint variants match {method_variant}/{ACTION_METHOD_VARIANT}/"
        f"{action_loss_method_variant}/{steering_robustness_method}/"
        f"{predictor_target_normalization}"
        if matches
        else "BC checkpoint uses incompatible method/action/loss variants; rebuild it for "
        f"{expected!r}/{ACTION_METHOD_VARIANT!r}/"
        f"{config.behavioral_cloning.action_loss_method_variant!r}/"
        f"{config.behavioral_cloning.steering_robustness_method_variant!r}/"
        f"{config.behavioral_cloning.predictor_target_normalization_method_variant!r}"
    )
    return ReadinessCheck(
        "checkpoint_variant",
        "ok" if matches else "error",
        message,
    )


def _installation_checks(config: ProjectConfig) -> list[ReadinessCheck]:
    installation = Path(config.assetto_corsa.installation_directory)
    car_path, track_path = configured_content_paths(config)
    track_model_path = configured_track_model_path(config)
    fast_lane_path = configured_fast_lane_path(config)
    bridge = (
        installation / "apps" / "python" / "assetto_corsa_rl_bridge" / "assetto_corsa_rl_bridge.py"
    )
    collision_bridge = (
        installation / "apps" / "lua" / "assetto_corsa_rl_csp" / "assetto_corsa_rl_csp.lua"
    )
    checks = [
        _file_check("assetto_corsa_executable", installation / "acs.exe"),
        _file_check("slip_bridge_app", bridge),
        _file_check("collision_bridge_app", collision_bridge),
        _file_check("custom_shaders_patch", installation / "dwrite.dll"),
        _file_check("controls_configuration", config.assetto_corsa.controls_path),
        _directory_check("configured_car", car_path),
        _directory_check("configured_track_ui", track_path),
        _file_check("configured_track", track_model_path),
        _ai_spline_check(fast_lane_path),
    ]
    if bridge.is_file() and config.assetto_corsa.reset_backend == "python_ext_reset_car_v1":
        checks.append(_reset_bridge_capability_check(bridge))
    if collision_bridge.is_file():
        checks.append(_collision_bridge_capability_check(collision_bridge))
        if config.assetto_corsa.reset_backend in {
            "csp_hotlap_spawn_v1",
            "csp_hotlap_reference_speed_v1",
            "csp_reference_pose_speed_v2",
        }:
            checks.append(
                _hotlap_reset_bridge_capability_check(
                    collision_bridge,
                    method_variant=config.assetto_corsa.reset_backend,
                )
            )
    return checks


def _ai_spline_check(path: Path) -> ReadinessCheck:
    if not path.is_file():
        return ReadinessCheck("configured_fast_lane", "error", f"required file is missing: {path}")
    try:
        spline = AssettoCorsaAISpline.from_file(path)
    except AISplineFormatError as error:
        return ReadinessCheck("configured_fast_lane", "error", f"invalid {path}: {error}")
    return ReadinessCheck(
        "configured_fast_lane",
        "ok",
        f"{path} ({spline.distance_m.size} points, {spline.track_length_m:.3f} m, "
        f"sha256={spline.source_sha256})",
    )


def _reset_bridge_capability_check(path: Path) -> ReadinessCheck:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return ReadinessCheck(
            "reset_bridge_capability",
            "error",
            f"cannot inspect {path}: {error}",
        )
    capable = "COMMAND_MAGIC" in source and "ext_resetCar" in source
    return ReadinessCheck(
        "reset_bridge_capability",
        "ok" if capable else "error",
        (
            "installed bridge supports the versioned acknowledged CSP reset command"
            if capable
            else "installed bridge is telemetry-only; install the updated reset-capable app"
        ),
    )


def _collision_bridge_capability_check(path: Path) -> ReadinessCheck:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return ReadinessCheck(
            "collision_bridge_capability",
            "error",
            f"cannot inspect {path}: {error}",
        )
    capable = "ACRL_CSP_STATE_V1" in source and "ac.onCarCollision(0" in source
    return ReadinessCheck(
        "collision_bridge_capability",
        "ok" if capable else "error",
        (
            "installed CSP Lua app publishes latched car-0 collision events"
            if capable
            else "installed CSP Lua app lacks the required latched collision callback"
        ),
    )


def _hotlap_reset_bridge_capability_check(
    path: Path,
    *,
    method_variant: str = "csp_hotlap_spawn_v1",
) -> ReadinessCheck:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return ReadinessCheck(
            "hotlap_reset_bridge_capability",
            "error",
            f"cannot inspect {path}: {error}",
        )
    shared_capabilities = (
        "physics.teleportCarTo(0, ac.SpawnSet.HotlapStart)" in source
        and "physics.resetCarState(0)" in source
    )
    if method_variant == "csp_reference_pose_speed_v2":
        capable = (
            shared_capabilities
            and "ACRL_HOTLAP_RESET_COMMAND_V3" in source
            and "physics.setCarPosition" in source
            and "pendingReset.targetHeadingRad" in source
            and "local placementDirection = -velocityDirection" in source
            and "pendingReset.posePlaced" in source
            and "resetCollisionSuppressionDuration" in source
            and "collisionSuppressionRemaining" in source
            and "physics.setWheelAngularVelocity" in source
            and "physics.setCarVelocity" in source
            and "pendingReset.velocityHold" in source
        )
        success_message = (
            "installed CSP Lua app supports reference-pose-aligned, human-speed HOTLAP_START resets"
        )
        failure_message = (
            "installed CSP Lua app lacks reference-pose/speed HOTLAP_START reset support"
        )
    elif method_variant == "csp_hotlap_reference_speed_v1":
        capable = (
            shared_capabilities
            and "ACRL_HOTLAP_RESET_COMMAND_V2" in source
            and "ac.getCar(0)" in source
            and "physics.setWheelAngularVelocity" in source
            and "physics.setCarVelocity" in source
            and "referenceSpeedHoldDuration" in source
            and "pendingReset.velocityHold" in source
        )
        success_message = (
            "installed CSP Lua app supports multi-frame human-reference-speed HOTLAP_START resets"
        )
        failure_message = (
            "installed CSP Lua app lacks multi-frame reference-speed HOTLAP_START reset support"
        )
    else:
        capable = (
            shared_capabilities
            and "ACRL_HOTLAP_RESET_COMMAND_V1" in source
            and "physics.engageGear(0, 1)" in source
        )
        success_message = (
            "installed CSP Lua app supports acknowledged deterministic HOTLAP_START "
            "resets with first-gear engagement"
        )
        failure_message = (
            "installed CSP Lua app lacks deterministic HOTLAP_START reset/gear support"
        )
    return ReadinessCheck(
        "hotlap_reset_bridge_capability",
        "ok" if capable else "error",
        success_message if capable else failure_message,
    )


def _control_checks(config: ProjectConfig) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    controls_path = config.assetto_corsa.controls_path
    if controls_path.is_file():
        parser = configparser.ConfigParser(
            interpolation=None,
            strict=False,
        )
        try:
            parser.read(controls_path, encoding="utf-8")
            input_method = parser.get("HEADER", "INPUT_METHOD", fallback="").upper()
        except configparser.Error as error:
            checks.append(
                ReadinessCheck(
                    "analog_input_method",
                    "error",
                    f"cannot parse {controls_path}: {error}",
                )
            )
        else:
            checks.append(
                ReadinessCheck(
                    "analog_input_method",
                    "ok" if input_method == "X360" else "error",
                    (
                        "Assetto Corsa uses the X360 analog input profile"
                        if input_method == "X360"
                        else f"INPUT_METHOD={input_method or 'missing'}; vgamepad requires X360"
                    ),
                )
            )
            if input_method == "X360":
                linear_profile = (
                    config.assetto_corsa.x360_control_profile == LINEAR_BC_X360_CONTROL_PROFILE
                )
                checks.append(
                    ReadinessCheck(
                        "x360_control_profile",
                        "ok",
                        (
                            "launched sessions temporarily apply the BC-calibrated linear "
                            "X360 steering profile; steering output gain="
                            f"{config.assetto_corsa.steering_output_gain:.3f}"
                            if linear_profile
                            else "launched sessions preserve the user's X360 steering curve; "
                            "steering output gain="
                            f"{config.assetto_corsa.steering_output_gain:.3f}"
                        ),
                    )
                )
    package_available = importlib.util.find_spec("vgamepad") is not None
    checks.append(
        ReadinessCheck(
            "vgamepad_python_package",
            "ok" if package_available else "error",
            (
                "vgamepad Python package is importable"
                if package_available
                else "vgamepad is not installed in the project environment"
            ),
        )
    )
    if package_available:
        try:
            importlib.import_module("vgamepad")
        except Exception as error:
            checks.append(
                ReadinessCheck(
                    "vgamepad_import",
                    "error",
                    f"vgamepad cannot be imported: {error}",
                )
            )
        else:
            checks.append(
                ReadinessCheck(
                    "vgamepad_import",
                    "ok",
                    "vgamepad imports; the virtual device is created only when training starts",
                )
            )
    return checks
