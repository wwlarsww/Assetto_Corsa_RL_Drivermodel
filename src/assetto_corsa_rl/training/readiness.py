"""Non-mutating preflight checks for mock and live RL training."""

from __future__ import annotations

import configparser
import importlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from assetto_corsa_rl.ac_integration.ai_spline import AISplineFormatError, AssettoCorsaAISpline
from assetto_corsa_rl.ac_integration.launcher import (
    configured_content_paths,
    configured_fast_lane_path,
    configured_track_model_path,
)
from assetto_corsa_rl.config import (
    ACTION_METHOD_VARIANT,
    BASELINE_METHOD_VARIANT,
    STABILIZED_PPO_METHOD_VARIANT,
    ProjectConfig,
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
                f"terminates after {stall_startup_grace_s:.2f} s; "
                + (
                    f"time-limit truncation after {config.experiment.max_episode_steps} steps"
                    if config.experiment.truncate_on_max_episode_steps
                    else "time-limit truncation disabled"
                )
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
                "live validation verifies first gear and forward motion"
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
    return TrainingReadinessReport(backend=backend, checks=tuple(checks))


def _artefact_checks(config: ProjectConfig) -> list[ReadinessCheck]:
    checks = [
        _file_check("fixed_reference", Path(config.reference.artifact_path)),
        _file_check(
            "bc_dataset",
            Path(config.behavioral_cloning.dataset_artifact_path),
        ),
    ]
    checkpoint_path = Path(config.ppo.initial_checkpoint_path)
    checks.append(_file_check("bc_checkpoint", checkpoint_path))
    metrics_path = Path(config.ppo.initial_checkpoint_metrics_path)
    checks.append(_file_check("bc_metrics", metrics_path))
    if metrics_path.is_file():
        checks.append(_checkpoint_variant_check(config, metrics_path))
        checks.append(_predictor_quality_check(config, metrics_path))
    return checks


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
    if expected == STABILIZED_PPO_METHOD_VARIANT:
        acceptable_method_variants.add(BASELINE_METHOD_VARIANT)
    matches = (
        method_variant in acceptable_method_variants
        and action_method_variant == ACTION_METHOD_VARIANT
    )
    message = (
        f"BC checkpoint variants match {method_variant}/{ACTION_METHOD_VARIANT}"
        if matches
        else "BC checkpoint uses incompatible method/action variants; rebuild it for "
        f"{expected!r}/{ACTION_METHOD_VARIANT!r}"
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
        if config.assetto_corsa.reset_backend == "csp_hotlap_spawn_v1":
            checks.append(_hotlap_reset_bridge_capability_check(collision_bridge))
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


def _hotlap_reset_bridge_capability_check(path: Path) -> ReadinessCheck:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return ReadinessCheck(
            "hotlap_reset_bridge_capability",
            "error",
            f"cannot inspect {path}: {error}",
        )
    capable = (
        "ACRL_HOTLAP_RESET_COMMAND_V1" in source
        and "physics.teleportCarTo(0, ac.SpawnSet.HotlapStart)" in source
        and "physics.resetCarState(0)" in source
        and "physics.engageGear(0, 1)" in source
    )
    return ReadinessCheck(
        "hotlap_reset_bridge_capability",
        "ok" if capable else "error",
        (
            "installed CSP Lua app supports acknowledged deterministic HOTLAP_START resets "
            "with first-gear engagement"
            if capable
            else "installed CSP Lua app lacks deterministic HOTLAP_START reset/gear support"
        ),
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
