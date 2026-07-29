"""Non-mutating preflight checks for mock and live RL training."""

from __future__ import annotations

import configparser
import importlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from assetto_corsa_rl.config import ProjectConfig


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
        return not any(check.severity == "error" for check in self.checks)


def assess_training_readiness(
    config: ProjectConfig,
    *,
    backend: str,
) -> TrainingReadinessReport:
    """Inspect required local state without starting or controlling the game."""

    if backend not in {"mock", "assetto-corsa"}:
        raise ValueError("backend must be mock or assetto-corsa")
    checks = [
        ReadinessCheck(
            "method_variant",
            "ok",
            f"named baseline: {config.ppo.method_variant}",
        )
    ]
    minimum_horizon_steps = int(5.0 * config.environment.control_frequency_hz) + 1
    if (
        config.ppo.predictor_loss_coefficient > 0
        and config.ppo.rollout_steps < minimum_horizon_steps
    ):
        checks.append(
            ReadinessCheck(
                "rollout_horizon",
                "error",
                "rollout_steps is shorter than the 5 s auxiliary target horizon",
            )
        )
    else:
        checks.append(
            ReadinessCheck(
                "rollout_horizon",
                "ok",
                f"{config.ppo.rollout_steps} steps cover the 5 s predictor horizon",
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
            "live_environment_backend",
            "error",
            "the write-capable live Gymnasium backend and automatic episode reset "
            "are not implemented yet; shared memory is currently diagnostic/read-only",
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
        checks.append(_predictor_quality_check(config, metrics_path))
    return checks


def _file_check(name: str, path: Path) -> ReadinessCheck:
    if path.is_file():
        return ReadinessCheck(name, "ok", f"{path} ({path.stat().st_size} bytes)")
    return ReadinessCheck(name, "error", f"required file is missing: {path}")


def _predictor_quality_check(
    config: ProjectConfig,
    metrics_path: Path,
) -> ReadinessCheck:
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        validation = payload["final_validation"]
        ade_m = float(validation["predictor_ade_m"])
        fde_m = float(validation["predictor_fde_m"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
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


def _installation_checks(config: ProjectConfig) -> list[ReadinessCheck]:
    installation = Path(config.assetto_corsa.installation_directory)
    bridge = (
        installation / "apps" / "python" / "assetto_corsa_rl_bridge" / "assetto_corsa_rl_bridge.py"
    )
    return [
        _file_check("assetto_corsa_executable", installation / "acs.exe"),
        _file_check("slip_bridge_app", bridge),
        _file_check("controls_configuration", config.assetto_corsa.controls_path),
    ]


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
                    "vigembus_driver",
                    "error",
                    f"vgamepad cannot connect to ViGEmBus: {error}",
                )
            )
        else:
            checks.append(
                ReadinessCheck(
                    "vigembus_driver",
                    "ok",
                    "vgamepad connected to the system ViGEmBus driver",
                )
            )
    return checks
