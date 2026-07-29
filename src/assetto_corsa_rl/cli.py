"""Command-line entry points for mock verification and future training tools."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from assetto_corsa_rl.ac_integration.diagnostics import (
    TelemetryDiagnosticSummary,
    capture_telemetry,
    summarise_capture,
    write_diagnostic_artifacts,
)
from assetto_corsa_rl.ac_integration.shared_memory import (
    AssettoCorsaSharedMemoryReader,
    AssettoCorsaTelemetryUnavailableError,
)
from assetto_corsa_rl.ac_integration.slip_bridge import (
    SLIP_BRIDGE_PORT,
    AssettoCorsaSlipBridgeFileReader,
    AssettoCorsaSlipBridgeReceiver,
)
from assetto_corsa_rl.ac_integration.slip_diagnostics import (
    SlipDiagnosticCapture,
    SlipDiagnosticSummary,
    SlipPacketReader,
    capture_slip_telemetry,
    summarise_slip_capture,
    write_slip_diagnostic_artifacts,
)
from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.data.assetto_corsa_gym import (
    DATASET_REVISION,
    SOURCE_REVISION,
    download_initial_dataset,
)
from assetto_corsa_rl.data.bc_dataset import (
    BehavioralCloningDataset,
    build_bc_dataset,
)
from assetto_corsa_rl.data.fixed_reference import build_fixed_reference
from assetto_corsa_rl.envs.factory import create_mock_environment
from assetto_corsa_rl.experiment import ExperimentRecorder
from assetto_corsa_rl.policies import MockDriverPolicy
from assetto_corsa_rl.training.behavioral_cloning import train_behavioral_cloning
from assetto_corsa_rl.training.ppo import train_ppo
from assetto_corsa_rl.training.readiness import (
    TrainingReadinessReport,
    assess_training_readiness,
)

LOGGER = logging.getLogger("assetto_corsa_rl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ac-rl-driver",
        description="Assetto Corsa RL driver research scaffold.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    mock = subparsers.add_parser(
        "mock-run",
        help="Run a deterministic episode without Assetto Corsa.",
    )
    mock.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mock.toml"),
        help="TOML experiment configuration.",
    )
    mock.add_argument(
        "--no-record",
        action="store_true",
        help="Run without writing JSON/CSV artefacts.",
    )
    download = subparsers.add_parser(
        "download-data",
        help="Download the pinned human F317/Barcelona subset.",
    )
    download.add_argument(
        "--destination",
        type=Path,
        default=Path("data/external/assetto_corsa_gym"),
        help="Local dataset root.",
    )
    download.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/assetto_corsa_gym_f317_barcelona.json"),
        help="Content-hash manifest written after verification.",
    )
    build_reference = subparsers.add_parser(
        "build-reference",
        help="Build one fixed reference from the downloaded human recordings.",
    )
    build_reference.add_argument(
        "--config",
        type=Path,
        default=Path("configs/f317_barcelona_fixed.toml"),
        help="Fixed-reference experiment configuration.",
    )
    build_bc = subparsers.add_parser(
        "build-bc-dataset",
        help="Export session-disjoint 77-feature BC tensors from human laps.",
    )
    build_bc.add_argument(
        "--config",
        type=Path,
        default=Path("configs/f317_barcelona_fixed.toml"),
        help="Fixed-reference BC configuration.",
    )
    build_bc.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the configured BC dataset artefact path.",
    )
    build_bc.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Optional bounded development subset; omit for the baseline dataset.",
    )
    train_bc = subparsers.add_parser(
        "bc-train",
        help="Jointly warm-start the policy and trajectory predictor.",
    )
    train_bc.add_argument(
        "--config",
        type=Path,
        default=Path("configs/f317_barcelona_fixed.toml"),
        help="Fixed-reference BC configuration.",
    )
    train_bc.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Override the configured BC dataset artefact path.",
    )
    train_bc.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device; CPU is the reproducible baseline.",
    )
    rl_ready = subparsers.add_parser(
        "rl-ready",
        help="Check whether mock or live PPO training can start safely.",
    )
    rl_ready.add_argument(
        "--config",
        type=Path,
        default=Path("configs/f317_barcelona_fixed.toml"),
        help="PPO experiment configuration.",
    )
    rl_ready.add_argument(
        "--backend",
        choices=("mock", "assetto-corsa"),
        default="assetto-corsa",
        help="Training backend to validate.",
    )
    rl_train = subparsers.add_parser(
        "rl-train",
        help="Train the fixed-reference policy with synchronous PPO.",
    )
    rl_train.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mock.toml"),
        help="PPO experiment configuration.",
    )
    rl_train.add_argument(
        "--backend",
        choices=("mock", "assetto-corsa"),
        default="mock",
        help="Training backend. Live mode stays gated until all checks pass.",
    )
    rl_train.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device; CPU is the reproducible baseline.",
    )
    diagnose = subparsers.add_parser(
        "telemetry-diagnose",
        help="Validate the read-only Assetto Corsa shared-memory transport.",
    )
    diagnose.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Capture duration in seconds.",
    )
    diagnose.add_argument(
        "--poll-hz",
        type=float,
        default=100.0,
        help="Shared-memory polling frequency.",
    )
    diagnose.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory; defaults to a timestamped directory below runs/.",
    )
    diagnose.add_argument(
        "--no-record",
        action="store_true",
        help="Validate without writing JSON/CSV artefacts.",
    )
    slip_diagnose = subparsers.add_parser(
        "slip-diagnose",
        help="Validate the local SlipRatio/SlipAngle Python-app bridge.",
    )
    slip_diagnose.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Capture duration in seconds.",
    )
    slip_diagnose.add_argument(
        "--transport",
        choices=("file", "udp"),
        default="file",
        help="Bridge transport; vanilla Assetto Corsa requires file.",
    )
    slip_diagnose.add_argument(
        "--bridge-file",
        type=Path,
        default=None,
        help="Override the default per-user bridge file.",
    )
    slip_diagnose.add_argument(
        "--port",
        type=int,
        default=SLIP_BRIDGE_PORT,
        help="Loopback UDP port when --transport=udp.",
    )
    slip_diagnose.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory; defaults to a timestamped directory below runs/.",
    )
    slip_diagnose.add_argument(
        "--no-record",
        action="store_true",
        help="Validate without writing JSON/CSV artefacts.",
    )
    slip_diagnose.add_argument(
        "--with-shared-memory",
        action="store_true",
        help="Pair each tyre packet with controls and motion from shared memory.",
    )
    return parser


def seed_everything(seed: int) -> None:
    """Seed standard RNGs used by this scaffold."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_mock(config: ProjectConfig, *, record: bool, command: list[str]) -> dict[str, object]:
    """Exercise the complete environment loop with a simple controller."""

    seed_everything(config.experiment.seed)
    environment = create_mock_environment(config)
    policy = MockDriverPolicy(
        pedal_delta_limit=config.actions.pedal_delta_limit,
        steering_delta_limit=config.actions.steering_delta_limit,
    )
    recorder = (
        ExperimentRecorder.create(
            output_directory=config.experiment.output_directory,
            config=config,
            command=command,
        )
        if record
        else None
    )

    total_reward = 0.0
    step = 0
    terminated = False
    truncated = False
    final_info: dict[str, object] = {}
    try:
        observation, _ = environment.reset(seed=config.experiment.seed)
        while not (terminated or truncated):
            action = policy.act(observation)
            observation, reward, terminated, truncated, info = environment.step(action)
            total_reward += reward
            step += 1
            final_info = info
            if recorder is not None:
                telemetry = info["telemetry"]
                reward_terms = info["reward_terms"]
                recorder.record_step(
                    {
                        "step": step,
                        **telemetry,
                        **reward_terms,
                        "episode_reward": total_reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "termination_reason": info["termination_reason"] or "",
                    }
                )
    finally:
        environment.close()

    telemetry = final_info.get("telemetry", {})
    summary: dict[str, object] = {
        "seed": config.experiment.seed,
        "steps": step,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": final_info.get("termination_reason"),
        "final_telemetry": telemetry,
    }
    if recorder is not None:
        recorder.write_summary(summary)
        recorder.close()
        summary["run_directory"] = str(recorder.run_directory)
    return summary


def run_telemetry_diagnostic(
    *,
    duration_s: float,
    poll_frequency_hz: float,
    output_directory: Path | None,
    record: bool,
) -> tuple[TelemetryDiagnosticSummary, Path | None]:
    """Capture live read-only telemetry and optionally persist diagnostics."""

    with AssettoCorsaSharedMemoryReader.connect() as reader:
        capture = capture_telemetry(
            reader,
            duration_s=duration_s,
            poll_frequency_hz=poll_frequency_hz,
        )
    summary = summarise_capture(capture)
    destination: Path | None = None
    if record:
        destination = output_directory or Path("runs") / (
            "telemetry-diagnostic-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        )
        write_diagnostic_artifacts(destination, capture, summary)
    return summary, destination


def run_slip_diagnostic(
    *,
    duration_s: float,
    transport: str,
    bridge_file: Path | None,
    port: int,
    pair_shared_memory: bool,
    output_directory: Path | None,
    record: bool,
) -> tuple[SlipDiagnosticSummary, Path | None]:
    """Capture the local in-game bridge and optionally persist diagnostics."""

    def capture_from(reader: SlipPacketReader) -> SlipDiagnosticCapture:
        if pair_shared_memory:
            with AssettoCorsaSharedMemoryReader.connect() as shared_reader:
                return capture_slip_telemetry(
                    reader,
                    duration_s=duration_s,
                    shared_telemetry_reader=shared_reader,
                )
        return capture_slip_telemetry(reader, duration_s=duration_s)

    if transport == "file":
        with AssettoCorsaSlipBridgeFileReader.connect(bridge_file) as file_reader:
            capture = capture_from(file_reader)
    elif transport == "udp":
        with AssettoCorsaSlipBridgeReceiver.bind(port=port) as udp_reader:
            capture = capture_from(udp_reader)
    else:
        raise ValueError(f"unsupported slip-bridge transport: {transport}")
    summary = summarise_slip_capture(capture)
    destination: Path | None = None
    if record:
        destination = output_directory or Path("runs") / (
            "slip-diagnostic-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        )
        write_slip_diagnostic_artifacts(destination, capture, summary)
    return summary, destination


def _run_telemetry_diagnostic_command(arguments: argparse.Namespace) -> int:
    try:
        summary, destination = run_telemetry_diagnostic(
            duration_s=arguments.duration,
            poll_frequency_hz=arguments.poll_hz,
            output_directory=arguments.output,
            record=not arguments.no_record,
        )
    except AssettoCorsaTelemetryUnavailableError as error:
        LOGGER.error("%s", error)
        return 2
    LOGGER.info(
        "Telemetry: car=%s track=%s frames=%s observed_hz=%s speed=%.1f..%.1f km/h",
        summary.car_model,
        "/".join(filter(None, (summary.track, summary.track_configuration))),
        summary.unique_physics_frames,
        (
            f"{summary.observed_physics_frequency_hz:.1f}"
            if summary.observed_physics_frequency_hz is not None
            else "unresolved"
        ),
        summary.speed_min_kmh,
        summary.speed_max_kmh,
    )
    for issue in summary.issues:
        log = LOGGER.error if issue.severity == "error" else LOGGER.warning
        log("%s: %s", issue.code, issue.message)
    if destination is not None:
        LOGGER.info("Artefacts: %s", destination)
    return 0 if summary.is_usable_transport else 3


def _run_slip_diagnostic_command(arguments: argparse.Namespace) -> int:
    try:
        summary, destination = run_slip_diagnostic(
            duration_s=arguments.duration,
            transport=arguments.transport,
            bridge_file=arguments.bridge_file,
            port=arguments.port,
            pair_shared_memory=arguments.with_shared_memory,
            output_directory=arguments.output,
            record=not arguments.no_record,
        )
    except AssettoCorsaTelemetryUnavailableError as error:
        LOGGER.error("%s", error)
        return 2
    except OSError as error:
        LOGGER.error("cannot start the local slip-bridge reader: %s", error)
        return 2
    ratio_range = (
        f"{min(summary.slip_ratio_min):.3f}..{max(summary.slip_ratio_max):.3f}"
        if summary.slip_ratio_min is not None and summary.slip_ratio_max is not None
        else "unavailable"
    )
    angle_range = (
        f"{min(summary.slip_angle_raw_min):.3f}..{max(summary.slip_angle_raw_max):.3f}"
        if summary.slip_angle_raw_min is not None and summary.slip_angle_raw_max is not None
        else "unavailable"
    )
    LOGGER.info(
        "Slip bridge: packets=%s observed_hz=%s shared_pairs=%s ratio=%s angle_raw=%s",
        summary.unique_packets,
        (
            f"{summary.observed_frequency_hz:.1f}"
            if summary.observed_frequency_hz is not None
            else "unresolved"
        ),
        summary.shared_memory_pairs,
        ratio_range,
        angle_range,
    )
    LOGGER.info(
        "Slip calibration: name=%s angle_unit=%s angle_scale=%+.8f axle=%s",
        summary.slip_calibration_name,
        summary.slip_angle_raw_unit,
        summary.slip_angle_scale_rad_per_raw,
        summary.axle_aggregation,
    )
    for issue in summary.issues:
        log = LOGGER.error if issue.severity == "error" else LOGGER.warning
        log("%s: %s", issue.code, issue.message)
    if destination is not None:
        LOGGER.info("Artefacts: %s", destination)
    return 0 if summary.is_usable_transport else 3


def _log_readiness(report: TrainingReadinessReport) -> None:
    for check in report.checks:
        log = {
            "ok": LOGGER.info,
            "warning": LOGGER.warning,
            "error": LOGGER.error,
        }[check.severity]
        log("%s: %s", check.name, check.message)
    LOGGER.info(
        "Training readiness (%s): %s",
        report.backend,
        "READY" if report.is_ready else "BLOCKED",
    )


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0911, PLR0912, PLR0915
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)
    if arguments.command == "mock-run":
        config = ProjectConfig.from_toml(arguments.config)
        command = ["ac-rl-driver", *(argv if argv is not None else sys.argv[1:])]
        summary = run_mock(config, record=not arguments.no_record, command=command)
        LOGGER.info(
            "Mock run complete: steps=%s reward=%.3f reason=%s",
            summary["steps"],
            summary["total_reward"],
            summary["termination_reason"] or "time_limit",
        )
        if "run_directory" in summary:
            LOGGER.info("Artefacts: %s", summary["run_directory"])
        return 0
    if arguments.command == "download-data":
        manifest = download_initial_dataset(
            destination=arguments.destination,
            manifest_path=arguments.manifest,
        )
        LOGGER.info(
            "Dataset ready: files=%s bytes=%s manifest=%s",
            manifest["file_count"],
            manifest["total_bytes"],
            arguments.manifest,
        )
        return 0
    if arguments.command == "build-reference":
        config = ProjectConfig.from_toml(arguments.config)
        if config.reference.mode != "fixed":
            raise ValueError("build-reference requires reference.mode='fixed'")
        recordings = [
            path
            for path in config.dataset.recordings_directory.rglob("*.ld")
            if "_SAC" not in path.as_posix().upper()
        ]
        reference_report = build_fixed_reference(
            recordings=recordings,
            racing_line_csv=config.dataset.racing_line_path,
            output_path=Path(config.reference.artifact_path),
            track_length_m=config.environment.track_length_m,
            provenance={
                "track_id": config.dataset.track_id,
                "car_id": config.dataset.car_id,
                "dataset_revision": DATASET_REVISION,
                "source_revision": SOURCE_REVISION,
            },
        )
        LOGGER.info(
            "Fixed reference ready: recordings=%s used_samples=%s speed=%.1f..%.1f m/s artefact=%s",
            reference_report.recordings,
            reference_report.samples_used,
            reference_report.speed_min_mps,
            reference_report.speed_max_mps,
            config.reference.artifact_path,
        )
        return 0
    if arguments.command == "build-bc-dataset":
        config = ProjectConfig.from_toml(arguments.config)
        _, bc_report = build_bc_dataset(
            config,
            output_path=arguments.output,
            maximum_recordings=arguments.max_recordings,
        )
        LOGGER.info(
            "BC dataset ready: recordings=%s laps=%s sessions=%s "
            "samples=%s/%s/%s predictor_train=%s artefact=%s",
            bc_report.recordings,
            bc_report.complete_laps,
            bc_report.sessions,
            bc_report.train_samples,
            bc_report.validation_samples,
            bc_report.test_samples,
            bc_report.predictor_valid_train_samples,
            bc_report.destination,
        )
        return 0
    if arguments.command == "bc-train":
        config = ProjectConfig.from_toml(arguments.config)
        dataset_path = arguments.dataset or Path(config.behavioral_cloning.dataset_artifact_path)
        dataset = BehavioralCloningDataset.from_npz(dataset_path)
        command = ["ac-rl-driver", *(argv if argv is not None else sys.argv[1:])]
        bc_result = train_behavioral_cloning(
            config,
            dataset,
            command=command,
            device=arguments.device,
        )
        LOGGER.info(
            "BC complete: best_epoch=%s validation_action_mse=%.6f "
            "test_action_mse=%.6f test_ADE=%.3f m artefacts=%s",
            bc_result.best_epoch,
            bc_result.validation.action_mse,
            bc_result.test.action_mse,
            bc_result.test.predictor_ade_m,
            bc_result.run_directory,
        )
        return 0
    if arguments.command == "rl-ready":
        config = ProjectConfig.from_toml(arguments.config)
        report = assess_training_readiness(config, backend=arguments.backend)
        _log_readiness(report)
        return 0 if report.is_ready else 4
    if arguments.command == "rl-train":
        config = ProjectConfig.from_toml(arguments.config)
        report = assess_training_readiness(config, backend=arguments.backend)
        _log_readiness(report)
        if not report.is_ready:
            return 4
        if arguments.backend != "mock":
            raise RuntimeError("live training passed readiness without a live backend")
        environment = create_mock_environment(config)
        command = ["ac-rl-driver", *(argv if argv is not None else sys.argv[1:])]
        try:
            ppo_result = train_ppo(
                config,
                environment,
                command=command,
                backend_name=arguments.backend,
                device=arguments.device,
            )
        finally:
            environment.close()
        LOGGER.info(
            "PPO complete: steps=%s iterations=%s policy_version=%s artefacts=%s",
            ppo_result.environment_steps,
            ppo_result.iterations,
            ppo_result.policy_version,
            ppo_result.run_directory,
        )
        return 0
    if arguments.command == "telemetry-diagnose":
        return _run_telemetry_diagnostic_command(arguments)
    if arguments.command == "slip-diagnose":
        return _run_slip_diagnostic_command(arguments)
    raise RuntimeError(f"unhandled command: {arguments.command}")
