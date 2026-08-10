"""CLI boundary for local data preparation, training, and diagnostics."""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from assetto_corsa_monitoring import MonitoringConfig, create_tracker
from assetto_corsa_rl.ac_integration.diagnostics import (
    TelemetryDiagnosticSummary,
    capture_telemetry,
    summarise_capture,
    write_diagnostic_artifacts,
)
from assetto_corsa_rl.ac_integration.launcher import (
    AssettoCorsaLaunchError,
    AssettoCorsaSessionLauncher,
    configured_fast_lane_path,
)
from assetto_corsa_rl.ac_integration.live_validation import (
    LiveValidationIssue,
    validate_live_environment,
    write_live_validation_artifacts,
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
from assetto_corsa_rl.ac_integration.track_assets import (
    TrackAssetBuildError,
    ensure_track_assets,
)
from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.control.vgamepad import (
    VGamepadControlWriter,
    VirtualGamepadUnavailableError,
)
from assetto_corsa_rl.data.assetto_corsa_gym import (
    DATASET_REVISION,
    SOURCE_REVISION,
    download_initial_dataset,
)
from assetto_corsa_rl.data.baseline_reference import build_baseline_reference
from assetto_corsa_rl.data.bc_dataset import (
    BehavioralCloningDataset,
    build_bc_dataset,
)
from assetto_corsa_rl.envs.factory import (
    create_assetto_corsa_environment,
    create_mock_environment,
)
from assetto_corsa_rl.experiment import ExperimentRecorder
from assetto_corsa_rl.policies import MockDriverPolicy
from assetto_corsa_rl.training.behavioral_cloning import train_behavioral_cloning
from assetto_corsa_rl.training.checkpoints import load_checkpoint
from assetto_corsa_rl.training.evaluation import evaluate_policy
from assetto_corsa_rl.training.ppo import (
    validate_initial_checkpoint,
    build_driver_network,
    train_ppo,
)
from assetto_corsa_rl.training.readiness import (
    TrainingReadinessReport,
    assess_training_readiness,
)

LOGGER = logging.getLogger("assetto_corsa_rl")


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915 - declarative CLI options
    """Build the command tree for local data, training, and diagnostic workflows.

    Returns:
        Parser configured with every supported CLI command and option.
    """

    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        prog="ac-rl-driver",
        description="Prepare data, validate integrations, and run local AC-RL experiments.",
        epilog="Research training is local-only; GitHub Actions never invokes training commands.",
        formatter_class=formatter,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="commands",
        metavar="COMMAND",
    )

    mock = subparsers.add_parser(
        "mock-run",
        help="Run a deterministic episode without Assetto Corsa.",
        description="Exercise the environment loop against the deterministic mock backend.",
        formatter_class=formatter,
    )
    mock.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mock.toml"),
        help="TOML configuration for the mock environment and episode.",
    )
    mock.add_argument(
        "--no-record",
        action="store_true",
        help="Do not write the usual JSON/CSV run artefacts.",
    )

    download = subparsers.add_parser(
        "download-data",
        help="Download the pinned human BMW Z4 GT3/Monza subset.",
        description="Download and verify the pinned public demonstration-data subset.",
        formatter_class=formatter,
    )
    download.add_argument(
        "--destination",
        type=Path,
        default=Path("data/external/assetto_corsa_gym"),
        help="Directory in which downloaded source data is stored.",
    )
    download.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/assetto_corsa_gym_bmw_z4_gt3_monza.json"),
        help="SHA-256 manifest written after the download is verified.",
    )
    build_reference = subparsers.add_parser(
        "build-reference",
        help="Build the baseline reference from downloaded human recordings.",
        description="Build the deterministic speed-parameterised baseline reference.",
        formatter_class=formatter,
    )
    build_reference.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bmw_z4_gt3_monza_baseline.toml"),
        help="TOML configuration containing dataset and baseline-reference paths.",
    )
    build_track_assets = subparsers.add_parser(
        "build-track-assets",
        help="Build cached track geometry and reference from Assetto Corsa.",
        description=(
            "Read the configured track's ai/fast_lane.ai and generate reusable "
            "centreline, boundaries and reference artefacts."
        ),
        formatter_class=formatter,
    )
    build_track_assets.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bmw_z4_gt3_monza_baseline.toml"),
        help="TOML configuration defining track, layout and cache settings.",
    )
    build_track_assets.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when the cached source SHA-256 still matches.",
    )
    build_bc = subparsers.add_parser(
        "build-bc-dataset",
        help="Export session-disjoint 77-feature BC tensors from human laps.",
        description="Create train/validation/test BC tensors and frozen Z-score statistics.",
        formatter_class=formatter,
    )
    build_bc.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bmw_z4_gt3_monza_baseline.toml"),
        help="TOML configuration defining features, splits, and output paths.",
    )
    build_bc.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the dataset to this path instead of the configured path.",
    )
    build_bc.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Limit input recordings for a quick development build; changes the baseline dataset.",
    )

    train_bc = subparsers.add_parser(
        "bc-train",
        help="Locally warm-start the policy and trajectory predictor with BC.",
        description="Train the shared policy/predictor network on a prepared BC dataset.",
        formatter_class=formatter,
    )
    train_bc.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bmw_z4_gt3_monza_baseline.toml"),
        help="TOML configuration defining the BC model, losses, and run output.",
    )
    train_bc.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Read this BC dataset instead of the configured artefact.",
    )
    train_bc.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device, for example cpu or cuda.",
    )
    evaluate_bc_live = subparsers.add_parser(
        "bc-live-evaluate",
        help="Evaluate the BC checkpoint deterministically in Assetto Corsa.",
        description=(
            "Load the configured BC warm start without PPO updates and record "
            "deterministic live-policy trajectories."
        ),
        formatter_class=formatter,
    )
    evaluate_bc_live.add_argument(
        "--config",
        type=Path,
        default=Path("configs/assetto_corsa_real_extended_smoke.toml"),
        help="TOML configuration defining the BC checkpoint and live session.",
    )
    evaluate_bc_live.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device, for example cpu or cuda.",
    )
    evaluate_bc_live.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of deterministic live episodes to record.",
    )
    evaluate_bc_live.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Run directory; otherwise use a timestamped runs/bc-live-evaluation path.",
    )
    evaluate_bc_live.add_argument(
        "--launch-game",
        action="store_true",
        help="Prepare and launch the configured local AC Hotlap session.",
    )
    rl_ready = subparsers.add_parser(
        "rl-ready",
        help="Check whether mock or live PPO training can start safely.",
        description="Run non-mutating readiness checks before starting a PPO process.",
        formatter_class=formatter,
    )
    rl_ready.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bmw_z4_gt3_monza_baseline.toml"),
        help="TOML configuration whose PPO prerequisites are checked.",
    )
    rl_ready.add_argument(
        "--backend",
        choices=("mock", "assetto-corsa"),
        default="assetto-corsa",
        help="Backend whose prerequisites should be validated.",
    )
    live_validate = subparsers.add_parser(
        "live-validate",
        help="Validate AC lifecycle, resets, controls and transports without PPO.",
        description=(
            "Run staged, write-capable acceptance checks against a real Assetto Corsa "
            "Hotlap session without loading or updating a policy."
        ),
        formatter_class=formatter,
    )
    live_validate.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bmw_z4_gt3_monza_live_smoke.toml"),
        help="TOML configuration defining the expected live session and transports.",
    )
    live_validate.add_argument(
        "--session",
        choices=("attach", "auto", "launch"),
        default="attach",
        help=(
            "Attach to a matching session, attach-or-launch automatically, or require "
            "a newly launched session."
        ),
    )
    live_validate.add_argument(
        "--resets",
        type=int,
        default=8,
        help="Number of reset positions used for the repeatability check.",
    )
    live_validate.add_argument(
        "--soak-steps",
        type=int,
        default=250,
        help="Frame-synchronised neutral steps used for the connection soak check.",
    )
    live_validate.add_argument(
        "--input-hold-steps",
        type=int,
        default=4,
        help="Control periods held for each throttle, brake and steering input probe.",
    )
    live_validate.add_argument(
        "--drive-off-steps",
        type=int,
        default=40,
        help="Control periods available for the forward-gear acceleration probe.",
    )
    live_validate.add_argument(
        "--drive-off-throttle",
        type=float,
        default=0.35,
        help="Bounded throttle command used by the forward-gear acceleration probe.",
    )
    live_validate.add_argument(
        "--minimum-forward-gear",
        type=int,
        default=1,
        help="Highest real forward gear AC must report during the acceleration probe.",
    )
    live_validate.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination directory; otherwise use a timestamped runs/live-validation path.",
    )
    live_validate.add_argument(
        "--no-record",
        action="store_true",
        help="Report the result without writing JSON/CSV evidence.",
    )
    live_validate.add_argument(
        "--leave-launched-game-running",
        action="store_true",
        help="Do not perform the default graceful stop of a session launched by this command.",
    )
    rl_train = subparsers.add_parser(
        "rl-train",
        help="Locally train baseline_v1 with synchronous PPO.",
        description="Run guarded PPO with the deterministic mock or live Assetto Corsa backend.",
        formatter_class=formatter,
    )
    rl_train.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mock.toml"),
        help="TOML configuration defining PPO, rewards, and run output.",
    )
    rl_train.add_argument(
        "--backend",
        choices=("mock", "assetto-corsa"),
        default="mock",
        help="Backend to use; the live backend requires an already running on-track session.",
    )
    rl_train.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device, for example cpu or cuda.",
    )
    rl_train.add_argument(
        "--launch-game",
        action="store_true",
        help="Prepare and launch the configured local AC Hotlap session before live training.",
    )

    diagnose = subparsers.add_parser(
        "telemetry-diagnose",
        help="Validate the read-only Assetto Corsa shared-memory transport.",
        description="Capture shared memory while Assetto Corsa is in an on-track session.",
        formatter_class=formatter,
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
        help="Requested shared-memory polling frequency in Hz.",
    )
    diagnose.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for diagnostic JSON/CSV files; otherwise use runs/<timestamp>.",
    )
    diagnose.add_argument(
        "--no-record",
        action="store_true",
        help="Capture and report only; do not write diagnostic artefacts.",
    )
    slip_diagnose = subparsers.add_parser(
        "slip-diagnose",
        help="Validate the local SlipRatio/SlipAngle Python-app bridge.",
        description="Capture tyre slip channels published by the enabled in-game bridge app.",
        formatter_class=formatter,
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
        help="Bridge transport; the original Assetto Corsa normally uses file.",
    )
    slip_diagnose.add_argument(
        "--bridge-file",
        type=Path,
        default=None,
        help="Read this bridge file instead of the default per-user path.",
    )
    slip_diagnose.add_argument(
        "--port",
        type=int,
        default=SLIP_BRIDGE_PORT,
        help="Loopback UDP port used only with --transport=udp.",
    )
    slip_diagnose.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for diagnostic JSON/CSV files; otherwise use runs/<timestamp>.",
    )
    slip_diagnose.add_argument(
        "--no-record",
        action="store_true",
        help="Capture and report only; do not write diagnostic artefacts.",
    )
    slip_diagnose.add_argument(
        "--with-shared-memory",
        action="store_true",
        help="Pair tyre packets with controls and vehicle motion from shared memory.",
    )
    x360_setup = subparsers.add_parser(
        "x360-setup",
        help="Expose a neutral virtual X360 device while configuring Assetto Corsa.",
        description=(
            "Create the virtual controller, keep every axis neutral, and release it "
            "after a bounded setup window."
        ),
        formatter_class=formatter,
    )
    x360_setup.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Seconds to keep the neutral virtual controller visible.",
    )
    return parser


def seed_everything(seed: int) -> None:
    """Seed standard RNGs used by this scaffold."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_x360_setup(*, duration_s: float) -> None:
    """Keep a neutral virtual controller visible for a bounded setup window."""

    if duration_s <= 0:
        raise ValueError("x360 setup duration must be positive")
    writer = VGamepadControlWriter()
    try:
        LOGGER.info(
            "Neutral virtual X360 controller is visible for %.1f s; "
            "configure and save Assetto Corsa controls now",
            duration_s,
        )
        time.sleep(duration_s)
    finally:
        writer.close()


def run_mock(config: ProjectConfig, *, record: bool, command: list[str]) -> dict[str, object]:
    """Exercise the complete environment loop with a simple controller."""

    seed_everything(config.experiment.seed)
    environment = create_mock_environment(config)
    policy = MockDriverPolicy()
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


def _copy_file_atomically(source: Path, destination: Path) -> None:
    """Publish one generated artefact without exposing a partial destination."""

    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _publish_bc_warm_start(
    config: ProjectConfig,
    *,
    checkpoint_path: Path,
    summary_path: Path,
) -> tuple[Path | None, Path | None]:
    """Publish BC outputs at the stable paths consumed by PPO readiness."""

    checkpoint_destination = (
        Path(config.ppo.initial_checkpoint_path)
        if config.ppo.initial_checkpoint_path.strip()
        else None
    )
    metrics_destination = (
        Path(config.ppo.initial_checkpoint_metrics_path)
        if config.ppo.initial_checkpoint_metrics_path.strip()
        else None
    )
    if checkpoint_destination is not None:
        _copy_file_atomically(checkpoint_path, checkpoint_destination)
    if metrics_destination is not None:
        _copy_file_atomically(summary_path, metrics_destination)
    return checkpoint_destination, metrics_destination


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0911, PLR0912, PLR0915
    """Execute one CLI command and translate its result into a process exit code.

    Args:
        argv: Arguments without the executable name. ``None`` reads ``sys.argv``.

    Returns:
        Zero on success and a non-zero command-specific status on failure.
    """

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
    if arguments.command == "build-track-assets":
        config = ProjectConfig.from_toml(arguments.config)
        try:
            bundle = ensure_track_assets(config, force=arguments.force)
        except (FileNotFoundError, TrackAssetBuildError, OSError, ValueError) as error:
            LOGGER.error("track assets could not be built: %s", error)
            return 2
        LOGGER.info(
            "Track assets ready: source=%s geometry=%s reference=%s "
            "geometry_rebuilt=%s reference_rebuilt=%s",
            bundle.source_fast_lane_path,
            bundle.paths.geometry_path,
            bundle.paths.reference_path,
            bundle.rebuilt_geometry,
            bundle.rebuilt_reference,
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
        reference_report = build_baseline_reference(
            recordings=recordings,
            fast_lane_path=configured_fast_lane_path(config),
            output_path=Path(config.reference.artifact_path),
            provenance={
                "track_id": config.dataset.track_id,
                "car_id": config.dataset.car_id,
                "dataset_revision": DATASET_REVISION,
                "source_revision": SOURCE_REVISION,
            },
        )
        LOGGER.info(
            "Baseline reference ready: recordings=%s used_samples=%s "
            "speed=%.1f..%.1f m/s artefact=%s",
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
        checkpoint_artifact, metrics_artifact = _publish_bc_warm_start(
            config,
            checkpoint_path=bc_result.checkpoint_path,
            summary_path=bc_result.run_directory / "summary.json",
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
        if checkpoint_artifact is not None:
            LOGGER.info("Published BC checkpoint: %s", checkpoint_artifact)
        if metrics_artifact is not None:
            LOGGER.info("Published BC metrics: %s", metrics_artifact)
        return 0
    if arguments.command == "bc-live-evaluate":
        config = ProjectConfig.from_toml(arguments.config)
        if arguments.episodes <= 0:
            LOGGER.error("--episodes must be positive")
            return 2
        if config.track_assets.auto_build:
            try:
                bundle = ensure_track_assets(config)
                LOGGER.info(
                    "track_assets: geometry=%s reference=%s source_sha256=%s",
                    bundle.paths.geometry_path,
                    bundle.paths.reference_path,
                    bundle.source_sha256,
                )
            except (FileNotFoundError, TrackAssetBuildError, OSError, ValueError) as error:
                LOGGER.error("track assets are not ready: %s", error)
                return 4
        report = assess_training_readiness(config, backend="assetto-corsa")
        _log_readiness(report)
        if not report.is_ready:
            return 4
        evaluation_config = replace(
            config,
            evaluation=replace(
                config.evaluation,
                enabled=True,
                episodes=arguments.episodes,
                deterministic=True,
                record_trajectories=True,
            ),
        )
        target_device = torch.device(arguments.device)
        checkpoint_path = Path(evaluation_config.ppo.initial_checkpoint_path)
        model = build_driver_network(evaluation_config)
        _state, checkpoint_configuration = load_checkpoint(
            checkpoint_path,
            model=model,
            map_location=target_device,
        )
        validate_initial_checkpoint(
            checkpoint_configuration,
            config=evaluation_config,
        )
        model.to(target_device)
        model.eval()
        output_directory = arguments.output or Path(
            evaluation_config.experiment.output_directory
        ) / ("bc-live-evaluation-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
        bc_writer: VGamepadControlWriter | None = None
        bc_launcher: AssettoCorsaSessionLauncher | None = None
        bc_environment = None
        try:
            bc_writer = VGamepadControlWriter(
                steering_output_sign=(evaluation_config.assetto_corsa.steering_output_sign),
                steering_output_gain=(evaluation_config.assetto_corsa.steering_output_gain),
            )
            bc_launcher = (
                AssettoCorsaSessionLauncher(
                    evaluation_config,
                    controller_index=getattr(bc_writer, "device_index", None),
                )
                if arguments.launch_game
                else None
            )
            launch_context = bc_launcher if bc_launcher is not None else nullcontext()
            with launch_context:
                try:
                    bc_environment = create_assetto_corsa_environment(
                        evaluation_config,
                        control_writer=bc_writer,
                    )
                    result = evaluate_policy(
                        config=evaluation_config,
                        model=model,
                        device=target_device,
                        training_run_directory=output_directory,
                        checkpoint_path=checkpoint_path,
                        checkpoint_update=0,
                        environment_steps=0,
                        backend_name="assetto-corsa",
                        tracker=None,
                        environment=bc_environment,
                    )
                finally:
                    bc_writer.neutral()
                    if bc_environment is not None:
                        bc_environment.close()
                    if bc_launcher is not None and bc_launcher.stop_owned_session():
                        LOGGER.info("Stopped the Assetto Corsa session owned by BC evaluation")
        except AssettoCorsaLaunchError as error:
            LOGGER.error("cannot launch Assetto Corsa: %s", error)
            return 2
        finally:
            if bc_writer is not None:
                bc_writer.close()
        LOGGER.info(
            "BC live evaluation complete: episodes=%s mean_reward=%.3f artefacts=%s",
            len(result.episodes),
            float(result.metrics["eval/reward/mean"]),
            result.output_directory,
        )
        return 0
    if arguments.command == "rl-ready":
        config = ProjectConfig.from_toml(arguments.config)
        if arguments.backend == "assetto-corsa" and config.track_assets.auto_build:
            try:
                bundle = ensure_track_assets(config)
                LOGGER.info(
                    "track_assets: geometry=%s reference=%s source_sha256=%s",
                    bundle.paths.geometry_path,
                    bundle.paths.reference_path,
                    bundle.source_sha256,
                )
            except (FileNotFoundError, TrackAssetBuildError, OSError, ValueError) as error:
                LOGGER.error("track assets are not ready: %s", error)
                return 4
        report = assess_training_readiness(config, backend=arguments.backend)
        _log_readiness(report)
        return 0 if report.is_ready else 4
    if arguments.command == "live-validate":
        config = ProjectConfig.from_toml(arguments.config)
        if config.track_assets.auto_build:
            try:
                ensure_track_assets(config)
            except (FileNotFoundError, TrackAssetBuildError, OSError, ValueError) as error:
                LOGGER.error("track assets are not ready: %s", error)
                return 4
        report = assess_training_readiness(config, backend="assetto-corsa")
        _log_readiness(report)
        if not report.is_ready:
            return 4
        writer: VGamepadControlWriter | None = None
        launcher: AssettoCorsaSessionLauncher | None = None
        environment = None
        validation_result = None
        stop_requested = False
        stopped: bool | None = None
        lifecycle_issue: LiveValidationIssue | None = None
        try:
            writer = VGamepadControlWriter(
                steering_output_sign=config.assetto_corsa.steering_output_sign,
                steering_output_gain=config.assetto_corsa.steering_output_gain,
            )
            if arguments.session != "attach":
                launcher = AssettoCorsaSessionLauncher(
                    config,
                    reuse_existing_session=arguments.session == "auto",
                    controller_index=getattr(writer, "device_index", None),
                )
                launcher.launch()
                session_origin = launcher.session_origin
            else:
                session_origin = "attached"
            environment = create_assetto_corsa_environment(
                config,
                control_writer=writer,
            )
            validation_result = validate_live_environment(
                environment,
                session_origin=session_origin,
                reset_method_variant=config.assetto_corsa.reset_backend,
                controller_index=getattr(writer, "device_index", None),
                control_frequency_hz=config.environment.control_frequency_hz,
                reset_count=arguments.resets,
                soak_steps=arguments.soak_steps,
                input_hold_steps=arguments.input_hold_steps,
                drive_off_steps=arguments.drive_off_steps,
                drive_off_throttle=arguments.drive_off_throttle,
                minimum_forward_gear=arguments.minimum_forward_gear,
                expected_simulator_steering_sign=config.assetto_corsa.steering_output_sign,
            )
        except Exception as error:
            LOGGER.error("live validation could not run: %s", error)
            return 2
        finally:
            if environment is not None:
                environment.close()
            if launcher is not None:
                stop_requested = (
                    launcher.session_origin == "launched"
                    and not arguments.leave_launched_game_running
                )
                if stop_requested:
                    try:
                        stopped = launcher.stop_owned_session()
                    except AssettoCorsaLaunchError as error:
                        stopped = False
                        lifecycle_issue = LiveValidationIssue(
                            "lifecycle",
                            "graceful_stop_failed",
                            str(error),
                        )
                launcher.restore_user_configuration()
            if writer is not None:
                writer.close()
        live_summary, reset_samples, transport_samples = validation_result
        issues = live_summary.issues
        if lifecycle_issue is not None:
            issues = (*issues, lifecycle_issue)
        live_summary = replace(
            live_summary,
            game_stop_requested=stop_requested,
            game_stopped=stopped,
            issues=issues,
        )
        destination = arguments.output
        if not arguments.no_record:
            if destination is None:
                destination = Path("runs") / (
                    "live-validation-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                )
            write_live_validation_artifacts(
                destination,
                live_summary,
                reset_samples,
                transport_samples,
            )
        LOGGER.info(
            "Live validation: %s; origin=%s resets=%s position_spread=%s m "
            "drive_off_gear=%s drive_off_speed=%s m/s drive_off_progress=%s m "
            "drive_off_max_rpm=%s drive_off_max_abs_slip=%s "
            "soak=%s/%s artefacts=%s",
            "PASS" if live_summary.passed else "FAIL",
            live_summary.session_origin,
            live_summary.reset_count,
            (
                f"{live_summary.reset_position_spread_m:.3f}"
                if live_summary.reset_position_spread_m is not None
                else "n/a"
            ),
            (
                live_summary.drive_off.maximum_observed_forward_gear
                if live_summary.drive_off is not None
                else "n/a"
            ),
            (
                f"{live_summary.drive_off.maximum_speed_mps:.3f}"
                if live_summary.drive_off is not None
                else "n/a"
            ),
            (
                f"{live_summary.drive_off.forward_progress_m:.3f}"
                if live_summary.drive_off is not None
                else "n/a"
            ),
            (live_summary.drive_off.maximum_rpms if live_summary.drive_off is not None else "n/a"),
            (
                f"{live_summary.drive_off.maximum_abs_slip_ratio:.3f}"
                if live_summary.drive_off is not None
                else "n/a"
            ),
            live_summary.soak_steps_completed,
            live_summary.soak_steps_requested,
            destination,
        )
        for issue in live_summary.issues:
            LOGGER.error("%s/%s: %s", issue.stage, issue.code, issue.message)
        return 0 if live_summary.passed else 3
    if arguments.command == "rl-train":
        config = ProjectConfig.from_toml(arguments.config)
        monitoring_config = MonitoringConfig.from_toml(arguments.config)
        if arguments.backend == "assetto-corsa" and config.track_assets.auto_build:
            try:
                bundle = ensure_track_assets(config)
                LOGGER.info(
                    "track_assets: geometry=%s reference=%s source_sha256=%s",
                    bundle.paths.geometry_path,
                    bundle.paths.reference_path,
                    bundle.source_sha256,
                )
            except (FileNotFoundError, TrackAssetBuildError, OSError, ValueError) as error:
                LOGGER.error("track assets are not ready: %s", error)
                return 4
        tracker = create_tracker(monitoring_config)
        if arguments.launch_game and arguments.backend != "assetto-corsa":
            LOGGER.error("--launch-game is supported only with --backend assetto-corsa")
            return 2
        report = assess_training_readiness(config, backend=arguments.backend)
        _log_readiness(report)
        if not report.is_ready:
            return 4
        command = ["ac-rl-driver", *(argv if argv is not None else sys.argv[1:])]
        evaluation_environment_factory = (
            (lambda: create_mock_environment(config))
            if config.evaluation.enabled and arguments.backend == "mock"
            else None
        )
        evaluation_uses_training_environment = (
            config.evaluation.enabled and arguments.backend == "assetto-corsa"
        )
        launch_control_writer = (
            VGamepadControlWriter(
                steering_output_sign=config.assetto_corsa.steering_output_sign,
                steering_output_gain=config.assetto_corsa.steering_output_gain,
            )
            if arguments.launch_game
            else None
        )
        try:
            session_launcher = (
                AssettoCorsaSessionLauncher(
                    config,
                    controller_index=(
                        getattr(launch_control_writer, "device_index", None)
                        if launch_control_writer is not None
                        else None
                    ),
                )
                if arguments.launch_game
                else None
            )
            launch_context = session_launcher if session_launcher is not None else nullcontext()
            with launch_context:
                environment = None
                try:
                    environment = (
                        create_mock_environment(config)
                        if arguments.backend == "mock"
                        else create_assetto_corsa_environment(
                            config,
                            control_writer=launch_control_writer,
                        )
                    )
                    ppo_result = train_ppo(
                        config,
                        environment,
                        command=command,
                        backend_name=arguments.backend,
                        device=arguments.device,
                        tracker=tracker,
                        tracking_run_name=monitoring_config.run_name,
                        evaluation_environment_factory=(evaluation_environment_factory),
                        evaluation_uses_training_environment=(evaluation_uses_training_environment),
                    )
                finally:
                    try:
                        if launch_control_writer is not None:
                            launch_control_writer.neutral()
                    finally:
                        try:
                            if environment is not None:
                                environment.close()
                        finally:
                            if (
                                session_launcher is not None
                                and session_launcher.stop_owned_session()
                            ):
                                LOGGER.info("Stopped the Assetto Corsa session owned by training")
        except AssettoCorsaLaunchError as error:
            LOGGER.error("cannot launch Assetto Corsa: %s", error)
            return 2
        finally:
            if launch_control_writer is not None:
                launch_control_writer.close()
        LOGGER.info(
            "PPO complete: steps=%s iterations=%s policy_version=%s evaluations=%s artefacts=%s",
            ppo_result.environment_steps,
            ppo_result.iterations,
            ppo_result.policy_version,
            ppo_result.evaluations_completed,
            ppo_result.run_directory,
        )
        return 0
    if arguments.command == "x360-setup":
        try:
            run_x360_setup(duration_s=arguments.duration)
        except VirtualGamepadUnavailableError as error:
            LOGGER.error("%s", error)
            return 2
        return 0
    if arguments.command == "telemetry-diagnose":
        return _run_telemetry_diagnostic_command(arguments)
    if arguments.command == "slip-diagnose":
        return _run_slip_diagnostic_command(arguments)
    raise RuntimeError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
