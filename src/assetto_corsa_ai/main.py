"""Application composition root for Assetto Corsa RL."""

from __future__ import annotations

import os
import argparse
import sys
from collections.abc import Callable
from dataclasses import asdict, replace
from math import hypot, isfinite
from pathlib import Path

from stable_baselines3.common.logger import HumanOutputFormat, Logger
from torch import nn
from torch.optim import Adam

from assetto_corsa_ai.assetto_corsa.session_manager import (
    AssettoCorsaSessionManager,
    SimulatorScenario,
    AssistMode,
)
from assetto_corsa_ai.assetto_corsa.simulator_controller import (
    AssettoCorsaSimulatorController,
)
from assetto_corsa_ai.assetto_corsa.static_info_reader import (
    StaticInfoReader,
)
from assetto_corsa_ai.assetto_corsa.telemetry_reader import (
    TelemetryReader,
)
from assetto_corsa_ai.assetto_corsa.track_reader import (
    AssettoCorsaTrackReader,
)
from assetto_corsa_ai.observation_builder import (
    ObservationBuilder,
)
from assetto_corsa_ai.RacingEnvironment import (
    RacingEnvironment, StartSampler,
)
from assetto_corsa_ai.agent import PPOAgent, SharedFeatureExtractor
from assetto_corsa_ai.track_perception import TrackPerceptionCalculator
from assetto_corsa_ai.tyre_dynamics import TyreDynamicsCalculator
from assetto_corsa_ai.training_config import TrainingConfig, checkpoint_config_path
from assetto_corsa_ai.training_callbacks import RunMetricsWriter, TrainingRunCallback
from assetto_corsa_ai.training_run import create_run_directory
from assetto_corsa_ai.reward import (
    RewardCalculator,
)
from assetto_corsa_ai.static_info import (
    SimulatorStaticInfo,
)
from assetto_corsa_ai.termination import (
    EpisodeTerminationChecker,
)
from assetto_corsa_ai.track_geometry import (
    TrackGeometry,
)
from assetto_corsa_ai.track_projection import (
    TrackProjector,
)

# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

DEFAULT_ASSETTO_CORSA_DIRECTORY = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
)

ASSETTO_CORSA_DIRECTORY = Path(
    os.environ.get(
        "ASSETTO_CORSA_DIRECTORY",
        DEFAULT_ASSETTO_CORSA_DIRECTORY,
    )
)


DEFAULT_SCENARIO = SimulatorScenario(
    car="bmw_z4_gt3",
    track="monza",
)


# Current Observation:
#
# vehicle dynamics:          8
# left boundary:        7 * 2 = 14
# right boundary:       7 * 2 = 14
# reference waypoints: 20 * 2 = 40
# applied controls:           2
#
# total = 78
OBSERVATION_SIZE = (
    10 + 4 * len(TrackPerceptionCalculator.DEFAULT_LOOKAHEAD_DISTANCES)
    + 2 * len(TrackPerceptionCalculator.DEFAULT_REFERENCE_DISTANCES)
)


CONTROL_INTERVAL_SECONDS = 0.05

MAX_EPISODE_TIME_SECONDS = 120.0


# ---------------------------------------------------------------------------
# Track geometry
# ---------------------------------------------------------------------------


def track_ai_path(
    static_info: SimulatorStaticInfo,
    assetto_corsa_directory: Path = ASSETTO_CORSA_DIRECTORY,
) -> Path:
    """Resolve the fast_lane.ai belonging to the loaded AC track."""

    track_directory = (
        assetto_corsa_directory / "content" / "tracks" / static_info.track_name
    )

    # Some AC tracks have multiple layouts.
    track_configuration = getattr(
        static_info,
        "track_configuration",
        None,
    )

    if track_configuration:
        layout_path = track_directory / track_configuration / "ai" / "fast_lane.ai"

        if not layout_path.is_file():
            raise FileNotFoundError(f"Missing AI spline for loaded layout: {layout_path}")
        return layout_path

    default_path = track_directory / "ai" / "fast_lane.ai"

    if not default_path.is_file():
        raise FileNotFoundError(
            "Could not find fast_lane.ai for loaded track: "
            f"{static_info.track_name!r}. "
            f"Expected: {default_path}"
        )

    return default_path


def load_track_geometry(
    static_info: SimulatorStaticInfo,
) -> TrackGeometry:
    """Load TrackGeometry for the currently active AC session."""

    path = track_ai_path(static_info)

    reader = AssettoCorsaTrackReader(path)

    return reader.read()


# ---------------------------------------------------------------------------
# Environment component factories
# ---------------------------------------------------------------------------


def create_observation_builder(
    static_info: SimulatorStaticInfo,
    track_geometry: TrackGeometry,
    *,
    front_axle_distance: float,
    rear_axle_distance: float,
) -> ObservationBuilder:
    """Create session-dependent observation processing.

    static_info and track_geometry are currently supplied for future
    session-dependent configuration even if ObservationBuilder itself does
    not need them during construction.
    """

    del static_info
    del track_geometry

    return ObservationBuilder(
        TyreDynamicsCalculator(front_axle_distance, rear_axle_distance),
        TrackPerceptionCalculator(),
    )


def create_termination_checker(
    projector: TrackProjector,
) -> EpisodeTerminationChecker:
    """Create fresh stateful termination logic for a session."""

    return EpisodeTerminationChecker(projector)


def create_reward_calculator() -> RewardCalculator:
    """Create the reward configuration used for training."""

    return RewardCalculator()


# ---------------------------------------------------------------------------
# Composition root
# ---------------------------------------------------------------------------


class SessionStaticInfoReader:
    """Open fresh metadata for each poll and release it even when reading fails."""

    def read(self) -> SimulatorStaticInfo:
        reader = StaticInfoReader()
        try:
            return reader.read()
        finally:
            reader.close()


def estimate_axle_distances() -> tuple[float, float]:
    """Estimate axle offsets using AC's car position as the vehicle origin.

    Explicit CG-to-axle measurements should override these estimates when
    the reported car origin differs from the centre of gravity.
    """
    reader = TelemetryReader()
    try:
        frame = reader.read()
    finally:
        reader.close()
    fl, fr, rl, rr = frame.tyre_contact_points
    front = ((fl[0] + fr[0]) / 2, (fl[2] + fr[2]) / 2)
    rear = ((rl[0] + rr[0]) / 2, (rl[2] + rr[2]) / 2)
    dx, dz = front[0] - rear[0], front[1] - rear[1]
    wheelbase = hypot(dx, dz)
    if not isfinite(wheelbase) or wheelbase <= 0:
        raise ValueError("Cannot estimate axle distances from invalid tyre positions.")
    x, _, z = frame.car_coordinates
    distances = (
        ((front[0] - x) * dx + (front[1] - z) * dz) / wheelbase,
        ((x - rear[0]) * dx + (z - rear[1]) * dz) / wheelbase,
    )
    if any(not isfinite(value) or value <= 0 for value in distances):
        raise ValueError("Car origin must lie between the axles; configure explicit axle distances.")
    return distances


def create_environment(
    *,
    assetto_corsa_directory: Path = ASSETTO_CORSA_DIRECTORY,
    documents_directory: Path | None = None,
    car: str = DEFAULT_SCENARIO.car,
    track: str = DEFAULT_SCENARIO.track,
    track_layout: str | None = None,
    front_axle_distance: float | None = None,
    rear_axle_distance: float | None = None,
    drive_ready_check: Callable[[], bool] | None = None,
    start_sampler: StartSampler | None = None,
) -> RacingEnvironment:
    """Create the live environment without launching AC until reset.

    Missing axle distances are estimated after launch. Explicit values are
    distances from the CG in metres; environment variables provide defaults.
    """

    # This reader is used only by SessionManager to detect when the requested
    # AC session is available. It must only OPEN AC-owned shared memory and
    # never create it.
    distances: list[float | None] = []
    for value, name in ((front_axle_distance, "FRONT_AXLE_DISTANCE"),
                        (rear_axle_distance, "REAR_AXLE_DISTANCE")):
        if value is None:
            configured = os.environ.get(name)
            value = float(configured) if configured is not None else None
        if value is not None and (not isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be a finite positive distance in metres.")
        distances.append(value)

    def build_observation(info: SimulatorStaticInfo, geometry: TrackGeometry) -> ObservationBuilder:
        front, rear = distances
        if front is None or rear is None:
            estimated_front, estimated_rear = estimate_axle_distances()
            front = estimated_front if front is None else front
            rear = estimated_rear if rear is None else rear
        return create_observation_builder(
            info, geometry, front_axle_distance=front, rear_axle_distance=rear,
        )

    def load_geometry(info: SimulatorStaticInfo) -> TrackGeometry:
        return AssettoCorsaTrackReader(track_ai_path(info, Path(assetto_corsa_directory))).read()

    session_manager = AssettoCorsaSessionManager(
        assetto_corsa_directory=Path(assetto_corsa_directory),
        documents_directory=documents_directory,
        static_info_reader=SessionStaticInfoReader(),
        drive_ready_check=drive_ready_check,
    )

    simulator_controller = AssettoCorsaSimulatorController()

    reward_calculator = create_reward_calculator()

    environment = RacingEnvironment(
        session_manager=session_manager,
        simulator_controller=simulator_controller,
        # Runtime resources must be created only after AC has started.
        telemetry_reader_factory=TelemetryReader,
        static_info_reader_factory=StaticInfoReader,
        track_geometry_loader=load_geometry,
        observation_builder_factory=build_observation,
        termination_checker_factory=create_termination_checker,
        reward_calculator=reward_calculator,
        observation_size=OBSERVATION_SIZE,
        default_scenario=SimulatorScenario(car, track, track_layout=track_layout),
        start_sampler=start_sampler,
        control_interval=CONTROL_INTERVAL_SECONDS,
        max_episode_time=MAX_EPISODE_TIME_SECONDS,
        reset_position_tolerance=1.0,
        reset_heading_tolerance_degrees=5.0,
        reset_speed_tolerance_kmh=1.0,
        reset_timeout=5.0,
        reset_poll_interval=0.05,
    )

    return environment


# ---------------------------------------------------------------------------
# Minimal executable entry point
# ---------------------------------------------------------------------------


def create_training_environment(config: TrainingConfig) -> RacingEnvironment:
    """Compose training exclusively from the validated run configuration."""
    config.validate()
    settings, simulator = config.environment, config.simulator
    ac_directory = Path(simulator.assetto_corsa_directory)

    def load_geometry(info: SimulatorStaticInfo) -> TrackGeometry:
        return AssettoCorsaTrackReader(track_ai_path(info, ac_directory)).read()

    def build_observation(info: SimulatorStaticInfo, geometry: TrackGeometry) -> ObservationBuilder:
        del info, geometry
        front, rear = settings.front_axle_distance, settings.rear_axle_distance
        if not front or not rear:
            estimated_front, estimated_rear = estimate_axle_distances()
            front, rear = front or estimated_front, rear or estimated_rear
        return ObservationBuilder(
            TyreDynamicsCalculator(front, rear, minimum_speed=settings.minimum_speed),
            TrackPerceptionCalculator(settings.lookahead_distances, settings.reference_distances),
        )

    def build_termination(projector: TrackProjector) -> EpisodeTerminationChecker:
        return EpisodeTerminationChecker(projector, **asdict(config.termination))

    session = AssettoCorsaSessionManager(
        assetto_corsa_directory=ac_directory,
        documents_directory=Path(simulator.documents_directory) if simulator.documents_directory else None,
        static_info_reader=SessionStaticInfoReader(), drive_ready_check=confirm_driving,
        startup_timeout=simulator.startup_timeout,
        drive_ready_timeout=simulator.drive_ready_timeout, poll_interval=simulator.poll_interval,
    )
    controller = AssettoCorsaSimulatorController(
        car_index=simulator.car_index, activation_timeout=simulator.activation_timeout,
        pulse_duration=simulator.pulse_duration,
    )
    reward_parameters = asdict(config.reward)
    # Future prediction reward settings are recorded, but never reach the
    # current calculator while that reward remains explicitly disabled.
    reward_parameters.pop("future_prediction")
    return RacingEnvironment(
        session_manager=session, simulator_controller=controller,
        telemetry_reader_factory=TelemetryReader, static_info_reader_factory=StaticInfoReader,
        track_geometry_loader=load_geometry, observation_builder_factory=build_observation,
        termination_checker_factory=build_termination,
        reward_calculator=RewardCalculator(**reward_parameters),
        observation_size=10 + 4 * len(settings.lookahead_distances) + 2 * len(settings.reference_distances),
        default_scenario=SimulatorScenario(
            car=settings.car, track=settings.track,
            track_layout=settings.track_layout or None, skin=settings.skin or None,
            abs_mode=AssistMode(settings.abs_mode),
            traction_control_mode=AssistMode(settings.traction_control_mode),
        ),
        start_sampler=RacingEnvironment.uniform_track_point_start,
        control_interval=settings.control_interval, max_episode_time=settings.max_episode_time,
        max_episode_steps=settings.max_episode_steps or None,
        reset_position_tolerance=settings.reset_position_tolerance,
        reset_heading_tolerance_degrees=settings.reset_heading_tolerance_degrees,
        reset_speed_tolerance_kmh=settings.reset_speed_tolerance_kmh,
        reset_timeout=settings.reset_timeout, reset_poll_interval=settings.reset_poll_interval,
        seed=config.training.seed,
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reset", "train", "drive"), nargs="?", default="reset")
    parser.add_argument("--ac-directory", type=Path, default=ASSETTO_CORSA_DIRECTORY)
    parser.add_argument("--documents-directory", type=Path)
    parser.add_argument("--car", default=DEFAULT_SCENARIO.car)
    parser.add_argument("--track", default=DEFAULT_SCENARIO.track)
    parser.add_argument("--layout")
    parser.add_argument("--front-axle-distance", type=float)
    parser.add_argument("--rear-axle-distance", type=float)
    parser.add_argument("--model-directory", type=Path, default=Path("models/ppo"))
    parser.add_argument("--config", type=Path, help="Complete training TOML (required for train).")
    parser.add_argument("--runs-directory", type=Path, default=Path("runs"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.seed < 0:
        parser.error("episodes must be positive and seed non-negative")
    args.training_config = None
    if args.mode == "train":
        if args.config is None:
            parser.error("train requires --config; all training settings belong in that TOML")
        # Prevent CLI options and environment variables silently overriding the
        # single source of truth. Installation/scenario settings are in TOML too.
        supplied = sys.argv[1:] if argv is None else argv
        allowed = {"--config", "--runs-directory"}
        if any(token.startswith("--") and token.split("=", 1)[0] not in allowed for token in supplied):
            parser.error("train accepts only --config and --runs-directory")
        try:
            config = TrainingConfig.load(args.config)
            if config.training.resume_from:
                source = Path(config.training.resume_from).resolve()
                config.require_compatible_resume(TrainingConfig.load(checkpoint_config_path(source)))
                for filename in ("model.zip", "vecnormalize.pkl"):
                    if not (source / filename).is_file():
                        raise FileNotFoundError(f"Incomplete training snapshot: {source / filename}")
            args.training_config = config
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return args
    if args.config is not None:
        parser.error("--config is for train; drive reads the saved run config")
    for attr, variable in (("front_axle_distance", "FRONT_AXLE_DISTANCE"),
                           ("rear_axle_distance", "REAR_AXLE_DISTANCE")):
        value = getattr(args, attr)
        if value is None:
            try:
                configured = os.environ.get(variable)
                value = float(configured) if configured is not None else None
            except ValueError:
                parser.error(f"{variable} must be a distance in metres")
        if value is not None and (not isfinite(value) or value <= 0):
            parser.error(f"{attr} must be finite and positive")
        setattr(args, attr, value)
    return args


def confirm_driving() -> bool:
    input("Enter Drive in AC, then press Enter to connect the agent: ")
    return True


def create_agent(environment: RacingEnvironment, config: TrainingConfig) -> PPOAgent:
    """Use the same TOML-driven policy construction in training and validation."""
    config.validate()
    network = config.network
    activation = nn.Tanh if network.activation == "tanh" else nn.ReLU
    return PPOAgent(
        environment, config=config.agent_config(),
        predictor_config=config.predictor, control_interval=config.environment.control_interval,
        policy_kwargs={
            "features_extractor_class": SharedFeatureExtractor,
            "features_extractor_kwargs": {
                "layers": network.feature_layers, "features_dim": network.feature_dim,
                "activation": activation,
            },
            "net_arch": {"pi": list(network.policy_layers), "vf": list(network.value_layers)},
            "activation_fn": activation, "ortho_init": network.ortho_init,
            "log_std_init": network.log_std_init,
            "optimizer_class": Adam, "optimizer_kwargs": {
                "eps": network.optimizer_epsilon, "betas": network.optimizer_betas,
                "weight_decay": network.optimizer_weight_decay,
                "amsgrad": network.optimizer_amsgrad,
            },
            "share_features_extractor": True,
        },
    )


def run_agent(environment: RacingEnvironment, args: argparse.Namespace) -> None:
    """Train/resume PPO or evaluate with frozen observation statistics."""
    agent: PPOAgent | None = None
    logger: Logger | None = None
    try:
        if args.mode == "reset":
            observation, info = environment.reset(seed=args.seed)
            print(f"RacingEnvironment ready: shape={observation.shape}, dtype={observation.dtype}")
            print(f"Scenario: {info.get('scenario')}")
            return
        if args.mode == "train":
            config: TrainingConfig = args.training_config
            run = create_run_directory(
                Path(args.runs_directory), track=config.environment.track, car=config.environment.car,
            )
            run.save_config(config.toml_text)
            run.save_metadata(Path(__file__).resolve().parents[2])
            print(f"Training run: {run.directory}")
            if config.training.resume_from:
                agent = PPOAgent.load_snapshot(
                    Path(config.training.resume_from), environment,
                    config=config.agent_config(), training=True,
                )
            else:
                agent = create_agent(environment, config)
            logger = Logger(None, [RunMetricsWriter(run)])
            if config.training.verbose:
                logger.output_formats.append(HumanOutputFormat(sys.stdout))
            agent.model.set_logger(logger)
            callback = TrainingRunCallback(run, config.training.checkpoint_interval)
            agent.learn(
                config.training.total_timesteps, callback=callback,
                reset_num_timesteps=not bool(config.training.resume_from),
            )
            run.save_final_model(agent.model, agent.vec_env)
            print(f"Model and normalization saved to {run.directory / 'final'}")
        else:
            source = Path(args.model_directory)
            if (source / PPOAgent.CONFIG_FILENAME).is_file():
                # Existing pre-run bundles remain usable for evaluation.
                agent = PPOAgent.load(source, environment, device=args.device)
            else:
                saved_config = TrainingConfig.load(checkpoint_config_path(source))
                agent = PPOAgent.load_snapshot(
                    source, environment,
                    config=replace(saved_config.agent_config(), device=args.device),
                )
            agent.set_evaluation_mode()
            for episode in range(args.episodes):
                observation, _ = environment.reset(seed=args.seed + episode)
                total_reward = 0.0
                while True:
                    observation, reward, terminated, truncated, info = environment.step(
                        agent.predict(observation, deterministic=True)
                    )
                    total_reward += reward
                    if terminated or truncated:
                        print(f"Episode {episode + 1}: reward={total_reward:.3f}, termination={info.get('termination')}")
                        break
    finally:
        try:
            if agent is not None:
                agent.close()
            else:
                environment.close()
        finally:
            if logger is not None:
                logger.close()


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    if args.mode == "train":
        environment = create_training_environment(args.training_config)
    elif args.mode == "drive" and not (args.model_directory / PPOAgent.CONFIG_FILENAME).is_file():
        environment = create_training_environment(TrainingConfig.load(checkpoint_config_path(args.model_directory)))
    else:
        environment = create_environment(
            assetto_corsa_directory=args.ac_directory,
            documents_directory=args.documents_directory,
            car=args.car, track=args.track, track_layout=args.layout,
            front_axle_distance=args.front_axle_distance,
            rear_axle_distance=args.rear_axle_distance,
            drive_ready_check=confirm_driving,
        )
    try:
        run_agent(environment, args)
    except (KeyboardInterrupt, EOFError):
        print("Stopped; simulator resources released. Completed checkpoints remain available.")


if __name__ == "__main__":
    main()
