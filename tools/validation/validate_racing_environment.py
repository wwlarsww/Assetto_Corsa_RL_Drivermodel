"""Validate the live Gymnasium RacingEnvironment end to end.

The script deliberately obtains the environment from a project factory instead
of duplicating the complete Assetto Corsa wiring here.

Default factory:
    assetto_corsa_ai.main:create_environment

The factory must be a zero-argument callable returning a RacingEnvironment.

Examples:
    python tools/validate.py environment

    python tools/validate.py environment \
        --factory assetto_corsa_ai.main:create_environment \
        --steps 10 \
        --action 0.0 0.15

    # Runs Stable-Baselines3's checker. Warning: check_env() performs resets
    # and sampled actions, so use this only while Assetto Corsa is ready.
    python tools/validate.py environment --check-env
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import importlib
from pathlib import Path
from math import isfinite
from typing import Any, cast

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from assetto_corsa_ai.RacingEnvironment import EpisodeStart, RacingEnvironment, StartSampler
from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.track_geometry import TrackGeometry
from assetto_corsa_ai.vehicle_geometry import vehicle_forward

DEFAULT_FACTORY = "assetto_corsa_ai.main:create_environment"
DEFAULT_SEED = 42


# Keep this validator independent from the project's optional implementation
# module.  The factory is still expected to return the project's environment,
# but all of the interface validated here is defined by Gymnasium.
Environment = RacingEnvironment
EnvironmentFactory = Callable[..., object]


def load_factory(path: str) -> EnvironmentFactory:
    """Load ``module:function`` and verify that it is callable."""

    module_name, separator, attribute_name = path.partition(":")

    if not separator or not module_name or not attribute_name:
        raise ValueError(
            "Factory must use the form 'module:function', for example "
            "'assetto_corsa_ai.main:create_environment'."
        )

    module = importlib.import_module(module_name)

    try:
        factory = getattr(module, attribute_name)
    except AttributeError as error:
        raise ValueError(
            f"Module '{module_name}' has no attribute '{attribute_name}'."
        ) from error

    if not callable(factory):
        raise TypeError(
            f"Factory '{path}' is not callable."
        )

    return factory


def _box(space: object) -> spaces.Box[np.float32]:
    if not isinstance(space, spaces.Box):
        raise TypeError("RacingEnvironment requires Box action and observation spaces.")
    return cast(spaces.Box[np.float32], space)


def validate_info(info: object) -> None:
    if not isinstance(info, dict):
        raise TypeError("'info' must be a dict.")


def validate_spaces(env: Environment) -> None:
    """Validate the static Gymnasium interface without driving the car."""

    action_space = _box(env.action_space)
    observation_space = _box(env.observation_space)

    if action_space.shape != (2,):
        raise ValueError(
            f"Expected action shape (2,), got {action_space.shape}."
        )

    if action_space.dtype != np.dtype(np.float32):
        raise TypeError(
            f"Expected float32 action space, got {action_space.dtype}."
        )

    expected_low = np.full((2,), -1.0, dtype=np.float32)
    expected_high = np.full((2,), 1.0, dtype=np.float32)

    if not np.array_equal(action_space.low, expected_low):
        raise ValueError(
            f"Expected action lower bounds {expected_low}, got {action_space.low}."
        )

    if not np.array_equal(action_space.high, expected_high):
        raise ValueError(
            f"Expected action upper bounds {expected_high}, got {action_space.high}."
        )

    if len(observation_space.shape) != 1 or observation_space.shape[0] <= 0:
        raise ValueError(
            "Observation space must be a flat one-dimensional vector; "
            f"got shape {observation_space.shape}."
        )

    if observation_space.dtype != np.dtype(np.float32):
        raise TypeError(
            f"Expected float32 observation space, got {observation_space.dtype}."
        )

    print(
        "Gymnasium spaces OK:"
        f"\n  observation_space={observation_space}"
        f"\n  action_space={action_space}"
    )


def validate_observation(
    env: Environment,
    observation: np.ndarray,
) -> None:
    """Validate one observation returned by reset() or step()."""

    if observation.dtype != np.float32:
        raise TypeError(
            f"Observation must have dtype float32; got {observation.dtype}."
        )

    if not env.observation_space.contains(observation):
        raise ValueError(
            "Observation is outside the declared observation_space: "
            f"shape={observation.shape}, dtype={observation.dtype}."
        )

    if not np.isfinite(observation).all():
        bad_indices = np.flatnonzero(~np.isfinite(observation))
        raise ValueError(
            "Observation contains NaN or infinity at indices "
            f"{bad_indices.tolist()}."
        )


def validate_action(
    env: Environment,
    action: np.ndarray,
) -> None:
    """Validate one action before sending it to the live environment."""

    if action.dtype != np.float32:
        raise TypeError(
            f"Action must have dtype float32; got {action.dtype}."
        )

    if not env.action_space.contains(action):
        raise ValueError(
            "Action must match the declared action_space; "
            f"got {action}."
        )


def print_reset_result(
    observation: np.ndarray,
    info: dict[str, Any],
) -> None:
    """Print the most useful reset diagnostics."""

    print(
        "\nReset OK"
        f"\n  observation shape={observation.shape}"
        f"\n  dtype={observation.dtype}"
        f"\n  min={float(np.min(observation)):+.4f}"
        f"\n  max={float(np.max(observation)):+.4f}"
    )

    for key in (
        "scenario",
        "episode_start",
        "projection",
        "session_changed",
    ):
        if key in info:
            print(f"  {key}={info[key]}")


def print_step_result(
    index: int,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> None:
    """Print compact diagnostics for one environment step."""

    fields = [
        f"step={index}",
        f"action=[{action[0]:+.3f}, {action[1]:+.3f}]",
        f"reward={reward:+.6f}",
        f"terminated={terminated}",
        f"truncated={truncated}",
    ]

    delta_s = info.get("delta_s")

    if isinstance(delta_s, (int, float)) and isfinite(float(delta_s)):
        fields.append(
            f"delta_s={float(delta_s):+.4f} m"
        )

    dt = info.get("dt")

    if isinstance(dt, (int, float)) and isfinite(float(dt)):
        fields.append(
            f"dt={float(dt):.4f} s"
        )

    termination = info.get("termination")

    if termination is not None:
        reason = getattr(
            termination,
            "reason",
            None,
        )

        if reason is not None:
            fields.append(
                f"reason={reason}"
            )

    print(" | ".join(fields))


def run_smoke_test(
    env: Environment,
    *,
    seed: int,
    steps: int,
    action: np.ndarray,
) -> None:
    """Run one real reset followed by a short fixed-action rollout."""

    print("\nStarting live environment reset...")

    validate_action(env, action)

    observation, info = env.reset(
        seed=seed
    )

    validate_observation(
        env,
        observation,
    )
    validate_info(info)

    print_reset_result(
        observation,
        info,
    )

    for index in range(
        1,
        steps + 1,
    ):
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        validate_observation(
            env,
            observation,
        )

        if not isinstance(reward, float):
            raise TypeError(
                f"Reward must be a Python float; got {type(reward).__name__}."
            )

        if not isfinite(reward):
            raise ValueError(
                f"Reward must be finite; got {reward}."
            )

        if type(terminated) is not bool:
            raise TypeError(
                "'terminated' must be a Python bool."
            )

        if type(truncated) is not bool:
            raise TypeError(
                "'truncated' must be a Python bool."
            )

        validate_info(info)

        print_step_result(
            index=index,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        if terminated or truncated:
            print(
                "Episode ended during smoke test."
            )
            break


def run_sb3_check(
    env: Environment,
) -> None:
    """Run Stable-Baselines3's official custom-environment checker."""

    print(
        "\nRunning Stable-Baselines3 check_env()."
        "\nWARNING: check_env() calls reset() and performs sampled actions."
    )

    check_env(
        env,
        warn=True,
        skip_render_check=True,
    )

    print(
        "Stable-Baselines3 check_env() passed."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--factory",
        default=DEFAULT_FACTORY,
        help=(
            "Zero-argument environment factory as module:function "
            f"(default: {DEFAULT_FACTORY})."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Gymnasium reset seed (default: {DEFAULT_SEED}).",
    )

    parser.add_argument("--ac-directory", type=Path, help="AC installation directory (default factory only).")
    parser.add_argument("--config", type=Path, help="Validate the TOML-configured environment, including reset pre-roll.")
    parser.add_argument("--documents-directory", type=Path, help="AC user configuration directory (default factory only).")
    parser.add_argument("--car", help="Car ID (default factory: bmw_z4_gt3).")
    parser.add_argument("--track", help="Track ID (default factory: monza).")
    parser.add_argument("--layout", help="Track layout ID (default factory only).")
    parser.add_argument("--front-axle-distance", type=float, help="CG to front axle in metres (default factory only).")
    parser.add_argument("--rear-axle-distance", type=float, help="CG to rear axle in metres (default factory only).")

    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of live smoke-test steps after reset (default: 5).",
    )

    parser.add_argument(
        "--action",
        type=float,
        nargs=2,
        metavar=("STEERING", "LONGITUDINAL"),
        default=(0.0, 0.0),
        help=(
            "Fixed smoke-test action in [-1, 1]. "
            "Default is neutral: 0 0."
        ),
    )

    parser.add_argument(
        "--check-env",
        action="store_true",
        help=(
            "Also run stable_baselines3.common.env_checker.check_env(). "
            "This performs additional resets and sampled actions."
        ),
    )

    parser.add_argument(
        "--keep-session",
        action="store_true",
        help="Leave the Assetto Corsa session running when validation ends.",
    )

    return parser.parse_args()


def confirm_driving() -> bool:
    """Keep CSP disconnected until the driver has entered the loaded session."""
    input("Enter Drive/Start in AC and keep the car stationary, then press Enter here: ")
    return True


def captured_start_sampler() -> StartSampler:
    """Reuse one known live pose for all validation resets, including check_env."""
    captured: EpisodeStart | None = None

    def sample(track: TrackGeometry, rng: np.random.Generator) -> EpisodeStart:
        nonlocal captured
        if captured is None:
            reader = TelemetryReader()
            try:
                frame = reader.read()
            finally:
                reader.close()
            if not isfinite(frame.speed_kmh) or abs(frame.speed_kmh) > 1.0:
                raise ValueError("Stop the car before capturing the validation start pose.")
            captured = EpisodeStart(frame.car_coordinates, vehicle_forward(frame.tyre_contact_points))
            print(f"Validation start pose: {captured}", flush=True)
        return captured

    return sample


def main() -> None:
    args = parse_arguments()

    if args.steps < 0:
        raise SystemExit(
            "--steps must be >= 0."
        )

    action = np.asarray(
        args.action,
        dtype=np.float32,
    )
    if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.0):
        raise SystemExit("--action requires two finite values in [-1, 1].")
    if args.seed < 0:
        raise SystemExit("--seed must be >= 0.")

    factory = load_factory(
        args.factory
    )

    factory_options: dict[str, Any] = {
        name: value for name, value in {
            "assetto_corsa_directory": args.ac_directory,
            "documents_directory": args.documents_directory,
            "car": args.car,
            "track": args.track,
            "track_layout": args.layout,
            "front_axle_distance": args.front_axle_distance,
            "rear_axle_distance": args.rear_axle_distance,
        }.items() if value is not None
    }
    if factory_options and args.factory != DEFAULT_FACTORY:
        raise SystemExit("Installation and scenario options require the default factory.")
    if args.factory == DEFAULT_FACTORY:
        factory_options["drive_ready_check"] = confirm_driving
        factory_options["start_sampler"] = captured_start_sampler()
    if args.config is not None:
        if args.factory != DEFAULT_FACTORY or any(name not in ("drive_ready_check", "start_sampler") for name in factory_options):
            raise SystemExit("--config cannot be combined with factory or scenario overrides.")
        from assetto_corsa_ai.main import create_training_environment
        from assetto_corsa_ai.training_config import TrainingConfig
        env = create_training_environment(TrainingConfig.load(args.config))
    else:
        env = factory(**factory_options)
    if not isinstance(env, RacingEnvironment):
        raise TypeError("Factory must return a RacingEnvironment.")

    print(
        f"Environment factory: {args.factory}"
    )

    try:
        validate_spaces(
            env
        )

        run_smoke_test(
            env,
            seed=args.seed,
            steps=args.steps,
            action=action,
        )

        if args.check_env:
            run_sb3_check(
                env
            )

        print(
            "\nRacingEnvironment validation passed."
        )

    finally:
        env.close(
            stop_session=not args.keep_session
        )


if __name__ == "__main__":
    main()
