"""Visually validate reward and termination behaviour in Assetto Corsa."""

# Matplotlib leaves some method signatures incomplete.
# Keep strict checks, but exclude unknown member types in this plotting script.
# pyright: reportUnknownMemberType=false

from collections import deque
from math import hypot, isfinite
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Affine2D

from assetto_corsa_ai.assetto_corsa.static_info_reader import StaticInfoReader
from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.assetto_corsa.track_reader import AssettoCorsaTrackReader, track_ai_path
from assetto_corsa_ai.reward import (
    Reward,
    RewardCalculator,
    calculate_progress_delta,
)
from assetto_corsa_ai.telemetry import TelemetryFrame
from assetto_corsa_ai.termination import (
    EpisodeTerminationChecker,
    TerminationResult,
)
from assetto_corsa_ai.track_geometry import TrackGeometry
from assetto_corsa_ai.track_projection import TrackProjection, TrackProjector

# ---------------------------------------------------------------------------
# Validation configuration
# ---------------------------------------------------------------------------

PERFORMANCE_WEIGHT = 1.0

STYLE_WEIGHT = 1.0
STYLE_SIGMA = 1.0

OFF_TRACK_PENALTY = -10.0
STUCK_PENALTY = -5.0
COLLISION_PENALTY = -10.0

STUCK_TIME = 4.0
MINIMUM_PROGRESS = 2.0

HISTORY_SECONDS = 15.0

VIEW_LATERAL_METRES = 35.0
VIEW_FORWARD_METRES = 80.0
VIEW_BEHIND_METRES = 20.0

STYLE_CURVE_MAX_DISTANCE = 5.0
STYLE_CURVE_SAMPLES = 200


def _extract_track_geometry(
    track_geometry: TrackGeometry,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Extract complete track geometry for plotting."""

    center_x = [point.position[0] for point in track_geometry.points]

    center_z = [point.position[2] for point in track_geometry.points]

    left_x = [point.left_boundary[0] for point in track_geometry.points]

    left_z = [point.left_boundary[2] for point in track_geometry.points]

    right_x = [point.right_boundary[0] for point in track_geometry.points]

    right_z = [point.right_boundary[2] for point in track_geometry.points]

    # Close all lines at start/finish.
    center_x.append(center_x[0])
    center_z.append(center_z[0])

    left_x.append(left_x[0])
    left_z.append(left_z[0])

    right_x.append(right_x[0])
    right_z.append(right_z[0])

    return (
        center_x,
        center_z,
        left_x,
        left_z,
        right_x,
        right_z,
    )


def _update_vehicle_view_transform(
    transform: Affine2D,
    telemetry: TelemetryFrame,
    previous_forward: tuple[float, float],
) -> tuple[float, float]:
    """Update world-to-vehicle plotting transformation.

    The actual track data remains in Assetto Corsa world coordinates.
    Only its display transform is changed.
    """

    (
        front_left,
        front_right,
        rear_left,
        rear_right,
    ) = telemetry.tyre_contact_points

    heading_x = (front_left[0] + front_right[0] - rear_left[0] - rear_right[0]) / 2.0

    heading_z = (front_left[2] + front_right[2] - rear_left[2] - rear_right[2]) / 2.0

    heading_length = hypot(
        heading_x,
        heading_z,
    )

    forward_x, forward_z = previous_forward

    if isfinite(heading_length) and heading_length > 1e-6:
        forward_x = heading_x / heading_length
        forward_z = heading_z / heading_length

    # Vehicle-right direction perpendicular to forward.
    right_x = -forward_z
    right_z = forward_x

    vehicle_x = telemetry.car_coordinates[0]
    vehicle_z = telemetry.car_coordinates[2]

    transform.set_matrix(
        Affine2D.from_values(
            right_x,
            forward_x,
            right_z,
            forward_z,
            -(vehicle_x * right_x + vehicle_z * right_z),
            -(vehicle_x * forward_x + vehicle_z * forward_z),
        ).get_matrix()
    )

    return forward_x, forward_z


def _tyre_track_state(
    telemetry: TelemetryFrame,
    projector: TrackProjector,
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[float],
]:
    """Separate tyres into inside and outside track groups."""

    inside: list[tuple[float, float, float]] = []
    outside: list[tuple[float, float, float]] = []
    clearances: list[float] = []

    for tyre_position in telemetry.tyre_contact_points:
        projection = projector.project(tyre_position)

        minimum_clearance = min(
            projection.left_clearance,
            projection.right_clearance,
        )

        clearances.append(minimum_clearance)

        if projection.left_clearance < 0.0 or projection.right_clearance < 0.0:
            outside.append(tyre_position)
        else:
            inside.append(tyre_position)

    return (
        inside,
        outside,
        clearances,
    )


def _set_marker_positions(
    marker: Line2D,
    points: list[tuple[float, float, float]],
) -> None:
    """Update a marker from world-coordinate points."""

    marker.set_data(
        [point[0] for point in points],
        [point[2] for point in points],
    )


def _build_style_curve(
    reward_calculator: RewardCalculator,
) -> tuple[list[float], list[float]]:
    """Generate the actual style-reward curve from RewardCalculator."""

    no_termination = TerminationResult(
        terminated=False,
        reason=None,
    )

    distances = [
        STYLE_CURVE_MAX_DISTANCE * index / (STYLE_CURVE_SAMPLES - 1)
        for index in range(STYLE_CURVE_SAMPLES)
    ]

    scores = [
        reward_calculator.calculate(
            delta_s=0.0,
            lateral_offset=distance,
            termination=no_termination,
        ).style_step
        for distance in distances
    ]

    return distances, scores


def _update_reward_history(
    history_axis: Axes,
    episode_time: float,
    time_history: deque[float],
    performance_history: deque[float],
    style_history: deque[float],
    total_history: deque[float],
    penalty_history: deque[float],
    performance_line: Line2D,
    style_line: Line2D,
    total_line: Line2D,
    penalty_line: Line2D,
) -> None:
    """Update rolling reward history."""

    performance_line.set_data(
        time_history,
        performance_history,
    )

    style_line.set_data(
        time_history,
        style_history,
    )

    total_line.set_data(
        time_history,
        total_history,
    )

    penalty_line.set_data(
        time_history,
        penalty_history,
    )

    right_limit = max(
        HISTORY_SECONDS,
        episode_time,
    )

    left_limit = max(
        0.0,
        right_limit - HISTORY_SECONDS,
    )

    history_axis.set_xlim(
        left_limit,
        right_limit,
    )

    history_axis.relim()
    history_axis.autoscale_view(
        scalex=False,
        scaley=True,
    )


def _trim_history(
    episode_time: float,
    time_history: deque[float],
    histories: tuple[deque[float], ...],
) -> None:
    """Keep only the configured history window."""

    cutoff = episode_time - HISTORY_SECONDS

    while time_history and time_history[0] < cutoff:
        time_history.popleft()

        for history in histories:
            history.popleft()


def _update_status(
    status_text: Text,
    *,
    active: bool,
    projection: TrackProjection,
    delta_s: float,
    reward: Reward | None,
    termination: TerminationResult,
    tyre_clearances: list[float],
    episode_time: float,
) -> None:
    """Update the textual validation status."""

    inside_tyres = sum(clearance >= 0.0 for clearance in tyre_clearances)

    minimum_tyre_clearance = min(tyre_clearances) if tyre_clearances else float("nan")

    if reward is None:
        performance = 0.0
        style = 0.0
        weighted_style = 0.0
        penalty = 0.0
        total = 0.0
        prediction_performance = 0.0
        prediction_style = 0.0
    else:
        performance = reward.performance_step
        style = reward.style_step

        weighted_style = STYLE_WEIGHT * reward.style_step

        penalty = reward.termination_penalty

        total = reward.total

        prediction_performance = reward.performance_prediction

        prediction_style = reward.style_prediction

    reason = termination.reason.name if termination.reason is not None else "NONE"

    state = "RUNNING" if active else "TERMINATED - press R to restart"

    status_text.set_text(
        "\n".join(
            (
                f"STATE: {state}",
                "",
                "TRACK / PERFORMANCE",
                "-------------------",
                (f"s:               " f"{projection.distance_along_track:8.2f} m"),
                (f"progress:        " f"{projection.normalized_track_position:8.2%}"),
                (f"delta_s:         " f"{delta_s:+8.3f} m"),
                (f"performance:     " f"{performance:+8.3f}"),
                "",
                "STYLE / IMITATION",
                "-----------------",
                (f"lateral offset:  " f"{projection.lateral_offset:+8.3f} m"),
                (f"style score:     " f"{style:8.3f}"),
                (f"lambda_style:    " f"{STYLE_WEIGHT:8.3f}"),
                (f"weighted style:  " f"{weighted_style:+8.3f}"),
                "",
                "FUTURE PREDICTOR",
                "----------------",
                (f"performance:     " f"{prediction_performance:+8.3f}"),
                (f"style:           " f"{prediction_style:+8.3f}"),
                "",
                "TERMINATION",
                "-----------",
                (f"tyres inside:    " f"{inside_tyres}/4"),
                (f"min clearance:   " f"{minimum_tyre_clearance:+8.3f} m"),
                (f"reason:          " f"{reason}"),
                (f"penalty:         " f"{penalty:+8.3f}"),
                "",
                "TOTAL",
                "-----",
                (f"reward:          " f"{total:+8.3f}"),
                (f"episode time:    " f"{episode_time:8.2f} s"),
                "",
                "Controls:",
                "R = restart validation episode",
                "C = simulate strong collision",
                "Q = close",
            )
        )
    )


def main() -> None:
    """Run live reward and termination validation."""

    # ------------------------------------------------------------------
    # Assetto Corsa data
    # ------------------------------------------------------------------

    static_reader = StaticInfoReader()
    telemetry_reader = TelemetryReader()

    static_info = static_reader.read()

    track_reader = AssettoCorsaTrackReader(track_ai_path(static_info, Path(r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa")))

    track_geometry = track_reader.read()

    projector = TrackProjector(track_geometry)

    print(f"Track: {static_info.track_name}")

    print(f"Track length: " f"{projector.track_length:.1f} m")

    # ------------------------------------------------------------------
    # Reward and termination
    # ------------------------------------------------------------------

    reward_calculator = RewardCalculator(
        performance_weight=PERFORMANCE_WEIGHT,
        style_weight=STYLE_WEIGHT,
        style_sigma=STYLE_SIGMA,
        off_track_penalty=OFF_TRACK_PENALTY,
        stuck_penalty=STUCK_PENALTY,
        collision_penalty=COLLISION_PENALTY,
    )

    termination_checker = EpisodeTerminationChecker(
        track_projector=projector,
        stuck_time=STUCK_TIME,
        minimum_progress=MINIMUM_PROGRESS,
    )

    # ------------------------------------------------------------------
    # Complete track geometry
    # ------------------------------------------------------------------

    (
        center_x,
        center_z,
        left_x,
        left_z,
        right_x,
        right_z,
    ) = _extract_track_geometry(track_geometry)

    # ------------------------------------------------------------------
    # Plot layout
    # ------------------------------------------------------------------

    plt.ion()

    figure, axes = plt.subplot_mosaic(
        [
            ["track", "history"],
            ["track", "style"],
            ["track", "status"],
        ],
        figsize=(15, 9),
        gridspec_kw={
            "width_ratios": [1.45, 1.0],
            "height_ratios": [1.1, 1.0, 1.3],
        },
    )

    track_axis = axes["track"]
    history_axis = axes["history"]
    style_axis = axes["style"]
    status_axis = axes["status"]

    # ------------------------------------------------------------------
    # Spatial track view
    # ------------------------------------------------------------------

    (full_left_line,) = track_axis.plot(
        left_x,
        left_z,
        linewidth=1.2,
        color="0.55",
        label="Left boundary",
        zorder=1,
    )

    (full_right_line,) = track_axis.plot(
        right_x,
        right_z,
        linewidth=1.2,
        color="0.55",
        label="Right boundary",
        zorder=1,
    )

    (reference_line,) = track_axis.plot(
        center_x,
        center_z,
        linestyle="--",
        linewidth=1.2,
        color="0.25",
        label="Reference spline",
        zorder=2,
    )

    (vehicle_marker,) = track_axis.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=9,
        color="tab:blue",
        label="Vehicle",
        zorder=8,
    )

    (inside_tyre_marker,) = track_axis.plot(
        [],
        [],
        marker="x",
        linestyle="None",
        markersize=8,
        markeredgewidth=2.5,
        color="tab:green",
        label="Tyre inside",
        zorder=10,
    )

    (outside_tyre_marker,) = track_axis.plot(
        [],
        [],
        marker="x",
        linestyle="None",
        markersize=10,
        markeredgewidth=3.0,
        color="tab:red",
        label="Tyre outside",
        zorder=11,
    )

    (projection_marker,) = track_axis.plot(
        [],
        [],
        marker="D",
        linestyle="None",
        markersize=7,
        color="tab:purple",
        label="Reference projection",
        zorder=9,
    )

    (projection_line,) = track_axis.plot(
        [],
        [],
        linestyle="-",
        linewidth=2.0,
        color="tab:purple",
        label="Lateral offset",
        zorder=7,
    )

    track_axis.set_title("Track projection and termination geometry")

    track_axis.set_xlabel("Vehicle right [m]")

    track_axis.set_ylabel("Vehicle forward [m]")

    track_axis.set_xlim(
        -VIEW_LATERAL_METRES,
        VIEW_LATERAL_METRES,
    )

    track_axis.set_ylim(
        -VIEW_BEHIND_METRES,
        VIEW_FORWARD_METRES,
    )

    track_axis.set_aspect(
        "equal",
        adjustable="box",
    )

    track_axis.grid(True)

    track_axis.legend(
        loc="upper left",
    )

    # All geometry remains in world coordinates.
    # The transform only changes how it is displayed.
    view_transform = Affine2D()

    world_artists = (
        full_left_line,
        full_right_line,
        reference_line,
        vehicle_marker,
        inside_tyre_marker,
        outside_tyre_marker,
        projection_marker,
        projection_line,
    )

    for artist in world_artists:
        artist.set_transform(view_transform + track_axis.transData)

    # ------------------------------------------------------------------
    # Reward history
    # ------------------------------------------------------------------

    (performance_line,) = history_axis.plot(
        [],
        [],
        label="Performance",
    )

    (style_history_line,) = history_axis.plot(
        [],
        [],
        label="Weighted style",
    )

    (total_line,) = history_axis.plot(
        [],
        [],
        linewidth=2.0,
        label="Total reward",
    )

    (penalty_line,) = history_axis.plot(
        [],
        [],
        linestyle="--",
        label="Termination penalty",
    )

    history_axis.set_title("Reward history")

    history_axis.set_xlabel("Episode time [s]")

    history_axis.set_ylabel("Reward")

    history_axis.grid(True)

    history_axis.legend(
        loc="upper left",
    )

    # ------------------------------------------------------------------
    # Style reward curve
    # ------------------------------------------------------------------

    (
        style_distances,
        style_scores,
    ) = _build_style_curve(reward_calculator)

    style_axis.plot(
        style_distances,
        style_scores,
        label="Style reward",
    )

    (style_current_marker,) = style_axis.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=9,
        label="Current state",
    )

    style_axis.axvline(
        STYLE_SIGMA,
        linestyle="--",
        linewidth=1.0,
        color="0.5",
        label="sigma",
    )

    style_axis.set_title("Style imitation reward")

    style_axis.set_xlabel("Absolute lateral offset [m]")

    style_axis.set_ylabel("Unweighted style score")

    style_axis.set_xlim(
        0.0,
        STYLE_CURVE_MAX_DISTANCE,
    )

    style_axis.set_ylim(
        -0.05,
        1.05,
    )

    style_axis.grid(True)

    style_axis.legend(
        loc="upper right",
    )

    # ------------------------------------------------------------------
    # Status panel
    # ------------------------------------------------------------------

    status_axis.axis("off")

    status_text = status_axis.text(
        0.0,
        1.0,
        "",
        va="top",
        family="monospace",
        fontsize=9,
        transform=status_axis.transAxes,
    )

    figure.suptitle(f"{static_info.track_name}: Reward and Termination Validation")

    figure.tight_layout()

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    time_history: deque[float] = deque()
    performance_history: deque[float] = deque()
    style_history: deque[float] = deque()
    total_history: deque[float] = deque()
    penalty_history: deque[float] = deque()

    # ------------------------------------------------------------------
    # Interactive validation state
    # ------------------------------------------------------------------

    episode_active = True
    reset_requested = False
    collision_requested = False

    previous_s: float | None = None

    episode_time = 0.0
    last_loop_time = perf_counter()

    last_reward: Reward | None = None

    last_termination = TerminationResult(
        terminated=False,
        reason=None,
    )

    last_delta_s = 0.0

    # Retain last valid heading if tyre geometry briefly becomes invalid.
    forward_direction = (
        0.0,
        -1.0,
    )

    def on_key(event: KeyEvent) -> None:
        """Handle validation keyboard shortcuts."""

        nonlocal reset_requested
        nonlocal collision_requested

        if event.key is None:
            return

        key = event.key.lower()

        if key == "r":
            reset_requested = True

        elif key == "c":
            collision_requested = True

        elif key == "q":
            plt.close(figure)

    figure.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    try:
        while plt.fignum_exists(figure.number):
            now = perf_counter()

            dt = max(
                now - last_loop_time,
                1e-6,
            )

            last_loop_time = now

            telemetry = telemetry_reader.read()

            projection = projector.project(telemetry.car_coordinates)

            # ----------------------------------------------------------
            # Vehicle-relative display transform
            # ----------------------------------------------------------

            forward_direction = _update_vehicle_view_transform(
                transform=view_transform,
                telemetry=telemetry,
                previous_forward=forward_direction,
            )

            # ----------------------------------------------------------
            # Spatial visualization
            # ----------------------------------------------------------

            vehicle_marker.set_data(
                [telemetry.car_coordinates[0]],
                [telemetry.car_coordinates[2]],
            )

            projection_marker.set_data(
                [projection.position[0]],
                [projection.position[2]],
            )

            projection_line.set_data(
                [
                    telemetry.car_coordinates[0],
                    projection.position[0],
                ],
                [
                    telemetry.car_coordinates[2],
                    projection.position[2],
                ],
            )

            (
                inside_tyres,
                outside_tyres,
                tyre_clearances,
            ) = _tyre_track_state(
                telemetry=telemetry,
                projector=projector,
            )

            _set_marker_positions(
                inside_tyre_marker,
                inside_tyres,
            )

            _set_marker_positions(
                outside_tyre_marker,
                outside_tyres,
            )

            # ----------------------------------------------------------
            # Manual episode reset
            # ----------------------------------------------------------

            if reset_requested:
                termination_checker.reset()

                previous_s = projection.distance_along_track

                episode_time = 0.0

                time_history.clear()
                performance_history.clear()
                style_history.clear()
                total_history.clear()
                penalty_history.clear()

                last_reward = None
                last_delta_s = 0.0

                last_termination = TerminationResult(
                    terminated=False,
                    reason=None,
                )

                episode_active = True
                collision_requested = False
                reset_requested = False

                last_loop_time = now

            # ----------------------------------------------------------
            # Reward and termination
            # ----------------------------------------------------------

            if episode_active:
                episode_time += dt

                current_s = projection.distance_along_track

                if previous_s is None:
                    delta_s = 0.0

                else:
                    delta_s = calculate_progress_delta(
                        current_s=current_s,
                        previous_s=previous_s,
                        track_length=projector.track_length,
                    )

                previous_s = current_s

                termination = termination_checker.check(
                    telemetry=telemetry,
                    delta_s=delta_s,
                    dt=dt,
                    strong_collision=collision_requested,
                )

                collision_requested = False

                reward = reward_calculator.calculate(
                    delta_s=delta_s,
                    lateral_offset=projection.lateral_offset,
                    termination=termination,
                )

                last_reward = reward
                last_delta_s = delta_s
                last_termination = termination

                # ------------------------------------------------------
                # Reward history
                # ------------------------------------------------------

                time_history.append(episode_time)

                performance_history.append(reward.performance_step)

                style_history.append(STYLE_WEIGHT * reward.style_step)

                total_history.append(reward.total)

                penalty_history.append(reward.termination_penalty)

                _trim_history(
                    episode_time=episode_time,
                    time_history=time_history,
                    histories=(
                        performance_history,
                        style_history,
                        total_history,
                        penalty_history,
                    ),
                )

                # ------------------------------------------------------
                # Termination freezes the current validation episode.
                # ------------------------------------------------------

                if termination.terminated:
                    episode_active = False

            # ----------------------------------------------------------
            # Reward history plot
            # ----------------------------------------------------------

            _update_reward_history(
                history_axis=history_axis,
                episode_time=episode_time,
                time_history=time_history,
                performance_history=performance_history,
                style_history=style_history,
                total_history=total_history,
                penalty_history=penalty_history,
                performance_line=performance_line,
                style_line=style_history_line,
                total_line=total_line,
                penalty_line=penalty_line,
            )

            # ----------------------------------------------------------
            # Style curve current operating point
            # ----------------------------------------------------------

            if last_reward is not None:
                style_current_marker.set_data(
                    [abs(projection.lateral_offset)],
                    [last_reward.style_step],
                )

            # ----------------------------------------------------------
            # Text status
            # ----------------------------------------------------------

            _update_status(
                status_text=status_text,
                active=episode_active,
                projection=projection,
                delta_s=last_delta_s,
                reward=last_reward,
                termination=last_termination,
                tyre_clearances=tyre_clearances,
                episode_time=episode_time,
            )

            figure.canvas.draw_idle()
            figure.canvas.flush_events()

            plt.pause(0.02)

    except KeyboardInterrupt:
        pass

    finally:
        telemetry_reader.close()
        static_reader.close()

        plt.close(figure)


if __name__ == "__main__":
    main()
