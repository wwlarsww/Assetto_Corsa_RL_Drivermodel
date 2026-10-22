"""Visually validate the complete RL observation in Assetto Corsa."""

# Matplotlib leaves some method signatures incomplete (kwargs and context managers).
# Keep strict checks, but exclude unknown member types in this plotting script.
# pyright: reportUnknownMemberType=false

from math import degrees
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Affine2D

from assetto_corsa_ai.assetto_corsa.static_info_reader import StaticInfoReader
from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.assetto_corsa.track_reader import AssettoCorsaTrackReader, track_ai_path
from assetto_corsa_ai.observation import Observation
from assetto_corsa_ai.observation_builder import ObservationBuilder
from assetto_corsa_ai.telemetry import TelemetryFrame
from assetto_corsa_ai.track_perception import TrackPerceptionCalculator
from assetto_corsa_ai.track_projection import TrackProjection, TrackProjector
from assetto_corsa_ai.vehicle_geometry import estimate_axle_distances

# Minimum visible lateral range on each side of the vehicle.
MIN_LATERAL_VIEW_METRES = 20.0

# Minimum forward range shown even when only short-distance points are present.
MIN_FORWARD_VIEW_METRES = 110.0


def _vehicle_plot_transform(telemetry: TelemetryFrame) -> Affine2D:
    """Map world X/Z to the same vehicle frame as the observation points."""

    (origin_x, origin_z), (forward_x, forward_z), (left_x, left_z) = TrackPerceptionCalculator.vehicle_frame(telemetry.tyre_contact_points)
    right_x, right_z = -left_x, -left_z

    return Affine2D.from_values(
        right_x,
        forward_x,
        right_z,
        forward_z,
        -(origin_x * right_x + origin_z * right_z),
        -(origin_x * forward_x + origin_z * forward_z),
    )


def _plot_coordinates(
    points: tuple[tuple[float, float], ...],
) -> tuple[list[float], list[float]]:
    """Convert observation coordinates into plotting coordinates.

    Observation coordinates are stored as:

        (forward_distance, left_distance)

    The plot uses:

        x = vehicle right
        y = vehicle forward

    Therefore the lateral coordinate must be negated.
    """

    x = [-left_distance for _, left_distance in points]

    y = [forward_distance for forward_distance, _ in points]

    return x, y


def _update_view(
    axis: Axes,
    observation: Observation,
) -> None:
    """Adjust the plot limits to keep all observation points visible."""

    all_points = (
        observation.left_boundary
        + observation.right_boundary
        + observation.reference_waypoints
    )

    if not all_points:
        return

    plot_x = [-left_distance for _, left_distance in all_points]

    plot_y = [forward_distance for forward_distance, _ in all_points]

    max_lateral = max(
        MIN_LATERAL_VIEW_METRES,
        max(abs(value) for value in plot_x) * 1.15,
    )

    minimum_forward = min(
        -10.0,
        min(plot_y) - 10.0,
    )

    maximum_forward = max(
        MIN_FORWARD_VIEW_METRES,
        max(plot_y) + 20.0,
    )

    axis.set_xlim(
        -max_lateral,
        max_lateral,
    )

    axis.set_ylim(
        minimum_forward,
        maximum_forward,
    )


def _update_status(
    status: Text,
    observation: Observation,
    projection: TrackProjection,
) -> None:
    """Update textual observation information."""

    left_correct = sum(point[1] > 0.0 for point in observation.left_boundary)

    right_correct = sum(point[1] < 0.0 for point in observation.right_boundary)

    status.set_text(
        "\n".join(
            (
                "VEHICLE STATE",
                "-------------",
                f"Speed:        {observation.speed:8.2f} m/s",
                ("Long. acc.:   " f"{observation.longitudinal_acceleration:+8.3f} g"),
                ("Lat. acc.:    " f"{observation.lateral_acceleration:+8.3f} g"),
                ("Yaw rate:     " f"{observation.yaw_rate:+8.3f} rad/s"),
                "",
                "TYRE DYNAMICS",
                "-------------",
                ("Slip ratio F: " f"{observation.front_slip_ratio:+8.3f}"),
                ("Slip ratio R: " f"{observation.rear_slip_ratio:+8.3f}"),
                ("Slip angle F: " f"{observation.front_slip_angle:+8.3f} rad"),
                ("              " f"{degrees(observation.front_slip_angle):+8.2f} deg"),
                ("Slip angle R: " f"{observation.rear_slip_angle:+8.3f} rad"),
                ("              " f"{degrees(observation.rear_slip_angle):+8.2f} deg"),
                "",
                "CONTROLS",
                "-------------",
                ("Steering:     " f"{observation.steering:+8.3f}"),
                ("Long. control:" f"{observation.longitudinal_control:+8.3f}"),
                "",
                "TRACK",
                "-------------",
                ("Track s:      " f"{projection.distance_along_track:8.1f} m"),
                ("Progress:     " f"{projection.normalized_track_position:8.2%}"),
                ("Lat. offset:  " f"{projection.lateral_offset:+8.2f} m"),
                "",
                (
                    "Left points:  "
                    f"{len(observation.left_boundary)} "
                    f"({left_correct} left)"
                ),
                (
                    "Right points: "
                    f"{len(observation.right_boundary)} "
                    f"({right_correct} right)"
                ),
                ("Reference:    " f"{len(observation.reference_waypoints)} points"),
            )
        )
    )


def main() -> None:
    """Visualize the complete RL observation in vehicle coordinates."""

    # --------------------------------------------------------------
    # Read static vehicle and track data
    # --------------------------------------------------------------

    static_reader = StaticInfoReader()
    static_info = static_reader.read()

    print(f"Track: {static_info.track_name}")

    track_reader = AssettoCorsaTrackReader(track_ai_path(static_info, Path(r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa")))

    track_geometry = track_reader.read()

    # --------------------------------------------------------------
    # Track projection
    # --------------------------------------------------------------

    projector = TrackProjector(track_geometry)

    print(f"Track length: " f"{projector.track_length:.1f} m")

    # --------------------------------------------------------------
    # Live telemetry
    # --------------------------------------------------------------

    telemetry_reader = TelemetryReader()
    figure: Figure | None = None

    try:
        initial_telemetry = telemetry_reader.read()

        front_axle_distance, rear_axle_distance = estimate_axle_distances(
            initial_telemetry
        )

        print("Front axle distance: " f"{front_axle_distance:.3f} m")

        print("Rear axle distance:  " f"{rear_axle_distance:.3f} m")

        # ----------------------------------------------------------
        # Observation pipeline
        # ----------------------------------------------------------

        observation_builder = ObservationBuilder.create(
            front_axle_distance=front_axle_distance, rear_axle_distance=rear_axle_distance,
        )

        # ----------------------------------------------------------
        # Create live plot
        # ----------------------------------------------------------

        plt.ion()

        figure, axis = plt.subplots(figsize=(12, 9))

        figure.subplots_adjust(right=0.70)

        # Cache full world boundaries once; only their display frame changes.
        boundary_transform = Affine2D()
        for boundary_name, color, label in (
            ("left_boundary", "tab:blue", "Left track edge"),
            ("right_boundary", "tab:orange", "Right track edge"),
        ):
            boundary = [getattr(point, boundary_name) for point in track_geometry.points]
            boundary.append(boundary[0])
            axis.plot(
                [point[0] for point in boundary],
                [point[2] for point in boundary],
                color=color,
                linewidth=1.2,
                alpha=0.5,
                zorder=1,
                label=label,
                transform=boundary_transform + axis.transData,
            )

        # Sampled observation points on the left track boundary.
        (left_boundary_line,) = axis.plot(
            [],
            [],
            marker="o",
            linestyle="--",
            linewidth=1.0,
            markersize=6,
            color="tab:blue",
            zorder=3,
            label="Left lookahead points",
        )

        # Sampled observation points on the right track boundary.
        (right_boundary_line,) = axis.plot(
            [],
            [],
            marker="o",
            linestyle="--",
            linewidth=1.0,
            markersize=6,
            color="tab:orange",
            zorder=3,
            label="Right lookahead points",
        )

        # Reference trajectory.
        (reference_line,) = axis.plot(
            [],
            [],
            marker="x",
            linestyle="-",
            linewidth=1.5,
            markersize=5,
            color="tab:green",
            zorder=3,
            label="Reference trajectory",
        )

        # The observation coordinate system is vehicle-relative,
        # therefore the vehicle always remains at the origin.
        axis.plot(
            0.0,
            0.0,
            marker="^",
            linestyle="None",
            markersize=12,
            label="Vehicle",
            zorder=10,
        )

        # Vehicle forward direction.
        axis.plot(
            [0.0, 0.0],
            [0.0, 10.0],
            linestyle="--",
            linewidth=1.0,
            label="Vehicle forward",
        )

        # Vehicle lateral and longitudinal axes.
        axis.axhline(
            0.0,
            linewidth=0.7,
            color="0.7",
        )

        axis.axvline(
            0.0,
            linewidth=0.7,
            color="0.7",
        )

        # Labels for the seven boundary lookahead distances.
        boundary_labels = [
            axis.text(
                0.0,
                0.0,
                "",
                fontsize=8,
            )
            for _ in (TrackPerceptionCalculator.DEFAULT_LOOKAHEAD_DISTANCES)
        ]

        # Observation status panel.
        status = figure.text(
            0.72,
            0.93,
            "",
            fontsize=9,
            va="top",
            family="monospace",
        )

        axis.set_title(f"{static_info.track_name}: RL observation validation")

        axis.set_xlabel("Vehicle right [m]")

        axis.set_ylabel("Vehicle forward [m]")

        axis.set_aspect(
            "equal",
            adjustable="box",
        )

        axis.grid(True)

        axis.legend(loc="upper left")

        # ----------------------------------------------------------
        # Live update
        # ----------------------------------------------------------

        while plt.fignum_exists(figure.number):
            telemetry = telemetry_reader.read()

            # Locate the vehicle on the AI spline.
            projection = projector.project(telemetry.car_coordinates)

            # Build the exact observation that will later be
            # provided to the RL agent.
            observation = observation_builder.build(
                telemetry=telemetry,
                static_info=static_info,
                track_geometry=track_geometry,
                current_s=projection.spline_distance(track_geometry, static_info.track_spline_length),
            )

            # ------------------------------------------------------
            # Convert observation coordinates to plot coordinates
            # ------------------------------------------------------

            left_x, left_y = _plot_coordinates(observation.left_boundary)

            right_x, right_y = _plot_coordinates(observation.right_boundary)

            reference_x, reference_y = _plot_coordinates(
                observation.reference_waypoints
            )

            # ------------------------------------------------------
            # Update geometry
            # ------------------------------------------------------

            boundary_transform.set_matrix(
                _vehicle_plot_transform(telemetry).get_matrix()
            )

            left_boundary_line.set_data(
                left_x,
                left_y,
            )

            right_boundary_line.set_data(
                right_x,
                right_y,
            )

            reference_line.set_data(
                reference_x,
                reference_y,
            )

            # ------------------------------------------------------
            # Label boundary lookahead distances
            # ------------------------------------------------------

            for (
                label,
                distance,
                x,
                y,
            ) in zip(
                boundary_labels,
                TrackPerceptionCalculator.DEFAULT_LOOKAHEAD_DISTANCES,
                left_x,
                left_y,
            ):
                label.set_position((x, y))

                label.set_text(f"{distance:g} m")

            # ------------------------------------------------------
            # Status information
            # ------------------------------------------------------

            _update_status(
                status=status,
                observation=observation,
                projection=projection,
            )

            # Keep all observed points visible.
            _update_view(
                axis=axis,
                observation=observation,
            )

            figure.canvas.draw_idle()
            figure.canvas.flush_events()

            plt.pause(0.02)

    except KeyboardInterrupt:
        pass

    finally:
        telemetry_reader.close()
        static_reader.close()

        if figure is not None:
            plt.close(figure)


if __name__ == "__main__":
    main()
