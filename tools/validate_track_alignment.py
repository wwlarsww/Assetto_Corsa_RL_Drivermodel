"""Visually validate Assetto Corsa track and vehicle coordinate alignment."""

# Matplotlib leaves some method signatures incomplete (kwargs and context managers).
# Keep strict checks, but exclude unknown member types in this plotting script.
# pyright: reportUnknownMemberType=false

from math import hypot, isfinite
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.assetto_corsa.track_reader import AssettoCorsaTrackReader
from assetto_corsa_ai.assetto_corsa.static_info_reader import StaticInfoReader
from assetto_corsa_ai.track_projection import TrackProjector


# Half the visible width and height around the vehicle, in world metres.
VIEW_RADIUS_METRES = 30.0


def main() -> None:
    """Plot track geometry and live vehicle position and projection onto the AI spline."""

    track_file = Path(
        r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
        r"\content\tracks\monza\ai\fast_lane.ai"
    )

    static_reader = StaticInfoReader()
    static_info = static_reader.read()

    print(f"Track: {static_info.track_name}")

    track_reader = AssettoCorsaTrackReader(track_file)
    track_geometry = track_reader.read()
    # Create a TrackProjector object with the track geometry.
    # The projector will be used to find the nearest point on the AI spline for the vehicle's position
    projector = TrackProjector(track_geometry)

    # Extract the X and Z coordinates of the track points, left boundaries, and right boundaries for plotting.
    center_x = [point.position[0] for point in track_geometry.points]

    center_z = [point.position[2] for point in track_geometry.points]

    left_x = [point.left_boundary[0] for point in track_geometry.points]

    left_z = [point.left_boundary[2] for point in track_geometry.points]

    right_x = [point.right_boundary[0] for point in track_geometry.points]

    right_z = [point.right_boundary[2] for point in track_geometry.points]

    # Include the closing segment, which the projector also searches.
    center_x.append(center_x[0])
    center_z.append(center_z[0])
    left_x.append(left_x[0])
    left_z.append(left_z[0])
    right_x.append(right_x[0])
    right_z.append(right_z[0])

    telemetry_reader = TelemetryReader()

    plt.ion()

    figure, axis = plt.subplots(figsize=(11, 8))

    axis.plot(
        center_x,
        center_z,
        label="AI spline",
        color="0.65",
    )

    axis.plot(
        left_x,
        left_z,
        label="Left boundary",
    )

    axis.plot(
        right_x,
        right_z,
        label="Right boundary",
    )

    (progress_line,) = axis.plot(
        [],
        [],
        color="tab:purple",
        linewidth=2.5,
        label="Spline start to projection",
        zorder=3,
    )

    (vehicle_marker,) = axis.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        label="Vehicle",
        color="tab:red",
        markersize=8,
        zorder=5,
    )

    (tyre_marker,) = axis.plot(
        [],
        [],
        marker="x",
        linestyle="None",
        markersize=6,
        markeredgewidth=3,
        label="Tyre contact points",
        zorder=20,
    )

    (projection_marker,) = axis.plot(
        [],
        [],
        marker="D",
        linestyle="None",
        color="tab:purple",
        markersize=7,
        label="Spline projection",
        zorder=6,
    )
    (projection_line,) = axis.plot(
        [], [], linestyle="--",
        color="0.3",
        label="Vehicle to projection",
        zorder=4
    )

    status = figure.text(0.02, 0.02, "", fontsize=10, va="bottom", family="monospace")
    figure.subplots_adjust(bottom=0.20)

    axis.set_title(f"{static_info.track_name}: vehicle and spline projection")
    axis.set_xlabel("Vehicle right [m]")
    axis.set_ylabel("Vehicle forward [m]")
    # Keep the requested limits while displaying equal world distances.
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-VIEW_RADIUS_METRES, VIEW_RADIUS_METRES)
    axis.set_ylim(-VIEW_RADIUS_METRES, VIEW_RADIUS_METRES)

    # Apply the same display transform to the road and every live marker.
    # All source data and the projector remain in native world coordinates.
    view_transform = Affine2D()
    for line in axis.lines:
        line.set_transform(view_transform + axis.transData)

    # Retain the last valid orientation if contact points are unavailable.
    forward_x, forward_z = 0.0, -1.0

    axis.legend()

    try:
        while plt.fignum_exists(figure.number):
            telemetry = telemetry_reader.read()
            projection = projector.project(telemetry.car_coordinates)
            projection_marker.set_data(
                [projection.position[0]], [projection.position[2]]
            )
            projection_line.set_data(
                [telemetry.car_coordinates[0], projection.position[0]],
                [telemetry.car_coordinates[2], projection.position[2]],
            )
            # This is progress in sample order, not the vehicle's driven path.
            # At start/finish the highlighted section resets with progress.
            progress_line.set_data(
                center_x[: projection.segment_index + 1] + [projection.position[0]],
                center_z[: projection.segment_index + 1] + [projection.position[2]],
            )
            status.set_text(
                f"Projected progress: {projection.distance_along_track:.1f} / "
                f"{projector.track_length:.1f} m "
                f"({projection.normalized_track_position:.2%})\n"
                f"AC telemetry progress: {telemetry.normalized_track_position:.2%} | "
                f"Segment: {projection.segment_index}\n"
                f"Lateral offset (+left): {projection.lateral_offset:+.2f} m | "
                f"3D distance to spline: {projection.distance_to_spline:.2f} m\n"
                f"Lateral margins L/R: {projection.left_clearance:+.2f} / "
                f"{projection.right_clearance:+.2f} m (negative = beyond edge)"
            )

            vehicle_marker.set_data(
                [telemetry.car_coordinates[0]],
                [telemetry.car_coordinates[2]],
            )
            tyre_marker.set_data(
                [point[0] for point in telemetry.tyre_contact_points],
                [point[2] for point in telemetry.tyre_contact_points],
            )

            vehicle_x, _, vehicle_z = telemetry.car_coordinates
            front_left, front_right, rear_left, rear_right = telemetry.tyre_contact_points
            heading_x = (
                front_left[0] + front_right[0] - rear_left[0] - rear_right[0]
            ) / 2.0
            heading_z = (
                front_left[2] + front_right[2] - rear_left[2] - rear_right[2]
            ) / 2.0
            heading_length = hypot(heading_x, heading_z)
            if isfinite(heading_length) and heading_length > 1e-6:
                forward_x = heading_x / heading_length
                forward_z = heading_z / heading_length

            right_x_direction, right_z_direction = -forward_z, forward_x
            view_transform.set_matrix(
                Affine2D.from_values(
                    right_x_direction,
                    forward_x,
                    right_z_direction,
                    forward_z,
                    -(vehicle_x * right_x_direction + vehicle_z * right_z_direction),
                    -(vehicle_x * forward_x + vehicle_z * forward_z),
                ).get_matrix()
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
