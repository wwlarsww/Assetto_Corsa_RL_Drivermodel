"""Visually validate Assetto Corsa track and vehicle coordinate alignment."""

# Matplotlib leaves some method signatures incomplete (kwargs and context managers).
# Keep strict checks, but exclude unknown member types in this plotting script.
# pyright: reportUnknownMemberType=false

from pathlib import Path

import matplotlib.pyplot as plt

from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.assetto_corsa.track_reader import AssettoCorsaTrackReader
from assetto_corsa_ai.assetto_corsa.static_info_reader import StaticInfoReader


def main() -> None:
    """Plot track geometry and live vehicle position."""

    track_file = Path(
        r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
        r"\content\tracks\monza\ai\fast_lane.ai"
    )

    static_reader = StaticInfoReader()
    static_info = static_reader.read()

    print(f"Track: {static_info.track_name}")

    track_reader = AssettoCorsaTrackReader(track_file)
    track_geometry = track_reader.read()

    center_x = [point.position[0] for point in track_geometry.points]

    center_z = [point.position[2] for point in track_geometry.points]

    left_x = [point.left_boundary[0] for point in track_geometry.points]

    left_z = [point.left_boundary[2] for point in track_geometry.points]

    right_x = [point.right_boundary[0] for point in track_geometry.points]

    right_z = [point.right_boundary[2] for point in track_geometry.points]

    telemetry_reader = TelemetryReader()

    plt.ion()

    figure, axis = plt.subplots()

    axis.plot(
        center_x,
        center_z,
        label="AI spline",
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

    (vehicle_marker,) = axis.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        label="Vehicle",
    )

    (tyre_marker,) = axis.plot(
        [],
        [],
        marker="x",
        linestyle="None",
        markersize=12,
        markeredgewidth=3,
        label="Tyre contact points",
        zorder=20,
    )

    axis.set_xlabel("World X [m]")
    axis.set_ylabel("World Z [m]")
    axis.axis("equal")

    # Visualization only: positive Assetto Corsa Z points in the
    # opposite direction to the conventional plot orientation.
    axis.invert_yaxis()

    axis.legend()

    try:
        while plt.fignum_exists(figure.number):
            telemetry = telemetry_reader.read()

            vehicle_marker.set_data(
                [telemetry.car_coordinates[0]],
                [telemetry.car_coordinates[2]],
            )
            tyre_marker.set_data(
                [point[0] for point in telemetry.tyre_contact_points],
                [point[2] for point in telemetry.tyre_contact_points],
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
