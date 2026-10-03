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
    """Plot the track spline and live vehicle position."""

    track_file = Path(
        r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
        r"\content\tracks\monza\ai\fast_lane.ai"
    )

    static_reader = StaticInfoReader()
    static_info = static_reader.read()
    print(f"Track:        {static_info.track_name}")    

    track_reader = AssettoCorsaTrackReader(track_file)
    positions = track_reader.read_positions()

    track_x = [position[0] for position in positions]
    track_z = [position[2] for position in positions]

    centerline, left_boundary, right_boundary = track_reader.read_border_lines()

    center_x = [point[0] for point in centerline]
    center_z = [point[2] for point in centerline]

    left_x = [point[0] for point in left_boundary]
    left_z = [point[2] for point in left_boundary]

    right_x = [point[0] for point in right_boundary]
    right_z = [point[2] for point in right_boundary]

    telemetry_reader = TelemetryReader()

    plt.ion()

    figure, axis = plt.subplots()

    axis.plot(
        track_x,
        track_z,
        label="Track spline",
    )

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

    axis.set_xlabel("World X [m]")
    axis.set_ylabel("World Z [m]")
    axis.axis("equal")
    axis.invert_yaxis() # Invert the y-axis to match the world coordinate system in Assetto Corsa
    axis.legend()

    try:
        while plt.fignum_exists(figure.number):
            telemetry = telemetry_reader.read()

            vehicle_marker.set_data(
                [telemetry.car_coordinates[0]],
                [telemetry.car_coordinates[2]],
            )

            figure.canvas.draw_idle()
            figure.canvas.flush_events()

            plt.pause(0.02)

    finally:
        telemetry_reader.close()
        plt.close(figure)


if __name__ == "__main__":
    main()
