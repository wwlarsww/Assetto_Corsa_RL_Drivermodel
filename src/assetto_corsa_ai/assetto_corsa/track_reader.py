"""Read track data from Assetto Corsa AI spline files."""


import struct
from pathlib import Path
from assetto_corsa_ai.track_geometry import TrackGeometry, TrackPoint


class AssettoCorsaTrackReader:
    """Read track geometry from an Assetto Corsa fast_lane.ai file."""

    def __init__(self, file_path: Path) -> None:
        """Store the path to the Assetto Corsa AI spline file."""

        self._file_path = file_path

    def read(self) -> TrackGeometry:
        """Read the complete track geometry."""

        with self._file_path.open("rb") as file:
            version = struct.unpack("<i", file.read(4))[0]
            point_count = struct.unpack("<i", file.read(4))[0]

            # Currently unused header values.
            _lap_time = struct.unpack("<i", file.read(4))[0]
            _sample_count = struct.unpack("<i", file.read(4))[0]

            # Validate the file version. The current implementation only supports version 7 of the fast_lane.ai format.
            if version != 7:
                raise ValueError(f"Unsupported fast_lane.ai version: {version}")

            # Read the base points, which contain the position and length of each sampled point along the track.
            base_points: list[
                tuple[
                    tuple[float, float, float],
                    float,
                ]
            ] = []

            for _ in range(point_count):
                x, y, z, length, _point_id = struct.unpack(
                    "<ffffi",
                    file.read(20),
                )

                base_points.append(
                    (
                        (float(x), float(y), float(z)),
                        float(length),
                    )
                )

            # Read the extra points, which contain additional information about each sampled point along the track.
            extra_count = struct.unpack("<i", file.read(4))[0]

            if extra_count != point_count:
                raise ValueError(
                    "Number of spline points does not match " "number of extra points."
                )

            points: list[TrackPoint] = []

            for index in range(point_count):
                values = struct.unpack(
                    "<18f",
                    file.read(18 * 4),
                )

                position, length = base_points[index]

                # Read the left and right side offsets from current position to the track
                side_left = float(values[5])
                side_right = float(values[6])

                # Read the normal forward vector
                normal = (
                    float(values[9]),
                    float(values[10]),
                    float(values[11]),
                )

                # Read the forward vector, which indicates the direction of the track at this point
                forward = (
                    float(values[13]),
                    float(values[14]),
                    float(values[15]),
                )

                points.append(
                    TrackPoint(
                        position=position,
                        length=length,
                        side_left=side_left,
                        side_right=side_right,
                        normal=normal,
                        forward=forward,
                        reference_speed=float(values[0]),
                    )
                )

        return TrackGeometry(points)
