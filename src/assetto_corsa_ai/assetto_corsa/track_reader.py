"""Read track data from Assetto Corsa AI spline files."""


from dataclasses import dataclass
import math
import struct
from pathlib import Path

@dataclass
class TrackPoint:
    """Represent one point of the Assetto Corsa AI spline."""

    position: tuple[float, float, float]
    length: float
    point_id: int

    side_left: float
    side_right: float

    normal: tuple[float, float, float]
    forward: tuple[float, float, float]


def _normalize(
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the normalized version of a 3D vector."""

    x, y, z = vector
    norm = math.sqrt(x * x + y * y + z * z)

    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector.")

    return (x / norm, y / norm, z / norm)


def _cross(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the cross product of two 3D vectors."""

    ax, ay, az = a
    bx, by, bz = b

    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    )


def _scale(
    vector: tuple[float, float, float],
    scalar: float,
) -> tuple[float, float, float]:
    """Scale a 3D vector by a scalar."""

    x, y, z = vector
    return (x * scalar, y * scalar, z * scalar)


def _add(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Add two 3D vectors."""

    return (
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
    )


def _sub(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Subtract vector b from vector a."""

    return (
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
    )


class AssettoCorsaTrackReader:
    """Read track geometry from an Assetto Corsa fast_lane.ai file."""

    def __init__(self, file_path: Path) -> None:
        """Store the path to the Assetto Corsa AI spline file."""

        self._file_path = file_path


    def inspect(self) -> None:
        """Debug function to read and print the header and first spline points."""

        with self._file_path.open("rb") as file:
            version = struct.unpack("<i", file.read(4))[0]
            point_count = struct.unpack("<i", file.read(4))[0]
            lap_time = struct.unpack("<i", file.read(4))[0]
            sample_count = struct.unpack("<i", file.read(4))[0]

            print(f"Version:      {version}")
            print(f"Point count:  {point_count}")
            print(f"Lap time:     {lap_time}")
            print(f"Sample count: {sample_count}")
            print()

            # Read only the first five spline points for inspection.
            for index in range(min(5, point_count)):
                x, y, z, length, point_id = struct.unpack(
                    "<ffffi",
                    file.read(20),
                )

                print(
                    f"Point {index}: "
                    f"position=({x:.3f}, {y:.3f}, {z:.3f}), "
                    f"length={length:.3f}, "
                    f"id={point_id}"
                )

            # Jump over the complete main spline point block.
            file.seek(16 + point_count * 20)

            extra_count = struct.unpack("<i", file.read(4))[0]

            print()
            print(f"Extra count:  {extra_count}")

            if extra_count != point_count:
                raise ValueError(
                    "Number of spline points does not match " "number of extra points."
                )


    def read_positions(self) -> list[tuple[float, float, float]]:
        """Read all spline positions from the Assetto Corsa AI line."""

        positions: list[tuple[float, float, float]] = []

        with self._file_path.open("rb") as file:
            # Skip version and read the number of spline points.
            file.read(4)
            point_count = struct.unpack("<i", file.read(4))[0]

            # Skip lap time and sample count.
            file.read(8)

            for _ in range(point_count):
                x, y, z, _length, _point_id = struct.unpack(
                    "<ffffi",
                    file.read(20),
                )

                positions.append(
                    (
                        float(x),
                        float(y),
                        float(z),
                    )
                )
        return positions
    
    def read_track_points(self) -> list[TrackPoint]:
        """Read all track points including extra AI spline data."""

        # Read the base points first, which contain position, length, and point ID.
        base_points: list[tuple[tuple[float, float, float], float, int]] = []

        with self._file_path.open("rb") as file:
            version = struct.unpack("<i", file.read(4))[0]
            point_count = struct.unpack("<i", file.read(4))[0]
            _lap_time = struct.unpack("<i", file.read(4))[0]
            _sample_count = struct.unpack("<i", file.read(4))[0]

            # Validate the file version. Assetto Corsa's fast_lane.ai format may change between versions, so it's important to ensure compatibility.
            if version != 7:
                raise ValueError(
                    f"Unsupported fast_lane.ai version: {version}"
                )

            # Read the base points, which contain position, length, and point ID.
            for _ in range(point_count):
                x, y, z, length, point_id = struct.unpack(
                    "<ffffi", # < = little-endian, f = float, i = int
                    file.read(20),
                )
                # Store the base point data for later use when reading the extra AI spline data.
                base_points.append(
                    (
                        (float(x), float(y), float(z)),
                        float(length),
                        int(point_id),
                    )
                )
                
            # Read the extra AI spline data, which contains side offsets, normals, and forward vectors.
            extra_count = struct.unpack("<i", file.read(4))[0]

            if extra_count != point_count:
                raise ValueError(
                    "Number of spline points does not match "
                    "number of extra points."
                )

            track_points: list[TrackPoint] = []

            for index in range(point_count):
                # Read 18 floats for each extra point, which include side offsets, normals, and forward vectors.
                values = struct.unpack("<18f", file.read(18 * 4))

                position, length, point_id = base_points[index]

                # Extract right and left side offsets from current position to the track boundaries
                side_left = float(values[5])
                side_right = float(values[6])

                # Extract normal vector, which is perpendicular to the track surface at the current point
                normal = (
                    float(values[9]),
                    float(values[10]),
                    float(values[11]),
                )

                # Extract forward vector, which points in the direction of travel at the current point
                forward = (
                    float(values[13]),
                    float(values[14]),
                    float(values[15]),
                )

                # Create a TrackPoint instance with all the extracted data and append it to the list of track points
                track_points.append(
                    TrackPoint(
                        position=position,
                        length=length,
                        point_id=point_id,
                        side_left=side_left,
                        side_right=side_right,
                        normal=normal,
                        forward=forward,
                    )
                )

        return track_points

    def read_border_lines(
        self,
    ) -> tuple[
        list[tuple[float, float, float]],
        list[tuple[float, float, float]],
        list[tuple[float, float, float]],
    ]:
        """Return centerline, left boundary, and right boundary as coordinates in world space. 
        Calculate left and right boundaries based on the normal and forward vectors of the track points."""

        track_points = self.read_track_points()

        centerline: list[tuple[float, float, float]] = []
        left_boundary: list[tuple[float, float, float]] = []
        right_boundary: list[tuple[float, float, float]] = []

        for point in track_points:
            centerline.append(point.position)

            normal = _normalize(point.normal)
            forward = _normalize(point.forward)

            # Candidate side direction.
            left_direction = _normalize(_cross(normal, forward))

            left_point = _add(
                point.position,
                _scale(left_direction, point.side_left),
            )

            right_point = _sub(
                point.position,
                _scale(left_direction, point.side_right),
            )

            left_boundary.append(left_point)
            right_boundary.append(right_point)

        return centerline, left_boundary, right_boundary    