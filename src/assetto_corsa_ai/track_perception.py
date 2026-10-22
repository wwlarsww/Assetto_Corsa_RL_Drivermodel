"""Calculate track perception features for the RL observation."""

from bisect import bisect_right
from dataclasses import dataclass
import math

from assetto_corsa_ai.telemetry import TelemetryFrame
from assetto_corsa_ai.track_geometry import TrackGeometry

Point2D = tuple[float, float]


@dataclass(frozen=True)
class TrackPerception:
    """Track perception features relative to the vehicle."""

    # Each point is stored as:
    # (forward_distance, left_distance)
    left_boundary: tuple[Point2D, ...]
    right_boundary: tuple[Point2D, ...]
    reference_waypoints: tuple[Point2D, ...]


class TrackPerceptionCalculator:
    """Calculate vehicle-relative track perception features."""

    DEFAULT_LOOKAHEAD_DISTANCES = (
        5.0,
        10.0,
        20.0,
        40.0,
        80.0,
        160.0,
        320.0,
    )

    DEFAULT_REFERENCE_DISTANCES = tuple(
        float(distance) for distance in range(5, 101, 5)
    )

    def __init__(
        self,
        lookahead_distances: tuple[float, ...] = DEFAULT_LOOKAHEAD_DISTANCES,
        reference_distances: tuple[float, ...] = DEFAULT_REFERENCE_DISTANCES,
    ) -> None:
        self._lookahead_distances = lookahead_distances
        self._reference_distances = reference_distances

    def calculate(
        self,
        telemetry: TelemetryFrame,
        track_geometry: TrackGeometry,
        current_s: float,
        track_length: float,
    ) -> TrackPerception:
        """Calculate track boundaries and reference waypoints ahead of the vehicle."""

        origin, forward, left = self._vehicle_frame(telemetry.tyre_contact_points)

        # --------------------------------------------------------------
        # Track boundaries
        # --------------------------------------------------------------

        left_boundary: list[Point2D] = []
        right_boundary: list[Point2D] = []

        for distance in self._lookahead_distances:
            target_s = (current_s + distance) % track_length

            left_world = self._sample_boundary(
                track_geometry=track_geometry,
                target_s=target_s,
                track_length=track_length,
                boundary_name="left_boundary",
            )

            right_world = self._sample_boundary(
                track_geometry=track_geometry,
                target_s=target_s,
                track_length=track_length,
                boundary_name="right_boundary",
            )

            left_boundary.append(
                self._world_to_vehicle(
                    point=left_world,
                    origin=origin,
                    forward=forward,
                    left=left,
                )
            )

            right_boundary.append(
                self._world_to_vehicle(
                    point=right_world,
                    origin=origin,
                    forward=forward,
                    left=left,
                )
            )

        # --------------------------------------------------------------
        # Reference trajectory
        # --------------------------------------------------------------

        reference_waypoints: list[Point2D] = []

        for distance in self._reference_distances:
            target_s = (current_s + distance) % track_length

            reference_world = self._sample_reference(
                track_geometry=track_geometry,
                target_s=target_s,
                track_length=track_length,
            )

            reference_waypoints.append(
                self._world_to_vehicle(
                    point=reference_world,
                    origin=origin,
                    forward=forward,
                    left=left,
                )
            )

        return TrackPerception(
            left_boundary=tuple(left_boundary),
            right_boundary=tuple(right_boundary),
            reference_waypoints=tuple(reference_waypoints),
        )

    @staticmethod
    def vehicle_frame(
        tyre_contact_points: tuple[tuple[float, float, float], ...],
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Expose the same horizontal frame to read-only diagnostic consumers."""
        return TrackPerceptionCalculator._vehicle_frame(tyre_contact_points)

    @staticmethod
    def _vehicle_frame(
        tyre_contact_points: tuple[tuple[float, float, float], ...],
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        """Determine vehicle position and orientation from tyre positions.

        Wheel order:
            0 = front left
            1 = front right
            2 = rear left
            3 = rear right

        Only Assetto Corsa's horizontal x/z plane is used.

             FL -------- FR
              |          |
              |    ↑     |
              | forward  |
              |          |
             RL -------- RR
        """

        if len(tyre_contact_points) != 4:
            raise ValueError("Expected tyre contact points for four wheels.")

        fl, fr, rl, rr = tyre_contact_points

        front_center = (
            (fl[0] + fr[0]) / 2.0,
            (fl[2] + fr[2]) / 2.0,
        )

        rear_center = (
            (rl[0] + rr[0]) / 2.0,
            (rl[2] + rr[2]) / 2.0,
        )

        left_center = (
            (fl[0] + rl[0]) / 2.0,
            (fl[2] + rl[2]) / 2.0,
        )

        right_center = (
            (fr[0] + rr[0]) / 2.0,
            (fr[2] + rr[2]) / 2.0,
        )

        # Vehicle origin = centre of the four tyre contact points.
        origin = (
            (front_center[0] + rear_center[0]) / 2.0,
            (front_center[1] + rear_center[1]) / 2.0,
        )

        # Direction in which the vehicle points.
        forward = TrackPerceptionCalculator._normalize(
            (
                front_center[0] - rear_center[0],
                front_center[1] - rear_center[1],
            )
        )

        # Direction towards the left side of the vehicle.
        left = TrackPerceptionCalculator._normalize(
            (
                left_center[0] - right_center[0],
                left_center[1] - right_center[1],
            )
        )

        return origin, forward, left

    @staticmethod
    def _world_to_vehicle(
        point: tuple[float, float, float],
        origin: tuple[float, float],
        forward: tuple[float, float],
        left: tuple[float, float],
    ) -> Point2D:
        """Transform an AC world point into vehicle-relative coordinates."""

        dx = point[0] - origin[0]
        dz = point[2] - origin[1]

        forward_distance = dx * forward[0] + dz * forward[1]

        left_distance = dx * left[0] + dz * left[1]

        return forward_distance, left_distance

    @staticmethod
    def _sample_boundary(
        track_geometry: TrackGeometry,
        target_s: float,
        track_length: float,
        boundary_name: str,
    ) -> tuple[float, float, float]:
        """Interpolate a boundary point at a given track distance."""

        points = track_geometry.points

        if len(points) < 2:
            raise ValueError("Track geometry requires at least two points.")

        lengths = [point.length for point in points]

        index = bisect_right(lengths, target_s) - 1
        index = max(index, 0)

        next_index = (index + 1) % len(points)

        first = points[index]
        second = points[next_index]

        s0 = first.length

        if next_index == 0:
            s1 = track_length
        else:
            s1 = second.length

        if s1 == s0:
            interpolation = 0.0
        else:
            interpolation = (target_s - s0) / (s1 - s0)

        p0 = getattr(first, boundary_name)
        p1 = getattr(second, boundary_name)

        return (
            p0[0] + interpolation * (p1[0] - p0[0]),
            p0[1] + interpolation * (p1[1] - p0[1]),
            p0[2] + interpolation * (p1[2] - p0[2]),
        )

    @staticmethod
    def _sample_reference(
        track_geometry: TrackGeometry,
        target_s: float,
        track_length: float,
    ) -> tuple[float, float, float]:
        """Interpolate a reference waypoint from the fast_lane.ai spline."""

        points = track_geometry.points

        if len(points) < 2:
            raise ValueError("Track geometry requires at least two points.")

        lengths = [point.length for point in points]

        index = bisect_right(lengths, target_s) - 1
        index = max(index, 0)

        next_index = (index + 1) % len(points)

        first = points[index]
        second = points[next_index]

        s0 = first.length

        if next_index == 0:
            s1 = track_length
        else:
            s1 = second.length

        if s1 == s0:
            interpolation = 0.0
        else:
            interpolation = (target_s - s0) / (s1 - s0)

        p0 = first.position
        p1 = second.position

        return (
            p0[0] + interpolation * (p1[0] - p0[0]),
            p0[1] + interpolation * (p1[1] - p0[1]),
            p0[2] + interpolation * (p1[2] - p0[2]),
        )

    @staticmethod
    def _normalize(
        vector: tuple[float, float],
    ) -> tuple[float, float]:
        """Normalize a two-dimensional vector."""

        length = math.hypot(
            vector[0],
            vector[1],
        )

        if length == 0.0:
            raise ValueError("Cannot normalize zero-length vector.")

        return (
            vector[0] / length,
            vector[1] / length,
        )
