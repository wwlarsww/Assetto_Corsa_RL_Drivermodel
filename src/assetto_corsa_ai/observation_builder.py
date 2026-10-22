"""Build simulator-independent observations for the RL agent."""

from math import isfinite

from assetto_corsa_ai.observation import Observation
from assetto_corsa_ai.static_info import SimulatorStaticInfo
from assetto_corsa_ai.telemetry import TelemetryFrame
from assetto_corsa_ai.track_geometry import TrackGeometry
from assetto_corsa_ai.track_perception import TrackPerceptionCalculator
from assetto_corsa_ai.tyre_dynamics import TyreDynamicsCalculator
from assetto_corsa_ai.vehicle_geometry import estimate_axle_distances

class ObservationBuilder:
    """Build an RL observation from vehicle and track state.

    The builder converts validated simulator data into the compact,
    simulator-independent representation used by the RL agent.

    Complex calculations are delegated to specialized components such as
    TyreDynamicsCalculator and TrackPerceptionCalculator.
    """

    def __init__(
        self,
        tyre_dynamics_calculator: TyreDynamicsCalculator,
        track_perception_calculator: TrackPerceptionCalculator,
    ) -> None:
        self._tyre_dynamics_calculator = tyre_dynamics_calculator
        self._track_perception_calculator = track_perception_calculator

    @classmethod
    def create(
        cls, *, front_axle_distance: float = 0.0, rear_axle_distance: float = 0.0,
        telemetry: TelemetryFrame | None = None, minimum_speed: float = 1.0,
        lookahead_distances: tuple[float, ...] = TrackPerceptionCalculator.DEFAULT_LOOKAHEAD_DISTANCES,
        reference_distances: tuple[float, ...] = TrackPerceptionCalculator.DEFAULT_REFERENCE_DISTANCES,
    ) -> "ObservationBuilder":
        """Compose the configured calculators; zero axle offsets request estimation."""
        if any(not isfinite(value) or value < 0 for value in (front_axle_distance, rear_axle_distance)):
            raise ValueError("Axle distances must be finite and non-negative.")
        if not front_axle_distance or not rear_axle_distance:
            if telemetry is None:
                raise ValueError("Automatic axle estimation requires telemetry.")
            front, rear = estimate_axle_distances(telemetry)
            front_axle_distance, rear_axle_distance = front_axle_distance or front, rear_axle_distance or rear
        return cls(
            TyreDynamicsCalculator(front_axle_distance, rear_axle_distance, minimum_speed=minimum_speed),
            TrackPerceptionCalculator(lookahead_distances, reference_distances),
        )

    def build(
        self,
        telemetry: TelemetryFrame,
        static_info: SimulatorStaticInfo,
        track_geometry: TrackGeometry,
        current_s: float,
    ) -> Observation:
        """Create an observation from the current simulator state.

        Track coordinates are expected to already be expressed relative to
        the vehicle, for example as ``(forward, lateral)`` coordinates.

        Slip ratios and slip angles are accepted as derived values because
        their calculation depends on additional vehicle-dynamics logic and
        should not be hidden inside the observation builder.
        """
        tyre_dynamics = self._tyre_dynamics_calculator.calculate(
            telemetry=telemetry,
            static_info=static_info,
        )

        track_perception = self._track_perception_calculator.calculate(
            telemetry=telemetry,
            track_geometry=track_geometry,
            current_s=current_s,
            track_length=static_info.track_spline_length,
        )
        return Observation(
            speed=self._speed_mps(telemetry),
            longitudinal_acceleration=self._longitudinal_acceleration(telemetry),
            lateral_acceleration=self._lateral_acceleration(telemetry),
            yaw_rate=self._yaw_rate(telemetry),
            front_slip_ratio=tyre_dynamics.front_slip_ratio,
            rear_slip_ratio=tyre_dynamics.rear_slip_ratio,
            front_slip_angle=tyre_dynamics.front_slip_angle,
            rear_slip_angle=tyre_dynamics.rear_slip_angle,
            left_boundary=track_perception.left_boundary,
            right_boundary=track_perception.right_boundary,
            reference_waypoints=track_perception.reference_waypoints,
            steering=telemetry.steering_angle,
            longitudinal_control=self._longitudinal_control(
                throttle=telemetry.gas,
                brake=telemetry.brake,
            ),
        )

    @staticmethod
    def _speed_mps(telemetry: TelemetryFrame) -> float:
        """Convert vehicle speed from km/h to m/s."""

        return telemetry.speed_kmh / 3.6

    @staticmethod
    def _longitudinal_acceleration(
        telemetry: TelemetryFrame,
    ) -> float:
        """Return longitudinal acceleration in g."""

        # Validated AC mapping:
        # acceleration_g[2] = longitudinal acceleration
        return telemetry.acceleration_g[2]

    @staticmethod
    def _lateral_acceleration(
        telemetry: TelemetryFrame,
    ) -> float:
        """Return lateral acceleration in g."""

        # Validated AC mapping:
        # acceleration_g[0] = lateral acceleration
        return telemetry.acceleration_g[0]

    @staticmethod
    def _yaw_rate(telemetry: TelemetryFrame) -> float:
        """Return vehicle yaw rate in rad/s."""

        # Validated AC mapping:
        # local_angular_velocity[1] = yaw rate
        return telemetry.local_angular_velocity[1]

    @staticmethod
    def _longitudinal_control(
        throttle: float,
        brake: float,
    ) -> float:
        """Combine throttle and brake into one normalized control value.

        Positive values represent throttle and negative values represent
        braking:

            -1.0 = full brake
             0.0 = coast
            +1.0 = full throttle
        """

        return throttle - brake
