"""Build simulator-independent observations for the RL agent."""

from assetto_corsa_ai.observation import Observation
from assetto_corsa_ai.telemetry import TelemetryFrame


class ObservationBuilder:
    """Build an RL observation from vehicle and track state.

    The builder converts validated simulator data into the compact,
    simulator-independent representation used by the RL agent.

    More complex calculations such as track projection, boundary sampling,
    coordinate transformations, and slip-angle estimation should be performed
    outside this class and passed in as already derived features.
    """

    def build(
        self,
        telemetry: TelemetryFrame,
        left_boundary: tuple[tuple[float, float], ...],
        right_boundary: tuple[tuple[float, float], ...],
        reference_waypoints: tuple[tuple[float, float], ...],
        front_slip_ratio: float = 0.0,
        rear_slip_ratio: float = 0.0,
        front_slip_angle: float = 0.0,
        rear_slip_angle: float = 0.0,
    ) -> Observation:
        """Create an observation from the current simulator state.

        Track coordinates are expected to already be expressed relative to
        the vehicle, for example as ``(forward, lateral)`` coordinates.

        Slip ratios and slip angles are accepted as derived values because
        their calculation depends on additional vehicle-dynamics logic and
        should not be hidden inside the observation builder.
        """

        return Observation(
            # Vehicle dynamics
            speed=self._speed_mps(telemetry),
            longitudinal_acceleration=self._longitudinal_acceleration(telemetry),
            lateral_acceleration=self._lateral_acceleration(telemetry),
            yaw_rate=self._yaw_rate(telemetry),
            # Tyre dynamics
            front_slip_ratio=front_slip_ratio,
            rear_slip_ratio=rear_slip_ratio,
            front_slip_angle=front_slip_angle,
            rear_slip_angle=rear_slip_angle,
            # Track perception
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            reference_waypoints=reference_waypoints,
            # Applied controls
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
        """Return longitudinal vehicle acceleration.

        The exact component mapping of Assetto Corsa's acceleration vector
        should be kept in one place and validated before RL training.
        """
        
        return telemetry.acceleration_g[2]

    @staticmethod
    def _lateral_acceleration(
        telemetry: TelemetryFrame,
    ) -> float:
        """Return lateral vehicle acceleration.

        The exact component mapping should be validated against Assetto Corsa
        before this feature is used for training.
        """

        return telemetry.acceleration_g[0]

    @staticmethod
    def _yaw_rate(telemetry: TelemetryFrame) -> float:
        """Return vehicle yaw rate.

        The local angular-velocity component order has not yet been fully
        validated, so the selected component must be confirmed before
        training.
        """

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

        The current action mapping prevents throttle and brake from being
        applied simultaneously.
        """

        return throttle - brake
