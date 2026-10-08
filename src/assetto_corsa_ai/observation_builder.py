"""Build simulator-independent observations for the RL agent."""

from typing import Any

from assetto_corsa_ai.observation import Observation
from assetto_corsa_ai.telemetry import TelemetryFrame
from assetto_corsa_ai.track_geometry import TrackGeometry
from assetto_corsa_ai.track_perception import TrackPerceptionCalculator
from assetto_corsa_ai.tyre_dynamics import TyreDynamicsCalculator

class ObservationBuilder:
    """Build an RL observation from vehicle and track state.

    The builder converts validated simulator data into the compact,
    simulator-independent representation used by the RL agent.

    More complex calculations such as track projection, boundary sampling,
    coordinate transformations, and slip-angle estimation should be performed
    outside this class and passed in as already derived features.
    """
    def __init__(
        self,
        tyre_dynamics_calculator: TyreDynamicsCalculator,
        track_perception_calculator: TrackPerceptionCalculator,
    ) -> None:
        self._tyre_dynamics_calculator = tyre_dynamics_calculator
        self._track_perception_calculator = track_perception_calculator

    def build(
        self,
        telemetry: TelemetryFrame,
        static_info: Any,
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
            track_length=getattr(static_info, "track_spline_length"),
        )
        return Observation(
            speed=telemetry.speed_kmh / 3.6,

            longitudinal_acceleration=telemetry.acceleration_g[2],
            lateral_acceleration=telemetry.acceleration_g[0],

            yaw_rate=telemetry.local_angular_velocity[1],

            front_slip_ratio=tyre_dynamics.front_slip_ratio,
            rear_slip_ratio=tyre_dynamics.rear_slip_ratio,

            front_slip_angle=tyre_dynamics.front_slip_angle,
            rear_slip_angle=tyre_dynamics.rear_slip_angle,

            left_boundary=track_perception.left_boundary,
            right_boundary=track_perception.right_boundary,
            
            reference_waypoints=track_perception.reference_waypoints,

            steering=telemetry.steering_angle,
            longitudinal_control=telemetry.gas - telemetry.brake,
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
