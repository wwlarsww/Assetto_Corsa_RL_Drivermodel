from dataclasses import dataclass
import math

from assetto_corsa_ai.static_info import SimulatorStaticInfo
from assetto_corsa_ai.telemetry import TelemetryFrame


@dataclass(frozen=True)
class TyreDynamics:
    """Derived tyre dynamics for the front and rear axle."""

    front_slip_ratio: float
    rear_slip_ratio: float

    # Slip angles will be added once their calculation is implemented.
    front_slip_angle: float
    rear_slip_angle: float 

class TyreDynamicsCalculator:
    """Calculate tyre dynamics from vehicle telemetry.
    The calculator derives slip ratios and slip angles from the current
    vehicle state and the static vehicle configuration.
    """

    def __init__(
        self,
        front_axle_distance: float,
        rear_axle_distance: float,
        minimum_speed: float = 1.0,
    ) -> None:
        """Initialize the calculator.

        Args:
            front_axle_distance:
                Distance from the center of gravity to the front axle in meters.

            rear_axle_distance:
                Distance from the center of gravity to the rear axle in meters.
                
            minimum_speed:
                Vehicle speed in m/s below which slip ratio is set to zero.
                This prevents numerical instability close to standstill.
        """
        self._front_axle_distance = front_axle_distance
        self._rear_axle_distance = rear_axle_distance
        self._minimum_speed = minimum_speed

    def calculate(
        self,
        telemetry: TelemetryFrame,
        static_info: SimulatorStaticInfo,
    ) -> TyreDynamics:
        """Calculate the current front and rear tyre dynamics.
        The results are in si units: slip ratios are dimensionless,
        slip angles are in radians."""
        

        lateral_velocity = telemetry.local_velocity[2]
        longitudinal_velocity = telemetry.local_velocity[1]
        yaw_rate = telemetry.local_angular_velocity[1]

        slip_ratios = tuple(
            self._slip_ratio(
                wheel_angular_speed=angular_speed,
                tyre_radius=radius,
                longitudinal_velocity=longitudinal_velocity,
            )
            for angular_speed, radius in zip(
                telemetry.wheel_angular_speed,
                static_info.tyre_radius,
            )
        )

        front_slip_ratio = self._average(
            slip_ratios[0],
            slip_ratios[1],
        )

        rear_slip_ratio = self._average(
            slip_ratios[2],
            slip_ratios[3],
        )

        front_slip_angle = self._slip_angle(
            longitudinal_velocity=longitudinal_velocity,
            lateral_velocity=lateral_velocity,
            yaw_rate=yaw_rate,
            axle_distance=self._front_axle_distance,
        )

        rear_slip_angle = self._slip_angle(
            longitudinal_velocity=longitudinal_velocity,
            lateral_velocity=lateral_velocity,
            yaw_rate=yaw_rate,
            axle_distance=-self._rear_axle_distance,
        )

        return TyreDynamics(
            front_slip_ratio=front_slip_ratio,
            rear_slip_ratio=rear_slip_ratio,
            front_slip_angle=front_slip_angle, 
            rear_slip_angle=rear_slip_angle,
        )

    def _slip_ratio(
        self,
        wheel_angular_speed: float,
        tyre_radius: float,
        longitudinal_velocity: float,
    ) -> float:
        """Calculate longitudinal slip ratio for one wheel.

        The tyre circumferential speed is

            v_wheel = omega * radius

        and is compared with the longitudinal vehicle velocity.

        Positive slip represents wheel spin during acceleration.
        Negative slip represents wheel slip during braking.
        """

        if abs(longitudinal_velocity) < self._minimum_speed:
            return 0.0

        wheel_velocity = wheel_angular_speed * tyre_radius

        return (wheel_velocity - longitudinal_velocity) / abs(longitudinal_velocity)

    def _slip_angle(
        self,
        longitudinal_velocity: float,
        lateral_velocity: float,
        yaw_rate: float,
        axle_distance: float,
    ) -> float:
        """Calculate slip angle at an axle centre.
        

        Positive axle distance represents the front axle.
        Negative axle distance represents the rear axle.
        """

        if abs(longitudinal_velocity) < self._minimum_speed:
            return 0.0

        axle_lateral_velocity = (
            lateral_velocity
            + yaw_rate * axle_distance
        )

        return math.atan2(
            axle_lateral_velocity,
            longitudinal_velocity,
        )

    @staticmethod
    def _average(first: float, second: float) -> float:
        """Return the arithmetic mean of two values."""

        return (first + second) / 2.0
