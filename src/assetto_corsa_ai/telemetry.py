"""Simulator independent telemetry data models."""

from dataclasses import dataclass

Vector3 = tuple[float, float, float]
DamageValues = tuple[float, float, float, float, float]

@dataclass
class TelemetryFrame:
    """Represent one snapshot of the vehicle telemetry."""

    speed_kmh: float
    gas: float # 0 -> no gas, 1 -> full gas
    brake: float # 0 -> no brake, 1 -> full brake
    fuel: float # in litres
    gear: int # 0 -> backward, 1 -> neurtal, 2 -> first gear, ...
    rpm: int
    steering_angle: float # 1 -> full right, -1 -> full left

    acceleration_g: Vector3  # in local coordinates
    # 0 → front left, 1 → front right
    # 2 → back left, 3 → back right
    wheel_slip: tuple[float, float, float, float]
    wheel_angular_speed: tuple[float, float, float, float]
    tyre_contact_points: tuple[Vector3,Vector3,Vector3,Vector3]
    
    local_velocity: Vector3  # vertical_velocity, longitudinal_velocity, lateral_velocity
    
    local_angular_velocity: Vector3 # Pitch Rate, Yaw Rate, Roll Rate in rad/s

    car_coordinates: Vector3 # X, Y, Z coordinates in world space
    normalized_track_position: float  # progress of the car on the track (0->1)

    # Raw damage channels in simulator order; not normalized percentages.
    car_damage: DamageValues
