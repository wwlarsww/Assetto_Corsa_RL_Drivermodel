"""Simulator independent telemetry data models."""

from dataclasses import dataclass

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
    velocity: tuple[float, float, float] # in world coordinates, in m/s, x, y, z
    acceleration_g: tuple[float, float, float]  # in world coordinates, in m/s, x, y, z
    # 0 → front left, 1 → front right
    # 2 → back left, 3 → back right
    wheel_slip: tuple[float, float, float, float]
