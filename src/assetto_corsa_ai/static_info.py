"""Simulator independent static information models."""

from dataclasses import dataclass


@dataclass
class SimulatorStaticInfo:
    """Represent static vehicle and track information."""

    car_model: str
    track_name: str
    track_configuration: str

    max_rpm: int
    max_fuel: float

    tyre_radius: tuple[float, float, float, float]

    track_spline_length: float
