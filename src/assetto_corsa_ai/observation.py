from dataclasses import dataclass

Point2D = tuple[float, float]

@dataclass(frozen=True)
class Observation:
    """Represent one observation of the vehicle state and track perception."""

    # Vehicle dynamics
    speed: float
    longitudinal_acceleration: float
    lateral_acceleration: float
    yaw_rate: float

    front_slip_ratio: float
    rear_slip_ratio: float
    front_slip_angle: float
    rear_slip_angle: float

    # Track perception
    left_boundary: tuple[Point2D, ...]
    right_boundary: tuple[Point2D, ...]
    reference_waypoints: tuple[Point2D, ...]

    # Actually applied controls
    steering: float
    longitudinal_control: float
