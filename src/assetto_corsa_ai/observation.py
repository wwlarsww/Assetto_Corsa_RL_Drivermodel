from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Point2D = tuple[float, float]

@dataclass(frozen=True)
class Observation:
    """Vehicle state, road perception and applied controls for one timestep.

    Speed is in m/s, accelerations in g, yaw rate in rad/s and slip angles in
    radians. Boundary and reference points use local (forward, left) metres.
    Steering and longitudinal control use [-1, 1]; positive longitudinal
    control means throttle and negative means brake.

    to_array() fixes the feature ordering and returns unnormalized float32
    values, with the two applied controls last.
    """

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

    def to_array(self) -> NDArray[np.float32]:
        """Return the observation as a flat neural-network input vector."""

        values = [
            self.speed,
            self.longitudinal_acceleration,
            self.lateral_acceleration,
            self.yaw_rate,
            self.front_slip_ratio,
            self.rear_slip_ratio,
            self.front_slip_angle,
            self.rear_slip_angle,
        ]

        for point in self.left_boundary:
            values.extend(point)

        for point in self.right_boundary:
            values.extend(point)

        for point in self.reference_waypoints:
            values.extend(point)

        values.extend(
            [
                self.steering,
                self.longitudinal_control,
            ]
        )

        return np.asarray(
            values,
            dtype=np.float32,
        )
