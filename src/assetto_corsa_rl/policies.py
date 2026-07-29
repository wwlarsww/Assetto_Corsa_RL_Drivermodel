"""Small policies for integration tests; not training algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.features.observation import ObservationLayout
from assetto_corsa_rl.types import REFERENCE_TIME_OFFSETS_S, FloatArray


@dataclass(slots=True)
class MockDriverPolicy:
    """Deterministic feedback controller for exercising the mock pipeline."""

    pedal_delta_limit: float
    steering_delta_limit: float
    _layout: ObservationLayout = field(default_factory=ObservationLayout, init=False)

    def act(self, observation: npt.ArrayLike) -> FloatArray:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (self._layout.size,):
            raise ValueError(f"observation must have shape ({self._layout.size},)")

        speed = float(values[self._layout.vehicle_state.start])
        front_sideslip = float(values[5])
        left_boundary = values[self._layout.left_track_boundary].reshape(7, 2)
        right_boundary = values[self._layout.right_track_boundary].reshape(7, 2)
        reference = values[self._layout.reference_line].reshape(20, 2)
        current_action = values[self._layout.previous_action]

        track_centre_in_ego_x = 0.5 * (left_boundary[0, 0] + right_boundary[0, 0])
        lateral_offset = -float(track_centre_in_ego_x)
        target_speed = float(reference[0, 1] / REFERENCE_TIME_OFFSETS_S[0])
        desired_pedal = float(np.clip(0.08 * (target_speed - speed), -1.0, 1.0))
        desired_steering = float(np.clip(-0.15 * lateral_offset - 1.5 * front_sideslip, -1.0, 1.0))

        desired = np.asarray([desired_pedal, desired_steering], dtype=np.float32)
        limits = np.asarray(
            [self.pedal_delta_limit, self.steering_delta_limit],
            dtype=np.float32,
        )
        return np.clip((desired - current_action) / limits, -1.0, 1.0).astype(np.float32)
