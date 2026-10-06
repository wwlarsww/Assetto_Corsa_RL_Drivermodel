"""Control Assetto Corsa using a virtual vJoy device.
Covers the translation of our meaningful actions into vJoy values and 
the application of those values to the vJoy device."""

from assetto_corsa_ai.action import VehicleAction
from assetto_corsa_ai.assetto_corsa.vjoy_interface import (
    VJoyAxis,
    VJoyInterface,
)


class AssettoCorsaVJoyController:
    """Apply vehicle actions to Assetto Corsa through vJoy."""

    def __init__(
        self,
        vjoy: VJoyInterface | None = None,
    ) -> None:
        # Use the provided vJoy interface or create a new one for device 1
        self._vjoy = vjoy or VJoyInterface(device_id=1)

        # Get the axis ranges for steering, throttle, and brake
        self._steering_range = self._vjoy.get_axis_range(VJoyAxis.X)
        self._throttle_range = self._vjoy.get_axis_range(VJoyAxis.Y)
        self._brake_range = self._vjoy.get_axis_range(VJoyAxis.Z)

        self.reset()

    def apply(self, action: VehicleAction) -> None:
        """Apply steering, throttle and brake commands."""

        steering = self._map_range(
            action.steering,
            -1.0,
            1.0,
            *self._steering_range,
        )

        throttle = self._map_range(
            action.throttle,
            0.0,
            1.0,
            *self._throttle_range,
        )

        brake = self._map_range(
            action.brake,
            0.0,
            1.0,
            *self._brake_range,
        )

        self._vjoy.set_axis(
            VJoyAxis.X,
            steering,
        )

        self._vjoy.set_axis(
            VJoyAxis.Y,
            throttle,
        )

        self._vjoy.set_axis(
            VJoyAxis.Z,
            brake,
        )

    def reset(self) -> None:
        """Set all controls to their neutral state."""

        self.apply(
            VehicleAction(
                steering=0.0,
                throttle=0.0,
                brake=0.0,
            )
        )

    def close(self) -> None:
        """Reset controls and release the vJoy device."""

        self.reset()
        self._vjoy.close()

    @staticmethod
    def _map_range(
        value: float,
        input_min: float,
        input_max: float,
        output_min: int,
        output_max: int,
    ) -> int:
        """Map a value from one numerical range to another."""

        normalized = (value - input_min) / (input_max - input_min)

        return round(output_min + normalized * (output_max - output_min))
