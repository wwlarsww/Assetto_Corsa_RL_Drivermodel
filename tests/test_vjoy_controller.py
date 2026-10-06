from unittest.mock import Mock, call

from assetto_corsa_ai.action import VehicleAction
from assetto_corsa_ai.assetto_corsa.vjoy_controller import (
    AssettoCorsaVJoyController,
)
from assetto_corsa_ai.assetto_corsa.vjoy_interface import (
    VJoyAxis,
    VJoyInterface,
)


def create_vjoy_mock() -> Mock:
    """Create a simulated vJoy interface."""

    vjoy = Mock(spec=VJoyInterface)

    vjoy.get_axis_range.side_effect = [
        (0, 1000),  # Steering
        (0, 1000),  # Throttle
        (0, 1000),  # Brake
    ]

    return vjoy


def test_apply_maps_vehicle_action_to_vjoy_axes() -> None:
    vjoy = create_vjoy_mock()

    controller = AssettoCorsaVJoyController(
        vjoy=vjoy,
    )

    # Ignore calls caused by controller initialization/reset.
    vjoy.set_axis.reset_mock()

    controller.apply(
        VehicleAction(
            steering=0.5,
            throttle=0.25,
            brake=0.75,
        )
    )

    assert vjoy.set_axis.call_args_list == [
        call(VJoyAxis.X, 750),
        call(VJoyAxis.Y, 250),
        call(VJoyAxis.Z, 750),
    ]


def test_reset_sets_controls_to_neutral() -> None:
    vjoy = create_vjoy_mock()

    controller = AssettoCorsaVJoyController(
        vjoy=vjoy,
    )

    vjoy.set_axis.reset_mock()

    controller.reset()

    assert vjoy.set_axis.call_args_list == [
        call(VJoyAxis.X, 500),
        call(VJoyAxis.Y, 0),
        call(VJoyAxis.Z, 0),
    ]
