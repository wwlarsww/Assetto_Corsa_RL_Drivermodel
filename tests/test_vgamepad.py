from __future__ import annotations

import pytest

from assetto_corsa_rl.control.vgamepad import VGamepadControlWriter


class _FakeGamepad:
    def __init__(self) -> None:
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.updates = 0

    @staticmethod
    def get_index() -> int:
        return 3

    def left_joystick_float(
        self,
        *,
        x_value_float: float,
        y_value_float: float,
    ) -> None:
        self.steering = x_value_float
        assert y_value_float == 0.0

    def right_trigger_float(self, *, value_float: float) -> None:
        self.throttle = value_float

    def left_trigger_float(self, *, value_float: float) -> None:
        self.brake = value_float

    def update(self) -> None:
        self.updates += 1


def test_vgamepad_maps_combined_pedal_and_neutralises_on_close() -> None:
    gamepad = _FakeGamepad()
    writer = VGamepadControlWriter(gamepad)
    writer.apply([0.75, -0.25])
    assert writer.device_index == 2
    assert (gamepad.throttle, gamepad.brake, gamepad.steering) == (0.75, 0.0, 0.25)
    writer.apply([-0.5, 0.5])
    assert (gamepad.throttle, gamepad.brake, gamepad.steering) == (0.0, 0.5, -0.5)
    writer.close()
    assert (gamepad.throttle, gamepad.brake, gamepad.steering) == (0.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="closed"):
        writer.apply([0.0, 0.0])


def test_vgamepad_steering_sign_is_explicit() -> None:
    gamepad = _FakeGamepad()
    writer = VGamepadControlWriter(gamepad, steering_output_sign=1.0)
    writer.apply([0.0, 0.4])
    assert gamepad.steering == pytest.approx(0.4)
    with pytest.raises(ValueError, match="steering_output_sign"):
        VGamepadControlWriter(gamepad, steering_output_sign=0.0)
