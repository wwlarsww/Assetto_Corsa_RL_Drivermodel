"""Optional Xbox 360 control writer for the future live backend."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.envs.backend import validate_backend_action


class VirtualGamepadUnavailableError(RuntimeError):
    """Raised when the optional Windows controller stack is unavailable."""


class VGamepadControlWriter:
    """Map absolute combined pedal/steering commands to an Xbox 360 report.

    The class deliberately owns only analog report generation. Session reset,
    frame synchronisation and safety termination belong to the live simulator
    backend rather than to this transport adapter.
    """

    def __init__(self, gamepad: Any | None = None) -> None:
        if gamepad is None:
            try:
                module = importlib.import_module("vgamepad")
                gamepad = module.VX360Gamepad()
            except (ImportError, OSError) as error:
                raise VirtualGamepadUnavailableError(
                    "vgamepad/ViGEmBus is unavailable; install the optional live "
                    "dependencies and the signed Windows driver"
                ) from error
        self._gamepad = gamepad
        self._closed = False
        self.neutral()

    def apply(self, absolute_action: npt.ArrayLike) -> None:
        """Send one finite command with independent analog throttle and brake."""

        if self._closed:
            raise RuntimeError("virtual gamepad is closed")
        action = validate_backend_action(
            np.asarray(absolute_action, dtype=np.float32),
        )
        pedal, steering = (float(action[0]), float(action[1]))
        self._gamepad.left_joystick_float(
            x_value_float=steering,
            y_value_float=0.0,
        )
        self._gamepad.right_trigger_float(value_float=max(pedal, 0.0))
        self._gamepad.left_trigger_float(value_float=max(-pedal, 0.0))
        self._gamepad.update()

    def neutral(self) -> None:
        """Release both pedals and centre steering."""

        if self._closed:
            return
        self._gamepad.left_joystick_float(
            x_value_float=0.0,
            y_value_float=0.0,
        )
        self._gamepad.right_trigger_float(value_float=0.0)
        self._gamepad.left_trigger_float(value_float=0.0)
        self._gamepad.update()

    def close(self) -> None:
        """Return to neutral once and make future writes fail."""

        if self._closed:
            return
        self.neutral()
        self._closed = True
