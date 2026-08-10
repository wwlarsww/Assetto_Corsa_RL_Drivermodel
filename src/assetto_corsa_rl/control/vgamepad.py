"""Optional Xbox 360 control writer for the future live backend."""

from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.envs.backend import validate_backend_action


class VirtualGamepadUnavailableError(RuntimeError):
    """Raised when the optional Windows controller stack is unavailable."""


def _is_finite_real(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


class VGamepadControlWriter:
    """Map absolute combined pedal/steering commands to an Xbox 360 report.

    The class deliberately owns only analog report generation. Session reset,
    frame synchronisation and safety termination belong to the live simulator
    backend rather than to this transport adapter.
    """

    def __init__(
        self,
        gamepad: Any | None = None,
        *,
        steering_output_sign: float = -1.0,
        steering_output_gain: float = 1.0,
    ) -> None:
        if steering_output_sign not in {-1.0, 1.0}:
            raise ValueError("steering_output_sign must be -1.0 or 1.0")
        if (
            not _is_finite_real(steering_output_gain)
            or not 0.0 < steering_output_gain <= 2.0
        ):
            raise ValueError("steering_output_gain must be finite and lie in (0, 2]")
        if gamepad is None:
            try:
                module = importlib.import_module("vgamepad")
                gamepad = module.VX360Gamepad()
            except (ImportError, OSError) as error:
                raise VirtualGamepadUnavailableError(
                    "vgamepad/ViGEmBus is unavailable; install the optional live "
                    "dependencies and the signed Windows driver"
                ) from error
        if gamepad is None:
            raise VirtualGamepadUnavailableError("vgamepad returned no controller instance")
        self._gamepad = gamepad
        self._steering_output_sign = steering_output_sign
        self._steering_output_gain = float(steering_output_gain)
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
            x_value_float=float(
                np.clip(
                    self._steering_output_sign * self._steering_output_gain * steering,
                    -1.0,
                    1.0,
                )
            ),
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

    @property
    def device_index(self) -> int | None:
        """Return AC's zero-based XInput index for the one-based ViGEm target."""

        get_index = getattr(self._gamepad, "get_index", None)
        if get_index is None:
            return None
        index = int(get_index()) - 1
        if not 0 <= index <= 3:
            raise VirtualGamepadUnavailableError(
                f"virtual X360 device received invalid ViGEm/XInput index {index + 1}"
            )
        return index

    def close(self) -> None:
        """Return to neutral once and make future writes fail."""

        if self._closed:
            return
        self.neutral()
        self._closed = True
