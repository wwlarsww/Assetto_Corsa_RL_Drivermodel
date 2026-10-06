"""Provide low-level access to the vJoy interface DLL."""

import ctypes
from enum import IntEnum
from pathlib import Path


class VJoyAxis(IntEnum):
    """vJoy HID axis identifiers."""

    X = 0x30
    Y = 0x31
    Z = 0x32


class VJoyInterface:
    """Provide low-level access to one vJoy device."""

    def __init__(
        self,
        device_id: int = 1,
        dll_path: Path = Path(r"C:\Program Files\vJoy\x64\vJoyInterface.dll"),
    ) -> None:
        self._device_id = device_id

        if not dll_path.exists():
            raise FileNotFoundError(f"vJoy interface DLL not found: {dll_path}")

        self._dll = ctypes.CDLL(str(dll_path))

        self._configure_functions()

        if not self._dll.vJoyEnabled():
            raise RuntimeError("vJoy driver is not enabled.")

        if not self._dll.AcquireVJD(self._device_id):
            raise RuntimeError(f"Could not acquire vJoy device {self._device_id}.")

    def _configure_functions(self) -> None:
        """Configure ctypes signatures for the used vJoy functions."""

        self._dll.vJoyEnabled.restype = ctypes.c_int

        self._dll.AcquireVJD.argtypes = [
            ctypes.c_uint,
        ]
        self._dll.AcquireVJD.restype = ctypes.c_int

        self._dll.RelinquishVJD.argtypes = [
            ctypes.c_uint,
        ]
        self._dll.RelinquishVJD.restype = None

        self._dll.SetAxis.argtypes = [
            ctypes.c_long,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        self._dll.SetAxis.restype = ctypes.c_int

        self._dll.GetVJDAxisMin.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_long),
        ]
        self._dll.GetVJDAxisMin.restype = ctypes.c_int

        self._dll.GetVJDAxisMax.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_long),
        ]
        self._dll.GetVJDAxisMax.restype = ctypes.c_int

    def get_axis_range(
        self,
        axis: VJoyAxis,
    ) -> tuple[int, int]:
        """Return minimum and maximum values of a vJoy axis."""

        minimum = ctypes.c_long()
        maximum = ctypes.c_long()

        if not self._dll.GetVJDAxisMin(
            self._device_id,
            axis,
            ctypes.byref(minimum),
        ):
            raise RuntimeError(f"Could not read minimum for axis {axis.name}.")

        if not self._dll.GetVJDAxisMax(
            self._device_id,
            axis,
            ctypes.byref(maximum),
        ):
            raise RuntimeError(f"Could not read maximum for axis {axis.name}.")

        return minimum.value, maximum.value

    def set_axis(
        self,
        axis: VJoyAxis,
        value: int,
    ) -> None:
        """Set one vJoy axis value."""

        if not self._dll.SetAxis(
            value,
            self._device_id,
            axis,
        ):
            raise RuntimeError(f"Could not set vJoy axis {axis.name}.")

    def close(self) -> None:
        """Release the vJoy device."""

        self._dll.RelinquishVJD(self._device_id)
