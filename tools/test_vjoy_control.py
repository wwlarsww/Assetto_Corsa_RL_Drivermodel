"""Test direct control of a vJoy device.
Open: vJoy Moniror to see the effect of this script on the vJoy device.
Requires vJoy to be installed and a vJoy device to be configured."""

import ctypes
import time

VJOY_DLL = r"C:\Program Files\vJoy\x64\vJoyInterface.dll"  # default path to vJoy DLL

# vJoy device ID to use for testing
DEVICE_ID = 1

# official HID usage codes for vJoy axes
HID_USAGE_X = 0x30
HID_USAGE_Y = 0x31
HID_USAGE_Z = 0x32


def main() -> None:
    """Move vJoy axes and return them to a neutral state."""

    vjoy = ctypes.CDLL(VJOY_DLL)

    vjoy.vJoyEnabled.restype = ctypes.c_int

    vjoy.AcquireVJD.argtypes = [ctypes.c_uint]
    vjoy.AcquireVJD.restype = ctypes.c_int

    vjoy.RelinquishVJD.argtypes = [ctypes.c_uint]
    vjoy.RelinquishVJD.restype = None

    vjoy.ResetVJD.argtypes = [ctypes.c_uint]
    vjoy.ResetVJD.restype = ctypes.c_int

    vjoy.SetAxis.argtypes = [
        ctypes.c_long,
        ctypes.c_uint,
        ctypes.c_uint,
    ]
    vjoy.SetAxis.restype = ctypes.c_int

    if not vjoy.vJoyEnabled():
        raise RuntimeError("vJoy driver is not enabled.")

    if not vjoy.AcquireVJD(DEVICE_ID):
        raise RuntimeError(f"Could not acquire vJoy device {DEVICE_ID}.")

    axis_min = 1
    axis_max = 32768
    axis_center = (axis_min + axis_max) // 2

    try:
        print("Center steering")

        vjoy.SetAxis(
            axis_center,
            DEVICE_ID,
            HID_USAGE_X,
        )

        time.sleep(2)

        print("Steering right")

        vjoy.SetAxis(
            axis_max,
            DEVICE_ID,
            HID_USAGE_X,
        )

        time.sleep(2)

        print("Steering left")

        vjoy.SetAxis(
            axis_min,
            DEVICE_ID,
            HID_USAGE_X,
        )

        time.sleep(2)

        print("Back to center")

        vjoy.SetAxis(
            axis_center,
            DEVICE_ID,
            HID_USAGE_X,
        )

    finally:
        vjoy.ResetVJD(DEVICE_ID)
        vjoy.RelinquishVJD(DEVICE_ID)


if __name__ == "__main__":
    main()
