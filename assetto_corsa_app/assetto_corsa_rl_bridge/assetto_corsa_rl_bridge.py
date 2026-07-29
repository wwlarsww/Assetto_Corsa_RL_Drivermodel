"""Assetto Corsa Python app that publishes tyre slip values to a local file.

This file intentionally targets Assetto Corsa's old embedded Python runtime:
no third-party packages, annotations, dataclasses, or f-strings are used.
"""

import ac

_import_error = ""
try:
    import math
    import os
    import struct
    import time

    import acsys
except Exception as error:
    _import_error = type(error).__name__ + ": " + str(error)
    ac.log("[assetto_corsa_rl_bridge] import error: " + _import_error)

APP_NAME = "Assetto Corsa RL Bridge"
BRIDGE_FILE_NAME = "assetto_corsa_rl_bridge.bin"
BRIDGE_FILE_PATH = os.path.join(
    os.environ.get("TEMP") or os.environ.get("TMP") or ".",
    BRIDGE_FILE_NAME,
)
PROTOCOL_MAGIC = b"ACRL"
PROTOCOL_VERSION = 1
PACKET_FORMAT = "<4sBId8f"
SEND_INTERVAL_S = 0.01

_app_window = 0
_status_label = 0
_bridge_file = None
_sequence = 0
_accumulator_s = 0.0
_last_error = ""


def _four_finite_values(state_name, values):
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(state_name + " did not return four numeric values") from error
    if len(result) != 4:
        raise ValueError(state_name + " did not return FL, FR, RL, RR")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(state_name + " returned a non-finite value")
    return result


def _encode_packet(sequence, source_time_s, slip_ratio, slip_angle):
    values = slip_ratio + slip_angle
    return struct.pack(
        PACKET_FORMAT,
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        sequence,
        source_time_s,
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
    )


def _set_status(text):
    if _status_label:
        ac.setText(_status_label, text)


def acMain(ac_version):
    global _app_window, _status_label, _bridge_file
    _app_window = ac.newApp(APP_NAME)
    ac.setSize(_app_window, 360, 65)
    _status_label = ac.addLabel(_app_window, "Starting local telemetry...")
    ac.setPosition(_status_label, 12, 32)
    if _import_error:
        _set_status("IMPORT ERROR: " + _import_error[:45])
        return APP_NAME
    _bridge_file = open(BRIDGE_FILE_PATH, "w+b", 0)
    _set_status("Local telemetry file - waiting for acUpdate")
    ac.log("[assetto_corsa_rl_bridge] initialised for AC " + str(ac_version))
    return APP_NAME


def acUpdate(delta_t):
    global _accumulator_s, _last_error, _sequence
    if _import_error or _bridge_file is None:
        return
    _accumulator_s += max(float(delta_t), 0.0)
    if _accumulator_s < SEND_INTERVAL_S:
        return
    _accumulator_s = _accumulator_s % SEND_INTERVAL_S
    try:
        slip_ratio = _four_finite_values("SlipRatio", ac.getCarState(0, acsys.CS.SlipRatio))
        slip_angle = _four_finite_values("SlipAngle", ac.getCarState(0, acsys.CS.SlipAngle))
        payload = _encode_packet(_sequence, time.time(), slip_ratio, slip_angle)
        _bridge_file.seek(0)
        _bridge_file.write(payload)
        _bridge_file.flush()
        _sequence = (_sequence + 1) & 0xFFFFFFFF
        if _sequence % 100 == 0:
            _set_status("Sending tyre telemetry - packet " + str(_sequence))
        _last_error = ""
    except Exception as error:
        message = type(error).__name__ + ": " + str(error)
        _set_status("ERROR: " + message[:52])
        if message != _last_error:
            ac.log("[assetto_corsa_rl_bridge] " + message)
            _last_error = message


def acShutdown():
    global _bridge_file
    if _bridge_file is not None:
        _bridge_file.close()
        _bridge_file = None
    try:
        os.remove(BRIDGE_FILE_PATH)
    except OSError:
        pass
