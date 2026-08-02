"""Assetto Corsa Python app for tyre telemetry and acknowledged CSP resets.

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
COMMAND_FILE_NAME = "assetto_corsa_rl_command.bin"
ACK_FILE_NAME = "assetto_corsa_rl_ack.bin"
BRIDGE_FILE_PATH = os.path.join(
    os.environ.get("TEMP") or os.environ.get("TMP") or ".",
    BRIDGE_FILE_NAME,
)
COMMAND_FILE_PATH = os.path.join(
    os.environ.get("TEMP") or os.environ.get("TMP") or ".",
    COMMAND_FILE_NAME,
)
ACK_FILE_PATH = os.path.join(
    os.environ.get("TEMP") or os.environ.get("TMP") or ".",
    ACK_FILE_NAME,
)
PROTOCOL_MAGIC = b"ACRL"
PROTOCOL_VERSION = 1
PACKET_FORMAT = "<4sBId8f"
COMMAND_MAGIC = b"ACRC"
ACK_MAGIC = b"ACRA"
COMMAND_FORMAT = "<4sBIB"
ACK_FORMAT = "<4sBIBd"
RESET_ACTION = 1
ACK_SUCCESS = 1
ACK_CSP_UNAVAILABLE = 2
ACK_FAILED = 3
SEND_INTERVAL_S = 0.01
ACK_REPLACE_RETRY_COUNT = 20
ACK_REPLACE_RETRY_DELAY_S = 0.001

_app_window = 0
_status_label = 0
_bridge_file = None
_sequence = 0
_accumulator_s = 0.0
_last_error = ""
_last_command_sequence = None


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


def _write_reset_ack(sequence, status):
    payload = struct.pack(
        ACK_FORMAT,
        ACK_MAGIC,
        PROTOCOL_VERSION,
        sequence,
        status,
        time.time(),
    )
    temporary_path = ACK_FILE_PATH + ".tmp"
    handle = open(temporary_path, "wb")
    try:
        handle.write(payload)
        handle.flush()
    finally:
        handle.close()
    for attempt in range(ACK_REPLACE_RETRY_COUNT):
        try:
            os.replace(temporary_path, ACK_FILE_PATH)
            return
        except OSError:
            if attempt + 1 >= ACK_REPLACE_RETRY_COUNT:
                raise
            time.sleep(ACK_REPLACE_RETRY_DELAY_S)


def _process_reset_command():
    global _last_command_sequence, _last_error
    try:
        handle = open(COMMAND_FILE_PATH, "rb")
        try:
            payload = handle.read()
        finally:
            handle.close()
    except OSError:
        return
    try:
        if len(payload) != struct.calcsize(COMMAND_FORMAT):
            return
        magic, version, sequence, action = struct.unpack(COMMAND_FORMAT, payload)
        if magic != COMMAND_MAGIC or version != PROTOCOL_VERSION or action != RESET_ACTION:
            return
        if sequence == _last_command_sequence:
            return
        _last_command_sequence = sequence
        try:
            os.remove(COMMAND_FILE_PATH)
        except OSError:
            pass
        if not hasattr(ac, "ext_resetCar"):
            _write_reset_ack(sequence, ACK_CSP_UNAVAILABLE)
            _set_status("RESET ERROR: Custom Shaders Patch required")
            return
        try:
            ac.ext_resetCar()
        except Exception as error:
            _write_reset_ack(sequence, ACK_FAILED)
            message = type(error).__name__ + ": " + str(error)
            ac.log("[assetto_corsa_rl_bridge] reset failed: " + message)
            _set_status("RESET ERROR: " + message[:45])
            _last_error = message
            return
        _write_reset_ack(sequence, ACK_SUCCESS)
        _set_status("Reset acknowledged - packet " + str(sequence))
    except Exception as error:
        message = type(error).__name__ + ": " + str(error)
        if message != _last_error:
            ac.log("[assetto_corsa_rl_bridge] command error: " + message)
            _last_error = message


def acMain(ac_version):
    """Create the in-game status app and initialise the local bridge file.

    Args:
        ac_version: Assetto Corsa Python API version supplied by the game.

    Returns:
        Application name expected by Assetto Corsa's app loader.
    """

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
    """Publish slip telemetry and process reset commands on the game callback.

    Args:
        delta_t: Elapsed simulation time since the preceding callback.
    """

    global _accumulator_s, _last_error, _sequence
    if _import_error or _bridge_file is None:
        return
    _process_reset_command()
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
    """Close the bridge file when Assetto Corsa unloads the app."""

    global _bridge_file
    if _bridge_file is not None:
        _bridge_file.close()
        _bridge_file = None
    try:
        os.remove(BRIDGE_FILE_PATH)
    except OSError:
        pass
