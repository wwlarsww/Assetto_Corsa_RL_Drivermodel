"""Assetto Corsa simulator control through CSP Custom AI shared memory.
This module provides a controller for Assetto Corsa that communicates
directly with the CSP Custom AI extension using Windows shared memory."""

from __future__ import annotations

from collections.abc import Callable
import ctypes
from ctypes import wintypes
import math
import mmap
import os
import struct
import threading
import time
from types import TracebackType

Vector3 = tuple[float, float, float]


class SimulatorControllerError(RuntimeError):
    """Raised when communication with CSP Custom AI fails."""


class _Float3(ctypes.Structure):
    """Three-dimensional vector used by the CSP Custom AI API."""

    _pack_ = 4

    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]


class _CarControls(ctypes.Structure):
    """CSP ``cai_car_controls`` shared-memory structure.
    Structure layout according to the CSP Custom AI API.
    """

    _pack_ = 4

    _fields_ = [
        # Continuous vehicle controls
        ("gas", ctypes.c_float),
        ("brake", ctypes.c_float),
        ("clutch", ctypes.c_float),
        ("steer", ctypes.c_float),
        ("handbrake", ctypes.c_float),
        # Discrete controls
        ("gear_up", ctypes.c_uint8),
        ("gear_dn", ctypes.c_uint8),
        ("drs", ctypes.c_uint8),
        ("kers", ctypes.c_uint8),
        ("brake_balance_up", ctypes.c_uint8),
        ("brake_balance_dn", ctypes.c_uint8),
        ("abs_up", ctypes.c_uint8),
        ("abs_dn", ctypes.c_uint8),
        ("tc_up", ctypes.c_uint8),
        ("tc_dn", ctypes.c_uint8),
        ("turbo_up", ctypes.c_uint8),
        ("turbo_dn", ctypes.c_uint8),
        ("engine_brake_up", ctypes.c_uint8),
        ("engine_brake_dn", ctypes.c_uint8),
        ("mguk_delivery_up", ctypes.c_uint8),
        ("mguk_delivery_dn", ctypes.c_uint8),
        ("mguk_recovery_up", ctypes.c_uint8),
        ("mguk_recovery_dn", ctypes.c_uint8),
        ("mguh_mode", ctypes.c_uint8),
        ("headlights", ctypes.c_uint8),
        # Teleport / automatic control options
        ("teleport_to", ctypes.c_uint8),
        ("autoclutch_on_start", ctypes.c_uint8),
        ("autoclutch_on_change", ctypes.c_uint8),
        ("autoblip_active", ctypes.c_uint8),
        ("teleport_pos", _Float3),
        ("teleport_dir", _Float3),
        ("autoshift_active", ctypes.c_uint8),
    ]


class _SimulationControl(ctypes.Structure):
    """CSP ``cai_sim_control`` shared-memory structure."""

    _pack_ = 4

    _fields_ = [
        ("pause", ctypes.c_uint8),
        ("restart_session", ctypes.c_uint8),
        ("disable_collisions", ctypes.c_uint8),
        ("extra_sleep_ms", ctypes.c_uint8),
    ]


# Guard against accidental structure-layout changes.
_EXPECTED_CAR_CONTROLS_SIZE = 72
_EXPECTED_SIM_CONTROL_SIZE = 4

if ctypes.sizeof(_CarControls) != _EXPECTED_CAR_CONTROLS_SIZE:
    raise RuntimeError(
        "Unexpected CSP car-controls structure size: "
        f"{ctypes.sizeof(_CarControls)} bytes."
    )

if ctypes.sizeof(_SimulationControl) != _EXPECTED_SIM_CONTROL_SIZE:
    raise RuntimeError(
        "Unexpected CSP simulation-control structure size: "
        f"{ctypes.sizeof(_SimulationControl)} bytes."
    )


class AssettoCorsaSimulatorController:
    """Control Assetto Corsa through CSP Custom AI.

    The controller communicates directly with CSP using Windows
    shared memory.

    The public API covers driving with automatic gears, episode resets,
    session restarts and simulation pause.

    Telemetry remains separate and is handled by ``TelemetryReader``.
    """

    def __init__(
        self,
        car_index: int = 0,
        activation_timeout: float = 3.0,
        pulse_duration: float = 0.02,
    ) -> None:
        if car_index < 0:
            raise ValueError("car_index must be >= 0.")

        if not math.isfinite(activation_timeout) or activation_timeout <= 0.0:
            raise ValueError("activation_timeout must be positive.")

        if not math.isfinite(pulse_duration) or pulse_duration <= 0.0:
            raise ValueError("pulse_duration must be positive.")

        self._car_index = car_index
        self._activation_timeout = activation_timeout
        self._pulse_duration = pulse_duration

        self._controls_name = (
            "AcTools.CSP.NewBehaviour.CustomAI." f"CarControls{car_index}.v0"
        )

        self._car_state_name = "AcTools.CSP.NewBehaviour.CustomAI." f"Car{car_index}.v0"

        self._simulation_name = "AcTools.CSP.NewBehaviour.CustomAI." "SimState.v0"

        self._controls_map: mmap.mmap | None = None
        self._simulation_map: mmap.mmap | None = None

        self._controls: _CarControls | None = None
        self._simulation: _SimulationControl | None = None

        self._lock = threading.RLock()

    @property
    def is_connected(self) -> bool:
        """Return True if the shared-memory mappings are open."""

        return (
            self._controls_map is not None
            and self._simulation_map is not None
            and self._controls is not None
            and self._simulation is not None
        )

    def connect(self) -> None:
        """Connect to CSP Custom AI.

        Creating the car-control memory map tells CSP that an external
        controller wants to take control of the selected vehicle.

        CSP should then create the corresponding car-state memory map.

        Raises:
            SimulatorControllerError:
                If CSP Custom AI does not become active within the configured
                timeout.
        """

        if os.name != "nt":
            raise SimulatorControllerError(
                "CSP Custom AI shared memory is only available on Windows."
            )

        if self.is_connected:
            return

        try:
            self._controls_map = mmap.mmap(
                -1,
                ctypes.sizeof(_CarControls),
                tagname=self._controls_name,
                access=mmap.ACCESS_WRITE,
            )

            self._controls = _CarControls.from_buffer(self._controls_map)

            self._simulation_map = mmap.mmap(
                -1,
                ctypes.sizeof(_SimulationControl),
                tagname=self._simulation_name,
                access=mmap.ACCESS_WRITE,
            )

            self._simulation = _SimulationControl.from_buffer(self._simulation_map)

            self._initialize_shared_memory()

            self._wait_for_csp_activation()

        except BaseException:
            self._release_mappings()
            raise

    def close(self) -> None:
        """Release the CSP shared-memory connection.

        Before closing, safe controls are written so the car does not keep
        accelerating if CSP keeps the mapping alive briefly.
        """

        if not self.is_connected:
            self._release_mappings()
            return

        with self._lock:
            self._set_safe_controls_unlocked()

            assert self._simulation is not None

            self._simulation.pause = 0
            self._simulation.restart_session = 0
            self._simulation.disable_collisions = 0
            self._simulation.extra_sleep_ms = 0

        self._release_mappings()

    def set_controls(
        self,
        steering: float,
        throttle: float,
        brake: float,
    ) -> None:
        """Apply continuous vehicle controls.

        Args:
            steering:
                Normalized steering command in ``[-1, 1]``.

            throttle:
                Throttle command in ``[0, 1]``.

            brake:
                Brake command in ``[0, 1]``.
        """

        self._require_connection()

        steering = self._validate_range(
            steering,
            -1.0,
            1.0,
            "steering",
        )

        throttle = self._validate_range(
            throttle,
            0.0,
            1.0,
            "throttle",
        )

        brake = self._validate_range(
            brake,
            0.0,
            1.0,
            "brake",
        )

        with self._lock:
            assert self._controls is not None

            self._controls.steer = steering
            self._controls.gas = throttle
            self._controls.brake = brake
            self._controls.clutch = 0.0
            self._controls.handbrake = 0.0

    def neutral_controls(self) -> None:
        """Release the pedals and centre the steering."""

        self._require_connection()
        with self._lock:
            self._set_safe_controls_unlocked()
            assert self._controls is not None
            self._controls.brake = 0.0

    def set_position(
        self,
        position: Vector3,
        forward: Vector3,
    ) -> None:
        """Teleport the vehicle to a world-space position.

        Args:
            position:
                Assetto Corsa world coordinates ``(x, y, z)``.

            forward:
                Desired vehicle-forward vector in world coordinates.
                The vector is normalized automatically.

        Notes:
            CSP handles the actual teleport asynchronously. The environment
            should verify the resulting vehicle position using telemetry
            before starting the next episode.
        """

        self._require_connection()

        position = self._validate_vector(
            position,
            "position",
        )

        forward = self._normalize_vector(
            forward,
            "forward",
        )

        with self._lock:
            assert self._controls is not None

            self._controls.teleport_pos.x = position[0]
            self._controls.teleport_pos.y = position[1]
            self._controls.teleport_pos.z = position[2]

            self._controls.teleport_dir.x = forward[0]
            self._controls.teleport_dir.y = forward[1]
            self._controls.teleport_dir.z = forward[2]

            self._controls.teleport_to = 2

    def debug_teleport_memory(self) -> None:
        """Print the current CSP teleport fields for live diagnostics."""

        self._require_connection()
        with self._lock:
            assert self._controls is not None
            print("teleport_to =", self._controls.teleport_to)
            print("teleport_pos =", (
                self._controls.teleport_pos.x,
                self._controls.teleport_pos.y,
                self._controls.teleport_pos.z,
            ))
            print("teleport_dir =", (
                self._controls.teleport_dir.x,
                self._controls.teleport_dir.y,
                self._controls.teleport_dir.z,
            ))
            print("autoshift =", self._controls.autoshift_active)

    def debug_raw_control_memory(self) -> None:
        """Decode an independent snapshot at the documented v0 byte offsets."""

        self._require_connection()
        with self._lock:
            assert self._controls_map is not None
            raw = self._controls_map[:72]
        print("teleport_to raw:", raw[40])
        print("teleport_pos raw:", struct.unpack_from("<3f", raw, 44))
        print("teleport_dir raw:", struct.unpack_from("<3f", raw, 56))
        print("autoshift raw:", raw[68])

    def clear_position_request(
        self,
        ) -> None:
        """Clear an active CSP teleport request."""

        self._require_connection()

        with self._lock:
            assert self._controls is not None
            self._controls.teleport_to = 0

    def reset_episode(
        self,
        position: Vector3,
        forward: Vector3,
        *,
        is_ready: Callable[[], bool],
        timeout: float = 5.0,
        poll_interval: float = 0.01,
    ) -> None:
        """Prepare the vehicle for a new RL episode.

        The reset procedure:

        1. applies a safe stationary control state
        2. teleports the vehicle to the requested position
        3. waits until the caller confirms a valid simulator state
        4. releases the brake and returns controls to neutral

        The readiness check is supplied by the environment so this controller
        remains independent of TelemetryReader and the RL domain layer.

        Args:
            position:
                Target Assetto Corsa world position ``(x, y, z)``.

            forward:
                Desired vehicle-forward direction in world coordinates.

            is_ready:
                Callback returning True once telemetry confirms that the vehicle
                has reached a valid episode-start state.

            timeout:
                Maximum time to wait for the simulator reset.

            poll_interval:
                Delay between readiness checks.

        Raises:
            SimulatorControllerError:
                If the vehicle does not reach a valid state before timeout.
        """

        self._require_connection()

        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout must be positive.")

        if not math.isfinite(poll_interval) or poll_interval <= 0.0:
            raise ValueError("poll_interval must be positive.")

        position = self._validate_vector(
            position,
            "position",
        )

        forward = self._normalize_vector(
            forward,
            "forward",
        )

        # Prepare safe controls and activate teleport request.
        with self._lock:
            self._set_safe_controls_unlocked()

            assert self._controls is not None

            self._controls.teleport_pos.x = position[0]
            self._controls.teleport_pos.y = position[1]
            self._controls.teleport_pos.z = position[2]

            self._controls.teleport_dir.x = forward[0]
            self._controls.teleport_dir.y = forward[1]
            self._controls.teleport_dir.z = forward[2]

            self._controls.teleport_to = 2

        deadline = time.monotonic() + timeout

        try:
            while time.monotonic() < deadline:

                if is_ready():
                    # CSP has actually moved the vehicle.
                    with self._lock:
                        assert self._controls is not None

                        self._controls.teleport_to = 0

                    # Release brake only after successful reset.
                    self.neutral_controls()

                    return

                time.sleep(
                    min(
                        poll_interval,
                        max(
                            0.0,
                            deadline - time.monotonic(),
                        ),
                    )
                )

        except BaseException:
            # Stop requesting teleport, but keep vehicle safely braked.
            with self._lock:
                assert self._controls is not None
                self._controls.teleport_to = 0

            raise

        # Timeout:
        # stop requesting teleport but intentionally keep brake applied.
        with self._lock:
            assert self._controls is not None
            self._controls.teleport_to = 0

        raise SimulatorControllerError(
            "Vehicle did not reach a valid episode-start state "
            f"within {timeout:.2f} seconds."
        )

    def restart_session(self) -> None:
        """Request a restart of the current Assetto Corsa session."""

        self._require_connection()

        with self._lock:
            assert self._simulation is not None

            self._simulation.restart_session = 1

            try:
                time.sleep(self._pulse_duration)
            finally:
                self._simulation.restart_session = 0

    def set_paused(
        self,
        paused: bool,
    ) -> None:
        """Pause or resume the simulation."""

        self._require_connection()

        with self._lock:
            assert self._simulation is not None

            self._simulation.pause = int(paused)

    def _initialize_shared_memory(self) -> None:
        """Initialize both CSP structures to a deterministic safe state."""

        assert self._controls is not None
        assert self._simulation is not None

        ctypes.memset(
            ctypes.addressof(self._controls),
            0,
            ctypes.sizeof(_CarControls),
        )

        ctypes.memset(
            ctypes.addressof(self._simulation),
            0,
            ctypes.sizeof(_SimulationControl),
        )

        # Prevent the vehicle from moving as soon as CSP takes control.
        self._controls.brake = 1.0

        # Sensible defaults for RL control.
        self._controls.autoclutch_on_start = 1
        self._controls.autoclutch_on_change = 1
        self._controls.autoblip_active = 1
        self._controls.autoshift_active = 1

    def _wait_for_csp_activation(self) -> None:
        """Wait until CSP creates the corresponding car-state mapping."""

        deadline = time.monotonic() + self._activation_timeout

        while time.monotonic() < deadline:
            if self._windows_mapping_exists(self._car_state_name):
                return

            time.sleep(0.01)

        raise SimulatorControllerError(
            "CSP Custom AI did not become active. "
            "Check that CUSTOM_AI is enabled in "
            "'extension/config/new_behaviour.ini', that extended physics "
            "is enabled for the track, and that "
            "ALLOW_CUSTOM_AI_MANIPULATION=1 is set in surfaces.ini."
        )

    def _set_safe_controls_unlocked(
        self,
    ) -> None:
        """Apply a safe stationary control state.

        Caller must hold ``self._lock``.
        """

        assert self._controls is not None

        self._controls.steer = 0.0
        self._controls.gas = 0.0
        self._controls.brake = 1.0
        self._controls.clutch = 0.0
        self._controls.handbrake = 0.0

    def _require_connection(self) -> None:
        """Ensure that CSP Custom AI is connected."""

        if not self.is_connected:
            raise SimulatorControllerError(
                "Simulator controller is not connected. " "Call connect() first."
            )

    def _release_mappings(self) -> None:
        """Release ctypes views and Windows memory mappings."""

        self._controls = None
        self._simulation = None

        if self._controls_map is not None:
            self._controls_map.close()
            self._controls_map = None

        if self._simulation_map is not None:
            self._simulation_map.close()
            self._simulation_map = None

    @staticmethod
    def _validate_range(
        value: float,
        minimum: float,
        maximum: float,
        name: str,
    ) -> float:
        """Validate a finite floating-point value inside a range."""

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between " f"{minimum} and {maximum}.")

        return value

    @staticmethod
    def _validate_vector(
        value: Vector3,
        name: str,
    ) -> Vector3:
        """Validate a finite three-dimensional vector."""

        if len(value) != 3:
            raise ValueError(f"{name} must contain exactly three values.")

        result = (
            float(value[0]),
            float(value[1]),
            float(value[2]),
        )

        if not all(math.isfinite(component) for component in result):
            raise ValueError(f"{name} must contain only finite values.")

        return result

    @classmethod
    def _normalize_vector(
        cls,
        value: Vector3,
        name: str,
    ) -> Vector3:
        """Validate and normalize a three-dimensional vector."""

        vector = cls._validate_vector(
            value,
            name,
        )

        length = math.hypot(*vector)

        if length < 1e-9:
            raise ValueError(f"{name} must not be the zero vector.")

        return (
            vector[0] / length,
            vector[1] / length,
            vector[2] / length,
        )

    def teleport_to_pits(
        self,
    ) -> None:
        """Request CSP teleport to pits."""

        self._require_connection()

        with self._lock:
            assert self._controls is not None
            # Flag 1 uses CSP's pit pose; custom position/direction need flag 2.
            self._controls.teleport_to = 1

    @staticmethod
    def _windows_mapping_exists(
        mapping_name: str,
    ) -> bool:
        """Check whether a named Windows memory mapping already exists."""

        if os.name != "nt":
            return False

        file_map_read = 0x0004

        kernel32 = ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        )

        open_file_mapping = kernel32.OpenFileMappingW

        open_file_mapping.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]

        open_file_mapping.restype = wintypes.HANDLE

        close_handle = kernel32.CloseHandle

        close_handle.argtypes = [
            wintypes.HANDLE,
        ]

        close_handle.restype = wintypes.BOOL

        handle = open_file_mapping(
            file_map_read,
            False,
            mapping_name,
        )

        if not handle:
            return False

        close_handle(handle)

        return True

    def __enter__(
        self,
    ) -> "AssettoCorsaSimulatorController":
        """Connect when used as a context manager."""

        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close when leaving a context manager."""

        self.close()
