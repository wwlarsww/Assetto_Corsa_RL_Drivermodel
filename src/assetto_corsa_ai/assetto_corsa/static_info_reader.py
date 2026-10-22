"""Read static information from Assetto Corsa shared memory."""

import ctypes
import mmap

from assetto_corsa_ai.static_info import SimulatorStaticInfo
from assetto_corsa_ai.assetto_corsa.shared_memory import open_readonly_mapping

_STATIC_MAPPING_NAME = "Local\\acpmf_static"


class _StaticPage(ctypes.Structure):
    """Describe Assetto Corsa's static shared memory layout."""

    _pack_ = 4

    _fields_ = [
        ("smVersion", ctypes.c_wchar * 15),
        ("acVersion", ctypes.c_wchar * 15),
        ("numberOfSessions", ctypes.c_int32),
        ("numCars", ctypes.c_int32),
        ("carModel", ctypes.c_wchar * 33),
        ("track", ctypes.c_wchar * 33),
        ("playerName", ctypes.c_wchar * 33),
        ("playerSurname", ctypes.c_wchar * 33),
        ("playerNick", ctypes.c_wchar * 33),
        ("sectorCount", ctypes.c_int32),
        ("maxTorque", ctypes.c_float),
        ("maxPower", ctypes.c_float),
        ("maxRpm", ctypes.c_int32),
        ("maxFuel", ctypes.c_float),
        ("suspensionMaxTravel", ctypes.c_float * 4),
        ("tyreRadius", ctypes.c_float * 4),
        ("maxTurboBoost", ctypes.c_float),
        ("deprecated_1", ctypes.c_float),
        ("deprecated_2", ctypes.c_float),
        ("penaltiesEnabled", ctypes.c_int32),
        ("aidFuelRate", ctypes.c_float),
        ("aidTireRate", ctypes.c_float),
        ("aidMechanicalDamage", ctypes.c_float),
        ("aidAllowTyreBlankets", ctypes.c_int32),
        ("aidStability", ctypes.c_float),
        ("aidAutoClutch", ctypes.c_int32),
        ("aidAutoBlip", ctypes.c_int32),
        ("hasDRS", ctypes.c_int32),
        ("hasERS", ctypes.c_int32),
        ("hasKERS", ctypes.c_int32),
        ("kersMaxJ", ctypes.c_float),
        ("engineBrakeSettingsCount", ctypes.c_int32),
        ("ersPowerControllerCount", ctypes.c_int32),
        ("trackSPlineLength", ctypes.c_float),
        ("trackConfiguration", ctypes.c_wchar * 33),
        ("ersMaxJ", ctypes.c_float),
        ("isTimedRace", ctypes.c_int32),
        ("hasExtraLap", ctypes.c_int32),
        ("carSkin", ctypes.c_wchar * 33),
        ("reversedGridPositions", ctypes.c_int32),
        ("PitWindowStart", ctypes.c_int32),
        ("PitWindowEnd", ctypes.c_int32),
    ]


class StaticInfoReader:
    """Read static vehicle and track information from Assetto Corsa."""

    def __init__(self) -> None:
        """Defer opening the memory until AC has created it."""

        self._memory: mmap.mmap | None = None
        self._closed = False

    @staticmethod
    def _open_memory() -> mmap.mmap:
        """Open existing memory without creating a competing AC mapping."""

        return open_readonly_mapping(_STATIC_MAPPING_NAME, ctypes.sizeof(_StaticPage))

    def read(self) -> SimulatorStaticInfo:
        """Read static information; raise FileNotFoundError until AC is available."""

        if self._closed:
            raise ValueError("Static info reader is closed.")
        if self._memory is None:
            self._memory = self._open_memory()

        self._memory.seek(0)

        raw_data = self._memory.read(ctypes.sizeof(_StaticPage))

        static = _StaticPage.from_buffer_copy(raw_data)

        return SimulatorStaticInfo(
            car_model=static.carModel,
            track_name=static.track,
            track_configuration=static.trackConfiguration,
            max_rpm=int(static.maxRpm),
            max_fuel=float(static.maxFuel),
            tyre_radius=(
                float(static.tyreRadius[0]),
                float(static.tyreRadius[1]),
                float(static.tyreRadius[2]),
                float(static.tyreRadius[3]),
            ),
            track_spline_length=float(static.trackSPlineLength),
        )

    def close(self) -> None:
        """Close the static shared memory connection."""

        if self._memory is not None:
            self._memory.close()
            self._memory = None
        self._closed = True
