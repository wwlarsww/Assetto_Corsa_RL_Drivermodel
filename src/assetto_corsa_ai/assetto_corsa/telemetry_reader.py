"""Read telemetry data from Assetto Corsa shared memory."""

import ctypes #  
import mmap 


class _PhysicsPage(ctypes.Structure):
    """Describe the relevant part of Assetto Corsa's physics memory layout."""

    # Assetto Corsa uses 4 byte alignment for its shared memory structures.
    _pack_ = 4

    # Field order and data types must exactly match the Assetto Corsa
    # shared memory specification. Otherwise, values are read from
    # incorrect memory offsets.
    _fields_ = [
        ("packetId", ctypes.c_int32),
        ("gas", ctypes.c_float),
        ("brake", ctypes.c_float),
        ("fuel", ctypes.c_float),
        ("gear", ctypes.c_int32),
        ("rpms", ctypes.c_int32),
        ("steerAngle", ctypes.c_float),
        ("speedKmh", ctypes.c_float),
    ]


class TelemetryReader:
    """Read vehicle telemetry from Assetto Corsa's physics shared memory."""

    def __init__(self)  -> None:
        """Open the Assetto Corsa physics shared memory."""
        # Assetto Corsa exposes physics data through a named Windows
        # shared memory region. Access is read only because this class
        # is responsible only for receiving telemetry.
        self._memory = mmap.mmap(
            fileno=-1,  # anonymous, not backed by a file
            length=ctypes.sizeof(_PhysicsPage),
            tagname="Local\\acpmf_physics", # named shared memory region
            access=mmap.ACCESS_READ, 
        )

    def read_speed(self) -> float:
        """Return the current vehicle speed in kilometres per hour."""

        # Reset the memory cursor because each telemetry frame starts
        # at the beginning of the shared memory region.
        self._memory.seek(0)

        raw_data = self._memory.read(ctypes.sizeof(_PhysicsPage))

        # Convert the raw bytes into the structure defined above so that
        # individual telemetry fields can be accessed by name.
        physics = _PhysicsPage.from_buffer_copy(raw_data)

        return float(physics.speedKmh)

    def close(self) -> None:
        """Close the shared memory connection."""

        self._memory.close()
