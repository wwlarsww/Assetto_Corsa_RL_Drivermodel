"""Read telemetry data from Assetto Corsa shared memory."""

import ctypes
import mmap 
from assetto_corsa_ai.telemetry import TelemetryFrame

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
        # +3 -> Array with x, y, z float values
        ("velocity", ctypes.c_float * 3),
        ("accG", ctypes.c_float * 3),
        # components of the wheel slip
        # 0 → vorne links,1 → vorne rechts
        # 2 → hinten links, 3 → hinten rechts
        ("wheelSlip", ctypes.c_float * 4),
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

    def read(self) -> TelemetryFrame:
        """Return the current vehicle telemetry."""

        # Reset the memory cursor because each telemetry frame starts
        # at the beginning of the shared memory region.
        self._memory.seek(0)

        raw_data = self._memory.read(ctypes.sizeof(_PhysicsPage))

        # Interpret the raw bytes according to Assetto Corsa's memory layout.
        physics = _PhysicsPage.from_buffer_copy(raw_data)

        # Convert Assetto Corsa specific field names into our internal
        # simulator independent telemetry representation.
        return TelemetryFrame(
            speed_kmh=float(physics.speedKmh),
            gas=float(physics.gas),
            brake=float(physics.brake),
            fuel=float(physics.fuel),
            gear=int(physics.gear),
            rpm=int(physics.rpms),
            steering_angle=float(physics.steerAngle),
            velocity=(
                float(physics.velocity[0]),
                float(physics.velocity[1]),
                float(physics.velocity[2]),
            ),
            acceleration_g=(
                float(physics.accG[0]),
                float(physics.accG[1]),
                float(physics.accG[2]),
            ),
            wheel_slip=(
                float(physics.wheelSlip[0]),
                float(physics.wheelSlip[1]),
                float(physics.wheelSlip[2]),
                float(physics.wheelSlip[3]),
            )
        )

    def close(self) -> None:
        """Close the shared memory connection."""

        self._memory.close()
