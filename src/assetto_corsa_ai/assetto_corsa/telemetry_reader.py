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
        ("velocity", ctypes.c_float * 3), # world coordinates
        ("accG", ctypes.c_float * 3), # local coordinates
        # components of the wheel slip
        # 0 → vorne links,1 → vorne rechts
        # 2 → hinten links, 3 → hinten rechts
        ("wheelSlip", ctypes.c_float * 4),
        ("wheelLoad", ctypes.c_float * 4),
        ("wheelsPressure", ctypes.c_float * 4),
        ("wheelAngularSpeed", ctypes.c_float * 4),
        ("tyreWear", ctypes.c_float * 4),
        ("tyreDirtyLevel", ctypes.c_float * 4),
        ("tyreCoreTemperature", ctypes.c_float * 4),
        ("camberRAD", ctypes.c_float * 4),
        ("suspensionTravel", ctypes.c_float * 4),

        ("drs", ctypes.c_float),
        ("tc", ctypes.c_float),
        ("heading", ctypes.c_float),
        ("pitch", ctypes.c_float),
        ("roll", ctypes.c_float),
        ("cgHeight", ctypes.c_float),

        ("carDamage", ctypes.c_float * 5),

        ("numberOfTyresOut", ctypes.c_int32),
        ("pitLimiterOn", ctypes.c_int32),

        ("abs", ctypes.c_float),
        ("kersCharge", ctypes.c_float),
        ("kersInput", ctypes.c_float),

        ("autoShifterOn", ctypes.c_int32),

        ("rideHeight", ctypes.c_float * 2),
        ("turboBoost", ctypes.c_float),
        ("ballast", ctypes.c_float),
        ("airDensity", ctypes.c_float),
        ("airTemp", ctypes.c_float),
        ("roadTemp", ctypes.c_float),

        ("localAngularVel", ctypes.c_float * 3),

        ("finalFF", ctypes.c_float),
        ("performanceMeter", ctypes.c_float),

        ("engineBrake", ctypes.c_int32),
        ("ersRecoveryLevel", ctypes.c_int32),
        ("ersPowerLevel", ctypes.c_int32),
        ("ersHeatCharging", ctypes.c_int32),
        ("ersIsCharging", ctypes.c_int32),

        ("kersCurrentKJ", ctypes.c_float),

        ("drsAvailable", ctypes.c_int32),
        ("drsEnabled", ctypes.c_int32),

        ("brakeTemp", ctypes.c_float * 4),
        ("clutch", ctypes.c_float),

        ("tyreTempI", ctypes.c_float * 4),
        ("tyreTempM", ctypes.c_float * 4),
        ("tyreTempO", ctypes.c_float * 4),

        ("isAIControlled", ctypes.c_int32),

        ("tyreContactPoint", (ctypes.c_float * 3) * 4),
        ("tyreContactNormal", (ctypes.c_float * 3) * 4),
        ("tyreContactHeading", (ctypes.c_float * 3) * 4),

        ("brakeBias", ctypes.c_float),

        ("localVelocity", ctypes.c_float * 3),
    ]


class _GraphicsPage(ctypes.Structure):
    """Describe Assetto Corsa's graphics shared memory layout."""

    _pack_ = 4

    _fields_ = [
        ("packetId", ctypes.c_int32),
        ("status", ctypes.c_int32),
        ("session", ctypes.c_int32),
        ("currentTime", ctypes.c_wchar * 15),
        ("lastTime", ctypes.c_wchar * 15),
        ("bestTime", ctypes.c_wchar * 15),
        ("split", ctypes.c_wchar * 15),
        ("completedLaps", ctypes.c_int32),
        ("position", ctypes.c_int32),  # placement
        ("iCurrentTime", ctypes.c_int32),
        ("iLastTime", ctypes.c_int32),
        ("iBestTime", ctypes.c_int32),
        ("sessionTimeLeft", ctypes.c_float),
        ("distanceTraveled", ctypes.c_float),
        ("isInPit", ctypes.c_int32),
        ("currentSectorIndex", ctypes.c_int32),
        ("lastSectorTime", ctypes.c_int32),
        ("numberOfLaps", ctypes.c_int32),
        ("tyreCompound", ctypes.c_wchar * 33),
        ("replayTimeMultiplier", ctypes.c_float),
        (
            "normalizedCarPosition",
            ctypes.c_float,
        ),  # progress of the car on the track (0->1)
        ("carCoordinates", ctypes.c_float * 3),
        ("penaltyTime", ctypes.c_float),
        ("flag", ctypes.c_int32),
        ("idealLineOn", ctypes.c_int32),
        ("isInPitLane", ctypes.c_int32),
        ("surfaceGrip", ctypes.c_float),
        ("mandatoryPitDone", ctypes.c_int32),
        ("windSpeed", ctypes.c_float),
        ("windDirection", ctypes.c_float),
    ]


class TelemetryReader:
    """Read vehicle telemetry from Assetto Corsa's physics shared memory."""

    def __init__(self)  -> None:
        """Open the Assetto Corsa physics shared memory."""
        # Assetto Corsa exposes physics data through a named Windows
        # shared memory region. Access is read only because this class
        # is responsible only for receiving telemetry.
        self._physics_memory = mmap.mmap(
            fileno=-1,  # anonymous, not backed by a file
            length=ctypes.sizeof(_PhysicsPage),
            tagname="Local\\acpmf_physics",  # named shared memory region
            access=mmap.ACCESS_READ,
        )

        self._graphics_memory = mmap.mmap(
            fileno=-1,
            length=ctypes.sizeof(_GraphicsPage),
            tagname="Local\\acpmf_graphics",
            access=mmap.ACCESS_READ,
        )

    def read(self) -> TelemetryFrame:
        """Return the current vehicle telemetry."""

        # Reset the memory cursor because each telemetry frame starts
        # at the beginning of the shared memory region.
        self._physics_memory.seek(0)

        # Interpret the raw bytes according to Assetto Corsa's memory layout.
        physics_raw_data = self._physics_memory.read(
            ctypes.sizeof(_PhysicsPage)
        )
        # Read the physics data from the shared memory according to the Assetto Corsa memory layout.
        physics = _PhysicsPage.from_buffer_copy(
            physics_raw_data
        )
        self._graphics_memory.seek(0)

        graphics_raw_data = self._graphics_memory.read(
            ctypes.sizeof(_GraphicsPage)
        )
        graphics = _GraphicsPage.from_buffer_copy(graphics_raw_data)

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
            ),
            local_velocity=(
                float(physics.localVelocity[0]),
                float(physics.localVelocity[1]),
                float(physics.localVelocity[2]),
            ),
            local_angular_velocity=(
                float(physics.localAngularVel[0]),
                float(physics.localAngularVel[1]),
                float(physics.localAngularVel[2]),
            ),
            tyre_contact_points=(
                (
                    float(physics.tyreContactPoint[0][0]),
                    float(physics.tyreContactPoint[0][1]),
                    float(physics.tyreContactPoint[0][2]),
                ),
                (
                    float(physics.tyreContactPoint[1][0]),
                    float(physics.tyreContactPoint[1][1]),
                    float(physics.tyreContactPoint[1][2]),
                ),
                (
                    float(physics.tyreContactPoint[2][0]),
                    float(physics.tyreContactPoint[2][1]),
                    float(physics.tyreContactPoint[2][2]),
                ),
                (
                    float(physics.tyreContactPoint[3][0]),
                    float(physics.tyreContactPoint[3][1]),
                    float(physics.tyreContactPoint[3][2]),
                ),
            ),
            wheel_angular_speed=(
                float(physics.wheelAngularSpeed[0]),
                float(physics.wheelAngularSpeed[1]),
                float(physics.wheelAngularSpeed[2]),
                float(physics.wheelAngularSpeed[3]),
                
            ),
            # Graphics data is used for world position and normalized track position
            car_coordinates=(
                float(graphics.carCoordinates[0]),
                float(graphics.carCoordinates[1]),
                float(graphics.carCoordinates[2]),
            ),
            normalized_track_position=float(graphics.normalizedCarPosition),
            car_damage=(
                float(physics.carDamage[0]),
                float(physics.carDamage[1]),
                float(physics.carDamage[2]),
                float(physics.carDamage[3]),
                float(physics.carDamage[4]),
            ),
        )

    def close(self) -> None:
        """Close the shared memory connection."""

        self._physics_memory.close()
        self._graphics_memory.close()
