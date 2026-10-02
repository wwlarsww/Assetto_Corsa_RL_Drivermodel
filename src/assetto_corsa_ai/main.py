"""Main entry point for the Assetto Corsa AI."""

from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from time import sleep


def main() -> None:
    """Read and print the current vehicle speed from Assetto Corsa."""
    
    reader = TelemetryReader()
    try:
        while True:
            telemetry = reader.read()

            print(
                f"Speed:       {telemetry.speed_kmh:7.2f} km/h\n"
                f"RPM:         {telemetry.rpm:7d}\n"
                f"Gear:        {telemetry.gear:7d}\n"
                f"Gas:         {telemetry.gas:7.3f}\n"
                f"Brake:       {telemetry.brake:7.3f}\n"
                f"Fuel:        {telemetry.fuel:7.2f}\n"
                f"Steering:    {telemetry.steering_angle:7.3f}\n"
                f"Velocity:    {telemetry.velocity}\n"
                f"Acceleration:{telemetry.acceleration_g}\n"
                f"Wheel Slip:  {telemetry.wheel_slip}\n"
                "----------------------------------------"
            )

            sleep(0.1)
    except KeyboardInterrupt:
        pass

    finally:
        reader.close()

if __name__ == "__main__":
    main()
