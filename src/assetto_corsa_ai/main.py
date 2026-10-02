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
                f"Car Coordinates: {telemetry.car_coordinates}\n"
            )
            print(f"Track Position: {telemetry.normalized_track_position:.4f}")
            sleep(0.1)
            
    except KeyboardInterrupt:
        pass

    finally:
        reader.close()

if __name__ == "__main__":
    main()
