"""Main entry point for the Assetto Corsa AI."""

from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader

def main() -> None:
    """Read and print the current vehicle speed from Assetto Corsa."""
    
    telemetry_reader = TelemetryReader()
    speed = telemetry_reader.read_speed()
    print(f"Current vehicle speed: {speed:.2f} km/h")
    print("Assetto Corsa RL")


if __name__ == "__main__":
    main()
