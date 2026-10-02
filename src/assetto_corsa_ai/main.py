"""Main entry point for the Assetto Corsa AI."""

from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.assetto_corsa.static_info_reader import StaticInfoReader
from time import sleep


def main() -> None:
    
    # Initialize the static info reader
    static_reader = StaticInfoReader()
    static_info = static_reader.read()

    print(f"Car:          {static_info.car_model}")
    print(f"Track:        {static_info.track_name}")
    print(f"Configuration:{static_info.track_configuration}")
    print(f"Max RPM:      {static_info.max_rpm}")
    print(f"Max Fuel:     {static_info.max_fuel:.1f} L")
    print(f"Tyre Radius:  {static_info.tyre_radius}")
    print(f"Track Length: {static_info.track_spline_length:.1f} m")
    static_reader.close()


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
