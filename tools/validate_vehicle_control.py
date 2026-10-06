"""Validate vehicle control against Assetto Corsa telemetry."""

import time

from assetto_corsa_ai.action import VehicleAction
from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.assetto_corsa.vjoy_controller import (
    AssettoCorsaVJoyController,
)


def apply_and_measure(
    controller: AssettoCorsaVJoyController,
    telemetry_reader: TelemetryReader,
    action: VehicleAction,
    label: str,
) -> None:
    """Apply an action and print the resulting simulator telemetry."""

    controller.apply(action)

    # Give Assetto Corsa time to process the new input.
    time.sleep(1)

    telemetry = telemetry_reader.read()

    print()
    print(label)

    print(
        f"Command:   "
        f"steering={action.steering:+.2f}, "
        f"throttle={action.throttle:.2f}, "
        f"brake={action.brake:.2f}"
    )

    print(
        f"Telemetry: "
        f"steering_angle={telemetry.steering_angle:+.3f}, "
        f"gas={telemetry.gas:.2f}, "
        f"brake={telemetry.brake:.2f}"
    )


def test_steering(
    controller: AssettoCorsaVJoyController,
    telemetry_reader: TelemetryReader,
) -> None:
    """Measure Assetto Corsa steering response over the full input range."""

    steering_values = [
        0.0,
        0.25,
        0.50,
        0.75,
        1.00,
        0.0,
        -0.25,
        -0.50,
        -0.75,
        -1.00,
        0.0,
    ]

    print()
    print("=== Steering sweep ===")

    for steering in steering_values:
        apply_and_measure(
            controller,
            telemetry_reader,
            VehicleAction(
                steering=steering,
                throttle=0.0,
                brake=0.0,
            ),
            f"Steering {steering:+.2f}",
        )


def test_pedals(
    controller: AssettoCorsaVJoyController,
    telemetry_reader: TelemetryReader,
) -> None:
    """Measure throttle and brake response."""

    print()
    print("=== Pedal test ===")

    apply_and_measure(
        controller,
        telemetry_reader,
        VehicleAction(
            steering=0.0,
            throttle=0.30,
            brake=0.0,
        ),
        "30 % throttle",
    )

    apply_and_measure(
        controller,
        telemetry_reader,
        VehicleAction(
            steering=0.0,
            throttle=0.0,
            brake=0.30,
        ),
        "30 % brake",
    )


def main() -> None:
    controller = AssettoCorsaVJoyController()
    telemetry_reader = TelemetryReader()

    try:
        test_steering(
            controller,
            telemetry_reader,
        )

        test_pedals(
            controller,
            telemetry_reader,
        )

    finally:
        controller.close()
        telemetry_reader.close()


if __name__ == "__main__":
    main()
