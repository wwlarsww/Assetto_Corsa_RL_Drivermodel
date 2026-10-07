"""Validate Assetto Corsa vehicle dynamics telemetry.

The script performs simple driving manoeuvres through vJoy and records the
raw acceleration and angular velocity components.

The goal is to identify:

- longitudinal acceleration
- lateral acceleration
- yaw rate
"""

import time

from assetto_corsa_ai.action import VehicleAction
from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.assetto_corsa.vjoy_controller import (
    AssettoCorsaVJoyController,
)

SAMPLE_TIME = 0.1


def run_test(
    controller: AssettoCorsaVJoyController,
    telemetry_reader: TelemetryReader,
    action: VehicleAction,
    duration: float,
    label: str,
) -> None:
    """Apply one control action and print the resulting telemetry."""

    print()
    print("=" * 80)
    print(label)
    print("=" * 80)

    controller.apply(action)

    end_time = time.time() + duration

    while time.time() < end_time:
        telemetry = telemetry_reader.read()

        acc = telemetry.acceleration_g
        angular = telemetry.local_angular_velocity

        print(
            f"speed={telemetry.speed_kmh:6.1f} km/h | "
            f"acc "
            f"[0]={acc[0]:+7.3f} "
            f"[1]={acc[1]:+7.3f} "
            f"[2]={acc[2]:+7.3f} | "
            f"angular "
            f"[0]={angular[0]:+7.3f} "
            f"[1]={angular[1]:+7.3f} "
            f"[2]={angular[2]:+7.3f}"
            f"wheel_slip"
            f"[0]={telemetry.wheel_slip[0]:+7.3f} "
            f"[1]={telemetry.wheel_slip[1]:+7.3f} "
            f"[2]={telemetry.wheel_slip[2]:+7.3f} "
            f"[3]={telemetry.wheel_slip[3]:+7.3f}"

        )

        time.sleep(SAMPLE_TIME)


def main() -> None:
    controller = AssettoCorsaVJoyController()
    telemetry_reader = TelemetryReader()

    try:
        # --------------------------------------------------------------
        # 1. Stationary / neutral
        # --------------------------------------------------------------
        run_test(
            controller,
            telemetry_reader,
            VehicleAction(
                steering=0.0,
                throttle=0.0,
                brake=0.0,
            ),
            duration=3.0,
            label="TEST 1 - Neutral",
        )

        # --------------------------------------------------------------
        # 2. Straight-line acceleration
        # --------------------------------------------------------------
        run_test(
            controller,
            telemetry_reader,
            VehicleAction(
                steering=0.0,
                throttle=0.6,
                brake=0.0,
            ),
            duration=5.0,
            label="TEST 2 - Straight-line acceleration",
        )

        # --------------------------------------------------------------
        # 3. Straight-line braking
        # --------------------------------------------------------------
        run_test(
            controller,
            telemetry_reader,
            VehicleAction(
                steering=0.0,
                throttle=0.0,
                brake=0.7,
            ),
            duration=3.0,
            label="TEST 3 - Straight-line braking",
        )

        # Accelerate again before cornering.
        run_test(
            controller,
            telemetry_reader,
            VehicleAction(
                steering=0.0,
                throttle=0.4,
                brake=0.0,
            ),
            duration=4.0,
            label="Prepare for cornering",
        )

        # --------------------------------------------------------------
        # 4. Left turn
        # --------------------------------------------------------------
        run_test(
            controller,
            telemetry_reader,
            VehicleAction(
                steering=-0.25,
                throttle=0.25,
                brake=0.0,
            ),
            duration=3.0,
            label="TEST 4 - Left turn",
        )

        # --------------------------------------------------------------
        # 5. Right turn
        # --------------------------------------------------------------
        run_test(
            controller,
            telemetry_reader,
            VehicleAction(
                steering=0.25,
                throttle=0.25,
                brake=0.0,
            ),
            duration=3.0,
            label="TEST 5 - Right turn",
        )

    finally:
        controller.close()
        telemetry_reader.close()


if __name__ == "__main__":
    main()
