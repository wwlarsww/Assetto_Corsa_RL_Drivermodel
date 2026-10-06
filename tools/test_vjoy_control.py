"""Test direct control of a vJoy device.
This script demonstrates how to control a vJoy device directly.
Open: vJoy Moniror to see the effect of this script on the vJoy device.
Requires vJoy to be installed and a vJoy device to be configured."""

import time
from assetto_corsa_ai.action import VehicleAction
from assetto_corsa_ai.assetto_corsa.vjoy_controller import (
    AssettoCorsaVJoyController,
)


def main() -> None:
    controller = AssettoCorsaVJoyController()

    try:
        print("Neutral")

        controller.apply(
            VehicleAction(
                steering=0.0,
                throttle=0.0,
                brake=0.0,
            )
        )

        time.sleep(2)

        print("Steering right")

        controller.apply(
            VehicleAction(
                steering=1.0,
                throttle=0.0,
                brake=0.0,
            )
        )

        time.sleep(2)

        print("Steering left")
        controller.apply(
            VehicleAction(
                steering=-1.0,
                throttle=0.0,
                brake=0.0,
            )
        )

        time.sleep(2)
        
        print("Steering left")
        
        
        controller.apply(
           VehicleAction(
               steering=0.0,
               throttle=1.0,
               brake=0.0,
           )
        )
        time.sleep(5)
        
        print("full throttle")
        
        controller.apply(
           VehicleAction(
               steering=0.0,
               throttle=0.0,
               brake=1.0,
           )
        )
        time.sleep(5)
        print("full brake")
        
    finally:
        controller.close()


if __name__ == "__main__":
    main()
