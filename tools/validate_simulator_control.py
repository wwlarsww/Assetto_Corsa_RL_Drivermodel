"""Interactively validate CSP simulator control against live telemetry."""

import argparse
from collections.abc import Callable
from math import acos, cos, degrees, dist, hypot, isfinite, radians
from pathlib import Path
from time import monotonic, sleep

from assetto_corsa_ai.assetto_corsa.simulator_controller import (
    AssettoCorsaSimulatorController,
    SimulatorControllerError,
)
from assetto_corsa_ai.assetto_corsa.session_manager import (
    AssettoCorsaSessionManager,
    SessionManagerError,
    SimulatorScenario,
)
from assetto_corsa_ai.assetto_corsa.static_info_reader import StaticInfoReader
from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.telemetry import TelemetryFrame, Vector3

SAMPLE_INTERVAL_SECONDS = 0.05
POSITION_TOLERANCE_METRES = 1.0
HEADING_TOLERANCE_DEGREES = 10.0
MAX_RESET_SPEED_KMH = 1.0
RESET_TIMEOUT_SECONDS = 5.0
PIT_SETTLE_SECONDS = 0.2
PIT_SETTLE_DISTANCE_METRES = 0.05


def vehicle_forward(telemetry: TelemetryFrame) -> Vector3:
    """Derive forward from front and rear tyre contacts, including at rest."""

    fl, fr, rl, rr = telemetry.tyre_contact_points
    vector = tuple((fl[i] + fr[i] - rl[i] - rr[i]) / 2.0 for i in range(3))
    length = hypot(*vector)
    if not isfinite(length) or length <= 1e-6:
        raise ValueError("No valid vehicle heading; enter a driving session first.")
    return vector[0] / length, vector[1] / length, vector[2] / length


def reset_ready(telemetry: TelemetryFrame, position: Vector3, forward: Vector3) -> bool:
    """Require the saved pose and low speed before accepting the reset."""

    if not isfinite(telemetry.speed_kmh) or abs(telemetry.speed_kmh) > MAX_RESET_SPEED_KMH:
        return False
    distance = dist(telemetry.car_coordinates, position)
    if not isfinite(distance) or distance > POSITION_TOLERANCE_METRES:
        return False
    try:
        actual_forward = vehicle_forward(telemetry)
    except ValueError:
        return False
    alignment = sum(actual_forward[i] * forward[i] for i in range(3))
    return alignment >= cos(radians(HEADING_TOLERANCE_DEGREES))


def reset_diagnostics(telemetry: TelemetryFrame, position: Vector3, forward: Vector3) -> str:
    """Explain which pose or speed condition prevents reset completion."""

    distance = dist(telemetry.car_coordinates, position)
    height_error = telemetry.car_coordinates[1] - position[1]
    try:
        actual_forward = vehicle_forward(telemetry)
        alignment = sum(actual_forward[i] * forward[i] for i in range(3))
        angle = degrees(acos(max(-1.0, min(1.0, alignment))))
        heading_text = f"{angle:.2f} deg (<= {HEADING_TOLERANCE_DEGREES:.1f})"
    except ValueError:
        heading_text = "invalid tyre geometry"
    return (
        f"Reset check: distance={distance:.3f} m (<= {POSITION_TOLERANCE_METRES:.1f}), "
        f"height error={height_error:+.3f} m, heading error={heading_text}, "
        f"speed={telemetry.speed_kmh:.2f} km/h (<= {MAX_RESET_SPEED_KMH:.1f})\n"
        f"  target={position}\n"
        f"  actual={telemetry.car_coordinates}"
    )


def print_telemetry(reader: TelemetryReader) -> None:
    """Show received controls, speed, gear, world position and raw damage."""

    frame = reader.read()
    print(
        f"speed={frame.speed_kmh:.1f} km/h | gear={frame.gear} | "
        f"steer={frame.steering_angle:+.3f} | gas={frame.gas:.3f} | "
        f"brake={frame.brake:.3f}\n"
        f"position={frame.car_coordinates} | damage={frame.car_damage}"
    )


def readiness_check(
    reader: TelemetryReader, position: Vector3, forward: Vector3,
    *, controller: AssettoCorsaSimulatorController | None = None,
) -> Callable[[], bool]:
    """Build a telemetry readiness callback with periodic diagnostic output."""

    next_report = 0.0
    memory_reported = False

    def is_ready() -> bool:
        nonlocal next_report, memory_reported
        if controller is not None and not memory_reported:
            print_teleport_memory(controller)
            memory_reported = True
        frame = reader.read()
        ready = reset_ready(frame, position, forward)
        now = monotonic()
        if ready or now >= next_report:
            print(reset_diagnostics(frame, position, forward))
            next_report = now + 0.5
        return ready

    return is_ready


def print_teleport_memory(controller: AssettoCorsaSimulatorController) -> None:
    """Compare typed fields with the corresponding raw shared-memory bytes."""

    print("\nCSP teleport fields:")
    controller.debug_teleport_memory()
    print("CSP raw control memory:")
    controller.debug_raw_control_memory()


def validate_teleport(
    controller: AssettoCorsaSimulatorController,
    reader: TelemetryReader,
    position: Vector3,
    forward: Vector3,
) -> None:
    """Keep the teleport request active until the pose is verified or timed out."""

    is_ready = readiness_check(reader, position, forward)
    controller.set_controls(0.0, 0.0, 1.0)
    try:
        controller.set_position(position, forward)
        print_teleport_memory(controller)
        deadline = monotonic() + RESET_TIMEOUT_SECONDS
        while monotonic() < deadline:
            if is_ready():
                print("Teleport verified; request cleared, brake remains applied.")
                return
            sleep(min(SAMPLE_INTERVAL_SECONDS, max(0.0, deadline - monotonic())))
        raise SimulatorControllerError(
            f"Teleport did not reach the saved pose within {RESET_TIMEOUT_SECONDS:.2f} seconds."
        )
    finally:
        controller.clear_position_request()


def measure_controls(
    controller: AssettoCorsaSimulatorController,
    reader: TelemetryReader,
    steering: float,
    throttle: float,
    brake: float,
    duration: float,
) -> None:
    """Apply a short command and return to a braked state even on interruption."""

    print(f"Command: steer={steering:+.2f}, gas={throttle:.2f}, brake={brake:.2f}")
    controller.set_controls(steering, throttle, brake)
    try:
        deadline = monotonic() + duration
        while monotonic() < deadline:
            print_telemetry(reader)
            sleep(0.2)
    finally:
        controller.set_controls(0.0, 0.0, 1.0)


def validate_pit_teleport(
    controller: AssettoCorsaSimulatorController, reader: TelemetryReader,
) -> None:
    """Wait for the pit pose, then turn the stationary vehicle by 180 degrees."""

    initial = reader.read()
    initial_forward = vehicle_forward(initial)
    if not all(isfinite(value) for value in initial.car_coordinates):
        raise ValueError("Vehicle position is not finite.")
    controller.set_controls(0.0, 0.0, 1.0)
    previous: TelemetryFrame | None = None
    settled_since: float | None = None
    pose_changed = False
    deadline = monotonic() + RESET_TIMEOUT_SECONDS
    try:
        controller.teleport_to_pits()
        while monotonic() < deadline:
            frame = reader.read()
            now = monotonic()
            if not all(isfinite(value) for value in frame.car_coordinates):
                previous, settled_since = None, None
                sleep(SAMPLE_INTERVAL_SECONDS)
                continue
            try:
                forward = vehicle_forward(frame)
            except ValueError:
                previous, settled_since = None, None
                sleep(SAMPLE_INTERVAL_SECONDS)
                continue
            alignment = sum(forward[i] * initial_forward[i] for i in range(3))
            pose_changed |= (
                dist(frame.car_coordinates, initial.car_coordinates) > POSITION_TOLERANCE_METRES
                or alignment < cos(radians(HEADING_TOLERANCE_DEGREES))
            )
            stable = (
                pose_changed and previous is not None
                and dist(frame.car_coordinates, previous.car_coordinates) <= PIT_SETTLE_DISTANCE_METRES
                and reset_ready(frame, previous.car_coordinates, vehicle_forward(previous))
            )
            if stable:
                if settled_since is None:
                    settled_since = now
                if now - settled_since >= PIT_SETTLE_SECONDS:
                    # On Monza/CSP 0.2.11 the native pit pose faces backwards.
                    # Clear flag 1 before placing the car at the same position;
                    # rotating around world Y preserves pitch on sloped ground.
                    controller.clear_position_request()
                    validate_teleport(
                        controller, reader, frame.car_coordinates,
                        (-forward[0], forward[1], -forward[2]),
                    )
                    print("Pit teleport and 180-degree heading correction verified.")
                    return
            else:
                settled_since = None
            previous = frame
            sleep(SAMPLE_INTERVAL_SECONDS)
        raise SimulatorControllerError("Pit teleport did not produce a new stationary pose before timeout.")
    finally:
        controller.clear_position_request()


def run_menu(controller: AssettoCorsaSimulatorController, reader: TelemetryReader) -> None:
    """Run individual tests selected by the user; no automatic driving sequence."""

    saved_pose: tuple[Vector3, Vector3] | None = None
    paused = False
    print(
        "\nCSP connected. Automatic gears enabled; brake initially applied.\n"
        "s: telemetry | l/r: steering | g: throttle | b: brake | n: neutral\n"
        "p: pause/resume | save: capture reset pose | t: teleport to saved pose\n"
        "reset: teleport and verify readiness | pit: teleport to pits and correct heading\n"
        "debug: teleport fields and raw bytes | clear: cancel teleport request\n"
        "restart: restart session | q: quit\n"
        "Reset checks pose and speed. Damage is displayed for separate validation."
    )
    while True:
        command = input("\nTest> ").strip().lower()
        try:
            if command == "q":
                return
            if command == "s":
                print_telemetry(reader)
            elif command in ("l", "r", "g", "b"):
                if paused:
                    print("Resume with p before testing controls.")
                    continue
                controls = {
                    "l": (-0.3, 0.0, 1.0, 1.0),
                    "r": (0.3, 0.0, 1.0, 1.0),
                    "g": (0.0, 0.5, 0.0, 2.0),
                    "b": (0.0, 0.0, 1.0, 1.0),
                }
                measure_controls(controller, reader, *controls[command])
            elif command == "n":
                controller.neutral_controls()
                print("Neutral: pedals released, steering centred.")
            elif command == "clear":
                controller.clear_position_request()
                print("Teleport request cleared.")
            elif command == "debug":
                print_teleport_memory(controller)
            elif command == "pit":
                if paused:
                    print("Resume with p before teleporting to pits.")
                    continue
                validate_pit_teleport(controller, reader)
                print_telemetry(reader)
            elif command == "p":
                paused = not paused
                controller.set_paused(paused)
                print("Pause requested." if paused else "Resume requested.")
            elif command == "save":
                frame = reader.read()
                if not isfinite(frame.speed_kmh) or abs(frame.speed_kmh) > MAX_RESET_SPEED_KMH:
                    print("Stop the vehicle before saving a reset pose.")
                    continue
                heading = vehicle_forward(frame)
                if not all(isfinite(value) for value in frame.car_coordinates):
                    raise ValueError("Vehicle position is not finite.")
                saved_pose = frame.car_coordinates, heading
                print(f"Saved position={saved_pose[0]}, forward={saved_pose[1]}")
            elif command in ("t", "reset"):
                if saved_pose is None:
                    print("Capture a starting pose with save first.")
                    continue
                if paused:
                    print("Resume with p before teleporting or resetting.")
                    continue
                position, forward = saved_pose
                if command == "t":
                    validate_teleport(controller, reader, position, forward)
                else:
                    controller.reset_episode(
                        position, forward,
                        is_ready=readiness_check(reader, position, forward, controller=controller),
                        timeout=RESET_TIMEOUT_SECONDS, poll_interval=SAMPLE_INTERVAL_SECONDS,
                    )
                    print("Reset verified: saved pose reached, speed <= 1 km/h; pedals released.")
                print_telemetry(reader)
            elif command == "restart":
                controller.set_controls(0.0, 0.0, 1.0)
                controller.set_paused(False)
                paused = False
                controller.restart_session()
                saved_pose = None
                print("Session restart requested. Capture a new pose after the session loads.")
            else:
                print("Unknown command. Use s, l, r, g, b, n, p, save, t, reset, pit, debug, clear, restart or q.")
        except (SimulatorControllerError, ValueError) as error:
            print(f"Test failed: {error}")
            if command in ("t", "reset") and saved_pose is not None:
                print(reset_diagnostics(reader.read(), *saved_pose))
                print("Brake remains applied. Pose, heading and speed must all meet their limits.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pulse-duration", type=float, default=0.02,
        help="Session restart pulse duration in seconds; teleport waits for telemetry (default: 0.02).",
    )
    parser.add_argument("--launch", action="store_true", help="Prepare flags and start AC through the session manager.")
    parser.add_argument(
        "--launch-only", action="store_true",
        help="Prepare and launch AC without connecting CSP controls; press Enter to stop it.",
    )
    parser.add_argument(
        "--ac-directory", type=Path,
        default=Path(r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"),
        help="Assetto Corsa installation directory.",
    )
    parser.add_argument("--documents-directory", type=Path, help="Assetto Corsa user configuration directory.")
    parser.add_argument("--car", default="bmw_z4_gt3")
    parser.add_argument("--track", default="monza")
    parser.add_argument("--layout", default=None)
    args = parser.parse_args()
    reader: TelemetryReader | None = None
    static_reader: StaticInfoReader | None = None
    session_manager: AssettoCorsaSessionManager | None = None
    controller = AssettoCorsaSimulatorController(pulse_duration=args.pulse_duration)
    try:
        if args.launch or args.launch_only:
            static_reader = StaticInfoReader()
            try:
                static_reader.read()
            except FileNotFoundError:
                pass  # AC creates the mapping after launch; the manager retries.
            else:
                raise SessionManagerError("Close the running Assetto Corsa session before using --launch.")
            session_manager = AssettoCorsaSessionManager(
                args.ac_directory, documents_directory=args.documents_directory,
                static_info_reader=static_reader,
            )
            print(f"Preparing and launching {args.car} on {args.track} through the session manager...", flush=True)
            session_manager.start_session(SimulatorScenario(
                car=args.car, track=args.track, track_layout=args.layout,
            ))
            print("Requested car/track/layout loaded. Enter Drive if AC shows the setup screen.")
            if args.launch_only:
                input("Launch test: CSP controls are disconnected. Inspect AC; press Enter to stop it. ")
                if not session_manager.is_running:
                    raise SessionManagerError("AC exited during the launch test before CSP controls were connected.")
                return
            # Static car/track data is available before driving/Custom AI is ready.
            input("Enter Drive in AC, then press Enter here to connect CSP controls. ")
        print("Connecting CSP controls...", flush=True)
        controller.connect()
        print("CSP controls connected. Opening telemetry...", flush=True)
        reader = TelemetryReader()
        run_menu(controller, reader)
    except (KeyboardInterrupt, EOFError):
        print("\nValidation stopped.")
    except (SimulatorControllerError, SessionManagerError, OSError, ValueError) as error:
        print(f"Validation failed: {error}", flush=True)
        raise SystemExit(f"Cannot run validation: {error}") from error
    finally:
        try:
            try:
                if controller.is_connected:
                    controller.clear_position_request()
            finally:
                controller.close()
        finally:
            try:
                if reader is not None:
                    reader.close()
            finally:
                try:
                    if static_reader is not None:
                        static_reader.close()
                finally:
                    if session_manager is not None:
                        if session_manager.is_running:
                            print("Stopping AC launched by this test script.", flush=True)
                        session_manager.stop_session()


if __name__ == "__main__":
    main()
