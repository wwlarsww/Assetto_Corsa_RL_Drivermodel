# Vehicle Control

CSP Custom AI is the current Assetto Corsa control backend. Vehicle control,
session setup and telemetry have separate responsibilities; RL logic uses
simulator-independent data.

## Control Path

The intended agent control flow is:

```text
AgentAction
    ↓
action mapping
    ↓
VehicleAction
    ↓
AssettoCorsaSimulatorController
    ↓
CSP Custom AI shared memory
    ↓
Assetto Corsa
```

An `AgentAction` contains steering and a longitudinal input in `[-1, 1]`.
The action mapping splits the latter into throttle and brake in `[0, 1]`.
The CSP controller currently accepts these three values through
`set_controls(steering, throttle, brake)`; the agent/environment loop is
still to be integrated.

## Responsibilities

| Component | Responsibility |
| --- | --- |
| [Session manager](../src/assetto_corsa_ai/assetto_corsa/session_manager.py) | Prepare car, track, assists and CSP permissions; launch and stop the owned AC process. |
| [Simulator controller](../src/assetto_corsa_ai/assetto_corsa/simulator_controller.py) | Apply controls, teleport, reset an episode, restart the current session and pause. |
| [Static info reader](../src/assetto_corsa_ai/assetto_corsa/static_info_reader.py) | Read the loaded car and track from memory created by AC. |
| [Telemetry reader](../src/assetto_corsa_ai/assetto_corsa/telemetry_reader.py) | Read vehicle state for observations and control validation. |

Session launch requires Windows, CSP Custom AI and unpacked track
`data/surfaces.ini`. The manager prepares the required flags and creates
`*.before-rl.bak` backups before changing existing INI files. It checks that
the requested car, track and layout have loaded. Entering the driving state
is a separate step before connecting the controller.

## Episode Reset

A reset applies the brake, requests a position and forward direction, then
waits for a readiness callback. The validation tool checks position, heading
and low speed using telemetry. Success clears the request and releases the
pedals; timeout or interruption clears the request and leaves the brake applied.

Changing car or track requires the session manager. Restarting the current
session belongs to the simulator controller. Damage repair through teleport
has not yet been verified.

## Live Validation

Run [the validation tool](../tools/validate_simulator_control.py) from the
repository root in the project virtual environment:

```powershell
# Connect to an existing driving session.
python tools/validate_simulator_control.py

# Prepare and launch a session, then connect the controller.
python tools/validate_simulator_control.py --launch --car bmw_z4_gt3 --track monza

# Test session launch without connecting CSP controls.
python tools/validate_simulator_control.py --launch-only --car bmw_z4_gt3 --track monza
```

With `--launch`, select Drive/Start in AC, then confirm in the terminal.
With `--launch-only`, inspect the game first; Enter ends the test.
The tool stops the AC process it launched when the test ends.

| Commands | Purpose |
| --- | --- |
| `s`, `l/r`, `g`, `b`, `n` | Telemetry, steering, throttle, brake and neutral controls. |
| `save`, `t`, `reset` | Capture a stationary pose, teleport to it or validate an episode reset. |
| `pit` | Teleport to the pits, correct the heading and verify the result. |
| `p`, `restart`, `q` | Pause/resume, restart the session and quit. |
| `debug`, `clear` | Inspect teleport memory or cancel a pending request. |

Session launch at the Hotlap position and the corrected pit teleport were
confirmed live with **BMW Z4 GT3, Monza and CSP 0.2.11**:

- Hotlap requires `TYPE=4` and `SPAWN_SET=HOTLAP_START`.
- Static telemetry memory must be created by AC before the reader opens it.
- The native pit pose faces backwards in this setup. The `pit` test waits
  for a stationary pose, clears the native request, then rotates the vehicle
  by 180 degrees at the same position and verifies its heading. This correction
  lives in the validation tool.

Tests cover session configuration, shared-memory startup, controller commands
and pit correction, including timeout and interruption. Structure layouts,
timing and further diagnostics are documented in the source.

For an automated manoeuvre sequence that prints raw acceleration, angular
velocity and wheel slip, run `python tools/validate_vehicle_dynamics.py`
with an active driving session.
