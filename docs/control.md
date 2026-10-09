# Output / Control Layer

The [`AssettoCorsaSimulatorController`](../src/assetto_corsa_ai/assetto_corsa/simulator_controller.py)
sends runtime commands to Assetto Corsa through CSP Custom AI. Session setup
and process management belong to the [session manager](session_manager.md);
telemetry is read separately.

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
The controller accepts these three values through
`set_controls(steering, throttle, brake)`;
[`RacingEnvironment`](racing_environment.md) maps the agent's action and
submits these controls on each step.

## Runtime Responsibilities

The controller connects to CSP, applies controls with automatic gears and
supports position teleporting, episode resets, pause and restart of the
current session. Car or track changes belong to the session manager.

A reset applies the brake, requests a position and forward direction, then
waits for a readiness callback supplied by the caller. The validation tool
checks position, heading and low speed using telemetry. Success clears the
request and releases the pedals; timeout or interruption clears the request
and leaves the brake applied. Damage repair through teleport is not yet verified.

## Live Validation

With an active driving session, run [the validation tool](../tools/validate_simulator_control.py)
from the repository root in the project virtual environment:

```powershell
python tools/validate_simulator_control.py
```

For launching AC through the same tool, see [session validation](session_manager.md#live-validation).

| Commands | Purpose |
| --- | --- |
| `s`, `l/r`, `g`, `b`, `n` | Telemetry, steering, throttle, brake and neutral controls. |
| `save`, `t`, `reset` | Capture a stationary pose, teleport to it or validate an episode reset. |
| `pit` | Teleport to the pits, correct the heading and verify the result. |
| `p`, `restart`, `q` | Pause/resume, restart the current session and quit. |
| `debug`, `clear` | Inspect teleport memory or cancel a pending request. |

The corrected pit teleport was confirmed live with **BMW Z4 GT3, Monza and
CSP 0.2.11**. The native pit pose faces backwards in this setup. The `pit`
test waits for a stationary pose, clears the native request, then turns the
vehicle by 180 degrees at the same position and verifies its heading.
This correction lives in the validation tool.

For an automated manoeuvre sequence that prints acceleration, angular velocity
and wheel slip, run `python tools/validate_vehicle_dynamics.py`.
Further API and shared-memory details are documented in the source.
