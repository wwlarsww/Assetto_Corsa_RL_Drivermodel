# Output / Control Layer

[AssettoCorsaSimulatorController](../src/assetto_corsa_ai/assetto_corsa/simulator_controller.py)
sends runtime commands through CSP Custom AI. Scenario setup and process
management belong to the [session manager](session_manager.md); telemetry is
read separately.

## Control Path

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

[Action mapping](../src/assetto_corsa_ai/action.py) splits the final longitudinal
input into throttle and brake. [RacingEnvironment](../src/assetto_corsa_ai/RacingEnvironment.py)
sends steering, throttle and brake through `set_controls()`. Policy deltas
are resolved before reaching this boundary. In delta mode, the agent's
[action filter](../src/assetto_corsa_ai/action_wrapper.py) applies the TOML's
per-step limits and longitudinal deadband before controls reach the environment.

## Runtime Responsibilities

The controller connects to CSP, applies controls with automatic gears and
supports teleporting, episode reset, pause and session restart. Car or track
changes belong to the session manager.

Reset applies the brake, requests a pose and waits for the caller's readiness
check. Success releases the request and pedals; timeout or interruption clears
the request and leaves the brake applied. Damage repair through teleport is
not yet verified. API and shared-memory details are in
[the controller source](../src/assetto_corsa_ai/assetto_corsa/simulator_controller.py).

## Live Validation

With an active driving session, run [the validation tool](../tools/validate_simulator_control.py)
from the repository root in the project virtual environment:

```powershell
python tools/validate_simulator_control.py
```

It supports control checks, pose capture/reset, pit teleport and inspection of
CSP fields. Commands and options are described in the tool. For launching AC,
see [session validation](session_manager.md#live-validation).

The corrected pit teleport was confirmed with BMW Z4 GT3, Monza and CSP 0.2.11.
In this setup the native pit pose faces backwards; the test turns the stationary
vehicle by 180 degrees and verifies its heading. The correction remains in
[the validation tool](../tools/validate_simulator_control.py).

[Vehicle dynamics validation](../tools/validate_vehicle_dynamics.py) provides
an automated manoeuvre sequence with acceleration, yaw and wheel-slip readouts.
