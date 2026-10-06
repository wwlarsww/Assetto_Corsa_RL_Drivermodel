Vehicle Control

Assetto Corsa is controlled through a virtual vJoy device.

Download here: https://sourceforge.net/projects/vjoystick/

The project uses normalized vehicle actions:

- `steering`: `[-1.0, 1.0]`
- `throttle`: `[0.0, 1.0]`
- `brake`: `[0.0, 1.0]`

The current vJoy mapping is:

| Vehicle action | vJoy axis |
|---|---|
| Steering | X |
| Throttle | Y |
| Brake | Z |

## Assetto Corsa configuration

The vJoy device is configured as a wheel controller in Assetto Corsa.

For a linear input mapping:

- Steering Gamma: `1.0`
- Steering Filter: `0`
- Steering Speed Sensitivity: `0`
- Brake Gamma: `1.0`

Automatic clutch and automatic shifting are enabled during the initial
development phase.

## Control validation

The control path was validated end to end:

```text
VehicleAction
    ↓
AssettoCorsaVJoyController
    ↓
vJoy
    ↓
Assetto Corsa
    ↓
Shared Memory
    ↓
TelemetryReader
```

The validation is implemented in:

```text
tools/validate_vehicle_control.py
```

The tool sends defined control commands through vJoy and compares them with
the values reported by Assetto Corsa telemetry.

### Throttle and brake

Throttle and brake show a linear mapping between the commanded values and the
simulator telemetry when the corresponding input gamma is set to `1.0`.

Example:

```text
Command throttle = 0.30
Telemetry gas     = 0.30

Command brake     = 0.30
Telemetry brake   = 0.30
```

A previous test with Brake Gamma set to `2.40` produced:

```text
Command brake = 0.30
Telemetry brake = 0.06
```

This is consistent with the nonlinear mapping:

```text
0.30 ^ 2.40 ≈ 0.056
```

Therefore, a gamma value of `1.0` is used for RL training so that the action
space remains linear.

## Steering range validation

The steering mapping depends on the relation between the configured virtual
wheel range and the steering range of the selected vehicle.

If the virtual wheel range is too large, the vehicle reaches full steering
lock before `VehicleAction.steering` reaches `±1.0`. This creates an unused
part of the action space.

For the current development vehicle, matching the configured wheel range to
the vehicle steering range resulted in a linear mapping over the full
`[-1.0, 1.0]` range.

For now, the wheel range is configured manually. Automatic vehicle-specific
steering scaling can be added later when support for multiple cars is needed.
