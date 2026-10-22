# Session Management

[AssettoCorsaSessionManager](../src/assetto_corsa_ai/assetto_corsa/session_manager.py)
prepares local AC sessions and owns the game process it launches. Runtime
commands belong to the [simulator controller](control.md).

## Session Lifecycle

A `SimulatorScenario` selects car, track, optional layout/skin and assists.
The manager validates it, prepares session files and CSP permissions, then
starts AC. An unchanged running scenario is reused; changes require a new
session. Ordinary episode resets instead reuse the session and teleport.

The manager waits for static telemetry to report the requested scenario.
A driving-readiness callback separately confirms active driving before CSP
controls connect. Startup failure stops the owned process; shutdown releases
that process.

## Configuration

Launch requires Windows, CSP Custom AI and unpacked track `data/surfaces.ini`.
The current setup uses single-car Hotlap. Modified INI files are backed up
as `*.before-rl.bak`. File locations, launch flags, startup sequencing and
timeouts are documented in
[the manager source](../src/assetto_corsa_ai/assetto_corsa/session_manager.py).
Training supplies these settings through
[config.toml](../configs/training.toml).

## Live Validation

Close AC first, then run [the simulator validator](../tools/validation/validate_simulator_control.py)
from the project virtual environment:

```powershell
python tools/validate.py controls --launch-only --car bmw_z4_gt3 --track monza
python tools/validate.py controls --launch --car bmw_z4_gt3 --track monza
```

With `--launch-only`, select Drive/Start and inspect the game; Enter in the
terminal ends the test. With `--launch`, confirm active driving in the terminal
to connect CSP. Exiting stops the game process launched by the tool.

Launch at the correct Hotlap position was confirmed with BMW Z4 GT3, Monza
and CSP 0.2.11.
