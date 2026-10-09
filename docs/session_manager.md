# Session Management

The [`AssettoCorsaSessionManager`](../src/assetto_corsa_ai/assetto_corsa/session_manager.py)
prepares and launches local Assetto Corsa sessions. It selects the scenario
and owns the game process it starts. Runtime commands belong to the
[simulator controller](control.md).

## Session Lifecycle

A `SimulatorScenario` selects car and track, with optional layout, skin and
assist settings. The manager validates the scenario, prepares session files
and CSP permissions, then starts AC.

With a static-info reader, it waits until the requested car, track and layout
are reported. An optional driving-readiness callback can additionally wait
for active driving. Loaded session data alone does not mean that driving
or CSP control is ready.

| Operation | Responsibility |
| --- | --- |
| `start_session()` | Prepare and launch a session. |
| `switch_session()` | Stop and launch for a changed scenario; reuse an identical running scenario. |
| `ensure_session()` | Reuse the current scenario or switch when needed. |
| `stop_session()` | Stop the AC process owned by this manager. |

A startup failure stops the owned process. Ordinary episode resets and a
restart of the current session are handled by the simulator controller.

## Configuration

Session launch requires Windows, CSP Custom AI and unpacked track
`data/surfaces.ini`. The manager prepares car/track selection, driving
assists, extended track physics and Custom AI permissions. Existing INI
files are backed up as `*.before-rl.bak` before modification.

The current setup uses a single-car Hotlap session. Two startup findings
are covered by regression tests:

- Hotlap requires `TYPE=4` and `SPAWN_SET=HOTLAP_START`.
- AC must create static telemetry memory before the reader opens it.

File locations, flags and timeout settings are documented in the source.

## Live Validation

From the repository root in the project virtual environment, close AC first:

```powershell
# Validate launch without connecting CSP controls.
python tools/validate_simulator_control.py --launch-only --car bmw_z4_gt3 --track monza

# Validate launch and then connect the simulator controller.
python tools/validate_simulator_control.py --launch --car bmw_z4_gt3 --track monza
```

With `--launch-only`, select Drive/Start and inspect the game; Enter in the
terminal ends the test. With `--launch`, select Drive/Start first, then confirm
in the terminal to connect CSP. Exiting the tool stops the game process it
launched.

Launch at the correct Hotlap position was confirmed live with **BMW Z4 GT3,
Monza and CSP 0.2.11**.
