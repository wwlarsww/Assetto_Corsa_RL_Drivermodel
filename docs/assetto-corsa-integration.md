# Assetto Corsa integration

The original Assetto Corsa connection is an outer adapter, not part of
environment or learning logic. A read-only shared-memory transport, a
telemetry-only in-game Python bridge for missing tyre signals, and diagnostic
commands are implemented. An optional X360 report writer is available as a
separately testable adapter. Reset/session automation, track-boundary geometry,
fresh-frame synchronisation, and the complete `SimulatorBackend` remain
separate stages.

The byte layouts follow the
[Assetto Corsa shared-memory reference](https://labs.assettocorsa.jp/documents/reference/shared_memory):
`acpmf_physics`, `acpmf_graphics`, and `acpmf_static`, with four-byte packing
and Windows UTF-16 strings.

## Run the telemetry diagnostic

The transport can be tested with any installed car and track. For the current
base-game installation, Monza and the BMW Z4 GT3 are sufficient; Barcelona and
the Dallara F317 are only required for the later target experiment.

1. Start Assetto Corsa and enter an on-track Practice or Hotlap session.
2. Drive for at least 15 seconds. Include a straight section above `5 km/h`,
   steering, throttle, and braking.
3. In a separate PowerShell terminal, from the repository root, run:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl telemetry-diagnose `
  --duration 15 `
  --poll-hz 100
```

The command writes a compact `summary.json` and raw `telemetry.csv` below a
timestamped `runs/telemetry-diagnostic-*` directory. Use `--no-record` for a
read-only check without files, or `--output <directory>` for an explicit
destination.

Exit codes are:

- `0`: transport usable; calibration warnings may remain;
- `2`: shared memory unavailable, normally because no on-track session exists;
- `3`: telemetry was readable but a transport-level validation failed.

The reader opens existing mappings through the Win32 read-only API. It never
creates an all-zero page when Assetto Corsa is absent.

## Install and validate the tyre-signal bridge

External shared memory does not provide the exact thesis inputs for tyre slip.
The repository therefore contains
`assetto_corsa_app/assetto_corsa_rl_bridge/assetto_corsa_rl_bridge.py`. The app
uses the installed Kunos Python API and writes only `SlipRatio` and `SlipAngle`
to `%TEMP%\assetto_corsa_rl_bridge.bin`. It receives no input and cannot
control the car.

Close Assetto Corsa before installing or changing `python.ini`. From the
repository root:

```powershell
$game = "C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
Copy-Item -Recurse -Force `
  "assetto_corsa_app\assetto_corsa_rl_bridge" `
  "$game\apps\python\assetto_corsa_rl_bridge"
```

Enable `Assetto Corsa RL Bridge` in the launcher's Python-app settings. If no
switch is shown, add the following to
`%USERPROFILE%\Documents\Assetto Corsa\cfg\python.ini` while the game is closed:

```ini
[ASSETTO_CORSA_RL_BRIDGE]
ACTIVE=1
```

Restart the game, enter an on-track session, and run:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl slip-diagnose `
  --duration 15 `
  --with-shared-memory
```

During this capture, drive both a left and a right turn, accelerate, and brake.
The output is written below `runs/slip-diagnostic-*` as `summary.json` and
`slip_telemetry.csv`. Wheel columns are named and retain the Kunos order
front-left, front-right, rear-left, rear-right. With
`--with-shared-memory`, every accepted bridge sequence is paired immediately
with speed, controls, local velocity, acceleration, and the external raw wheel
slip channels. Raw recordings remain diagnostic inputs and do not enter the
observation builder or reward directly.

The binary protocol is fixed little-endian and versioned. The default reader
uses bounded polling, accepts only a new sequence with a source timestamp no
more than one second old, and rejects invalid magic, version, length, and
non-finite values. Partial writes and stale files therefore never become
observations.

The file transport is required by the unmodified game: its embedded Python 3.3
installation contains `socket.py` but not the native `_socket` module. A
loopback UDP reader remains available as an explicit diagnostic transport, but
the vanilla in-game app cannot produce it.

## Required adapter behaviour

`reset(seed, reference)`
: Start or restart a controlled session, synchronise to the first valid frame
  and return a complete `TelemetryFrame`. Assetto Corsa itself may not expose a
  seed; record that limitation in backend metadata instead of claiming
  simulator determinism.

`step(absolute_action)`
: Apply combined pedal and steering, advance exactly one agent control period
  and return the next synchronised frame plus an optional termination reason.
  `TelemetryFrame.progress_m` is the current world position projected onto the
  prescribed reference line; the environment differences consecutive values.

`close()`
: Release shared-memory handles, virtual-controller devices, sockets and helper
  processes idempotently.

Run `rl-ready --backend assetto-corsa` before attempting control. The command
also rejects the current keyboard profile, missing virtual-controller support,
an unvalidated predictor checkpoint, or an incomplete live backend.

## Suggested adapter layers

Keep these game-specific pieces separate:

- telemetry reader: shared-memory physics/graphics/static pages or a local
  bridge;
- control writer: a documented virtual controller or in-game bridge;
- session controller: restart, car/track selection and readiness detection;
- track provider: boundaries and world-position projection onto the closed
  reference arc coordinate;
- synchroniser: maps higher-rate game frames to the configured control rate.

The first implementation should run on Windows, time out all blocking I/O,
detect stale frame counters and fail with component-specific errors.

## Telemetry mapping

The environment needs:

- speed and acceleration on vehicle axes x (lateral) and z (forward);
- front/rear slip ratios and side-slip angles (aggregate wheel signals with a
  documented rule);
- ego-centric left/right track boundaries at the seven preview distances;
- reference-track progress, lateral offset and heading error;
- simulation time and an ego or track-frame position for auxiliary targets.

Do not derive hidden vehicle-model state in the agent path. Store richer raw
telemetry in logs when useful, but keep the baseline observation explicit.

### Current feature readiness

| Required signal | Shared memory | Decision |
|---|---|---|
| speed | `physics.speedKmh` | direct after km/h to m/s conversion |
| longitudinal acceleration | `physics.accG[2]` | Kunos `gMeter` identifies z as longitudinal; the first straight-line capture also matched positive z with the velocity derivative |
| lateral acceleration | `physics.accG[0]` | Kunos `gMeter` identifies x as lateral; validate the sign with left/right turns |
| world pose | `graphics.carCoordinates[3]`, `physics.heading` | direct after axis/sign validation |
| track phase | `graphics.normalizedCarPosition` | direct; compare against reference projection |
| slip ratio | external `physics.wheelSlip[4]` is semantically insufficient | accepted `vanilla_ac_python_v1`: `acsys.CS.SlipRatio`, API sign preserved, arithmetic left/right axle mean |
| side-slip angle | not published in external shared memory | accepted `vanilla_ac_python_v1`: `acsys.CS.SlipAngle` in degrees, API sign preserved, arithmetic axle mean, multiply by `+π/180` |

The installed game confirms that its Python app API exposes both
`acsys.CS.SlipRatio` and `acsys.CS.SlipAngle`, and defines wheel order as
`FL, FR, RL, RR`. The bridge preserves all four values. The deterministic
feature adapter in `ac_integration/slip_features.py` owns the named conversion;
the transport never changes units or signs. The accepted rule also follows the
public
[AssettoCorsaGym car exporter](https://github.com/dasGringuen/assetto_corsa_gym/blob/ef0fa502bdd257fc2affad9361c02b3dbdfe1632/assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/structures.py)
and
[observation definition](https://github.com/dasGringuen/assetto_corsa_gym/blob/ef0fa502bdd257fc2affad9361c02b3dbdfe1632/assetto_corsa_gym/AssettoCorsaEnv/ac_env.py):
preserve API signs and use positive feature scales. The paired left/right
capture validates that wheels on an axle agree closely and that the
degree-scale angles change coherently with lateral acceleration.

Do not replace these channels with shared-memory `physics.wheelSlip`, and do
not add clipping. Large low-speed launch values remain visible to readiness
and safety logic. The offline ACTI/MoTeC source records slip ratio in percent,
so its separate data adapter must divide ratios by `100`; this scaling does not
apply to `CS.SlipRatio`.

## Safety and timing

Game pausing, menus and loading screens are not environment transitions. A
backend should emit a step only for a fresh physics frame. It should distinguish
lap completion from off-track, stalled and excessive-slip termination. External
timeouts become Gymnasium truncations or adapter errors, not successful
terminal states.

## Validation sequence

1. Run `telemetry-diagnose` and retain one raw diagnostic capture.
2. Confirm local velocity and acceleration axes with straight-line,
   acceleration, braking, and left/right steering manoeuvres.
3. Run `slip-diagnose` and verify the named slip calibration plus transport
   timing against the paired shared-memory data.
4. Project world pose onto the fixed reference and validate phase/progress.
5. Construct track boundaries, the twenty fixed-line points, and all 77
   unnormalised features.
6. Replay recorded telemetry through the complete adapter boundary.
7. Verify `gymnasium.utils.env_checker.check_env`.
8. Add and validate control signs at low speed in a safe session.
9. Record one deterministic-policy lap and inspect every observation channel.
10. Only then collect demonstrations or start RL.
