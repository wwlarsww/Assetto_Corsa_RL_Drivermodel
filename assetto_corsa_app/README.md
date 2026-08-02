# Assetto Corsa bridge apps

`assetto_corsa_rl_bridge` is a minimal telemetry-and-reset Python app for the
original Assetto Corsa. It supplies the four tyre-level `SlipRatio` and
`SlipAngle` values that are not available with sufficiently explicit semantics
through external shared memory and executes acknowledged CSP car resets.

The app:

- reads only `acsys.CS.SlipRatio` and `acsys.CS.SlipAngle`;
- preserves Assetto Corsa wheel order `FL, FR, RL, RR`;
- updates one fixed, versioned binary packet in
  `%TEMP%\assetto_corsa_rl_bridge.bin`;
- accepts only a versioned local reset command, calls `ac.ext_resetCar()`, and
  writes a matching acknowledgement;
- never accepts steering, throttle, brake, arbitrary code, or network input;
- uses only the standard library supported by Assetto Corsa's embedded Python.

Install it only while Assetto Corsa is closed:

```powershell
$game = "C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
Copy-Item -Recurse -Force `
  "assetto_corsa_app\assetto_corsa_rl_bridge" `
  "$game\apps\python\assetto_corsa_rl_bridge"
```

Enable `Assetto Corsa RL Bridge` in the launcher's Python-app settings and
restart the game. If the launcher does not expose the switch, add this section
to `%USERPROFILE%\Documents\Assetto Corsa\cfg\python.ini` while the game is
closed:

```ini
[ASSETTO_CORSA_RL_BRIDGE]
ACTIVE=1
```

Once an on-track session is running, validate the app from the project root:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl slip-diagnose `
  --duration 15 `
  --with-shared-memory
```

Drive through both a left and a right turn and include acceleration and
braking. The diagnostic records raw tyre values for unit, sign, and axle
aggregation validation and pairs them with controls and vehicle motion from
shared memory. The transport deliberately leaves them unchanged. The external
named adapter `vanilla_ac_python_v1` preserves the API signs, computes
arithmetic front/rear axle means, and converts `SlipAngle` from degrees to
radians with `+π/180`.

The file transport is intentional. Vanilla Assetto Corsa's embedded Python 3.3
does not ship the native `_socket` extension, so importing `socket.py` fails.
The external reader accepts only complete 49-byte packets with the expected
magic and version, a new sequence number, finite values, and a source timestamp
no more than one second old. A stale file left by a crash is therefore not a
valid observation.

Episode reset uses `%TEMP%\assetto_corsa_rl_command.bin` and
`%TEMP%\assetto_corsa_rl_ack.bin`. Both packets have fixed little-endian
formats, magic bytes, a protocol version, and a matching unsigned sequence.
Without Custom Shaders Patch the app returns an explicit unavailable status;
the external backend then aborts without starting an episode.

## CSP Lua collision app

`assetto_corsa_rl_csp` is a separate minimal Lua app for crash termination. It
registers `ac.onCarCollision(0, callback)`, increments a persistent unsigned
sequence when car 0 first contacts a wall or another car, and atomically writes
the following ASCII state to `%TEMP%\assetto_corsa_rl_csp_state.txt`:

```text
ACRL_CSP_STATE_V1
<track-id>
<collision-sequence>
<simulator-frame>
```

The sequence latches an event across the faster simulator callback and slower
agent loop. The Python backend records its reset-time baseline and terminates
on any later change. The app exposes no network input, policy control, damage,
or vehicle geometry.

Install it while Assetto Corsa is closed:

```powershell
$game = "C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
Copy-Item -Recurse -Force `
  "assetto_corsa_app\assetto_corsa_rl_csp" `
  "$game\apps\lua\assetto_corsa_rl_csp"
```

Enable `Assetto Corsa RL CSP Bridge` in Content Manager. `rl-ready` requires
the installed Lua source before allowing live training.
