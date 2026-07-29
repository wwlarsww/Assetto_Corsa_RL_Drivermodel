# Assetto Corsa Python app

`assetto_corsa_rl_bridge` is a minimal, telemetry-only Python app for the
original Assetto Corsa. It supplies the four tyre-level `SlipRatio` and
`SlipAngle` values that are not available with sufficiently explicit semantics
through external shared memory.

The app:

- reads only `acsys.CS.SlipRatio` and `acsys.CS.SlipAngle`;
- preserves Assetto Corsa wheel order `FL, FR, RL, RR`;
- updates one fixed, versioned binary packet in
  `%TEMP%\assetto_corsa_rl_bridge.bin`;
- accepts no input and sends no vehicle controls;
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
