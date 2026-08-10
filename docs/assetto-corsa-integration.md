# Assetto Corsa integration

The original Assetto Corsa connection is an outer adapter, not part of
environment or learning logic. The implemented live backend composes direct
read-only shared memory, local tyre/reset and collision apps, an X360 analog
report writer, installed AI-spline geometry, and bounded fresh-frame
synchronisation. Game
launch is an optional separate adapter; simulator I/O and learning remain
independent from process and `race.ini` management.

The byte layouts follow the
[Assetto Corsa shared-memory reference](https://labs.assettocorsa.jp/documents/reference/shared_memory):
`acpmf_physics`, `acpmf_graphics`, and `acpmf_static`, with four-byte packing
and Windows UTF-16 strings.

## Optional automatic session launch

Pass `--launch-game` to `rl-train --backend assetto-corsa` to launch the exact
configured car, track layout and Hotlap session. The launcher validates local
content, including the native `models_<layout>.ini` track manifest, moves the
current per-user `race.ini` to an `.acrl-backup`, writes a temporary
deterministic session, provides Assetto Corsa's Steam App ID `244210` while
starting `acs.exe`, and waits for matching live shared-memory identity before
the controller can be opened. A matching existing session is reused only while
its physics or graphics packet IDs continue to advance. If matching shared
memory is frozen, `--launch-game` locates processes by the exact configured
`acs.exe` path. It gracefully closes and replaces the session only when there
is exactly one such process. If that process has no closeable window or ignores
`WM_CLOSE` for 15 seconds, the launcher terminates that same verified process
and waits for its exit before restarting. It never terminates a healthy process
or guesses between multiple candidates. A newly launched session must likewise
publish advancing packets before it is accepted. Checking the model manifest
prevents Steam's DLC preview-only `ui` directories from being mistaken for
playable tracks. It
restores the original `race.ini` and removes a Steam App ID file that it created
after training. An existing valid Steam App ID file is preserved. A recovery
backup left by a hard crash is never overwritten; restore it before retrying.

When Steam is not ready yet, the directly started bootstrap `acs.exe` can exit
before the game process appears. This is treated as a Steam handoff rather than
an immediate failure. The launcher keeps waiting for up to the configured
`launch_timeout_s` (120 seconds in the live configuration), adopts exactly one
delayed process from the same full `acs.exe` path, and retains ownership so that
the Steam-started game is still closed cleanly after training. Multiple delayed
processes remain a safety error.

The launcher also requires the selected layout's `ai/fast_lane.ai`. This file,
not a TOML length or downloaded CSV, supplies the reference centreline, lap
length, and left/right boundary distances.

This option does not install CSP, cars, tracks, the Python bridge or controller
drivers. Those remain explicit one-time prerequisites.

## Run the telemetry diagnostic

The baseline and transport diagnostic both use the base-game combination Monza
and BMW Z4 GT3.

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
to `%TEMP%\assetto_corsa_rl_bridge.bin`. Its acknowledged `ac.ext_resetCar()`
command remains available only as the named legacy
`python_ext_reset_car_v1` recovery variant; it does not accept policy actions
or general vehicle controls.

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

Collision termination uses a separate CSP Lua app. Install it while the game
is closed:

```powershell
Copy-Item -Recurse -Force `
  "assetto_corsa_app\assetto_corsa_rl_csp" `
  "$game\apps\lua\assetto_corsa_rl_csp"
```

Enable `Assetto Corsa RL CSP Bridge` in Content Manager's Lua-app settings.
It registers `ac.onCarCollision(0, callback)` and publishes only the track ID,
a persistent collision sequence, and simulator frame. It neither controls the
car nor reads damage.

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

Reset uses a second versioned local-file exchange. The external controller
atomically writes a command with a sequence number; the app writes success,
`CSP_UNAVAILABLE`, or failure with the same sequence. The baseline uses the
explicit `csp_hotlap_spawn_v1` variant in the Lua app. It starts the offline
session if necessary, teleports car 0 to `ac.SpawnSet.HotlapStart`, repairs the
car, zeros velocity and re-enables user input. The backend starts no episode
until the matching acknowledgement and configured stable, non-pit frames have
both arrived. A configured neutral post-reset quiescence lets any reset-induced
collision callback arrive before the episode collision baseline is recorded.
The baseline then applies its configured `[+1, 0]` initial action (full
throttle, centred steering) and exposes the same value as the first
observation's previous action. This is reset synchronisation, not a grace
period after an agent action.

The separate named `csp_hotlap_reference_speed_v1` variant gives the episode a
flying start without changing `csp_hotlap_spawn_v1`. On the first reset of a
backend process it performs a stationary probe, projects the actual
`HOTLAP_START` world position onto the fixed reference, and reads the
interpolated `human_mean_speed_mps` value there. A V2 command sends that bounded
value to the CSP app. After teleport and repair, CSP aligns body velocity with
`car.look`, engages a forward gear and enables its automatic shifter. It holds
the chassis speed for 0.5 s and lets the tyre physics settle the four wheel
velocities instead of assigning one potentially incorrect wheel sign. Later
resets reuse the measured start coordinate.

The successor `csp_reference_pose_speed_v2` variant uses a V3 command carrying
the same human mean speed plus an explicit reference-line position and heading.
An optional signed `reset_reference_lateral_offset_m` shifts the pose along the
reference's left normal; negative values start to the right. CSP first performs
the native Hotlap teleport and repair, then places and settles the requested
pose before the speed-hold phase. Python requires the requested speed for the
configured number of low-slip frames and validates lateral and heading errors
before returning the first observation. Transient shared-memory and tyre-bridge
timeouts are retried only within the existing bounded reset timeout.

During either moving reset, the Lua app ignores reset-induced collision
callbacks while the state machine is active and for 1.0 s after its
acknowledgement. This covers delayed suspension and placement contacts only;
normal collision callbacks latch immediately afterward. The installed Lua app
and a rebuilt human reference artefact containing `human_mean_speed_mps` are
required for both moving variants.

Build the additional mean profile before enabling the variant, then select it
explicitly in a non-baseline configuration:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl build-reference `
  --config configs\bmw_z4_gt3_monza_baseline.toml
```

```toml
[assetto_corsa]
reset_backend = "csp_reference_pose_speed_v2"
reset_reference_speed_tolerance_mps = 2.0
reset_reference_lateral_offset_m = -0.5
reset_reference_heading_tolerance_rad = 0.05
reset_max_abs_slip_ratio = 0.2
reset_slip_stable_frames = 3
reset_reference_speed_max_attempts = 3
```

Each moving-reset attempt must independently receive a matching CSP
acknowledgement and pass the speed/slip checks. V3 pose resets must additionally
pass lateral and heading validation inside that same retry boundary. This
allows a short physics/shared-memory disturbance caused by moving or minimising
the AC window to trigger a new fully acknowledged reset instead of terminating
training after the first bad frame. The extended experiment permits three
attempts; persistent misalignment still fails hard. The default remains one
attempt for the unchanged baseline. It uses a 0.5 m right offset to remain close
to the demonstrated human start states. The
live-smoke configuration retains V2, while `bmw_z4_gt3_monza_baseline.toml`
deliberately retains the stationary thesis baseline.

The live transport also exposes `steering_output_gain` (default `1.0`) as an
explicit controller calibration. It scales only the X360 steering report and
clips it to `[-1, 1]`; it does not alter policy observations, BC targets or the
unchanged thesis baseline. The extended V6 experiment uses `1.2`, selected from
live reference-heading measurements after applying its temporary linear X360
profile.

Do not use `python_ext_reset_car_v1` as an episode reset: CSP documents it as
recovery back to the track. On the validated Monza setup repeated calls moved
the vehicle about 30 m backwards each time. A track spline's phase zero is not
necessarily its start line. Stability is
therefore checked from consecutive world positions (the configured position
tolerance is in metres), together with the speed threshold, instead of assuming
a particular normalized track phase. This permits recovery at a valid point on
the circuit while keeping the limitation explicit in the recorded metadata.

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
  prescribed reference line. The legacy timing differences consecutive
  returned values. The named `pre_action_progress_baseline_v1` live variant
  first samples a fresh projection immediately before applying the new action,
  so inference-time motion under the preceding control is not credited to the
  newly sampled action.

`close()`
: Release shared-memory handles, virtual-controller devices, sockets and helper
  processes idempotently.

Run `rl-ready --backend assetto-corsa` before attempting control. The command
also rejects a keyboard profile, missing virtual-controller support, a missing
CSP loader, an old telemetry-only bridge, or an unvalidated predictor
checkpoint.

## Implemented adapter layers

Keep these game-specific pieces separate:

- telemetry reader: shared-memory physics/graphics/static pages or a local
  bridge;
- control writer: a documented virtual controller or in-game bridge;
- session controller: restart, car/track selection and readiness detection;
- track provider: boundaries and world-position projection onto the closed
  `fast_lane.ai` reference arc coordinate;
- collision reader: consumes the latched CSP collision sequence without
  assuming that callback and agent frequencies match;
- synchroniser: maps higher-rate game frames to the configured control rate.

The live implementation runs on Windows, times out all blocking I/O, rejects
stale frame counters and mismatched car/track/AI-spline length, and neutralises
the controller after transport errors.

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
and observation diagnostics but never terminate an episode by themselves. The
offline ACTI/MoTeC source records slip ratio in percent,
so its separate data adapter must divide ratios by `100`; this scaling does not
apply to `CS.SlipRatio`.

## Safety and timing

Game pausing, menus and loading screens are not environment transitions. The
backend emits a step only after a full control period, a fresh live physics
frame, a fresh slip packet, and fresh collision state. For
`pre_action_progress_baseline_v1`, the full-period wait begins only after the
pre-action progress sample and control write. Diagnostics record the timing
variant, the reward start projection, observation-to-action delay and measured
action-hold duration. This corrects reward attribution but does not claim that
the real-time simulator can be paused during neural-network inference. For
`csp_hotlap_spawn_v1`, reset first waits for stable stationary frames and the
configured neutral collision-quiescence interval. For
the V2 reference-speed reset, CSP settles the teleported car before applying
the reference speed. The V3 pose-and-speed reset adds explicit reference pose
placement and Python-side pose validation. Both moving paths require
consecutive in-tolerance speed and low-slip frames, while the Lua bridge
suppresses only reset-generated collision callbacks for its bounded reset
window. All paths then record the current collision sequence. A callback
increment at any point afterward stays
latched until the next agent step and immediately produces crash termination.
This deliberately avoids damage thresholds, vehicle width/length, and wall
intersection calculations. The backend also distinguishes lap completion from
off-track, stalled and excessive vehicle body-sideslip termination. The
body-sideslip termination uses the calibrated shared-memory local
velocity axes and a configurable absolute threshold above 5 m/s. The baseline
uses 20 degrees and the extended experiment uses 30 degrees;
longitudinal tyre slip ratio is not a terminal condition. External timeouts
are adapter errors, not successful terminal states.
The live baseline has no ordinary time-limit truncation. Stall detection arms
after the car first reaches 5 km/h; five consecutive later low-speed frames
produce `STALLED`. A separate 25-step fallback terminates a car that never
drives off. The extended experiment records its separate 20 Hz choices: two
low-speed frames, a 20-step fallback and a 2,400-step safety truncation. The
next episode performs the deterministic reset.

The stationary deterministic reset explicitly engages first gear after placing
the car at `HOTLAP_START`; the reference-speed variant engages a forward gear
for its flying start and enables CSP's automatic shifter. For sessions started
by the launcher, the user's `assists.ini`
is backed up, `AUTO_SHIFTER=1` and `AUTO_CLUTCH=1` are installed before AC
starts, and the original bytes are restored after shutdown. An attached
session is never reconfigured behind AC's back; its effective gearbox state is
accepted only if the live drive-off probe observes a raw forward gear, speed,
and positive reference-line progress.

## Validation sequence

1. Run `telemetry-diagnose` and retain one raw diagnostic capture.
2. Run `live-validate --session launch`. It must start the configured session,
   reproduce the reset pose, report a forward gear and forward motion under low
   throttle, echo all analog controls, complete the connection soak and
   gracefully stop the process it owns.
   To verify the automatic first-to-second upshift explicitly, add
   `--drive-off-steps 300 --drive-off-throttle 0.5 --minimum-forward-gear 2`.
   The probe stops as soon as the requested gear, minimum speed, and positive
   progress have all been observed.
3. Start or deliberately leave the same session running and run
   `live-validate --session attach`. It must reconnect a new controller process
   and pass the same reset/input/transport invariants without owning the game.
4. Confirm that attach-only mode fails quickly when AC is closed and does not
   start a process.
5. Confirm local velocity and acceleration axes with straight-line,
   acceleration, braking, and left/right steering manoeuvres.
6. Run `slip-diagnose` and verify the named slip calibration plus transport
   timing against the paired shared-memory data.
7. Verify the configured `fast_lane.ai` digest, derived length, reference
   projection, and left/right widths.
8. Construct track boundaries, the twenty AI-spline points, and all 77
   unnormalised features.
9. Replay recorded telemetry through the complete adapter boundary.
10. Verify `gymnasium.utils.env_checker.check_env`.
11. Validate control signs and automatic upshifts at low speed in a safe
    session.
12. Trigger one low-speed wall contact and verify exactly one latched crash
   termination even when callback and agent frequencies differ.
13. Record one deterministic-policy lap and inspect every observation channel.
14. Only then collect demonstrations or start RL.
