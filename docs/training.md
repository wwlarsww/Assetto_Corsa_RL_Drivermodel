# Training workflow

The thesis baseline remains `ai_spline_absolute_v1`: one BMW Z4 GT3, one Monza
AI-spline reference and bounded absolute actions. Current PPO configurations use
the explicitly named `ai_spline_absolute_stabilized_v2` training variant. It
keeps the baseline observation/action geometry but adds the documented safety,
discounted prediction-reward and KL-control changes below.

## Safe preflight

Run the non-mutating check before every training campaign:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl rl-ready `
  --config configs\bmw_z4_gt3_monza_baseline.toml `
  --backend assetto-corsa
```

Exit code `0` means ready; exit code `4` means at least one required condition
failed. The report checks the fixed reference, BC tensors and checkpoint,
predictor validation ADE/FDE, game/CSP/reset/collision-bridge installation,
configured `fast_lane.ai`, analog input profile, controller package, rollout
horizon and live-backend availability. It
does not change game or Windows settings and does not create a virtual device.

The initial checkpoint is copied locally to
`data/processed/bmw_z4_gt3_monza/bc_checkpoint.pt`; generated data and
checkpoints remain ignored by Git.

## PPO smoke training

The complete learning path can be exercised without the game:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl rl-train `
  --config configs\mock.toml `
  --backend mock `
  --device cpu
```

Each synchronous iteration performs this fixed sequence:

1. collect a fragment tagged with the current policy version;
2. reject stale fragments at the collector boundary;
3. build actual future-path targets without crossing episodes;
4. decode the 10-output `bezier-5x2` predictor;
5. project every predicted horizon point onto the current fixed reference,
   discount later progress increments, and add the scaled prediction term;
6. compute GAE with separate termination and time-limit semantics;
7. optimise clipped PPO, the value head and the auxiliary predictor;
8. increment the policy version and atomically save a checkpoint.

During `rl-train`, the CLI prints live iteration summaries to the console at
INFO level, including total environment steps, average environment and shaped
rewards, policy/value losses, entropy, approximate KL divergence, clip fraction,
the adaptive learning rate and completed episode count. The append-only
`metrics.csv` additionally records pedal quantiles and throttle/brake/coast
fractions, direct gas-to-brake switches, speed range, progress without the
termination penalty, predictor share, per-iteration termination deltas, tyres
outside and the KL target band. Aim streams the same schema without becoming
the source of record.

Metrics and the full configuration are stored below `runs/ppo-<backend>/`.
Time-limit truncations bootstrap from their final observation but stop the GAE
trace; true terminations do not bootstrap. This remains supported for mock and
named finite-horizon experiments. The named termination variant
`any_tyre_armed_stall_v1` ends an episode as soon as one tyre is reported
outside. Stall detection is inactive at reset, arms after the car first reaches
5 km/h, and subsequently terminates after five consecutive low-speed frames
(0.2 s at 25 Hz). If the car never reaches 5 km/h, a separate 25-step startup
fallback terminates after 1 s. Crash, excessive vehicle body sideslip and lap
completion remain separate terminal conditions.

## Live-control prerequisites

Live control uses the implemented `AssettoCorsaLiveBackend` around an Xbox 360
virtual controller. Install its optional package only when you are ready to approve
the signed ViGEmBus Windows driver installer:

```powershell
$env:VGAMEPAD_SKIP_VIGEMBUS_INSTALL = "true"
.\.venv\Scripts\python.exe -m pip install `
  -c requirements\tested.txt `
  -e ".[live]"
```

The Python package is pinned to the tested official source revision because
the older PyPI source distribution starts its MSI during package-metadata
generation. Install the bundled x64 driver once from an interactive
PowerShell:

```powershell
$msi = Resolve-Path `
  ".venv\Lib\site-packages\vgamepad\win\vigem\install\x64\ViGEmBusSetup_x64.msi"
Start-Process msiexec.exe -Verb RunAs -Wait `
  -ArgumentList "/i", "`"$($msi.Path)`""
```

The optional dependency and driver behaviour follow the
[official vgamepad documentation](https://github.com/yannbouteiller/vgamepad).

With `--launch-game`, the CLI creates and neutralises this virtual controller
before starting Assetto Corsa, then hands the same device to the live backend.
This ensures the simulator can bind the configured X360 profile during device
initialisation without creating a second virtual controller. At training exit,
the CLI gracefully stops only a game process it launched itself and restores
the original user configuration; a matching session it merely attached to is
not stopped.

Then configure Assetto Corsa for the generated X360 controller and verify
steering, throttle and brake signs at low speed. Keyboard input is not accepted
for RL because it cannot represent the continuous commands used by the policy.

There is no supported external shared-memory write channel for steering,
throttle and brake in the original game. Shared memory is therefore used
directly for telemetry, while the continuous controls enter through the same
analog controller interface the game already supports. The in-game app does
not stream policy actions. The Lua bridge accepts the bounded, versioned
`csp_hotlap_spawn_v1` command and returns the car to the deterministic
`HOTLAP_START` spawn. The Python `ac.ext_resetCar()` recovery path is retained
only as the explicitly named legacy variant `python_ext_reset_car_v1`.

## Live PPO start sequence

1. Close Assetto Corsa, install the repository's current Python
   `assetto_corsa_rl_bridge` and Lua `assetto_corsa_rl_csp` apps, enable both,
   and ensure Custom Shaders Patch is installed.
2. In one terminal, expose a neutral virtual device with
   `python -m assetto_corsa_rl x360-setup --duration 120`. During that window,
   select the Xbox 360 controller in Assetto Corsa and save the controls. The
   resulting `controls.ini` must report `INPUT_METHOD=X360`.
3. Start the exact configured session manually: BMW Z4 GT3 on Monza,
   preferably Hotlap, and remain on track.
4. Run `telemetry-diagnose` and `slip-diagnose --with-shared-memory` once after
   changing a game, car, track, controller, CSP, or app version.
5. Run `live-validate --session launch`, then repeat with `--session attach`
   against the deliberately running matching session. Both reports must pass,
   including the raw forward-gear, acceleration, and positive-progress checks.
   Launch mode temporarily enables AC's automatic shifter and clutch and
   restores the previous `assists.ini` exactly when it stops the owned game.
6. Run `rl-ready --backend assetto-corsa`. Fix every `error`; warnings are
   recorded experiment assumptions.
7. Verify steering, throttle and brake signs plus an automatic upshift at low
   speed. The baseline uses policy-positive-left and
   `steering_output_sign = -1.0` for X360.
8. Start PPO from a second PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl rl-train `
  --config configs\bmw_z4_gt3_monza_baseline.toml `
  --backend assetto-corsa `
  --device cpu
```

When the `monitoring` extra is installed, PPO starts an independent CSV follower
and `aim up` web server automatically. The server uses the run directory as its
repository and remains available at `http://127.0.0.1:43800` after training
finishes. Monitoring never blocks or controls the learner. Disable it explicitly
with `--no-monitor-ui`:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl rl-train `
   --config configs\bmw_z4_gt3_monza_baseline.toml `
   --backend assetto-corsa `
   --device cpu `
   --no-monitor-ui
```

## Aim (Aimstack) integration

Aim 3.29.1 is the supported monitoring backend and no separate plotting package
is a project dependency. Upstream Aim wheels currently support CPython through 3.12, so use
Python 3.11 or 3.12 for an environment that needs the automatic web UI:

```powershell
.\.venv\Scripts\python.exe -m pip install `
  -c requirements\tested.txt `
  -e ".[dev,monitoring]"
```

`rl-train` then creates the Aim run and starts the UI automatically. Reopen or
follow the newest run manually with:

```powershell
.\.venv\Scripts\ac-rl-driver.exe training-monitor
```

Select a particular run with `--run`, or import an already completed CSV once:

```powershell
.\.venv\Scripts\ac-rl-driver.exe aim-migrate `
  --run runs\smoke\ppo-assetto-corsa\<run-id>
```

Python 3.13/3.14 remains supported for the simulator-independent code and test
suite, but Aim itself is unavailable there until upstream publishes compatible
wheels. Use `--no-monitor-ui` in such an environment. A supported Python 3.12
environment can still inspect the recorded CSV later.

Or select any running or completed run explicitly:

```powershell
.\.venv\Scripts\ac-rl-driver.exe training-monitor `
  --run runs\smoke\ppo-assetto-corsa\<run-id>
```

Without `--launch-game`, the command connects to an already running matching
session. To prepare `race.ini`, launch `acs.exe`, and wait for the configured
live car/track identity automatically, use:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl rl-train `
  --config configs\bmw_z4_gt3_monza_live_smoke.toml `
  --backend assetto-corsa `
  --launch-game
```

For an intermediate live run, `bmw_z4_gt3_monza_live_10k.toml` records 10,240
environment steps in ten 1,024-step rollouts with four update epochs. It is a
named robustness run between the 512-step smoke test and the one-million-step
thesis baseline; it does not redefine either baseline.

```powershell
.\.venv\Scripts\ac-rl-driver.exe rl-train `
  --config configs\bmw_z4_gt3_monza_live_10k.toml `
  --backend assetto-corsa `
  --device cpu `
  --launch-game
```

Neither mode changes controller settings or installs missing game content. At
every environment reset the backend neutralises the controller, requests
one deterministic CSP Hotlap reset, waits for the matching acknowledgement and three stable live
frames, then keeps controls neutral for the configured
`reset_collision_quiescence_s` and records the collision baseline. This absorbs
any reset-induced collision callback; after the
first agent action there is no crash grace period. Every action remains active for one configured
control period; only a newer live physics frame and a fresh tyre packet may
complete the step. The collision callback increments a persistent sequence,
so an event between two lower-frequency agent steps remains observable. Any
post-quiescence collision terminates the episode immediately; damage and inferred
vehicle dimensions are not used. Transport or reset timeouts neutralise the
controller and abort instead of fabricating an observation.
