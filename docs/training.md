# Training workflow

The thesis baseline remains `ai_spline_absolute_v1`: one BMW Z4 GT3, one Monza
AI-spline reference and bounded absolute actions. The extended live experiment
uses the explicitly named `ai_spline_absolute_policy_protected_v3` PPO variant.
It keeps the baseline observation/action geometry and the v2 safety, reward and
KL-control changes, while preventing value/predictor gradients from changing
the shared BC policy representation. Policy, value and predictor parameter
groups are gradient-clipped separately so the initially large value error does
not suppress or overwrite steering updates.

## Safe preflight

Run the non-mutating check before every training campaign:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl rl-ready `
  --config configs\bmw_z4_gt3_monza_baseline.toml `
  --backend assetto-corsa
```

Exit code `0` means ready; exit code `4` means at least one required condition
failed. The report checks the fixed reference, BC tensors and checkpoint,
requires the BC dataset and checkpoint metadata to carry the exact active
reference SHA-256,
predictor validation ADE/FDE, game/CSP/reset/collision-bridge installation,
configured `fast_lane.ai`, analog input profile, controller package, rollout
horizon and live-backend availability. It
does not change game or Windows settings and does not create a virtual device.

Changing or regenerating the fixed reference invalidates both the BC dataset
and its checkpoint, even when the source `fast_lane.ai` hash is unchanged.
Re-run `build-bc-dataset` and then `bc-train`; the preflight blocks PPO until
both artefacts match the active reference. The
`assetto_corsa_real_extended_smoke.toml` flying-start experiment uses the
named predictor normalization
`train_split_per_horizon_axis_std_floor_5m_v2`. It fits one physical scale for
each horizon and ego axis from valid targets in the unaugmented training split,
then floors each scale at 5 m. This keeps low-variance short-horizon lateral
targets from dominating the shared BC backbone; validation and test errors are
still reported in metres. Dataset, checkpoint and PPO all carry and reuse the
same fitted scales. Its clean BC action loss remains
`steering_weighted_action_mse_v1` with steering weight 20. The named
`permuted_causal_feedback_steering_v1` robustness branch randomly re-pairs
acceleration, slip and previous-action features inside each training batch
while retaining speed, boundaries and future reference points. It supervises
steering on that second branch as well. This breaks the teacher-forcing
shortcut in which the policy waits for lateral acceleration or a previous
steering command before initiating a turn. It starts PPO exploration at
`initial_log_std_after_bc=-3.1` (`std≈0.045`) and uses a small `0.0001` entropy
coefficient. These are recorded properties of the named stabilised experiment,
not baseline defaults.

The initial checkpoint is copied locally to
`data/processed/bmw_z4_gt3_monza/bc_checkpoint.pt`; generated data and
checkpoints remain ignored by Git.

Before allowing PPO to change the BC policy, run one deterministic closed-loop
BC evaluation and inspect its recorded steering and trajectory:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl bc-live-evaluate `
  --config .\configs\assetto_corsa_real_extended_smoke.toml `
  --device cpu `
  --episodes 1 `
  --launch-game
```

This command loads `ppo.initial_checkpoint_path`, validates its reference and
steering provenance, forces deterministic mean actions, performs no optimiser
step, and writes the usual evaluation summary, episode table and trajectory
NPZ below a timestamped `runs/bc-live-evaluation-*` directory. `episodes.csv`
stores the exact final termination reason and `summary.json` aggregates reason
counts, so reset faults, crashes and genuine off-track exits remain distinct.

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
6. for configurations selecting `successor_progress_alignment_v1`, attach
   reached-state lateral/heading alignment to the action that caused it;
7. compute GAE with separate termination and time-limit semantics;
8. optimise clipped PPO, the value head and the auxiliary predictor;
9. increment the policy version and atomically save a checkpoint.

The extended live configuration currently sets
`use_predicted_progress_reward=false` to isolate the successor-state reference
reward. Predictor targets, quality metrics and auxiliary loss remain available;
only their predicted-progress reward contribution is zero in that experiment.

During `rl-train`, the CLI prints live iteration summaries to the console at
INFO level, including total environment steps, average environment and shaped
rewards, policy/value losses, entropy, approximate KL divergence, clip fraction,
the adaptive learning rate and completed episode count. The append-only
`metrics.csv` additionally records pedal quantiles and throttle/brake/coast
fractions, direct gas-to-brake switches, speed range, progress without the
termination penalty, predictor share, per-iteration termination deltas, tyres
outside and the KL target band. MLflow receives the configured training metrics
without replacing the local CSV files as the source of record.

Metrics and the full configuration are stored below `runs/ppo-<backend>/`.
Time-limit truncations bootstrap from their final observation but stop the GAE
trace; true terminations do not bootstrap. This remains supported for mock and
named finite-horizon experiments. The named termination variant
`any_tyre_armed_stall_v1` ends an episode as soon as one tyre is reported
outside. Stall detection is inactive at reset and arms after the car first
reaches 5 km/h. The thesis baseline then requires five low-speed frames
(0.2 s at 25 Hz), with a separate 25-step/1 s never-drive-off fallback. The
extended 20 Hz experiment explicitly uses two low-speed frames (0.1 s) and a
20-step/1 s fallback. Crash, excessive vehicle body sideslip and lap completion
remain separate terminal conditions.

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
`csp_hotlap_spawn_v1` stationary command and the separately named
`csp_hotlap_reference_speed_v1` flying-start command. Its V3 successor,
`csp_reference_pose_speed_v2`, additionally carries the reference-line pose and
a configurable signed lateral offset. All moving variants use the arithmetic
mean speed from valid human recordings at the measured spawn coordinate. The
Python `ac.ext_resetCar()` recovery path is retained
only as the explicitly named legacy variant `python_ext_reset_car_v1`.

The extended live configuration also selects
`linear_bc_calibrated_x360_v2`. For sessions owned by `--launch-game`, the
launcher backs up `controls.ini`, selects the ViGEm controller and temporarily
sets X360 steering gamma to 1, speed sensitivity/deadzone to 0 and steering
speed to 1. A `0.7` temporal steering filter suppresses short BC transients
without changing a sustained command. The BC steer map is an absolute linear
calibration, so the usual human gamepad steering gamma and high-speed
attenuation would otherwise weaken the commands a second time. Pedal settings
remain unchanged. The original file is restored byte-for-byte when the session
exits. Reusing a manually started session does not reload this temporary
profile, so this experiment should be run with `--launch-game`.

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
   `steering_output_sign = -1.0` for X360. Non-baseline live experiments can
   additionally record a bounded `steering_output_gain`; its default is `1.0`.
8. Start PPO from a second PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl rl-train `
  --config configs\bmw_z4_gt3_monza_baseline.toml `
  --backend assetto-corsa `
  --device cpu
```

## MLflow experiment tracking

MLflow is the only supported optional monitoring backend. Tracking is decoupled
from the learner: by default `rl-train` writes directly to the local SQLite
tracking store and does not require a running MLflow server or browser UI.
Install the optional dependency once:

```powershell
.\.venv\Scripts\python.exe -m pip install `
  -c requirements\tested.txt `
  -e ".[dev,monitoring]"
```

Enable local tracking in the experiment TOML:

```toml
[monitoring]
enabled = true
backend = "mlflow"
tracking_uri = "sqlite:///mlflow.db"
experiment_name = "Assetto Corsa PPO"
run_name = "bmw-z4-gt3-monza"
log_interval_updates = 1
fail_on_error = false
```

`create_tracker` prepares the configured tracking destination before a live
controller or Assetto Corsa session is created. With `fail_on_error = false`, an
unavailable tracking destination disables monitoring for that training process
and the learner continues with its local CSV/checkpoint artefacts. Remote HTTP(S)
tracking URIs receive a short reachability preflight so a dead dashboard server
cannot leave a live vehicle waiting inside MLflow retry loops.

The UI is optional and can be started or stopped at any time in a separate
terminal while training continues to write to the same SQLite database:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui `
  --backend-store-uri sqlite:///mlflow.db `
  --host 127.0.0.1 `
  --port 5000
```

Open `http://127.0.0.1:5000` only when visualization is needed. With
`enabled = false` or `backend = "none"`, the no-operation tracker is used and
MLflow is not imported. `fail_on_error = true` remains available for deliberately
strict/offline workflows, but live training configurations should keep it false.

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
every environment reset the backend neutralises the controller and requests a
deterministic CSP Hotlap reset. The stationary baseline waits for stable live
frames and the configured `reset_collision_quiescence_s`. The named
reference-speed variants first discover the spawn coordinate once, then wait
for the requested human mean speed within
`reset_reference_speed_tolerance_mps` and consecutive low-slip frames. V3 also
validates the requested reference-line lateral offset and heading. The extended
experiment uses a neutral initial combined-pedal command because CSP already
owns the physical flying start; the stationary baseline remains full throttle.
The Lua bridge suppresses only reset-induced collision callbacks while a moving
reset is active and for one second after acknowledgement. The backend records
the collision baseline before the episode begins. After that bounded reset
window there is no crash grace period. Every action remains active for one configured
control period; only a newer live physics frame and a fresh tyre packet may
complete the step. The active flying-start configuration uses the named
`pre_action_progress_baseline_v1` timing variant: it samples current progress
immediately before the new controller command and uses that value as the
reward baseline. Motion during CPU inference under the previous command is
therefore not attributed to the new action. The collision callback increments a persistent sequence,
so an event between two lower-frequency agent steps remains observable. Any
later collision terminates the episode immediately; damage and inferred
vehicle dimensions are not used. Transport or reset timeouts neutralise the
controller and abort instead of fabricating an observation.

The extended configuration additionally selects
`unrecoverable_progress_stall_v2`. A rolling window begins at the reset frame;
after 1.0 s, less than 2.0 m of wrap-safe net reference progress terminates the
episode as `STALLED`. The terminal transition receives the normal safety
penalty, PPO increments the trajectory ID before reset, GAE stops its trace,
and observed predictor targets cannot cross into the next episode.
