# assetto-corsa-rl-driver

Research-oriented Python scaffold for an autonomous race driver in the original
Assetto Corsa. The project exposes the simulator as a Gymnasium environment and
keeps telemetry, feature construction, reward shaping, policy networks and
training orchestration behind explicit interfaces.

The repository is intentionally an engineering foundation, not a claim of a
finished racing agent. It includes a deterministic mock backend so that every
central component can be developed and tested without launching Assetto Corsa.

GitHub Actions is deliberately limited to code quality, deterministic tests,
coverage, and documentation publishing. Behavioural-cloning and PPO research
training always runs locally; see
[docs/continuous-integration.md](docs/continuous-integration.md).

## Method at a glance

- four observation groups: vehicle state, track-boundary preview, reference-line
  coordinates and the previous absolute action;
- a two-dimensional absolute action in `[-1, 1]` for combined pedal and
  steering, without a per-step delta limit;
- progress and safety rewards inside the environment, followed by predicted
  progress after a rollout; no imitation term is active in the initial variant;
- a two-layer shared feature extractor with Gaussian policy, value and
  auxiliary trajectory-prediction heads;
- a frozen `ai_spline_absolute_v1` thesis baseline plus the explicitly named
  `ai_spline_absolute_stabilized_v2` PPO variant whose line, lap length and
  boundaries come from the installed track's `ai/fast_lane.ai`;
- on-policy policy-version checks for PPO rollout collection;
- synchronous PPO/GAE training with auxiliary predictor updates and explicit
  live-training readiness gates.

See [docs/methodology.md](docs/methodology.md) for the exact mapping and
explicit assumptions. The compact 77-value data dictionary is in
[docs/feature-reference.md](docs/feature-reference.md).

## Quick start

Use Python 3.11 or newer in an isolated environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -c requirements/tested.txt -e ".[dev]"
python -m assetto_corsa_rl mock-run --config configs/mock.toml
pytest
```

The mock command writes standard JSON metadata and CSV step metrics below
`runs/`. No Assetto Corsa installation is required.

Exercise the complete post-rollout reward, GAE and PPO path with:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl rl-train `
  --config configs\mock.toml --backend mock
```

For a guarded real-time smoke test, `rl-train` with
`--backend assetto-corsa --launch-game` can prepare and launch the configured Hotlap session after
all live prerequisites pass. See [docs/training.md](docs/training.md) before
enabling control. When training finishes or fails, it gracefully closes only
the AC process it launched itself; an attached pre-existing session is left running.
When the optional `[monitoring]` configuration enables MLflow, `rl-train` logs
parameters, metrics and artefacts to the configured tracking server. The MLflow
server is started separately from the learner; see [docs/training.md](docs/training.md).

With Assetto Corsa running in an on-track session, validate its direct,
read-only telemetry transport with:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl telemetry-diagnose `
  --duration 15 --poll-hz 100
```

Before PPO, validate the write-capable boundary independently of any policy:

```powershell
.\.venv\Scripts\ac-rl-driver.exe live-validate `
  --config configs\bmw_z4_gt3_monza_live_smoke.toml `
  --session launch
```

The command checks owned process start/stop, repeatable `HOTLAP_START` resets,
first-gear engagement and forward motion, raw AC throttle/brake/steering echo,
unique physics/slip frames and connection continuity. Launched sessions
temporarily enable AC's automatic shifter and clutch and restore the original
`assists.ini` afterward. Use `--session attach` to test a matching session that
is already running. A failed report blocks the simulator acceptance step; it
is not a PPO performance result.

An explicit automatic-upshift probe can require second gear with
`--drive-off-steps 300 --drive-off-throttle 0.5 --minimum-forward-gear 2`.
It stops accelerating as soon as AC reports the requested gear together with
forward speed and progress.

See [docs/assetto-corsa-integration.md](docs/assetto-corsa-integration.md) for
the driving manoeuvres, outputs, exit codes, and remaining signal gaps.

The four exact tyre slip-ratio and slip-angle channels and acknowledged episode
reset use a small in-game Python app. After installing and enabling
`assetto_corsa_app/assetto_corsa_rl_bridge`, validate it with:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl slip-diagnose --duration 15
```

`requirements/tested.txt` constrains the direct dependencies to the versions
used by the latest local verification. GPU-specific PyTorch installations may
use a platform-appropriate wheel while keeping the same PyTorch version.

## Real-data baseline

The public-data experiment uses the BMW Z4 GT3 on Monza and one fixed reference
for every episode. Both simulator assets ship with the base game. The local
game installation is required for reference and BC construction because
`fast_lane.ai` is the geometry source of truth. The executable contract and
module-by-module program flow are described in
[docs/baseline.md](docs/baseline.md). Download and prepare it with:

```bash
python -m pip install -c requirements/tested.txt -e ".[data,dev]"
python -m assetto_corsa_rl download-data
python -m assetto_corsa_rl build-reference --config configs/bmw_z4_gt3_monza_baseline.toml
python -m assetto_corsa_rl build-bc-dataset --config configs/bmw_z4_gt3_monza_baseline.toml
python -m assetto_corsa_rl bc-train --config configs/bmw_z4_gt3_monza_baseline.toml
```

The selective download contains human raw MoTeC recordings,
and excludes the much larger SAC/replay-buffer data and preprocessed pickle
telemetry. See [docs/data-pipeline.md](docs/data-pipeline.md) for provenance,
filtering, baseline construction, the exact BC export, and split rules.
The guarded PPO commands and current live-control prerequisites are documented
in [docs/training.md](docs/training.md).

## What is implemented

- Gymnasium `reset`/`step` environment with correct terminated/truncated
  semantics
- typed simulator backend protocol and deterministic mock vehicle backend
- immutable-layout observation builder and static Z-score normalisation
- bounded absolute pedal/steering actions and a tanh-squashed Gaussian policy
- environment reward terms and rollout-time auxiliary reward post-processing
- thesis-shaped PyTorch policy/value/predictor network
- behavioural-cloning and auxiliary prediction losses
- episode-safe, ego-frame future trajectory target construction
- read-only AC 1.14 shared-memory decoder and live telemetry diagnostics
- pinned BMW Z4 GT3/Monza human-data downloader and SHA-256 manifest
- strict Assetto Corsa v7 `fast_lane.ai` reader plus a robust human-median
  speed-profile builder
- session-disjoint 77-feature BC export with frozen training-only Z-scores
- joint policy/predictor BC warm start with held-out metrics
- versioned local-file bridge for exact tyre-level slip signals
- named, deterministic live slip calibration compatible with BC telemetry
- named 20-degree vehicle body-sideslip safety termination from calibrated
  local velocity, independent of launch wheelspin
- frame-synchronised live backend with session identity and stale-frame checks
- AI-spline-derived offline/live reference, length and boundary geometry
- latched CSP collision events with immediate crash termination
- acknowledged CSP episode reset with bounded local command protocol
- policy-versioned trajectory fragments and on-policy collector
- synchronous clipped PPO, termination-aware GAE and atomic checkpoints
- post-rollout predicted-progress shaping before advantage estimation
- non-mutating mock/live training preflight reports
- optional, testable X360 analog control writer with explicit steering sign
- TOML configuration, deterministic seeding and traceable run artefacts
- unit tests, lint/type configuration and cross-platform CI

## What is deliberately not implemented

- interactive menu navigation beyond deterministic offline Hotlap launch;
- process-based asynchronous actors or GPU inference serving.

Their interfaces and integration boundaries are documented in
[docs/architecture.md](docs/architecture.md) and
[docs/assetto-corsa-integration.md](docs/assetto-corsa-integration.md).

## Repository map

```text
src/assetto_corsa_rl/
├── control/       # bounded absolute action contract
├── envs/          # Gymnasium environment and simulator backends
├── features/      # observation layout and static normalisation
├── models/        # policy, value and auxiliary heads
├── references/    # fixed AI-spline reference contract
├── rewards/       # step rewards and rollout-time reward shaping
└── training/      # rollout contracts, losses and checkpoints
```

## Project status

Early research scaffold. APIs can still change before version `1.0`.
Contributions and reproducibility reports are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. Assetto Corsa is a product of Kunos Simulazioni. This independent research
project is not affiliated with or endorsed by Kunos Simulazioni. Dataset and
format-reference attribution is collected in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
