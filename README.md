# assetto-corsa-rl-driver

Research-oriented Python scaffold for an autonomous race driver in the original
Assetto Corsa. The project exposes the simulator as a Gymnasium environment and
keeps telemetry, feature construction, reward shaping, policy networks and
training orchestration behind explicit interfaces.

The repository is intentionally an engineering foundation, not a claim of a
finished racing agent. It includes a deterministic mock backend so that every
central component can be developed and tested without launching Assetto Corsa.

## Method at a glance

- four observation groups: vehicle state, track-boundary preview, fixed-line
  coordinates and the previous integrated action;
- a two-dimensional continuous delta action for combined pedal and steering;
- progress and safety rewards inside the environment, followed by predicted
  progress after a rollout; no imitation term is active in the initial variant;
- a two-layer shared feature extractor with Gaussian policy, value and
  auxiliary trajectory-prediction heads;
- a probabilistic reference-line interface with a compact ProMP
  implementation;
- a named fixed-reference F317/Barcelona baseline that performs no ProMP
  sampling;
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

With Assetto Corsa running in an on-track session, validate its read-only
telemetry transport with:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl telemetry-diagnose `
  --duration 15 --poll-hz 100
```

See [docs/assetto-corsa-integration.md](docs/assetto-corsa-integration.md) for
the driving manoeuvres, outputs, exit codes, and remaining signal gaps.

The four exact tyre slip-ratio and slip-angle channels use a telemetry-only
in-game Python app. After installing and enabling
`assetto_corsa_app/assetto_corsa_rl_bridge`, validate it with:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl slip-diagnose --duration 15
```

`requirements/tested.txt` constrains the direct dependencies to the versions
used by the latest local verification. GPU-specific PyTorch installations may
use a platform-appropriate wheel while keeping the same PyTorch version.

## Initial real-data baseline

The first public-data experiment uses the Dallara F317 on Barcelona GP and one
fixed reference for every episode. Download and prepare it with:

```bash
python -m pip install -c requirements/tested.txt -e ".[data,dev]"
python -m assetto_corsa_rl download-data
python -m assetto_corsa_rl build-reference --config configs/f317_barcelona_fixed.toml
python -m assetto_corsa_rl build-bc-dataset --config configs/f317_barcelona_fixed.toml
python -m assetto_corsa_rl bc-train --config configs/f317_barcelona_fixed.toml
```

The selective download is about 550 MB, contains human raw MoTeC recordings,
and excludes the much larger SAC/replay-buffer data and preprocessed pickle
telemetry. See [docs/data-pipeline.md](docs/data-pipeline.md) for provenance,
filtering, fixed-reference construction, the exact BC export, and split rules.
The guarded PPO commands and current live-control prerequisites are documented
in [docs/training.md](docs/training.md).

## What is implemented

- Gymnasium `reset`/`step` environment with correct terminated/truncated
  semantics
- typed simulator backend protocol and deterministic mock vehicle backend
- fixed-layout observation builder and static Z-score normalisation
- cumulative, clipped delta-action integration
- environment reward terms and rollout-time auxiliary reward post-processing
- thesis-shaped PyTorch policy/value/predictor network
- behavioural-cloning and auxiliary prediction losses
- episode-safe, ego-frame future trajectory target construction
- read-only AC 1.14 shared-memory decoder and live telemetry diagnostics
- ProMP fit, mean-reference and sampling example
- pinned F317/Barcelona human-data downloader and SHA-256 manifest
- safe MoTeC reader and fixed human-median reference builder
- session-disjoint 77-feature BC export with frozen training-only Z-scores
- joint policy/predictor BC warm start with held-out metrics
- versioned local-file bridge for exact tyre-level slip signals
- named, deterministic live slip calibration compatible with BC telemetry
- policy-versioned trajectory fragments and on-policy collector
- synchronous clipped PPO, termination-aware GAE and atomic checkpoints
- post-rollout predicted-progress shaping before advantage estimation
- non-mutating mock/live training preflight reports
- optional, testable X360 analog control writer
- TOML configuration, deterministic seeding and traceable run artefacts
- unit tests, lint/type configuration and cross-platform CI

## What is deliberately not implemented

- the complete Assetto Corsa Gymnasium backend and control adapter;
- process-based asynchronous actors or GPU inference serving.

Their interfaces and integration boundaries are documented in
[docs/architecture.md](docs/architecture.md) and
[docs/assetto-corsa-integration.md](docs/assetto-corsa-integration.md).

## Repository map

```text
src/assetto_corsa_rl/
├── control/       # delta action integration
├── envs/          # Gymnasium environment and simulator backends
├── features/      # observation layout and static normalisation
├── models/        # policy, value and auxiliary heads
├── references/    # fixed and ProMP reference trajectories
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
