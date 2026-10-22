# Assetto Corsa RL Drivermodel

A modular reinforcement learning platform for autonomous driving in Assetto
Corsa. A Gymnasium environment connects simulator telemetry and CSP controls
to PPO, with a shared feature extractor for policy, value and an auxiliary
trajectory predictor.

**[Read the documentation](https://wwlarsww.github.io/Assetto_Corsa_RL_Drivermodel/)**
for the project overview, architecture, user guide and generated Python API
reference. The current source tree replaces the earlier implementation;
the earlier Git history remains available.

## Start here

| Goal | Guide |
| --- | --- |
| Understand the platform | [Project overview](docs/index.md), [architecture](docs/architecture.md) |
| Follow the learning loop | [A training step](docs/training_step.md) |
| Install and validate | [Getting started](docs/getting_started.md) |
| Configure an experiment | [Configuration](docs/configuration.md) |
| Train, resume or evaluate | [Training and evaluation](docs/agent_training.md) |
| Inspect driving data | [Diagnostics and replay](docs/diagnostics.md) |
| Build or publish the docs | [Documentation guide](docs/documentation.md) |

## Development

Python 3.12 or later is required. From an activated project environment:

```powershell
python -m pip install -e ".[dev,docs]"
python -m pytest -q
python -m mkdocs serve
```

Open <http://127.0.0.1:8000> for the documentation. To generate a static copy
and validate its links:

```powershell
python -m mkdocs build --strict
```

Tests use synthetic data and simulator mocks. Live driving requires Windows,
Assetto Corsa and CSP. Synthetic tests establish integration correctness;
driving performance and predictor accuracy need evaluation on real driving data.

## Training

Complete [live validation](docs/racing_environment.md#validation), then copy
and adjust [configs/training.toml](configs/training.toml). Set the step budget
and future reward schedule explicitly before launching:

```powershell
python -m assetto_corsa_ai.main train --config configs/training.toml
```

Runs preserve configuration, software provenance, metrics and paired model /
normalization checkpoints. See the [training guide](docs/agent_training.md)
for artifact layout and restoration.
