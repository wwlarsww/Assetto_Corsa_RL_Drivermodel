# Assetto Corsa RL

## Project Goal

Build a modular reinforcement learning platform for autonomous driving in
Assetto Corsa. The project grows incrementally, with clear responsibilities,
testable components and reproducible training runs.

## System Overview

```text
Assetto Corsa
    ↓ Telemetry
Python
    ↓ Vehicle State
RL Environment
    ↓ Observation
Agent
    ↓ Vehicle Actions
RL Environment
    ↓ Steuerbefehl
Assetto Corsa
```

## Current Scope

The system reads telemetry, reconstructs track geometry and exposes a Gymnasium
environment with CSP controls and session management. PPO uses a shared feature
extractor for policy, value and an auxiliary trajectory predictor. Optional
future rewards evaluate detached predictions; each run records configuration,
provenance, metrics and paired model/normalization checkpoints.

| Topic | Documentation | Implementation |
| --- | --- | --- |
| System responsibilities | [Architecture](docs/architecture.md) | [Composition root](src/assetto_corsa_ai/main.py) |
| Simulator input | [Read layer](docs/input.md) | [AC adapters](src/assetto_corsa_ai/assetto_corsa/) |
| Controls and sessions | [Control](docs/control.md), [sessions](docs/session_manager.md) | [Controller](src/assetto_corsa_ai/assetto_corsa/simulator_controller.py), [session manager](src/assetto_corsa_ai/assetto_corsa/session_manager.py) |
| Episodes and observations | [Environment](docs/racing_environment.md), [features](docs/observation.md) | [Environment](src/assetto_corsa_ai/RacingEnvironment.py), [builder](src/assetto_corsa_ai/observation_builder.py) |
| Reward and termination | [Reward](docs/reward.md), [termination](docs/termination.md) | [Reward calculator](src/assetto_corsa_ai/reward.py), [termination checker](src/assetto_corsa_ai/termination.py) |
| Learning and run records | [Training](docs/agent_training.md), [predictor](docs/predictor.md) | [Agent](src/assetto_corsa_ai/agent.py), [run management](src/assetto_corsa_ai/training_run.py) |

## Development

From the repository root, using the project virtual environment:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

Automated tests use synthetic data and mocked simulator adapters, including
short PPO updates and checkpoint restoration. They do not require a running
Assetto Corsa process. Live simulator integration requires Windows and CSP.

## Validation and Training

| Check | Tool |
| --- | --- |
| Following plot of spline, road edges, vehicle and projection | [validate_track_alignment.py](tools/validation/validate_track_alignment.py) |
| Runtime controls and session launch | [validate_simulator_control.py](tools/validation/validate_simulator_control.py) |
| Gymnasium/SB3 compatibility and live reset/step | [validate_racing_environment.py](tools/validation/validate_racing_environment.py) |
| Predictor accuracy and future reward contributions on test/selected data | [validate_predictor.py](tools/validation/validate_predictor.py) |
| Raw damage for collision calibration | [validate_damage.py](tools/validation/validate_damage.py) |

Before training, follow the [live environment validation](docs/racing_environment.md#validation).
Then copy and adjust [the training configuration](configs/training.toml):

```powershell
python -m assetto_corsa_ai.main train --config configs/training.toml
```

Set the step budget and future reward warmup explicitly for each run. Synthetic
tests establish integration correctness; predictor accuracy still needs real
driving data. [Training documentation](docs/agent_training.md) explains runs,
resume and evaluation.
