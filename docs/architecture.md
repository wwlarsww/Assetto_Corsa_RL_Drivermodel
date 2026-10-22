# Software Architecture

## Overview

The project separates Assetto Corsa integration, simulator-independent domain
calculations and learning. [The composition root](../src/assetto_corsa_ai/main.py)
builds the environment and agent from [the run configuration](../configs/training.toml).
The [project overview](index.md#system-overview) shows the simulator/agent loop.
For a concrete walkthrough, read [a training step](training_step.md).
Explore [automatically generated class diagrams](diagrams/index.md) to inspect
direct dependencies, dependents and links to the API reference.

## Repository map

| Location | Contents |
| --- | --- |
| `src/assetto_corsa_ai/` | Simulator-independent data, geometry, environment, learning and run management |
| `src/assetto_corsa_ai/assetto_corsa/` | Simulator-specific readers, controls and session lifecycle |
| `configs/` | TOML experiment settings |
| `tools/validation/` | Focused offline and live validators, selected through `tools/validate.py` |
| `tools/diagnostics/` | Independent live monitoring, recording and replay |
| `tests/` | Synthetic component tests and mocked integration tests |
| `docs/` | Maintained project and concept guides |
| `scripts/gen_reference.py` | Build-time API page generation and documentation link resolution |
| `runs/`, `recordings/` | Local experiment artifacts, excluded from Git |

## Component boundaries

| Responsibility | Concept | Code |
| --- | --- | --- |
| Read simulator data | [Read layer](input.md) translates shared memory and track files into domain models. | [AC adapters](../src/assetto_corsa_ai/assetto_corsa/) |
| Write controls | [Controller](control.md) sends runtime CSP commands. | [simulator_controller.py](../src/assetto_corsa_ai/assetto_corsa/simulator_controller.py) |
| Own the session | [Session manager](session_manager.md) launches AC and switches scenarios. | [session_manager.py](../src/assetto_corsa_ai/assetto_corsa/session_manager.py) |
| Derive state | Geometry, [observations](observation.md), [reward](reward.md) and [termination](termination.md) stay in dedicated components. | [track_projection.py](../src/assetto_corsa_ai/track_projection.py), [observation_builder.py](../src/assetto_corsa_ai/observation_builder.py), [reward.py](../src/assetto_corsa_ai/reward.py), [termination.py](../src/assetto_corsa_ai/termination.py) |
| Coordinate episodes | [Gymnasium environment](racing_environment.md) manages reset, step, timing and cleanup. | [RacingEnvironment.py](../src/assetto_corsa_ai/RacingEnvironment.py) |
| Learn and predict | [PPO and the auxiliary predictor](predictor.md) share one feature extractor and optimizer. | [agent.py](../src/assetto_corsa_ai/agent.py), [racing_policy.py](../src/assetto_corsa_ai/racing_policy.py), [racing_ppo.py](../src/assetto_corsa_ai/racing_ppo.py) |
| Record experiments | [Training runs](agent_training.md) preserve config, provenance, metrics and paired snapshots. | [training_run.py](../src/assetto_corsa_ai/training_run.py), [training_callbacks.py](../src/assetto_corsa_ai/training_callbacks.py) |

The environment contains no neural-network logic. The optional
[future reward wrapper](../src/assetto_corsa_ai/reward_wrapper.py) consumes a
detached prediction during collection and combines it with the actual step
reward before monitoring and normalization. The auxiliary loss uses actual
future positions from the current rollout; rewards remain fixed during updates.

## Coordinates and Visualization

Telemetry and [track geometry](../src/assetto_corsa_ai/track_geometry.py) retain
AC world coordinates `(x, y, z)`. The [projector](../src/assetto_corsa_ai/track_projection.py)
measures 3D polyline progress and signed lateral offset on the AI reference line.
Its geometric progress can differ from AC's spline metadata. Observation
waypoints and predicted trajectories use vehicle-relative `(forward, left)` metres.

[Track alignment validation](../tools/validation/validate_track_alignment.py) follows the
vehicle in a zoomed X/Z view with its front pointing up. Display rotation and
translation leave telemetry and projection calculations unchanged. Sampling,
frame conversion and projection details are documented in the linked code.
