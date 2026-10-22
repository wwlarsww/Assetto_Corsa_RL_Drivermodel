# API reference

The pages in this section are generated from Python source, signatures, type
annotations and docstrings at every documentation build. New modules under
`src/` and `tools/` are discovered automatically. Private names and inherited
library APIs are excluded to keep the reference focused on this project.

Use the [project overview](../index.md) and [training walkthrough](../training_step.md)
to understand the design before looking up individual methods here.

## Core entry points

| Responsibility | Reference | Concept |
| --- | --- | --- |
| Application composition and CLI | [main](assetto_corsa_ai/main.md) | [Getting started](../getting_started.md) |
| Gymnasium episodes | [RacingEnvironment](assetto_corsa_ai/RacingEnvironment.md) | [Environment and resets](../racing_environment.md) |
| Agent and normalization | [agent](assetto_corsa_ai/agent.md), [racing_policy](assetto_corsa_ai/racing_policy.md) | [Training](../agent_training.md) |
| PPO and auxiliary supervision | [racing_ppo](assetto_corsa_ai/racing_ppo.md), [predictor](assetto_corsa_ai/predictor.md) | [Predictor](../predictor.md) |
| Observation construction | [observation_builder](assetto_corsa_ai/observation_builder.md) | [Observations](../observation.md) |
| Actual and predicted reward | [reward](assetto_corsa_ai/reward.md), [future_reward](assetto_corsa_ai/future_reward.md) | [Reward](../reward.md) |
| Episode failure detection | [termination](assetto_corsa_ai/termination.md) | [Termination](../termination.md) |
| Configuration and artifacts | [training_config](assetto_corsa_ai/training_config.md), [training_run](assetto_corsa_ai/training_run.md) | [Configuration](../configuration.md) |

## Adapters

The `assetto_corsa` package contains simulator-specific formats and APIs:

- [TelemetryReader](assetto_corsa_ai/assetto_corsa/telemetry_reader.md)
- [StaticInfoReader](assetto_corsa_ai/assetto_corsa/static_info_reader.md)
- [AssettoCorsaTrackReader](assetto_corsa_ai/assetto_corsa/track_reader.md)
- [AssettoCorsaSimulatorController](assetto_corsa_ai/assetto_corsa/simulator_controller.md)
- [AssettoCorsaSessionManager](assetto_corsa_ai/assetto_corsa/session_manager.md)

Read [telemetry and track data](../input.md), [controls](../control.md) and
[sessions](../session_manager.md) for the boundaries between these components.

## Tools

The **Tools** navigation group covers the validation CLI and diagnostic modules.
Start with [validate](tools/validate.md),
[diagnostic sources](tools/diagnostics/sources.md) or
[recording and replay](tools/diagnostics/recording.md).

## How the reference is built

mkdocstrings analyzes the source statically; it does not instantiate a simulator
reader, launch Assetto Corsa, load a trained model or initialize Qt. The
documentation build therefore needs documentation packages, not the training
or GUI dependency stacks.

Generated API pages are build artifacts. Edit the Python docstring or type
annotation to improve a reference entry. See the
[documentation guide](../documentation.md) for the build process and conventions.
