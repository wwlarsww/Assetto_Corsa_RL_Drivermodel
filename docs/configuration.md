# Configuration

The training TOML is the single source of truth for an experiment. Copy
[configs/training.toml](../configs/training.toml) and adjust the copy. Training
accepts `--config` and an optional `--runs-directory`; scenario and model settings
come from the TOML rather than competing command-line overrides.

`TrainingConfig` validates the complete configuration before Assetto Corsa
starts. The exact input text is archived in each run, including resumed runs.

## Settings by responsibility

| Section | What it controls | Further reading |
| --- | --- | --- |
| `environment` | Car, track, control period, observation sampling and episode/reset limits | [Environment](racing_environment.md), [observations](observation.md) |
| `action` | Maximum executed control changes and longitudinal deadband | [Controls](control.md) |
| `start` | Optional rolling starts and pre-roll controller | [Resets](racing_environment.md#episodes) |
| `simulator` | Installation paths, car index and session/control timeouts | [Sessions](session_manager.md) |
| `reward` | Actual progress, reference-line style and termination penalties | [Reward](reward.md) |
| `reward.future_prediction` | Prediction reward weights, discount, warmup and ramp | [Future reward](reward.md#future-predictor-reward) |
| `termination` | Time window and required progress for stuck/wrong-way detection | [Termination](termination.md) |
| `ppo` | Optimization, exploration and rollout settings | [Training](agent_training.md) |
| `normalization` | Observation/reward normalization and clipping | [Agent input](observation.md#coordinates-and-agent-input) |
| `network` | Shared extractor, policy/value heads, activation and optimizer settings | [Predictor architecture](predictor.md) |
| `predictor` | Auxiliary task, trajectory horizons and supervised loss weight | [Predictor](predictor.md) |
| `training` | Seed, step budget, device, checkpoint cadence and resume source | [Run management](agent_training.md) |
| `trajectories` | Reserved trajectory-export switch | Currently reserved; use [diagnostic recordings](diagnostics.md#recording-and-replay) |
| `diagnostics` | Publish paired snapshots for the independent monitor | [Diagnostics](diagnostics.md) |

## Decisions to make before a run

1. Set the simulator installation path and verify the selected car/track layout.
2. Set a deliberate step budget. PPO collects whole rollouts and may exceed it.
3. Choose stationary or rolling starts. Validate spline speed units before
   enabling rolling starts.
4. Decide whether future reward should be active immediately. Warmup and ramp
   scale prediction reward weights; the predictor still learns during warmup.
5. Choose checkpoint frequency in **completed PPO phases**, not optimizer epochs.
6. Select CPU or CUDA deliberately when comparing experiment performance.

## Reading the template

The template uses delta actions, a 50 ms nominal control interval and ten
predictor horizons from 0.1 to 1.0 seconds. Each horizon must be an integer
multiple of the control interval.

With `n_steps = 1024` and `checkpoint_interval_updates = 4`, a fresh run writes
a regular checkpoint after 4,096 collected transitions and their completed
optimization phases. Diagnostics snapshots are a separate rotating cache.

Zero values are **setting-specific**. For example, zero disables an episode
limit or periodic checkpoints, but the learning rate must be positive. The
validator, rather than a general interpretation of template comments, defines
which values are accepted.

## Resume compatibility

Set `training.resume_from` to a completed checkpoint directory containing
`model.zip` and `vecnormalize.pkl`. The new experiment must be compatible with
its source configuration; see [resume and evaluation](agent_training.md#resume-and-evaluation).

Code: [TrainingConfig](../src/assetto_corsa_ai/training_config.py),
[PPOConfig](../src/assetto_corsa_ai/agent.py),
[StartConfig](../src/assetto_corsa_ai/episode_start.py).
