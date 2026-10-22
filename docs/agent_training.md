# Training Runs

[PPOAgent](../src/assetto_corsa_ai/agent.py) manages PPO, normalization and model
loading. [main.py](../src/assetto_corsa_ai/main.py) composes the environment,
agent and callbacks. [TrainingRun](../src/assetto_corsa_ai/training_run.py)
manages experiment files.

## Running an Experiment

First complete [live environment validation](racing_environment.md#validation).
Close AC before launch; select Drive/Start and confirm in the terminal when
requested. Copy and adjust [configs/training.toml](../configs/training.toml):

```powershell
python -m assetto_corsa_ai.main train --config configs/training.toml
python -m assetto_corsa_ai.main drive --model-directory runs/<run_name>/final --episodes 5
```

The TOML is the sole source of training settings, including environment,
network, PPO, predictor, reward and normalization.
[TrainingConfig](../src/assetto_corsa_ai/training_config.py) validates it before
AC starts. The template enables delta actions and the
[auxiliary predictor](predictor.md). `training.total_timesteps` sets the total
step budget; [future reward warmup](reward.md#future-predictor-reward) is configured separately.

The policy adds deltas to current controls. `ppo.initial_action_std` sets the
initial Gaussian exploration scale; it remains learnable during training.
`ppo.initial_longitudinal_mean` initializes the trainable longitudinal actor bias.
For stationary starts, a positive value encourages acceleration while Gaussian
sampling retains braking exploration. It changes only new-policy initialization;
loaded checkpoints retain their trained weights. The current template disables
pre-roll and starts at 0 km/h.
`[action]` limits executed steering/longitudinal changes per step and defines
the longitudinal deadband. At 50 ms, limits of 0.05/0.10 correspond to at most
1.0/2.0 control units per second. Small longitudinal commands settle to neutral
without exceeding those limits. [The action filter](../src/assetto_corsa_ai/action_wrapper.py)
keeps PPO's original samples and likelihoods intact.
Physical observation features are normalized; the two controls keep their
existing scale in delta mode. Architecture and normalization are implemented
in [racing_policy.py](../src/assetto_corsa_ai/racing_policy.py).

## Run Records and Checkpoints

Each invocation, including resume, creates a unique directory under `runs/`:

```text
runs/<YYYYMMDD_HHMMSS_track_car_random-id>/
├── config.toml
├── metadata.json
├── metrics.csv
├── checkpoints/
│   └── step_004096/
│       ├── model.zip
│       └── vecnormalize.pkl
└── final/
    ├── model.zip
    └── vecnormalize.pkl
```

The exact config and automatically collected Git/software provenance are
written once. A dirty Git status means the recorded commit alone cannot
reconstruct the implementation. Completed snapshots are never overwritten.
All policy, value, predictor and shared-extractor weights belong to
`model.zip`; `vecnormalize.pkl` must accompany it.

`training.checkpoint_interval_updates = 4` saves after every fourth completed
PPO phase, after optimization. A phase collects `ppo.n_steps` steps and then
optimizes that rollout; `n_epochs` does not determine the checkpoint cadence.
Zero disables periodic checkpoints; successful training still writes `final/`.
Directory names retain the total step count. The completed-phase counter is
saved in `model.zip`, continues across resume and appears in `metrics.csv`.

[Training callbacks](../src/assetto_corsa_ai/training_callbacks.py) append raw
episode rewards and separate PPO update rows to `metrics.csv`. Update rows
include predictor loss/accuracy, raw reward component means and future weight
scale. Unavailable values remain empty; field definitions live in
[training_run.py](../src/assetto_corsa_ai/training_run.py).

## Resume and Evaluation

Set `training.resume_from` to a completed checkpoint or `final/` in a new
TOML. Settings must match the source run; the additional budget, checkpoint
interval and verbosity may change. Loading restores weights, optimizer,
normalization and total timesteps, so warmup/ramp continue at the saved step.
An unfinished rollout and random-generator state are not restored.
Archived configs remain readable: legacy action behavior is preserved and
step-based checkpoint intervals round up to whole PPO phases. Older models
without the phase counter start a new checkpoint cadence at zero.

PPO completes whole rollouts, so it may exceed the requested budget.
Interrupted runs retain completed checkpoints; `final/` is saved after
successful training. Evaluation freezes normalization statistics. Resume
compatibility and snapshot loading are defined in
[training_config.py](../src/assetto_corsa_ai/training_config.py) and
[agent.py](../src/assetto_corsa_ai/agent.py).
