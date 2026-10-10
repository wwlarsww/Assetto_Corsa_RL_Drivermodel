# Training runs

`PPOAgent` handles PPO and observation normalization. `main.py` composes the
environment and training callbacks. `TrainingRun` only manages files; it has
no model architecture or PPO logic.

Before training, validate the live simulator with
[`validate_racing_environment.py`](../tools/validate_racing_environment.py).
Close AC before launching a run, select Drive, then confirm in the terminal.

```powershell
python -m assetto_corsa_ai.main train --config configs/training.toml
python -m assetto_corsa_ai.main drive --model-directory runs/<run_name>/final --episodes 5
```

Copy and edit [`configs/training.toml`](../configs/training.toml) for each
experiment. It contains all training settings: scenario and assists, environment
timing/reset/perception, reward/termination, PPO, normalization, shared feature
extractor and policy/value heads, seed, budget and checkpoint interval. Unknown
or missing settings fail before AC starts. Training accepts no parameter
overrides from the CLI or environment variables. The exact TOML is preserved
in the run. The template requests 4096 steps; PPO collects complete rollouts,
so a different budget may be exceeded to finish its last rollout.

Each invocation, including resumed training, creates a new directory:

```text
runs/<YYYYMMDD_HHMMSS_track_car_random-id>/
├── config.toml
├── metadata.json
├── metrics.csv
├── checkpoints/
│   └── step_010000/
│       ├── model.zip
│       └── vecnormalize.pkl
└── final/
    ├── model.zip
    └── vecnormalize.pkl
```

Config, metadata and completed snapshots are written once; existing artifacts
are never replaced. Metrics append episode rows with raw rewards and separate
PPO update rows. Unavailable losses stay empty, including `predictor_loss`.
Metadata records Git commit/dirty status, timezone-aware start time and software
versions, without duplicating training parameters. A dirty Git status signals
that the commit alone does not capture the working implementation.

To resume, set `training.resume_from` to a completed checkpoint or `final/` in
a new TOML and run `train --config` again. Settings must match the source run;
the additional timestep budget, checkpoint interval and verbosity may change.
Weights, optimizer state, normalization and step counter are restored; an
unfinished rollout and random-generator state are not restored. Checkpoints
are counted from the start of the new invocation and named by total steps.
Ctrl+C retains completed checkpoints. `final/` is saved after successful training.

The shared extractor, policy/value heads and enabled predictor head are included in `model.zip`;
`vecnormalize.pkl` is saved alongside it for every snapshot. Evaluation freezes
normalization statistics. Existing `models/` bundles remain loadable for
evaluation; new training always uses runs, which are ignored by Git.

The template enables the [auxiliary trajectory predictor](predictor.md) and
delta actions. Both objectives train one shared extractor with one optimizer;
predictor loss, valid horizon count and forward/lateral MAE join `metrics.csv`.
Delta controls retain their existing `[-1, 1]` scale in VecNormalize; physical
features are normalized. The policy adds its delta to the observed controls
before SB3 clips the final action. `action_mode = "absolute"` is also supported.

The template enables [future prediction reward](reward.md) with 20,000 warmup
steps and a 30,000-step linear ramp. Its initial 4096-step run therefore trains
the auxiliary predictor without adding future reward. The schedule follows
total model timesteps across resume; disabling the extension leaves step
rewards unchanged. Raw reward component means, weight scale and prediction
diagnostics join the PPO update rows in `metrics.csv`, before normalization.

Selected trajectory export is reserved for later and disabled. No full
rollouts, `architecture.txt` or separate predictor weights are saved.
