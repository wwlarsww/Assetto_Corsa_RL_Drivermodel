# BMW Z4 GT3/Monza data pipeline

The executable public-data baseline fixes one base-game car, one base-game
track, and one episode context:

| Item | Selection |
|---|---|
| dataset variant | `bmw_z4_gt3_monza_v1` |
| car | BMW Z4 GT3 (`bmw_z4_gt3`) |
| track | Monza (`monza`) |
| reference mode | fixed reference only |
| demonstrations | human MoTeC recordings only |
| dataset revision | `ad1df12c49cce10f159ba31967744750bb9edd8b` |
| source revision | `ef0fa502bdd257fc2affad9361c02b3dbdfe1632` |

The data comes from
[AssettoCorsaGym](https://huggingface.co/datasets/dasgringuen/assettoCorsaGym)
under CC BY 4.0. Its source configuration is MIT licensed.

## Reproduce the local data

```powershell
python -m pip install -c requirements/tested.txt -e ".[data,dev]"
python -m assetto_corsa_rl download-data
python -m assetto_corsa_rl build-reference `
  --config configs\bmw_z4_gt3_monza_baseline.toml
python -m assetto_corsa_rl build-bc-dataset `
  --config configs\bmw_z4_gt3_monza_baseline.toml
python -m assetto_corsa_rl bc-train `
  --config configs\bmw_z4_gt3_monza_baseline.toml
```

`download-data` selects human `.ld`/`.ldx` recordings and available lap
metadata below `data_sets/monza/bmw_z4_gt3`. SAC trajectories, replay buffers,
checkpoints, and pickle telemetry are excluded. Downloaded CSV geometry is not
used as runtime truth. Every selected file is size-checked and hashed in
`data/manifests/assetto_corsa_gym_bmw_z4_gt3_monza.json`.

Large source data, processed tensors, checkpoints, and runs remain ignored by
Git. Only source code, configuration, documentation, and the compact provenance
manifest belong in the repository.

## AI-spline reference construction

The configured track's installed `content/tracks/<track>/<layout>/ai/fast_lane.ai`
is the sole geometry source for live and offline paths. The strict v7 reader
extracts centre points, cumulative distance, and per-point left/right side
distances. From those values the pipeline derives the closed-lap length, the
world-space reference and heading, and the boundary-point feature geometry.

Unknown binary versions, invalid point IDs, missing side metadata, and
degenerate geometry are rejected rather than guessed. The reference NPZ stores
the exact source SHA-256; reference loading and BC export fail when it differs
from the configured installed file. Human telemetry contributes only the
target-speed profile, not alternative line geometry.

The builder reads raw MoTeC without enabling NumPy pickle loading. Pit, invalid,
off-track, non-finite, and implausible samples are discarded. Remaining speeds
are aggregated into 1 m distance bins, periodically interpolated where needed,
and smoothed with a circular 15 m window. The baseline `speed_mps` curve remains
the robust per-bin median. A separate `human_mean_speed_mps` curve stores the
arithmetic mean of the same valid samples for the named CSP reference-speed
reset; it does not change observations, rewards, or the thesis baseline. The
resulting safe NPZ is written to
`data/processed/bmw_z4_gt3_monza/fixed_reference.npz`.

## Behavioural-cloning export

The exporter accepts complete laps, resamples them to the configured 25 Hz
control rate, and constructs the exact 77-value observation. Sessions are kept
disjoint between training, validation, and test. Static Z-score statistics are
fitted on the training split only and stored with the exact feature order.

Absolute controls remain part of the observation. The next recorded absolute
combined-pedal and steering commands are the BC targets directly in `[-1, 1]`;
there is no delta scaling or action-rate filter. Raw ACTI/MoTeC steering is
positive-right, whereas the policy and ego frame use positive-left. Dataset
schema 6 therefore applies the named conversion
`motec_positive_right_to_policy_positive_left_v1` before normalisation and
stores both that conversion and the resulting `policy_positive_left_v1`
convention in its metadata. BC training and `rl-ready` reject older or missing
steering-calibration metadata. BMW steering then uses the pinned magnitude map
[`±320° <-> ±0.7119140625`](https://github.com/dasGringuen/assetto_corsa_gym/blob/main/assetto_corsa_gym/AssettoCorsaConfigs/cars/bmw_z4_gt3/steer_map.csv).
The predictor target is the actual future human path at twenty `0.25 s`
offsets, never a path crossing a lap or recording boundary.

## BC warm start

`bc-train` jointly optimises the shared feature extractor, action head, and
trajectory-predictor head with the configured loss weight. It reports held-out
action and predictor metrics and atomically publishes the accepted warm-start
artefacts at the stable paths consumed by PPO. The selected checkpoint records
the method name, absolute-action schema, observation schema, model configuration,
and dataset metadata.
The baseline action loss is `uniform_action_mse_v1`. The named
`steering_weighted_action_mse_v1` uses the configured steering weight, while
the optional `straight_steering_weighted_action_mse_v2` adds a tenfold weight
for steering targets with absolute magnitude at most `0.02`. The latter records
both constants in the metrics summary. It is an explicit experiment, not a
silent replacement for the baseline or the active weighted checkpoint.
Only a checkpoint whose validation thresholds pass `rl-ready` may initialise
live PPO.

Assetto Corsa need not be running for these offline stages, but the configured
installation and its `fast_lane.ai` must be available. Live diagnostics and
closed-loop PPO additionally require the game and bridge apps.
