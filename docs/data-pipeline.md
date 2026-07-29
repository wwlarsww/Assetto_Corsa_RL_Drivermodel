# F317/Barcelona data pipeline

The first real-data baseline deliberately fixes one car, one track, and one
episode context:

| Item | Selection |
|---|---|
| car | Dallara F317 (`dallara_f317`) |
| track | Barcelona GP (`ks_barcelona-layout_gp`) |
| reference mode | fixed; no ProMP sampling |
| demonstrations | human MoTeC recordings only |
| dataset revision | `ad1df12c49cce10f159ba31967744750bb9edd8b` |
| source revision | `ef0fa502bdd257fc2affad9361c02b3dbdfe1632` |

The data comes from
[AssettoCorsaGym](https://huggingface.co/datasets/dasgringuen/assettoCorsaGym).
The dataset is CC BY 4.0; its source code and track CSV files are MIT licensed.

## Reproduce the local data

Install the optional downloader and run the four explicit stages:

```bash
python -m pip install -c requirements/tested.txt -e ".[data,dev]"
python -m assetto_corsa_rl download-data
python -m assetto_corsa_rl build-reference --config configs/f317_barcelona_fixed.toml
python -m assetto_corsa_rl build-bc-dataset --config configs/f317_barcelona_fixed.toml
python -m assetto_corsa_rl bc-train --config configs/f317_barcelona_fixed.toml
```

`download-data` selects 70 human `.ld` recordings, their `.ldx` files, the
available lap metadata, the 1 m occupancy grid, and three official track source
files. It excludes SAC trajectories, replay buffers, checkpoints, and
preprocessed pickle telemetry. Every selected file is size-checked and hashed
in `data/manifests/assetto_corsa_gym_f317_barcelona.json`.

All large files below `data/` are ignored by Git. Only the provenance manifest
and documentation are tracked.

## Fixed-reference construction

The official AssettoCorsaGym configuration selects
`ks_barcelona-layout_gp-racing_line_fixed.csv` and gives an internal spline
length of `4591.88720703125 m`. For this initial baseline the line is the common
episode reference:

- lateral offset relative to that line is always zero;
- heading is the wrapped world-frame tangent of the official line;
- target speed is a robust fixed profile derived from human telemetry.

The builder reads the raw MoTeC format directly. It does not load the downloaded
Python pickle telemetry. Samples are rejected when they are in the pits, have a
tire outside the track, are marked as an invalid lap, or contain implausible
speed/phase values. The remaining speeds are aggregated into 1 m distance bins
with a 0.25 m/s histogram resolution. The median is taken per bin, missing bins
are interpolated periodically, and a circular 15 m moving average removes
quantisation noise.

For the pinned data currently downloaded, the artefact contains:

- 70 recordings;
- 1,692,986 raw samples inspected;
- 1,522,640 valid samples used;
- 4,592 distance samples;
- target speeds from 18.8 to 66.4 m/s.

The result is
`data/processed/f317_barcelona/fixed_reference.npz`. NumPy pickle loading is
disabled. The archive contains the model features, world-space line coordinates
for later simulator integration, build metadata, and the pinned source
revisions. With the pinned inputs and current builder it is byte-reproducible:
SHA-256
`ef00bcefe265558ce34ee86a1b868d6581863c94bd3e73a88c64454cbc222584`.

At runtime `load_reference_sampler` returns a `FixedReferenceSampler`. Calling
`sample()` therefore returns the same immutable reference for every episode.
The ProMP implementation remains available for a later named experiment but is
not involved in this baseline.

## Behavioural-cloning export

The exporter reads only complete MoTeC laps, accepts quantised duplicate lap
timestamps, and resamples to the configured 25 Hz control rate. It constructs
the exact 77-value observation:

1. absolute speed, acceleration x/z, and front/rear slip ratio and angle;
2. seven `(x, z)` points for each track boundary;
3. twenty `(x, z)` points on the prescribed fixed racing line at
   `0.25, 0.50, ..., 5.00 s`;
4. the most recent absolute combined pedal and steering values.

All geometry uses the vehicle frame `x=lateral-left, z=forward`. Reference
times are converted to distance with the complete fixed-reference speed
profile, then sampled from the prescribed world-space racing line. The
predictor target is intentionally different: it is the actual human vehicle
path over the same twenty future times.

Controls are mapped to combined pedal (`throttle - brake`) and steering.
Absolute commands remain in the observation; their next-step differences,
divided by the configured action limits, are the BC targets. The Dallara
steering conversion uses the
[pinned public steering map](https://github.com/dasGringuen/assetto_corsa_gym/blob/main/assetto_corsa_gym/AssettoCorsaConfigs/cars/dallara_f317/steer_map.csv):
`±240° <-> ±0.5341796875`.

Splits are disjoint by recording session and greedily balanced by usable sample
count. Static Z-score statistics are fitted on the training split only, stored
with all 77 feature names, and never updated from validation, test, or RL data.
The compressed NPZ loader disables pickle support.

For the pinned local data and current filtering, the complete export contains:

- 43 usable recordings/laps from 11 sessions;
- 87,409 training samples;
- 18,812 validation samples;
- 18,426 test samples;
- 75,003 training origins with a complete 5 s predictor target.

Generated dataset and run artefacts stay below ignored `data/processed/` and
`runs/` paths and must not be committed.

The tyre-channel mapping for that exporter is already fixed. ACTI labels
`Tire Slip Angle FL/FR/RL/RR` as degrees and `Tire Slip Ratio
FL/FR/RL/RR` as percent. Preserve their signs, take arithmetic `FL/FR` and
`RL/RR` means, convert angles with `+π/180`, and convert ratios with `/100`.
The result then has exactly the same feature semantics as the live
`vanilla_ac_python_v1` adapter. Neither path clips slip values.

## Joint BC/predictor warm start

`bc-train` trains the shared feature extractor, the action-mean head, and the
trajectory-predictor head together. The action loss is MSE on two normalised
action increments. The auxiliary loss is Smooth L1 between the decoded
`bezier-5x2` trajectory and the actual human future path. Validation and test
report action MSE/MAE plus predictor Smooth L1, ADE, and FDE.

The default three epochs are a pipeline smoke baseline, not a converged driving
model. The first verified 10-output-head run reached test action MAE `0.0959`
and predictor ADE `67.3 m`; the latter is still too large for a trusted RL reward signal.
Longer training and predictor hyperparameter validation are required before
closed-loop use.

Assetto Corsa is not needed for these offline stages. It remains necessary for
the live backend, control integration, and closed-loop validation.
