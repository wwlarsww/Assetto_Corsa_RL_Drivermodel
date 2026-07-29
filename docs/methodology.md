# Method contract

This document is the single source of truth for the public method scaffold. It
maps Sections 2.2.1–2.2.4 and Figures 2.1–2.3 of Lars Kilian's 2026 master's
thesis to code. The thesis deliberately describes some implementation details
only conceptually. Such details are labelled here instead of being presented as
if they were thesis requirements.

Three labels are used:

- **Thesis**: stated directly in the thesis text, equation, or figure.
- **Public convention**: needed for an executable open implementation, but not
  uniquely specified by the thesis.
- **Initial variant**: intentionally differs from the complete contextual
  method for the first Assetto Corsa experiment.

## Initial experiment

`f317_barcelona_fixed` starts with one Dallara F317, Barcelona GP, and one
reference in every episode. This is an **initial variant**: ProMP fitting and
sampling are disabled, while the observation, action, reward, and network
components are recorded under the named variant
`fixed_reference_no_promp_v1`. A later contextual experiment must use its own
named configuration and must not silently change this baseline.

The fixed reference combines the official world-space line with a robust
human-median speed profile. The profile converts the 20 future times to
positions on that same prescribed line. See
[data-pipeline.md](data-pipeline.md).

## Observation contract

The observation has 77 `float32` scalars in four groups:

| Slice | Size | Frame and unit |
|---|---:|---|
| `0:7` | 7 | vehicle dynamics in SI units |
| `7:21` | 14 | seven left-boundary `(x, z)` points in vehicle ego frame, metres |
| `21:35` | 14 | seven right-boundary `(x, z)` points in vehicle ego frame, metres |
| `35:75` | 40 | twenty fixed-reference `(x, z)` points in vehicle ego frame, metres |
| `75:77` | 2 | previous absolute pedal and steering commands, normalised |

**Thesis:** the vehicle features are absolute speed, acceleration in the
vehicle x and z directions, front and rear slip ratio, and front and rear
side-slip angle.
Boundary preview distances are `5, 10, 20, 40, 80, 160, 320 m`. The reference
contains 20 coordinate pairs on the prescribed racing line at `0.25 s`
intervals over `5 s`. The previous action is required because the policy
predicts increments.

**Public convention:** flat-track ego coordinates are
`x=lateral-left, z=forward`. The fixed observation order and every feature unit
are machine-readable in `features/schema.py` and explained in
[feature-reference.md](feature-reference.md).

The real-game tyre adapter uses the named **public convention**
`vanilla_ac_python_v1`: left/right wheel arithmetic means, Kunos API signs
preserved, and `SlipAngle` converted from degrees with `+π/180`. This matches
the public AssettoCorsaGym online channels and gives the BC exporter an exact
offline/live contract. It is not claimed as an unpublished thesis detail.

Reference time points are not approximated with one constant speed. They are
mapped to track distance with the complete reference speed profile and then to
world coordinates on the one fixed racing line. Every point is finally
transformed relative to the current vehicle pose.

All features use static Z-score normalisation:

```text
z_i = (x_i - mean_i) / max(std_i, epsilon)
```

**Thesis:** statistics are computed once from demonstrations and remain fixed
during RL. **Public convention:** a normalisation artefact may carry the exact
feature names; the observation builder rejects statistics with a different or
missing order. Identity statistics are allowed only for the transparent mock
environment.

## Reference and ProMP contract

**Thesis:** demonstrations are parameterised over track distance, projected
onto radial basis functions by ridge regression, and the flattened weights are
modelled by

```text
mu = (1/N) sum(w_i)
Sigma = (1/N) sum((w_i - mu)(w_i - mu)^T)
```

The implementation uses that population covariance (`1/N`), circular RBF
distance for a closed lap, and unwraps heading before fitting. A sampled
reference is fixed for the complete episode; deterministic evaluation uses the
distribution mean.

The current ProMP example encodes lateral offset, heading, and speed but has no
track-geometry conditioning interface yet. Prediction rewards in a real ProMP
run therefore additionally require world positions for every sampled
reference. The fixed-reference experiment already contains these positions.

## Action contract

The policy action is a two-dimensional increment:

1. combined pedal, positive for throttle and negative for braking;
2. steering.

The public adapter scales the dimensionless policy values by configured
per-step delta scales. The command state then follows thesis Equation 2.7:

```text
a[t+1] = clip(a[t] + delta_a[t], a_min, a_max)
```

Absolute commands are normalised to `[-1, 1]`; an Assetto Corsa backend maps
them to throttle, brake, and the car-specific steering range.

**Thesis:** the policy is Gaussian, its mean is bounded by `tanh`, and its log
standard deviation is state-independent. A Gaussian sample itself is not
silently clipped before collection. Only the integrated absolute command is
clipped. This keeps the sampled action consistent with the log probability
that PPO must store.

## Reward contract

The initial no-ProMP variant deliberately has no imitation/style component:

```text
r_fixed =
    c * delta_reference_progress
  + alpha_pred * predicted_progress
  + termination_penalty
```

For actual progress, each vehicle world position is projected onto the closed
prescribed reference line. The reward uses the signed, wrap-safe difference of
the current and previous projected arc coordinates:

```text
s_t = project_to_reference(position_t)
delta_reference_progress = wrap_signed(s_t - s_(t-1))
```

The environment adds a fixed negative penalty for safety termination. Lap
completion is not a safety violation. Off-track, stalled, and excessive-slip
termination are provided as explicit mock/test examples.

After the rollout, the predictor is decoded into twenty ego-frame Bézier
samples. The same frame contains a reference polyline comprising the exact
current projection point followed by the twenty prescribed future reference
points. The predicted endpoint is projected onto this polyline, and its
along-line arc coordinate is the predicted progress:

```text
reference_path = [current_reference_projection, future_reference_points...]
predicted_progress =
    project_to_reference_arc(bezier_samples[-1], reference_path)
```

Origins without a complete observed future horizon in the same fragment are
masked only for auxiliary predictor supervision. Predicted progress itself is
available at every rollout origin because the prediction and fixed future
reference are already known. This is not raw ego-frame `z` displacement:
bends in the reference line contribute their full arc length.
`PredictedProgressReward` contains no imitation cost or Lagrange update. The
separate `PredictionReward` and `StyleLagrangeController` implementations are
retained only for a later named contextual/ProMP variant.

Prediction terms are added **after rollout collection and before GAE**. They
must never be calculated inside simulator dynamics or appended after advantage
estimation.

## Network and predictor contract

Figure 2.3 specifies these widths:

| Module | Widths | Status |
|---|---|---|
| shared feature extractor | `512, 512` | Thesis |
| policy head | `128, 128, 2` | Thesis |
| value head | `512, 512, 1` | Thesis |
| predictor head | `128, 128, 10` | Thesis |

The policy, value, and predictor heads share the feature extractor. During the
warm start, BC action MSE and predictor Smooth L1 are optimised jointly. The
predictor target is the actually observed future human path, not the prescribed
reference points in the observation. During RL, predictor supervision remains
an auxiliary update against actually observed future positions.

The thesis fixes the Bézier-head width to 10 but does not disclose how those
scalars are packed. The scaffold therefore names its replaceable **public
convention** `bezier-5x2`: five ego-frame 2D control points are decoded into 20
time-ordered path points. Code and artefacts record this codec name; it must
not be described as the unpublished original encoding.

## Executable post-rollout sequence

For each current-policy fragment:

1. retain world position, vehicle heading, simulation time, a globally unique
   actor/episode trajectory ID, reference progress, policy action, value, and
   log probability;
2. interpolate 20 actually observed future positions at `0.25…5.0 s`;
3. reject targets that cross termination, truncation, or a fragment boundary;
4. transform future positions into each origin's ego frame;
5. build the current reference projection plus future reference points in
   exactly the same ego frames;
6. decode rollout predictions, project their endpoint, and compute predicted
   reference arc progress;
7. add predicted-progress rewards to environment rewards;
8. compute termination-aware GAE;
9. update clipped PPO, the value function and the predictor on valid observed
   targets.

`training/targets.py`, `models/bezier.py`, and `rewards/components.py` make
steps 2–8 independently testable with arrays and mock data.

## Traceability matrix

| Thesis item | Public implementation | Verification |
|---|---|---|
| Section 2.2.1, ProMP weight Gaussian | `references/promp.py` | fit/mean/seeded-sample tests |
| Section 2.2.2, 77 observation scalars | `features/schema.py`, `observation.py` | exact name/order/shape tests |
| Section 2.2.2, fixed demonstration Z-score | `features/normalizer.py` | round-trip and schema mismatch tests |
| Equation 2.7, cumulative action | `control/actions.py` | scale/integrate/saturation tests |
| Equation 2.8, reward timing | `rewards/components.py` | component and mask tests |
| Section 2.2.4, network widths | `models/driver_network.py` | output-shape tests |
| Section 2.2.4, observed future targets | `training/targets.py` | interpolation/frame/episode tests |
| Figure 2.3, 10 predictor outputs | `models/bezier.py` | named codec and endpoint tests |
| Section 2.2.5/4.3, on-policy fragments | `training/rollout.py` | stale-version rejection tests |
| Post-rollout optimisation | `training/gae.py`, `training/ppo.py` | terminal bootstrap and mock smoke tests |

## What is not yet validated

Passing unit tests establishes interface and mathematical consistency, not
driving quality. Speed, acceleration x/z, and the four tyre slip signals have
named live mappings; the offline demonstrations now provide fixed Z-score
statistics and a joint BC/predictor checkpoint. Real Assetto Corsa integration
still needs world-pose sign verification, reference projection, live
track-boundary extraction, and closed-loop control. The 35-epoch BC run passes
the configured held-out predictor gate, but that is not evidence that its
reward coefficient is optimal in closed loop. Reward coefficients, safety
thresholds, and predictor weighting require controlled experiments rather
than being inferred from a smoke run.
