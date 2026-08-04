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

`bmw_z4_gt3_monza_v1` starts with one BMW Z4 GT3, Monza, and one reference in
every episode. This is the **initial fixed-reference variant**: reference
sampling is disabled, while the observation, action, reward, and network
components are recorded under the named variant
`ai_spline_absolute_v1`. A later experiment must use its own
named configuration and must not silently change this baseline.

The current PPO configurations therefore use the distinct
`ai_spline_absolute_stabilized_v2` variant. It retains the baseline geometry,
observation and bounded-action contracts while explicitly selecting
`any_tyre_armed_stall_v1`, `discounted_progress_increments_v1`, and
`target_band_learning_rate_v1`. It is a stabilised experimental variant, not a
silent redefinition of the thesis baseline.

The fixed reference combines the installed track's `ai/fast_lane.ai`
world-space centreline with a robust human-median speed profile. The same file
provides lap length and side distances for boundary features. The speed profile
converts the 20 future times to positions on that same line. See
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
contains 20 coordinate pairs on the AI-spline reference at `0.25 s`
intervals over `5 s`. The previous absolute action makes actuator state
explicit to the policy.

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

## Reference contract

The initial method has no learned or sampled reference model. It reads the
configured track layout's binary v7 `ai/fast_lane.ai` and rejects unknown
versions. Centre points and cumulative distances define reference projection
and lap length; per-point left/right distances define the boundary previews.
The generated reference stores the source SHA-256 and cannot be loaded against
a different spline. Human telemetry changes only the target-speed profile.

## Action contract

The policy action is a two-dimensional absolute command:

1. combined pedal, positive for throttle and negative for braking;
2. steering.

Both components are bounded to `[-1, 1]`:

```text
a_t = tanh(u_t),  u_t ~ Normal(mu_t, sigma)
```

There is no delta limit or action integration. A direct switch from `-1` to
`+1` in one agent step is valid. An Assetto Corsa backend maps the combined
pedal to throttle/brake and steering to the simulator's normalised range.
The initial live variant uses Assetto Corsa's automatic shifter and automatic
clutch; gear selection is not part of the policy action. A deterministic reset
engages first gear, and live acceptance requires actual forward motion and
positive reference-line progress under a bounded throttle command.

The policy uses a tanh-squashed Gaussian with state-independent log standard
deviation. PPO stores the transformed log probability including the tanh
Jacobian, so the sampled action and rollout density remain consistent.

## Reward contract

The initial variant contains only progress, predicted progress, and a safety
termination penalty:

```text
r_fixed =
    c * delta_reference_progress
  + alpha_pred * discounted_predicted_progress
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
completion is not a safety violation. In live runs the collision baseline is
recorded only after the acknowledged deterministic `HOTLAP_START` reset, stable
frames and a configured neutral quiescence interval so reset-induced callbacks
cannot leak into the episode. Every subsequent CSP collision event terminates immediately as
`CRASH`; damage and inferred vehicle dimensions
are not part of the criterion. The explicitly named termination variant
`any_tyre_armed_stall_v1` terminates as soon as Assetto Corsa reports one tyre
outside; the earlier four-tyre condition is not used. Vehicle body-sideslip
remains an explicit additional safety condition. The body-sideslip
termination computes the body slip angle as
`beta = atan2(local_velocity_x, local_velocity_z)`: AC local x is the
calibrated lateral axis and local z is forward-positive. It terminates when
`abs(beta)` exceeds `critical_body_sideslip_angle_deg = 20.0` at or above
`minimum_body_sideslip_termination_speed_mps = 5.0`. Longitudinal tyre slip
ratio remains an observation but is no longer a termination criterion, so
launch wheelspin is permitted.
Every episode starts from the explicit absolute action `[+1, 0]`: full
throttle and centred steering. The live backend keeps the controller neutral
during reset validation and collision quiescence, then applies this initial
action immediately before returning the first observation. The same action is
stored in the observation's previous-action fields. This primes the actuator
without creating an unrecorded simulator transition; the policy may replace
both commands without a delta limit on its first regular step.
For the live configurations, `minimum_speed_mps = 1.3888888889` arms stall
detection after the first successful drive-off. Five consecutive later frames
below 5 km/h then terminate. A separate 25-step startup fallback terminates a
car that never drives off after 1 s at 25 Hz. These settings are recorded under
`any_tyre_armed_stall_v1`; they are not the former 250-step launch grace. Normal
`max_episode_steps` truncation is disabled: an episode ends by lap completion
or an explicit safety termination. Transport timeouts remain adapter errors
and never become successful environment transitions.

After the rollout, the predictor is decoded into twenty ego-frame Bézier
samples. The same frame contains a reference polyline comprising the exact
current projection point followed by the twenty prescribed future reference
points. Every predicted horizon point is projected onto this polyline. The
named `discounted_progress_increments_v1` reward discounts later increments:

```text
reference_path = [current_reference_projection, future_reference_points...]
p_i = project_to_reference_arc(bezier_samples[i], reference_path)
delta_p_i = p_i - p_(i-1), with p_(-1) = 0
discounted_predicted_progress = sum_i discount^i * delta_p_i
```

Origins without a complete observed future horizon in the same fragment are
masked only for auxiliary predictor supervision. Predicted progress itself is
available at every rollout origin because the prediction and fixed future
reference are already known. This is not raw ego-frame `z` displacement:
bends in the reference line contribute their full arc length.
The configured discount is `0.5` and `alpha_pred` is `0.01`, reduced from the
earlier endpoint scale of `0.05`. A discount of one exactly recovers endpoint
progress, which makes the method difference testable. `PredictedProgressReward`
contains no imitation cost or Lagrange update.

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

The network default remains the baseline `initial_log_std = -0.5`. When PPO
loads a checkpoint identified as a behavioural-cloning checkpoint, the
stabilised variant sets `initial_log_std_after_bc = -1.5`, corresponding to
`sigma ~= 0.223`. A resumed PPO checkpoint keeps its learned standard
deviation and is not reset. The named `target_band_learning_rate_v1` controller
measures approximate KL on the complete rollout after each update epoch. It
stops further epochs above the configured upper boundary and adjusts the next
iteration's learning rate: multiply by `0.5` above the band, by `1.1` below the
band (up to the configured base rate), and hold it inside the band. With target
`0.01` and tolerance `0.003`, the logged target band is `[0.007, 0.013]`.

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
6. decode rollout predictions, project every horizon point, and compute the
   discounted sum of reference-progress increments;
7. add predicted-progress rewards to environment rewards;
8. compute termination-aware GAE;
9. update clipped PPO, the value function and the predictor on valid observed
   targets.

`training/targets.py`, `models/bezier.py`, and `rewards/components.py` make
steps 2–8 independently testable with arrays and mock data.

## Traceability matrix

| Thesis item | Public implementation | Verification |
|---|---|---|
| Fixed AI-spline reference and provenance | `ac_integration/ai_spline.py`, `references/factory.py` | parser/hash/mismatch tests |
| Section 2.2.2, 77 observation scalars | `features/schema.py`, `observation.py` | exact name/order/shape tests |
| Section 2.2.2, fixed demonstration Z-score | `features/normalizer.py` | round-trip and schema mismatch tests |
| Bounded absolute action | `control/actions.py`, `models/driver_network.py` | full-switch/bounds/log-probability tests |
| Equation 2.8, reward timing | `rewards/components.py` | component and mask tests |
| Section 2.2.4, network widths | `models/driver_network.py` | output-shape tests |
| Section 2.2.4, observed future targets | `training/targets.py` | interpolation/frame/episode tests |
| Figure 2.3, 10 predictor outputs | `models/bezier.py` | named codec and endpoint tests |
| Section 2.2.5/4.3, on-policy fragments | `training/rollout.py` | stale-version rejection tests |
| Post-rollout optimisation | `training/gae.py`, `training/ppo.py` | terminal bootstrap and mock smoke tests |

## What is not yet validated

Passing unit tests establishes interface and mathematical consistency, not
driving quality. Speed, acceleration x/z, world pose, fixed-reference
projection, track-boundary previews, and the four tyre slip signals have named
live mappings. The backend and reset/control transports are implemented, but
their signs, timing and closed-loop driving still require the documented
low-speed and full-lap validation on the target installation. The 35-epoch BC run passes
the configured held-out predictor gate, but that is not evidence that its
reward coefficient is optimal in closed loop. Reward coefficients, safety
thresholds, and predictor weighting require controlled experiments rather
than being inferred from a smoke run.
