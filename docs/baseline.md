# `ai_spline_absolute_v1`

`ai_spline_absolute_v1` is the executable research method. Its initial dataset
variant uses the BMW Z4 GT3, Monza, one fixed AI-spline reference, and human
MoTeC demonstrations for the speed profile and behavioural-cloning warm start.

## Contract at a glance

| Part | Method contract |
|---|---|
| geometry source | installed track-layout `ai/fast_lane.ai`, binary version 7 |
| reference | AI-spline centreline plus human-derived target speed |
| boundaries | spline left/right side distances; no configured constant width |
| observation | 77 static-Z-score-normalised `float32` values |
| action | absolute combined pedal and steering, each in `[-1, 1]` |
| action rate | unrestricted; `-1` to `+1` is valid in one agent step |
| policy | tanh-squashed Gaussian with consistent bounded-action log probability |
| reward | actual reference progress + predicted reference progress + safety penalty |
| crash | latched CSP `ac.onCarCollision(0, callback)` event terminates immediately |
| learning | joint policy/predictor BC warm start, then synchronous PPO |

There is no style, imitation, ProMP, damage, vehicle-dimension, or inferred
wall-collision term in this method.

## Geometry provenance

`build-reference`, BC export, live reference loading, progress projection,
reference features, and boundary previews all use the same configured
`fast_lane.ai`. The parser obtains centre points, cumulative distance, and
left/right side distances directly from the file. It computes the closing
segment to obtain lap length and rejects unsupported versions or malformed
metadata.

The reference artefact stores the source file's SHA-256. Loading fails if that
digest or the derived lap length differs from the currently configured track.
No live track length or width is copied into TOML. The similarly named mock
length and half-width settings apply only to deterministic test dynamics.

No proprietary track file is copied into this repository or a run artefact.

## Environment step

1. The policy emits `[combined_pedal, steering]` in `[-1, 1]`.
2. The action adapter validates and clips only numerical boundary overshoot;
   it does not integrate or rate-limit the command.
3. The backend holds the absolute command for one control period and returns a
   fresh telemetry frame.
4. The environment computes wrap-safe progress along the AI-spline reference
   and a safety termination penalty when applicable.
5. The next observation contains the latest absolute command.

For the pedal, `+1` is full throttle and `-1` is full brake. Steering `+1` is
full left in the policy convention; the live X360 adapter applies the explicit
configured output sign.

## Collision timing

The CSP Lua app registers `ac.onCarCollision(0, callback)`. Its callback
increments a persistent unsigned sequence, while a small state file is
refreshed independently of the agent loop. At reset the backend records the
current sequence as the episode baseline. Any later sequence change is still
visible at the next 20/25 Hz agent step even if the collision happened between
two steps, and produces `TerminationReason.CRASH` immediately.

The signal covers wall and car contacts. Damage level, bounding boxes, vehicle
width/length, and geometric wall intersection are deliberately not consulted.

## PPO iteration

1. A frozen policy version collects one rollout fragment.
2. Each transition stores its bounded action and matching squashed-distribution
   log probability.
3. Complete future paths are interpolated at twenty `0.25 s` offsets; targets
   crossing an episode or fragment boundary are masked.
4. The network's ten Bézier parameters are decoded into twenty path points.
5. Predicted endpoint progress on the same reference is added after collection
   and before GAE.
6. GAE and clipped PPO update policy/value heads plus the auxiliary predictor.
7. The policy version increments only after optimisation; stale fragments are
   rejected.

## Safe extension rule

This method name freezes action semantics, geometry provenance, observation
order, reward terms, network codec, and target construction. A change to one of
those contracts requires a new method name and configuration. In particular,
legacy `fixed_reference_v1` delta-action checkpoints are incompatible and are
rejected rather than silently loaded.
