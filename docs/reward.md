# Reward

The reward function evaluates how well the agent progresses along the track while following the reference trajectory.

The current implementation consists of:

- a **performance reward** for forward progress
- a **style imitation reward** for staying close to the reference line
- a **termination penalty** when an episode ends because of an undesirable state

Prediction-based reward terms are already reserved for a future auxiliary trajectory predictor.

---

## Reward Structure

```mermaid
flowchart TD

    R[Total Reward]

    P[Performance Reward]
    S[Style / Imitation Reward]
    T[Termination Penalty]

    R --> P
    R --> S
    R --> T

    P --> PS[Step Performance]
    P --> PP[Prediction Performance - future]

    S --> SS[Step Style]
    S --> SP[Prediction Style - future]

    PS --> DS[Progress since previous step]
    PP --> PD[Predicted future progress]

    SS --> LO[Current distance to reference line]
    SP --> FP[Predicted distances to reference line]

    T --> OT[Offtrack]
    T --> ST[Stuck / Wrong Way]
    T --> CO[Strong Collision]
```

---

## Current Reward

The current reward is

\[
r_t =
r_{p,t}
+
\lambda_s r_{s,t}
+
p_{term}
\]

with:

- \(r_{p,t}\): performance reward
- \(r_{s,t}\): style imitation reward
- \(\lambda_s\): style weight
- \(p_{term}\): termination penalty

---

## 1. Performance Step Reward

The performance reward is based on the signed progress along the closed reference spline between two consecutive agent steps.

```text
delta_s = current_s - previous_s
```

Positive progress produces a positive reward, while backwards movement produces a negative reward.

Start/finish wraparound is corrected automatically so that crossing the lap boundary does not produce an artificial jump.

The reward term is:

\[
r_{p,t} = w_p \Delta s_t
\]

where:

- \(\Delta s_t\) is the signed progress in metres
- \(w_p\) is the performance weight

This term directly rewards actual progress along the reference line rather than vehicle speed alone.

---

## 2. Style Imitation Reward

The style reward encourages the vehicle to remain close to the reference trajectory.

The input is the lateral offset from the spline:

```text
lateral_offset = projection.lateral_offset
```

Only the magnitude of the offset matters, so deviations to the left and right are treated equally.

A Gaussian reward is used:

\[
r_{s,t}
=
\exp\left(
-\frac{d_t^2}{2\sigma_s^2}
\right)
\]

where:

- \(d_t\) is the lateral distance to the reference line
- \(\sigma_s\) controls how quickly the reward decreases with distance

The unweighted style score satisfies:

```text
0 < style_step <= 1
```

and reaches its maximum directly on the reference line.

Example for `style_sigma = 1.0`:

| Lateral distance | Style score |
|---:|---:|
| `0.0 m` | `1.000` |
| `0.5 m` | `0.882` |
| `1.0 m` | `0.607` |
| `2.0 m` | `0.135` |
| `3.0 m` | `0.011` |

The contribution to the total reward is weighted by `style_weight`.

---

## 3. Termination Penalty

If the current step terminates the episode, an additional penalty is added.

Current termination reasons are:

| Reason | Description |
|---|---|
| `OFF_TRACK` | At least one tyre contact point is outside the track boundaries. |
| `STUCK_OR_WRONG_WAY` | The vehicle makes insufficient net forward progress over a configurable time window. |
| `STRONG_COLLISION` | A strong collision has been detected externally. |

The exact penalty values are configurable in `RewardCalculator`.

---

## Reward Breakdown

Instead of returning only one scalar value, the implementation exposes the individual reward components:

```python
@dataclass(frozen=True)
class Reward:
    total: float

    performance_step: float
    style_step: float

    performance_prediction: float
    style_prediction: float

    termination_penalty: float
```

This makes reward tuning and training diagnostics easier because each component can be logged independently.

---

## Future Predictor Extension

A future auxiliary trajectory predictor will estimate several future vehicle positions.

The reward structure is already prepared for two additional terms:

- `performance_prediction`
- `style_prediction`

The intended formulation is:

\[
r_t =
r_{p,t}
+
\hat r_{p,t}
+
\lambda_s
\left(
r_{s,t}
+
\hat r_{s,t}
\right)
+
p_{term}
\]

The future predicted points can be projected onto the same reference spline and evaluated using the same progress and distance concepts as the current step.

Until the predictor is implemented:

```text
performance_prediction = 0.0
style_prediction       = 0.0
```

---

## Design Principle

The reward system remains independent of the simulator interface.

It consumes already derived quantities such as:

- signed progress
- lateral offset
- termination result

This keeps simulator-specific data handling separate from reward logic and makes the reward function easy to test.
