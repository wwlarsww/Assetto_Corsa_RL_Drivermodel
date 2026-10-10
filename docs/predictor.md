# Auxiliary trajectory predictor

The predictor learns future positions in metres, expressed as `(forward, left)`
relative to the vehicle pose at the observation. Horizon times, head sizes and
loss weight come from `[predictor]` in the run's TOML; the template predicts
10 points from 0.1 to 1.0 seconds.

```mermaid
flowchart TD
    O[Observation] --> N[VecNormalize]
    N --> F[Shared feature tensor]
    F --> P[Policy head: delta action]
    P --> A[Add current controls and clip final action]
    F --> V[Value head]
    F --> T[Predictor head: local trajectory]
    T --> D[Detach during collection]
    D --> R[Reference projection and future reward]
    R --> B[Frozen rollout rewards]
    B -->|Returns and advantages| L
    P --> L[PPO loss + weighted masked SmoothL1]
    V --> L
    T --> L
    L --> U[One optimizer updates all heads and shared extractor]
```

`RacingPPO` extends SB3 collection with pose recording and its PPO update with
the auxiliary loss. SB3 still owns rollouts, GAE, clipping and serialization.
Targets use only the current rollout. Each horizon has a mask: missing future
steps and points beyond an episode end are excluded, while shorter valid
horizons still train. The actual terminal endpoint is usable; the auto-reset
pose belongs to a different episode. Horizons must match the configured
control grid; timings currently use nominal control steps.

`RacingActorCriticPolicy` evaluates all three heads from the same feature
tensor. All weights and the common optimizer state live in `model.zip`;
matching normalization statistics remain in `vecnormalize.pkl`.
`PPOAgent.predict_trajectory(raw_observation)` returns detached local points
for inspection. With `[reward.future_prediction].enabled`, the action's shared
features also produce a detached snapshot. `FutureRewardWrapper` projects it
onto the active reference line and adds the [future reward](reward.md) before
Monitor and VecNormalize. Rewards stay fixed during PPO updates; actual future
poses remain supervised targets only. Disabling the extension preserves the
existing step reward and auxiliary loss.

```powershell
# Test data and a fresh, untrained model; checks plumbing, not learned accuracy.
python tools/validate_predictor.py --training-step 50000 --plot

# Inspect a trained run using its own config and normalization.
python tools/validate_predictor.py --checkpoint runs/<run_name>/final --plot
```

`--data selected.npz` accepts raw `observations` shaped `(steps, envs, features)`,
`vehicle_positions`, `vehicle_forwards` and `next_vehicle_positions` shaped
`(steps, envs, 3)`, and boolean `episode_ends` shaped `(steps, envs)`.
Endpoint positions must precede auto-reset. `--sample` selects a starting step;
the tool inspects environment 0 and prints predictions, targets, coordinate
errors, masked loss, each reference projection and all reward terms. With
`--data`, supply the matching `--reference-line path/to/fast_lane.ai`; synthetic
validation uses an independent test reference. `--training-step` inspects the
configured warmup/ramp (otherwise the saved model timestep is used).
`--save-plot trajectory.png` exports the comparison.
No rollout data is saved automatically. Game-free tests cover geometry,
masking, shared gradients, short PPO training and checkpoint restoration;
prediction accuracy still needs validation on real driving data.
