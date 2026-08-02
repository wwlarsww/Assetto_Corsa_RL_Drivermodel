# Observation feature reference

The `ai_spline_absolute_v1` input is a flat `float32[77]` vector. This page gives the
human-readable data contract; `features/schema.py` is its machine-readable
counterpart.

## Coordinate and sign conventions

- Geometry and trajectory ego frame: `x` points laterally left and `z` points
  forward. This matches the selected horizontal AC axes; vertical `y` is not
  part of the observation.
- World heading: radians, positive counter-clockwise, wrapped to `[-π, π]`.
- Combined pedal: `+1` is full throttle, `-1` is full brake.
- Steering: normalised command; positive means a left turn.
- Slip ratio: dimensionless. The Kunos Python API sign is preserved.
- Side-slip angle: radians. The Kunos Python API/MoTeC sign is preserved rather
  than inferred from a differently named local-velocity axis.

The named live calibration `vanilla_ac_python_v1` maps wheel order
`FL, FR, RL, RR` to front/rear arithmetic means. `CS.SlipRatio` is used without
unit or sign scaling. `CS.SlipAngle` is a degree value and is multiplied by
`+π/180`; it is not negated or clipped. This matches the public
AssettoCorsaGym observation path, which forwards the four API angle channels
without a sign transformation. The offline MoTeC adapter must use the same
result: average `Tire Slip Angle *` in degrees and multiply by `+π/180`, while
`Tire Slip Ratio *` is stored as percent and must additionally be divided by
`100`.

## Exact order

| Indices | Names | Unit |
|---|---|---|
| `0` | `vehicle.speed_mps` | `m/s` |
| `1` | `vehicle.acceleration_x_mps2` (lateral-left) | `m/s²` |
| `2` | `vehicle.acceleration_z_mps2` (forward) | `m/s²` |
| `3:5` | `vehicle.front_slip_ratio`, `vehicle.rear_slip_ratio` | `1` |
| `5:7` | `vehicle.front_sideslip_angle_rad`, `vehicle.rear_sideslip_angle_rad` | `rad` |
| `7:21` | left `(x, z)` at `5, 10, 20, 40, 80, 160, 320 m` | `m` |
| `21:35` | right `(x, z)` at the same distances | `m` |
| `35:75` | 20 `fast_lane.ai` reference `(x, z)` points | `m` |
| `75` | `action.previous_combined_pedal` | `1` |
| `76` | `action.previous_steering` | `1` |

Immediately after an episode reset these two raw fields are `[+1, 0]`, matching
the primed full-throttle and centred-steering controller state. Subsequent
observations contain the action actually applied by the preceding regular
environment step.

The front/rear side-slip observation features are tyre-level Kunos
`CS.SlipAngle` signals. They are distinct from the vehicle body sideslip beta
used by the body-sideslip termination, which is computed from
calibrated local lateral and forward velocity and is not added to the fixed
77-feature observation schema.

Reference point `k` corresponds to `t = 0.25 * (k + 1) s`; therefore the first
point occupies indices `35:37` and the final point `73:75`. These are always
coordinates sampled from the installed AI spline. They are not the
future human path used as the predictor's supervised target.

## Adapter acceptance checks

Before recording demonstrations or starting RL, verify with a stationary and a
slow manual test:

1. all 77 values are finite and in the exact schema order;
2. speed and accelerations use SI units;
3. left/right boundary signs agree with the ego convention;
4. increasing reference times advance along the baseline line;
5. the reference `z` coordinate advances and lateral `x` bends consistently;
6. throttle, brake, and steering signs match the action convention;
7. offline and live adapters produce the same front/rear axle means, units,
   and preserved Kunos signs for both slip quantities.

Normalisation statistics are accepted only when their feature-name sequence
matches the schema. This prevents a numerically valid but semantically shifted
77-vector from entering the policy.
