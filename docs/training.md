# Training workflow

The executable baseline is `fixed_reference_no_promp_v1`: one Dallara F317,
one Barcelona GP reference line, no ProMP sampling and no imitation reward.

## Safe preflight

Run the non-mutating check before every training campaign:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl rl-ready `
  --config configs\f317_barcelona_fixed.toml `
  --backend assetto-corsa
```

Exit code `0` means ready; exit code `4` means at least one required condition
failed. The report checks the fixed reference, BC tensors and checkpoint,
predictor validation ADE/FDE, game and bridge installation, analog input
profile, controller package, rollout horizon and live-backend availability. It
does not change game or Windows settings.

The initial checkpoint is copied locally to
`data/processed/f317_barcelona/bc_checkpoint.pt`; generated data and
checkpoints remain ignored by Git.

## PPO smoke training

The complete learning path can be exercised without the game:

```powershell
.\.venv\Scripts\python.exe -m assetto_corsa_rl rl-train `
  --config configs\mock.toml `
  --backend mock `
  --device cpu
```

Each synchronous iteration performs this fixed sequence:

1. collect a fragment tagged with the current policy version;
2. reject stale fragments at the collector boundary;
3. build actual future-path targets without crossing episodes;
4. decode the 10-output `bezier-5x2` predictor;
5. project its endpoint onto the current fixed reference and add predicted
   progress after collection;
6. compute GAE with separate termination and time-limit semantics;
7. optimise clipped PPO, the value head and the auxiliary predictor;
8. increment the policy version and atomically save a checkpoint.

Metrics and the full configuration are stored below `runs/ppo-<backend>/`.
Time-limit truncations bootstrap from their final observation but stop the GAE
trace; true terminations do not bootstrap.

## Live-control prerequisites

Live control uses a future `SimulatorBackend` around an Xbox 360 virtual
controller. Install its optional package only when you are ready to approve
the signed ViGEmBus Windows driver installer:

```powershell
.\.venv\Scripts\python.exe -m pip install `
  -c requirements\tested.txt `
  -e ".[live]"
```

The optional dependency and driver behaviour follow the
[vgamepad package documentation](https://pypi.org/project/vgamepad/).

Then configure Assetto Corsa for the generated X360 controller and verify
steering, throttle and brake signs at low speed. Keyboard input is not accepted
for RL because it cannot represent the continuous commands used by the policy.

The live training command stays deliberately blocked until the write-capable
backend, fresh-frame synchronisation and automatic episode reset have passed
their tests. Read-only telemetry and the slip bridge alone are insufficient
authority to control the car safely.
