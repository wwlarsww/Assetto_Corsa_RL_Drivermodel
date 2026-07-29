# Reproducibility

Reproducibility is a property of the complete experiment, not only a random
seed.

## Recorded automatically

`mock-run`, BC and PPO training write:

- the complete validated TOML-derived configuration;
- seed, command, UTC creation time and project version;
- Python, OS and machine metadata;
- flat CSV step metrics;
- a JSON episode summary.

Production training should add package lock hashes, Git commit and dirty state,
device/CUDA details, dataset/reference identifiers, normalisation artefact hash
and Assetto Corsa/car/track/setup versions.

The repository's `requirements/tested.txt` constrains direct dependencies to
the versions used by the latest verification. It is a cross-platform tested
baseline, not a hash-locked replacement for an experiment-specific CPU, CUDA
or ROCm environment lock.

## Deterministic evaluation

For `fixed_reference_no_promp_v1`, use the one prescribed reference,
deterministic policy mean and a fixed checkpoint. Keep the vehicle, setup,
circuit, simulation conditions and hyperparameters constant unless one is the
explicit independent variable.

Training evaluation and final inference campaigns have different purposes:

- fixed-reference evaluation tracks learning and checkpoint quality;
- sampled-reference inference belongs to a later named ProMP experiment.

Always report lap time together with reference offset. For setup studies,
include transfer tests in which a policy trained on one setup is evaluated on
other setups.

## Asynchronous variance

Independent actors change fragment arrival order even with identical seeds.
Policy-version checks preserve on-policy data but do not make actor scheduling
deterministic. Quantify this with repeated full runs and treat small isolated
differences cautiously.

## Checkpoints

`save_checkpoint` stores a versioned schema with model state, optimiser state,
environment-step count, iteration, policy version and complete configuration.
Only load checkpoints from trusted sources; model serialisation is not a
security boundary.

PPO checkpoints are written after each synchronous update. The stored policy
version is the version used by the next collector; fragments carrying an older
version are rejected.
