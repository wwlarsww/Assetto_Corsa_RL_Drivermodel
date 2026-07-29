# Contributing

Thank you for improving `assetto-corsa-rl-driver`.

## Development setup

```bash
python -m venv .venv
python -m pip install -c requirements/tested.txt -e ".[dev]"
pytest
ruff check .
ruff format --check .
mypy
```

Keep changes simulator-independent where possible. New integrations should
implement `SimulatorBackend` instead of leaking transport-specific details into
the environment, reward or model modules.

## Research changes

For changes to observations, rewards or training behaviour:

1. state the hypothesis and the affected configuration;
2. preserve the baseline method or introduce a new named variant;
3. record seeds, package versions and full configurations;
4. report lap time and reference tracking together;
5. include unit tests for deterministic transformations.

Do not commit game files, proprietary telemetry, driver data, checkpoints,
credentials or the local thesis PDF.

## Pull requests

Use focused commits, add or update tests and document public API changes. CI
must pass before review. By contributing, you agree that your work is licensed
under the repository's MIT license.
