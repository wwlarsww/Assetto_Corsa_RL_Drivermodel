# Repository guidance

This is research software for an RL race-driver model.

- Preserve the separation between simulator I/O, Gymnasium environment logic,
  deterministic feature/reward transformations and learning code.
- Keep the real Assetto Corsa integration optional; the full test suite must run
  against the mock backend.
- Do not silently change the thesis baseline. Add a named configuration or
  method variant and document the difference.
- Keep prediction-based reward terms in rollout post-processing, before
  advantage estimation, rather than inside simulator dynamics.
- PPO fragments must carry a policy version and stale fragments must be
  rejected.
- Never commit proprietary game assets, private telemetry, thesis PDFs,
  credentials, large checkpoints or generated run artefacts.
- Run `pytest`, `ruff check .`, `ruff format --check .` and `mypy` for code
  changes.
