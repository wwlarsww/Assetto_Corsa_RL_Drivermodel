# Assetto Corsa RL Driver

This project is a reproducible research scaffold for an autonomous racing
driver in the original Assetto Corsa. It separates simulator I/O, deterministic
feature and reward transformations, Gymnasium environment logic, and learning
code.

The current baseline uses one prescribed reference line for the BMW Z4 GT3 on
Monza. It supports behavioural cloning and a guarded PPO pipeline while
keeping every central component testable against deterministic mock data.

## Start here

- [Methodology](methodology.md) maps the thesis method to the implementation.
- [Feature reference](feature-reference.md) defines the 77-value observation.
- [Data pipeline](data-pipeline.md) explains demonstration and reference data.
- [Training](training.md) covers local BC and PPO commands.
- [Architecture](architecture.md) explains component and data-flow boundaries.
- [Continuous integration](continuous-integration.md) defines what GitHub
  Actions may and may not execute.

!!! note
    Behavioural-cloning and PPO research runs are local-only. GitHub Actions
    performs code-quality checks, deterministic tests, and documentation
    publishing; it never launches an experiment or controls Assetto Corsa.
