# Continuous integration

GitHub Actions protects the software baseline but is not a training service.
Research training always runs explicitly on the local machine.

## Workflows

| Workflow | Trigger | Responsibility |
| --- | --- | --- |
| Code quality and tests | push, pull request, manual | Ruff lint and format check, MyPy strict, Pylance/Pyright strict, security-oriented Ruff rules, unit tests, integration tests, mock simulations, and coverage |
| Documentation | push, pull request, manual | strict MkDocs build; deployment to GitHub Pages only after a push to `main` |

The test matrix covers supported Python versions on Linux and the primary
Python version on Windows. Assetto Corsa itself, Steam files, a controller,
private telemetry, and downloaded research datasets are not required.

Pyright uses strict diagnostics for production code in `src/` and standard
diagnostics for tests, where dynamic fixtures would otherwise add noise. The
separate in-game app is excluded because Assetto Corsa embeds Python 2.7,
whereas the research package targets modern Python.

Some tests perform a very small number of deterministic optimisation steps on
synthetic or mock data. These are bounded software integration tests: they
write only to temporary directories and do not create a research model,
consume demonstration data, launch Assetto Corsa, or persist run artefacts.

## Explicitly excluded from CI

The workflows must not call:

- `bc-train`;
- `rl-train`;
- live telemetry or controller commands;
- data download or preparation commands;
- experiment tracking or checkpoint upload.

Run actual training locally using the commands and readiness gates described
in [Training](training.md).

## Local equivalent

Before pushing a code change, run:

```powershell
.\.venv\Scripts\python.exe -m pip install -c requirements\tested.txt -e ".[dev,docs]"
.\.venv\Scripts\python.exe -m pytest --cov --cov-report=term-missing
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m pyright --pythonpath .\.venv\Scripts\python.exe
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

The formatter command checks formatting and does not rewrite source files.
To publish the site, configure the repository's **Settings → Pages → Build and
deployment → Source** as **GitHub Actions** once. Pull requests only validate
the documentation; only `main` is deployed.
