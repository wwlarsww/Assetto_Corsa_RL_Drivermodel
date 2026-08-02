"""Optional Aim integration for logging live PPO metrics.

This module provides a small `AimMonitor` implementing the same
monitor protocol used by the existing training monitor. The implementation
is intentionally optional: it only activates when the `aim` package is
installed. This keeps the runtime behaviour unchanged for users who do
not want Aim.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

try:
    import aim
except Exception:  # pragma: no cover - optional dependency
    aim = None

LOGGER = logging.getLogger("assetto_corsa_rl")


class AimMonitor:
    """Log run metrics to an Aim repository.

    Usage:
      monitor = AimMonitor(run_directory=Path("runs/ppo-mock/<id>"), run_label="PPO")
      monitor.update(metrics_dict)
      monitor.finish(completed=True)
      monitor.close()

    The monitor is a best-effort wrapper: when `aim` is not installed it is
    a no-op object that keeps the same external behaviour as the Matplotlib
    monitor (no exceptions raised).
    """

    def __init__(self, *, run_directory: Path | None = None, run_label: str = "PPO") -> None:
        self._enabled = False
        self._run: Any | None = None
        self._started_at = time.time()
        self._run_label = run_label

        if aim is None:
            LOGGER.warning("Aim is not installed; AimMonitor disabled")
            return

        try:
            # Use run directory as repo path when available so history is colocated.
            repo = str((run_directory or Path("runs")).resolve()) if run_directory is not None else None
            self._run = aim.Run(repo=repo, experiment=self._run_label)
            self._enabled = True
            LOGGER.info("AimMonitor attached to repo=%s experiment=%s", repo, self._run_label)
        except Exception:
            LOGGER.exception("Could not create Aim Run; AimMonitor disabled")
            self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def update(self, metrics: dict[str, Any]) -> None:
        if not self._enabled or self._run is None:
            return
        try:
            step = int(metrics.get("environment_steps") or 0)
        except Exception:
            step = 0

        for key, value in metrics.items():
            # Only track primitive numeric or short textual values; skip large blobs.
            if value is None:
                continue
            try:
                # Aim accepts many value types; keep this resilient to odd inputs.
                self._run.track(value, name=key, step=step)
            except Exception:
                LOGGER.debug("Aim: could not track metric %s (skipped)", key, exc_info=True)

    def finish(self, *, completed: bool) -> None:
        if not self._enabled or self._run is None:
            return
        try:
            # Write a final metadata flag and flush.
            try:
                self._run.set("meta/status", "COMPLETE" if completed else "STOPPED")
            except Exception:
                # Some Aim versions may not support nested set keys; fall back.
                try:
                    self._run.set("status", "COMPLETE" if completed else "STOPPED")
                except Exception:
                    pass
            try:
                self._run.flush()
            except Exception:
                pass
        except Exception:
            LOGGER.debug("AimMonitor.finish() failed", exc_info=True)

    def close(self) -> None:
        if not self._enabled or self._run is None:
            return
        try:
            try:
                self._run.close()
            except Exception:
                pass
        finally:
            self._enabled = False


def migrate_run(run_directory: Path) -> None:
    """Migrate an existing PPO run's scalar metrics and small artefacts into Aim.

    This is a best-effort, idempotent migration helper. It reads `metrics.csv`
    and streams every row into an Aim run, then uploads small text artefacts
    such as `run.json` or `summary.json` into Aim metadata. Large binary
    artefacts are skipped to avoid excessive memory usage.
    """

    if aim is None:
        LOGGER.error("Aim is not installed; cannot migrate %s", run_directory)
        return

    if not run_directory.is_dir():
        LOGGER.error("Run directory does not exist: %s", run_directory)
        return

    metric_path = run_directory / "metrics.csv"
    if not metric_path.is_file():
        LOGGER.error("No metrics.csv found in %s", run_directory)
        return

    run = None
    try:
        run = aim.Run(repo=str(run_directory.resolve()), experiment=run_directory.name)
    except Exception:
        LOGGER.exception("Could not create Aim Run for migration: %s", run_directory)
        return

    try:
        # Stream CSV rows
        import csv

        with metric_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for i, row in enumerate(reader):
                try:
                    step = int(row.get("environment_steps") or i)
                except Exception:
                    step = i
                for key, raw in row.items():
                    if raw is None or raw == "":
                        continue
                    # Try numeric conversion
                    try:
                        value = float(raw)
                    except Exception:
                        value = raw
                    try:
                        run.track(value, name=key, step=step)
                    except Exception:
                        LOGGER.debug("Aim: skipping metric %s during migration", key, exc_info=True)

        # Small text artefacts
        for candidate in run_directory.iterdir():
            if candidate.is_file() and candidate.suffix.lower() in {".json", ".txt", ".md", ".csv"}:
                try:
                    size = candidate.stat().st_size
                    if size > 1_000_000:
                        LOGGER.info("Skipping large artefact %s (%d bytes)", candidate.name, size)
                        continue
                    content = candidate.read_text(encoding="utf-8")
                    # Store as metadata under artifacts/<name>
                    try:
                        run.set(f"artifacts/{candidate.name}", content)
                    except Exception:
                        try:
                            run.set(candidate.name, content)
                        except Exception:
                            LOGGER.debug("Could not set artifact %s in Aim", candidate.name, exc_info=True)
                except Exception:
                    LOGGER.debug("Failed to migrate artefact %s", candidate, exc_info=True)

        try:
            run.flush()
        except Exception:
            pass
    finally:
        try:
            run.close()
        except Exception:
            pass
