"""Traceable run metadata and standard CSV metric persistence."""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from assetto_corsa_rl import __version__
from assetto_corsa_rl.config import ProjectConfig


@dataclass(slots=True)
class ExperimentRecorder:
    """Append-only recorder for one local experiment."""

    run_directory: Path
    _metric_file: Any = field(init=False, default=None, repr=False)
    _writer: csv.DictWriter[str] | None = field(init=False, default=None, repr=False)

    @classmethod
    def create(
        cls,
        *,
        output_directory: str | Path,
        config: ProjectConfig,
        command: list[str],
    ) -> ExperimentRecorder:
        """Create a unique run directory and persist immutable run metadata.

        Args:
            output_directory: Parent directory for the timestamped run.
            config: Fully validated configuration recorded with the run.
            command: Command-line invocation used to reproduce the run.

        Returns:
            Recorder bound to the newly created run directory.
        """

        timestamp = datetime.now(UTC)
        run_id = f"{timestamp:%Y%m%dT%H%M%SZ}-seed-{config.experiment.seed}"
        run_directory = Path(output_directory) / run_id
        run_directory.mkdir(parents=True, exist_ok=False)
        recorder = cls(run_directory=run_directory)
        recorder._write_json(
            "run.json",
            {
                "schema_version": 1,
                "created_at": timestamp.isoformat(),
                "project_version": __version__,
                "command": command,
                "runtime": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                },
                "configuration": config.to_dict(),
            },
        )
        return recorder

    def record_step(self, metrics: dict[str, Any]) -> None:
        """Append one flat step record with a stable schema."""

        if self._writer is None:
            metric_path = self.run_directory / "metrics.csv"
            self._metric_file = metric_path.open("w", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(
                self._metric_file,
                fieldnames=list(metrics),
                extrasaction="raise",
            )
            self._writer.writeheader()
        self._writer.writerow(metrics)
        self._metric_file.flush()

    def write_summary(self, summary: dict[str, Any]) -> None:
        """Persist the final scalar and artefact summary as JSON."""

        self._write_json("summary.json", summary)

    def close(self) -> None:
        """Flush and close the metrics stream idempotently."""

        if self._metric_file is not None:
            self._metric_file.close()
            self._metric_file = None
            self._writer = None

    def __enter__(self) -> ExperimentRecorder:
        return self

    def __exit__(self, *_errors: object) -> None:
        self.close()

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        destination = self.run_directory / name
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
