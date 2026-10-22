"""Launch the independent read-only Live / Replay desktop monitor."""

import argparse
from pathlib import Path
import sys
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training.toml"))
    parser.add_argument("--recording", type=Path, help="Open a recording directly in offline Replay mode")
    parser.add_argument("--run", type=Path, help="Follow this run's latest snapshot (otherwise follow the newest run)")
    parser.add_argument("--runs-directory", type=Path, default=Path("runs"))
    parser.add_argument("--recordings-directory", type=Path, default=Path("recordings"))
    args = parser.parse_args()
    try:
        from PySide6.QtWidgets import QApplication
        from tools.diagnostics.ui.main_window import MainWindow
    except ImportError as error:
        parser.exit(1, f"Diagnostics dependencies missing: {error}\nInstall: pip install -e \".[diagnostics]\"\n")
    # All configuration/model reads and inference run in the source worker.
    # A small CPU inference model needs no pool of competing BLAS workers.
    import torch
    torch.set_num_threads(1)
    settings: dict[str, Any] = {
        "config_file": args.config, "run_directory": args.run, "runs_directory": args.runs_directory,
    }
    application = QApplication(sys.argv)
    window = MainWindow(settings, args.recordings_directory, args.recording)
    window.show()
    sys.exit(application.exec())


if __name__ == "__main__":
    main()
