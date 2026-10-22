"""Start the optional desktop diagnostics monitor from the repository root."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.diagnostics.__main__ import main


if __name__ == "__main__":
    main()
