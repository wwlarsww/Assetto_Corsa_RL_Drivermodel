"""Optional offscreen smoke test; no simulator and no Qt imports during collection."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(importlib.util.find_spec("PySide6") is None or importlib.util.find_spec("pyqtgraph") is None,
                    reason="Optional diagnostics dependencies are not installed")
def test_desktop_live_replay_navigation_and_all_panels_offscreen(tmp_path: Path) -> None:
    script = r'''
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import sys, time
from pathlib import Path
from dataclasses import replace
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
sys.path.insert(0, "tests")
from test_diagnostics import frame
from tools.diagnostics.recording import save_lap
from tools.diagnostics.recorder import DiagnosticsRecorder
from tools.diagnostics.sources import LiveSource
from tools.diagnostics.ui.main_window import MainWindow

LiveSource._read_vehicle = lambda self, now: (None, None, "No Assetto Corsa telemetry available")
root = Path(sys.argv[1])
recorder = DiagnosticsRecorder(root, "track", "car")
frames = tuple(replace(frame(100.0 + i * 0.05), position=(i * 0.05, 0.0, 0.0),
                       left_boundary=np.asarray([[5., 5.], [20., 5.], [40., 8.]]),
                       right_boundary=np.asarray([[5., -5.], [20., -5.], [40., -2.]]),
                       reference_trajectory=np.asarray([[5., 0.], [20., 0.], [40., 3.]]),
                       raw_observation=np.arange(10, dtype=float), normalized_observation=np.arange(10, dtype=float))
               for i in range(30))
save_lap(recorder.directory / "lap_001.npz", frames)
recorder.close()
application = QApplication([])
window = MainWindow({"ac_directory": root, "runs_directory": root / "runs"}, root)
errors = []
def exception_hook(*args):
    errors.append(str(args[1]))
sys.excepthook = exception_hook
window.show()
def load():
    assert window.health.values.count() > 0
    assert "No Assetto Corsa" in window.vehicle.values.item(0, 1).text()
    window.mode.setCurrentText("REPLAY")
    window.worker.command("open", recorder.directory)
def navigate():
    assert window.replay_controls.laps.count() == 1
    assert window.observations.values.rowCount() == 10
    assert window.reward.values.rowCount() == 6
    assert window.track.curves["left_boundary"].xData.size == 3
    window.worker.command("seek", 7)
def inspect():
    assert window.replay_controls.timeline.value() == 7
    assert window.analysis.values.rowCount() >= 10
    assert not errors, errors
    window.close()
    application.quit()
QTimer.singleShot(200, load)
QTimer.singleShot(800, navigate)
QTimer.singleShot(1400, inspect)
QTimer.singleShot(4000, application.quit)
application.exec()
assert not errors, errors
assert not window.isVisible()
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
