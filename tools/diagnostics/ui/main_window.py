"""Read-only desktop monitor. Live and Replay share all diagnostic panels."""

from pathlib import Path
from typing import Any, cast

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QComboBox, QDockWidget, QFileDialog, QGroupBox, QHBoxLayout, QMainWindow, QPushButton, QSplitter, QVBoxLayout, QWidget

from tools.diagnostics.frame import Array
from tools.diagnostics.ui.action_panel import ActionPanel
from tools.diagnostics.ui.analysis_panel import AnalysisPanel
from tools.diagnostics.ui.health_panel import HealthPanel
from tools.diagnostics.ui.observation_panel import ObservationPanel
from tools.diagnostics.ui.predictor_panel import PredictorPanel
from tools.diagnostics.ui.replay_controls import ReplayControls
from tools.diagnostics.ui.reward_panel import RewardPanel
from tools.diagnostics.ui.track_panel import TrackPanel
from tools.diagnostics.ui.vehicle_panel import VehiclePanel
from tools.diagnostics.ui.worker import MonitorUpdate, MonitorWorker


class MainWindow(QMainWindow):
    def __init__(self, live_settings: dict[str, Any], recordings_root: Path, recording: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Assetto Corsa — Diagnostics Monitor (read-only)")
        self.resize(1560, 1000)
        self.mode = QComboBox()
        self.mode.addItems(["LIVE", "REPLAY"])
        record = QPushButton("Start / Stop Lap Recording")
        open_run = QPushButton("Select Run")
        open_run.clicked.connect(self._select_run)
        self.replay_controls = ReplayControls()
        central = QWidget()
        layout = QVBoxLayout(central)
        row = QHBoxLayout()
        row.addWidget(self.mode)
        row.addWidget(record)
        row.addWidget(open_run)
        row.addStretch()
        layout.addLayout(row)
        layout.addWidget(self.replay_controls)
        self.setCentralWidget(central)
        self.vehicle, self.action_panel, self.track = VehiclePanel(), ActionPanel(), TrackPanel()
        self.observations, self.predictor, self.reward, self.health = ObservationPanel(), PredictorPanel(), RewardPanel(), HealthPanel()
        body = QSplitter(Qt.Orientation.Horizontal)
        columns = ((("Vehicle", self.vehicle), ("Actions", self.action_panel), ("Health", self.health)),
                   (("Track / trajectories", self.track), ("Predictor", self.predictor)),
                   (("Observations", self.observations), ("Reward", self.reward)))
        for panels, sizes in zip(columns, ([450, 280, 160], [550, 340], [490, 400])):
            column = QSplitter(Qt.Orientation.Vertical)
            column.setChildrenCollapsible(False)
            for title, panel in panels:
                group = QGroupBox(title)
                QVBoxLayout(group).addWidget(panel)
                column.addWidget(group)
            column.setSizes(sizes)
            body.addWidget(column)
        body.setChildrenCollapsible(False)
        body.setSizes([450, 580, 500])
        layout.addWidget(body, 1)
        self.analysis = AnalysisPanel()
        self.analysis_dock = QDockWidget("Replay lap analysis", self)
        self.analysis_dock.setWidget(self.analysis)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysis_dock)
        self.analysis_dock.hide()
        self.menuBar().addMenu("View").addAction(self.analysis_dock.toggleViewAction())
        self.worker = MonitorWorker(live_settings, recordings_root)
        self.worker.message.connect(self.statusBar().showMessage)
        self.worker.updated.connect(self._receive)
        self.worker.recording_opened.connect(self._opened)
        self.worker.analysis_ready.connect(self._analysis)
        self.mode.currentTextChanged.connect(self._mode_changed)
        record.clicked.connect(lambda: self.worker.command("record"))
        self.replay_controls.open_requested.connect(self._open)
        self.replay_controls.lap_requested.connect(self._lap)
        self.replay_controls.seek_requested.connect(self._seek)
        self.replay_controls.step_requested.connect(self._step)
        self.replay_controls.play_requested.connect(self._play)
        self.replay_controls.speed_requested.connect(self._speed)
        self._pending: MonitorUpdate | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._render)
        self.timer.start()
        self.worker.start()
        self._mode_changed("LIVE")
        if recording:
            self.mode.setCurrentText("REPLAY")
            self.worker.command("open", recording)

    def _mode_changed(self, mode: str) -> None:
        self.replay_controls.setVisible(mode == "REPLAY")
        self.worker.command("mode", mode)

    def _lap(self, index: int) -> None:
        self.worker.command("lap", index)

    def _seek(self, index: int) -> None:
        self.worker.command("seek", index)

    def _step(self, delta: int) -> None:
        self.worker.command("step", delta)

    def _play(self, playing: bool) -> None:
        self.worker.command("play", playing)

    def _speed(self, speed: float) -> None:
        self.worker.command("speed", speed)

    def _open(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Open Recording", "recordings")
        if directory:
            self.worker.command("open", directory)

    def _select_run(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Training Run", "runs")
        if directory:
            self.worker.command("run", directory)

    def _opened(self, names: object) -> None:
        self.mode.setCurrentText("REPLAY")
        self.replay_controls.set_laps(cast(tuple[str, ...], names))
        self.statusBar().showMessage("Recording loaded" if names else "Recording has no complete laps")

    def _analysis(self, payload: object) -> None:
        statistics, x, series = cast(tuple[dict[str, float | int | None], Array, tuple[tuple[Array, ...], ...]], payload)
        self.analysis.update_analysis(statistics, x, series)

    def _receive(self) -> None:
        self._pending = self.worker.consume_update()

    def _render(self) -> None:
        update, self._pending = self._pending, None
        if update is None:
            return
        frame = update.frame
        self.vehicle.update_frame(frame, update.history)
        self.action_panel.update_frame(frame)
        self.track.update_frame(frame)
        self.observations.update_frame(frame)
        self.predictor.update_frame(frame)
        self.reward.update_frame(frame, update.history)
        self.health.update_checks(update.checks)
        self.replay_controls.update_position(update.index, update.count, update.playing)
        if frame.status:
            self.statusBar().showMessage(frame.status)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.timer.stop()
        self.worker.stop()
        super().closeEvent(event)
