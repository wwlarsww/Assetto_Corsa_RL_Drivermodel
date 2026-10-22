"""Replay navigation; signals only, no file reads on the GUI thread."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget


class ReplayControls(QWidget):
    open_requested = Signal()
    lap_requested = Signal(int)
    seek_requested = Signal(int)
    step_requested = Signal(int)
    play_requested = Signal(bool)
    speed_requested = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.laps = QComboBox()
        self.speed = QComboBox()
        self.speed.addItems(["1x", "2x", "5x"])
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.position = QLabel("No recording loaded")
        row = QHBoxLayout()
        open_button = QPushButton("Open Recording")
        open_button.clicked.connect(self.open_requested.emit)
        row.addWidget(open_button)
        row.addWidget(QLabel("Lap:"))
        row.addWidget(self.laps)
        for label, offset in (("◀ Frame", -1), ("Frame ▶", 1)):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, delta=offset: self.step_requested.emit(delta))
            row.addWidget(button)
        play, pause = QPushButton("▶ Play"), QPushButton("Pause")
        play.clicked.connect(lambda: self.play_requested.emit(True))
        pause.clicked.connect(lambda: self.play_requested.emit(False))
        row.addWidget(play)
        row.addWidget(pause)
        row.addWidget(self.speed)
        self.laps.currentIndexChanged.connect(self.lap_requested.emit)
        self.timeline.valueChanged.connect(self.seek_requested.emit)
        self.speed.currentTextChanged.connect(self._speed_changed)
        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(self.timeline)
        layout.addWidget(self.position)

    def _speed_changed(self, text: str) -> None:
        self.speed_requested.emit(float(text[:-1]))

    def set_laps(self, names: tuple[str, ...]) -> None:
        self.laps.blockSignals(True)
        self.laps.clear()
        self.laps.addItems(names)
        self.laps.blockSignals(False)

    def update_position(self, index: int, count: int, playing: bool) -> None:
        self.timeline.blockSignals(True)
        self.timeline.setRange(0, max(0, count - 1))
        self.timeline.setValue(index)
        self.timeline.blockSignals(False)
        self.position.setText(f"Frame {index + 1}/{count} · {'Playing' if playing else 'Paused'}" if count else "No recording loaded")
