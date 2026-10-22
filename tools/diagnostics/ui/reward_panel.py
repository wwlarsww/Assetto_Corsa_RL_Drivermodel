"""Reward decomposition and total reward history."""

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget

from tools.diagnostics.checks import reward_is_consistent
from tools.diagnostics.frame import DiagnosticsFrame
from tools.diagnostics.ui.widgets import Chart, ValueTable, unavailable_label


class RewardPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.status = unavailable_label()
        self.values = ValueTable(("Reward component", "Value"))
        self.chart = Chart("Total reward", ("total",))
        self.chart.widget.setMaximumHeight(160)
        self.values.setMinimumHeight(160)
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.values)
        layout.addWidget(self.chart.widget)

    def update_frame(self, frame: DiagnosticsFrame, history: tuple[DiagnosticsFrame, ...]) -> None:
        consistent = reward_is_consistent(frame)
        self.status.setText((frame.reward_source or "No reward transition available") +
                            f" · style weight {frame.reward_style_weight:g}" +
                            (" · weighted components consistent" if consistent else " · WARNING: inconsistent decomposition" if consistent is False else ""))
        self.values.fill([(label, getattr(frame.reward, field) if frame.reward else None) for label, field in
                          (("Performance step", "performance_step"), ("Performance look-ahead", "performance_prediction"),
                           ("Style step", "style_step"), ("Style look-ahead", "style_prediction"),
                           ("Termination penalty", "termination_penalty"), ("TOTAL", "total"))])
        self.chart.update(np.asarray([f.timestamp - frame.timestamp for f in history]),
                          (np.asarray([f.reward.total if f.reward else np.nan for f in history]),))
