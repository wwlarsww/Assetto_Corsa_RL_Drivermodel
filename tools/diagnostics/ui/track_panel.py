"""Vehicle-relative XY geometry shared by live and replay."""

from typing import Any, cast

import numpy as np
from PySide6.QtWidgets import QCheckBox, QLabel, QVBoxLayout, QWidget

from tools.diagnostics.frame import DiagnosticsFrame
from tools.diagnostics.ui.widgets import COLORS, display, pg


class TrackPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.centered = QCheckBox("Keep vehicle centered")
        self.values = QLabel()
        self.values.setWordWrap(True)
        self.plot: Any = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.setLabel("bottom", "left [m] (+ to the left)")
        self.plot.getViewBox().invertX(True)
        self.plot.setLabel("left", "forward [m]")
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.curves: dict[str, Any] = {}
        for i, (field, label) in enumerate((("left_boundary", "Left boundary"), ("right_boundary", "Right boundary"),
                                         ("reference_trajectory", "Reference"), ("predicted_trajectory", "Predicted"),
                                         ("actual_future_trajectory", "Actual future"))):
            self.curves[field] = self.plot.plot(pen=pg.mkPen(COLORS[i], width=2), name=label, connect="finite")
        self.plot.plot([0.0], [0.0], pen=None, symbol="t1", symbolSize=14, symbolBrush="w", name="Vehicle")
        layout = QVBoxLayout(self)
        layout.addWidget(self.values)
        layout.addWidget(self.centered)
        layout.addWidget(cast(QWidget, self.plot), 1)

    def update_frame(self, frame: DiagnosticsFrame) -> None:
        self.values.setText(f"Progress {display(frame.progress)} m | offset {display(frame.lateral_offset)} m | "
                            f"left/right clearance {display(frame.left_clearance)} / {display(frame.right_clearance)} m")
        extent = 20.0
        for field, curve in self.curves.items():
            points = getattr(frame, field)
            if points is None or not points.size:
                curve.setData([], [])
                continue
            curve.setData(points[:, 1], points[:, 0], connect="finite")
            finite = points[np.isfinite(points)]
            if finite.size:
                extent = max(extent, float(np.max(np.abs(finite))))
        if self.centered.isChecked():
            self.plot.setRange(xRange=(-extent, extent), yRange=(-extent, extent), padding=0.03)
        else:
            self.plot.enableAutoRange()
