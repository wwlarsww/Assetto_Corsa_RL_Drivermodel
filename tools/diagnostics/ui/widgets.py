"""Small shared plot/table widgets. Data preparation remains outside Qt."""

import importlib
from typing import Any, cast

import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QWidget

from tools.diagnostics.frame import Array

# PyQtGraph has no complete public type stubs; contain that boundary here.
pg: Any = importlib.import_module("pyqtgraph")
pg.setConfigOptions(antialias=False, background="#20242b", foreground="#dddddd")
COLORS = ("#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#e3b341", "#ffffff")


def display(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    return str(value)


class ValueTable(QTableWidget):
    def __init__(self, columns: tuple[str, ...], parent: QWidget | None = None) -> None:
        super().__init__(0, len(columns), parent)
        self.setHorizontalHeaderLabels(columns)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(22)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    def fill(self, rows: list[tuple[object, ...]], warnings: set[int] | None = None) -> None:
        self.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = self.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.setItem(row, column, item)
                item.setText(display(value))
                item.setToolTip(display(value))
                item.setBackground(QColor("#66451a") if warnings and row in warnings else QColor("transparent"))


class Chart:
    def __init__(self, title: str, labels: tuple[str, ...], *, x_label: str = "time [s]", y_label: str = "") -> None:
        self.plot: Any = pg.PlotWidget(title=title)
        self.widget = cast(QWidget, self.plot)
        self.widget.setMinimumHeight(95)
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", y_label)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.addLegend(offset=(5, 5))
        self.curves: list[Any] = [self.plot.plot(pen=pg.mkPen(COLORS[index % len(COLORS)], width=1.5), name=label,
                                               connect="finite") for index, label in enumerate(labels)]

    def update(self, x: Array, series: tuple[Array, ...]) -> None:
        for curve, values in zip(self.curves, series):
            curve.setData(x, values, connect="finite")


def unavailable_label() -> QLabel:
    label = QLabel("No checkpoint inference available")
    label.setWordWrap(True)
    return label
