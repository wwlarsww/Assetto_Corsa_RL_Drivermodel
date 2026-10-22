"""Whole-lap statistics and progress plots prepared by the replay worker."""

from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from tools.diagnostics.frame import Array
from tools.diagnostics.ui.widgets import Chart, ValueTable


class AnalysisPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.values = ValueTable(("Lap statistic", "Value"))
        self.charts = [Chart(title, names, x_label="unwrapped track progress [m]", y_label=unit) for title, names, unit in
                       (("Speed", ("speed",), "km/h"), ("Reward", ("total",), "reward / step"),
                        ("Lateral offset", ("offset",), "m"), ("Predictor error", ("forward MAE", "lateral MAE"), "m"),
                        ("Steering", ("steering",), "normalized"), ("Pedals", ("throttle", "brake"), "normalized"))]
        layout = QVBoxLayout(self)
        layout.addWidget(self.values)
        plots = QGridLayout()
        for index, chart in enumerate(self.charts):
            plots.addWidget(chart.widget, index // 2, index % 2)
        layout.addLayout(plots, 3)

    def update_analysis(self, statistics: dict[str, float | int | None], x: Array, series: tuple[tuple[Array, ...], ...]) -> None:
        self.values.fill([(key.replace("_", " "), value) for key, value in statistics.items()])
        for chart, values in zip(self.charts, series):
            chart.update(x, values)
