"""Render checks calculated by the source worker."""

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from tools.diagnostics.checks import HealthCheck


class HealthPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.values = QListWidget()
        QVBoxLayout(self).addWidget(self.values)

    def update_checks(self, checks: tuple[HealthCheck, ...]) -> None:
        self.values.clear()
        for check in checks:
            item = QListWidgetItem(f"{check.level.upper()}: {check.message}")
            item.setForeground(QColor("#a34e00" if check.level == "warning" else "#237531" if check.level == "ok" else "#555555"))
            self.values.addItem(item)
