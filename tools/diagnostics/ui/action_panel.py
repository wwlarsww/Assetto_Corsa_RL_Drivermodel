"""Distinguish checkpoint policy estimates from controls actually observed in AC."""

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from tools.diagnostics.frame import DiagnosticsFrame
from tools.diagnostics.ui.widgets import ValueTable, unavailable_label


class ActionPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.status = unavailable_label()
        self.values = ValueTable(("Action stage", "Steering", "Longitudinal"))
        layout = QVBoxLayout(self)
        chain = QLabel("Policy mean is recomputed from the checkpoint. Controls and deltas are observed in AC.")
        chain.setWordWrap(True)
        layout.addWidget(chain)
        layout.addWidget(self.status)
        layout.addWidget(self.values)

    def update_frame(self, frame: DiagnosticsFrame) -> None:
        self.status.setText("Longitudinal: + throttle / − brake; sampled PPO actions are unavailable" if frame.observed_action is not None else "No observed controls available")
        rows: list[tuple[object, ...]] = []
        for label, field in (("Checkpoint mean (final)", "policy_mean"), ("Previous observed controls", "previous_observed_action"),
                             ("Observed control delta", "observed_action_delta"), ("Current observed controls", "observed_action")):
            value = getattr(frame, field)
            rows.append((label, value[0] if value is not None else None, value[1] if value is not None else None))
        if frame.telemetry:
            rows.extend([("Observed steering / throttle", frame.telemetry.steering_angle, frame.telemetry.gas),
                         ("Observed brake", None, frame.telemetry.brake)])
        self.values.fill(rows)
