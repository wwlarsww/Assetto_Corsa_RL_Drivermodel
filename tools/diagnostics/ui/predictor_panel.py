"""Predictor metrics and errors against reconstructed replay futures."""

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget

from tools.diagnostics.frame import DiagnosticsFrame
from tools.diagnostics.recording import prediction_errors
from tools.diagnostics.ui.widgets import Chart, ValueTable, unavailable_label


class PredictorPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.status = unavailable_label()
        self.values = ValueTable(("Predictor metric", "Value"))
        self.errors = Chart("Replay error per horizon", ("forward", "lateral", "Euclidean"), x_label="horizon [s]", y_label="error [m]")
        self.errors.widget.setMaximumHeight(130)
        self.values.setMinimumHeight(140)
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.values)
        layout.addWidget(self.errors.widget)

    def update_frame(self, frame: DiagnosticsFrame) -> None:
        self.status.setText(f"Checkpoint inference at step {frame.step}; training losses are unavailable live" if frame.predicted_trajectory is not None else "No checkpoint inference available")
        self.values.fill([(name.replace("_", " "), getattr(frame, name)) for name in
                          ("predictor_forward_mae", "predictor_lateral_mae", "mean_predicted_progress", "mean_predicted_lateral_offset")])
        errors = prediction_errors(frame)
        if errors is not None and len(frame.horizons_seconds) == len(errors):
            self.errors.update(np.asarray(frame.horizons_seconds), (errors[:, 0], errors[:, 1], np.linalg.norm(errors, axis=1)))
        else:
            empty = np.empty(0, dtype=np.float64)
            self.errors.update(empty, (empty, empty, empty))
