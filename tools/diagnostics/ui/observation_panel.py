"""Raw and normalized features with invalid/clipped value highlights."""

import numpy as np
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from tools.diagnostics.frame import DiagnosticsFrame
from tools.diagnostics.ui.widgets import ValueTable


class ObservationPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.values = ValueTable(("Feature", "Raw", "Normalized", "Check"))
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Amber: NaN/Inf, |raw| > 10000 or near VecNormalize clipping"))
        layout.addWidget(self.values)

    def update_frame(self, frame: DiagnosticsFrame) -> None:
        raw, normalized = frame.raw_observation, frame.normalized_observation
        count = max(len(raw) if raw is not None else 0, len(normalized) if normalized is not None else 0)
        rows: list[tuple[object, ...]] = []
        warnings: set[int] = set()
        for index in range(count):
            r = float(raw[index]) if raw is not None and index < len(raw) else None
            n = float(normalized[index]) if normalized is not None and index < len(normalized) else None
            reason = ""
            if any(v is not None and not np.isfinite(v) for v in (r, n)):
                reason = "NaN/Inf"
            elif r is not None and abs(r) > 10_000:
                reason = "large"
            elif n is not None and frame.clip_observations is not None and index < count - 2 and abs(n) >= frame.clip_observations * 0.98:
                reason = "near clipping"
            if reason:
                warnings.add(index)
            name = frame.observation_names[index] if index < len(frame.observation_names) else f"feature_{index}"
            rows.append((name, r, n, reason))
        if not rows:
            rows.append(("No observation available", None, None, ""))
        self.values.fill(rows, warnings)
