"""Vehicle state and bounded telemetry history."""

import numpy as np
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from tools.diagnostics.frame import DiagnosticsFrame
from tools.diagnostics.ui.widgets import Chart, ValueTable


class VehiclePanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.values = ValueTable(("Vehicle / static state", "Value"))
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.speed = Chart("Speed", ("km/h",))
        self.controls = Chart("Observed controls", ("steering", "throttle", "brake"))
        self.speed.widget.setMaximumHeight(115)
        self.controls.widget.setMaximumHeight(115)
        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(self.values, 2)
        layout.addWidget(self.speed.widget)
        layout.addWidget(self.controls.widget)

    def update_frame(self, frame: DiagnosticsFrame, history: tuple[DiagnosticsFrame, ...]) -> None:
        telemetry = frame.telemetry
        rows: list[tuple[object, ...]] = []
        if telemetry is not None:
            self.summary.setText(f"Speed {telemetry.speed_kmh:.1f} km/h | RPM {telemetry.rpm} | AC gear {telemetry.gear}\n"
                                 f"Steering {telemetry.steering_angle:+.3f} | throttle {telemetry.gas:.3f} | brake {telemetry.brake:.3f}\n"
                                 f"Acceleration long/lat {telemetry.acceleration_g[2]:+.3f} / {telemetry.acceleration_g[0]:+.3f} g | "
                                 f"yaw {telemetry.local_angular_velocity[1]:+.3f} rad/s")
            for name in ("speed_kmh", "rpm", "gear", "steering_angle", "gas", "brake", "fuel",
                         "local_velocity", "acceleration_g", "local_angular_velocity", "wheel_slip",
                         "wheel_angular_speed", "tyre_contact_points", "car_coordinates", "car_damage"):
                rows.append((name, getattr(telemetry, name)))
            rows.extend([("longitudinal acceleration [g]", telemetry.acceleration_g[2]),
                         ("lateral acceleration [g]", telemetry.acceleration_g[0]),
                         ("yaw rate [rad/s]", telemetry.local_angular_velocity[1])])
        else:
            self.summary.setText("No Assetto Corsa telemetry available")
            rows.append(("Status", "No Assetto Corsa telemetry available"))
        if frame.static_info:
            self.summary.setText(self.summary.text() + f"\n{frame.static_info.track_name} / {frame.static_info.car_model}")
            for name in ("car_model", "track_name", "track_configuration", "max_rpm", "max_fuel", "tyre_radius", "track_spline_length"):
                rows.append((name, getattr(frame.static_info, name)))
        self.values.fill(rows)
        x = np.asarray([f.timestamp - frame.timestamp for f in history])
        series = tuple(np.asarray([getattr(f.telemetry, name) if f.telemetry else np.nan for f in history], dtype=np.float64)
                       for name in ("speed_kmh", "steering_angle", "gas", "brake"))
        self.speed.update(x, series[:1])
        self.controls.update(x, series[1:])
