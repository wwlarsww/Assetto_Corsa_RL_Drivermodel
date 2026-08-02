from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from assetto_corsa_rl.ac_integration.live_validation import (
    validate_live_environment,
    write_live_validation_artifacts,
)
from assetto_corsa_rl.envs.environment import AssettoCorsaEnv


class _LiveEnvironment:
    def __init__(self, *, reset_drift_m: float = 0.0) -> None:
        self.active_reference = SimpleNamespace(track_length_m=1000.0)
        self.reset_drift_m = reset_drift_m
        self.reset_count = 0
        self.packet = 0
        self.progress_m = 100.0

    @staticmethod
    def _info(
        *,
        progress_m: float,
        position_x_m: float,
        speed_mps: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "telemetry": {
                "progress_m": progress_m,
                "lateral_offset_m": 0.0,
                "heading_error_rad": 0.0,
                "speed_mps": speed_mps,
            },
            "position_xy_m": np.asarray([position_x_m, 2.0], dtype=np.float32),
            "vehicle_heading_rad": 0.5,
            "termination_reason": None,
        }

    def reset(self) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        self.reset_count += 1
        drift = self.reset_drift_m * (self.reset_count - 1)
        self.progress_m = 100.0 + drift
        return np.zeros(1, dtype=np.float32), self._info(
            progress_m=self.progress_m,
            position_x_m=1.0 + drift,
        )

    def step(
        self,
        action: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], float, bool, bool, dict[str, Any]]:
        self.packet += 1
        pedal, steering = (float(action[0]), float(action[1]))
        speed_mps = 2.0 if pedal > 0.0 else 0.0
        self.progress_m += speed_mps * 0.1
        info = self._info(
            progress_m=self.progress_m,
            position_x_m=1.0,
            speed_mps=speed_mps,
        )
        info["backend"] = {
            "physics_packet_id": self.packet,
            "graphics_packet_id": self.packet,
            "slip_sequence": self.packet,
            "collision_sequence": 0,
            "simulator_gas": max(pedal, 0.0),
            "simulator_brake": max(-pedal, 0.0),
            "simulator_steer_angle": 10.0 * steering,
            "simulator_gear": 2,
            "simulator_rpms": 2000,
            "front_slip_ratio": 0.0,
            "rear_slip_ratio": 0.0,
        }
        return np.zeros(1, dtype=np.float32), 0.0, False, False, info


def test_live_validation_accepts_repeatable_reset_input_echo_and_transport(
    tmp_path: Path,
) -> None:
    environment = cast(AssettoCorsaEnv, _LiveEnvironment())
    summary, resets, transport = validate_live_environment(
        environment,
        session_origin="attached",
        reset_method_variant="csp_hotlap_spawn_v1",
        reset_count=3,
        soak_steps=5,
        input_hold_steps=1,
    )
    assert summary.passed
    assert summary.unique_physics_packets == 5
    assert summary.unique_slip_packets == 5
    assert summary.reset_position_spread_m == 0.0
    assert summary.drive_off is not None
    assert summary.drive_off.maximum_observed_gear_raw == 2
    assert summary.drive_off.maximum_observed_forward_gear == 1
    assert summary.drive_off.forward_progress_m >= 1.0

    destination = tmp_path / "report"
    write_live_validation_artifacts(destination, summary, resets, transport)
    assert '"passed": true' in (destination / "summary.json").read_text(encoding="utf-8")
    assert (destination / "resets.csv").is_file()
    assert (destination / "transport.csv").is_file()


def test_live_validation_rejects_recovery_reset_drift() -> None:
    environment = cast(AssettoCorsaEnv, _LiveEnvironment(reset_drift_m=30.0))
    summary, _, _ = validate_live_environment(
        environment,
        session_origin="attached",
        reset_method_variant="python_ext_reset_car_v1",
        reset_count=3,
        soak_steps=2,
        input_hold_steps=1,
    )
    assert not summary.passed
    codes = {issue.code for issue in summary.issues}
    assert "position_not_repeatable" in codes
    assert "progress_not_repeatable" in codes


def test_live_validation_can_require_an_automatic_upshift() -> None:
    environment = cast(AssettoCorsaEnv, _LiveEnvironment())
    summary, _, _ = validate_live_environment(
        environment,
        session_origin="attached",
        reset_method_variant="csp_hotlap_spawn_v1",
        reset_count=2,
        soak_steps=2,
        input_hold_steps=1,
        minimum_forward_gear=2,
    )
    assert not summary.passed
    assert "automatic_upshift_missing" in {issue.code for issue in summary.issues}
