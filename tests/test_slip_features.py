from __future__ import annotations

import math

import pytest

from assetto_corsa_rl.ac_integration.slip_bridge import SlipBridgePacket
from assetto_corsa_rl.ac_integration.slip_features import (
    VANILLA_AC_SLIP_CALIBRATION_NAME,
    vanilla_ac_slip_features,
)


def test_vanilla_ac_slip_calibration_aggregates_and_converts_without_clipping() -> None:
    packet = SlipBridgePacket(
        sequence=1,
        source_time_s=10.0,
        received_monotonic_s=20.0,
        slip_ratio=(0.1, 0.3, -0.2, 0.2),
        slip_angle_raw=(10.0, 14.0, -8.0, -12.0),
    )
    features = vanilla_ac_slip_features(packet)
    assert VANILLA_AC_SLIP_CALIBRATION_NAME == "vanilla_ac_python_v1"
    assert features.front_slip_ratio == pytest.approx(0.2)
    assert features.rear_slip_ratio == pytest.approx(0.0)
    assert features.front_sideslip_angle_rad == pytest.approx(12.0 * math.pi / 180.0)
    assert features.rear_sideslip_angle_rad == pytest.approx(-10.0 * math.pi / 180.0)


def test_vanilla_ac_slip_calibration_preserves_large_ratio_for_safety_logic() -> None:
    packet = SlipBridgePacket(
        sequence=2,
        source_time_s=10.0,
        received_monotonic_s=20.0,
        slip_ratio=(0.0, 0.0, 7.0, 8.0),
        slip_angle_raw=(0.0, 0.0, 0.0, 0.0),
    )
    assert vanilla_ac_slip_features(packet).rear_slip_ratio == 7.5
