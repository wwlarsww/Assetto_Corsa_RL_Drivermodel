"""Deterministic conversion of Kunos tyre and body-slip signals."""

from __future__ import annotations

import math
from dataclasses import dataclass

from assetto_corsa_rl.ac_integration.slip_bridge import SlipBridgePacket, WheelValues

VANILLA_AC_SLIP_CALIBRATION_NAME = "vanilla_ac_python_v1"
VANILLA_AC_SLIP_ANGLE_RAW_UNIT = "degree"
VANILLA_AC_SLIP_ANGLE_RAD_PER_RAW = math.pi / 180.0
VANILLA_AC_SLIP_AXLE_AGGREGATION = "arithmetic_left_right_mean"
ASSETTO_CORSA_LOCAL_LATERAL_VELOCITY_AXIS = 0
ASSETTO_CORSA_LOCAL_FORWARD_VELOCITY_AXIS = 2


@dataclass(frozen=True, slots=True)
class AxleSlipFeatures:
    """Front/rear slip ratio and BC-compatible side-slip angle in radians."""

    front_slip_ratio: float
    rear_slip_ratio: float
    front_sideslip_angle_rad: float
    rear_sideslip_angle_rad: float


def _axle_means(values: WheelValues) -> tuple[float, float]:
    return (
        0.5 * (values[0] + values[1]),
        0.5 * (values[2] + values[3]),
    )


def vanilla_ac_slip_features(packet: SlipBridgePacket) -> AxleSlipFeatures:
    """Apply the accepted original-AC Python API calibration.

    Assetto Corsa exposes wheel order FL, FR, RL, RR. ``SlipRatio`` already has
    the desired physical sign: positive under driven-wheel spin and negative
    under braking lock. ``SlipAngle`` is reported in degrees. Its API sign is
    preserved to match AssettoCorsaGym observations and MoTeC demonstrations,
    then the axle mean is converted with ``pi / 180``. Values are deliberately
    not clipped.
    """

    front_ratio, rear_ratio = _axle_means(packet.slip_ratio)
    front_angle_degrees, rear_angle_degrees = _axle_means(packet.slip_angle_raw)
    return AxleSlipFeatures(
        front_slip_ratio=front_ratio,
        rear_slip_ratio=rear_ratio,
        front_sideslip_angle_rad=(front_angle_degrees * VANILLA_AC_SLIP_ANGLE_RAD_PER_RAW),
        rear_sideslip_angle_rad=rear_angle_degrees * VANILLA_AC_SLIP_ANGLE_RAD_PER_RAW,
    )


def assetto_corsa_body_sideslip_angle_rad(
    local_velocity_mps: tuple[float, float, float],
) -> float:
    """Return vehicle body sideslip beta from calibrated AC-local velocity.

    Telemetry diagnostics establish local axis 2 as forward-positive and axis
    0 as lateral for the supported original Assetto Corsa integration. The
    vertical component is deliberately ignored. ``atan2`` also maps reverse
    travel close to +/-pi instead of disguising it as a small sideslip angle.
    """

    if len(local_velocity_mps) != 3 or not all(math.isfinite(v) for v in local_velocity_mps):
        raise ValueError("local_velocity_mps must contain three finite values")
    lateral_mps = local_velocity_mps[ASSETTO_CORSA_LOCAL_LATERAL_VELOCITY_AXIS]
    forward_mps = local_velocity_mps[ASSETTO_CORSA_LOCAL_FORWARD_VELOCITY_AXIS]
    return math.atan2(lateral_mps, forward_mps)
