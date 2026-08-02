"""Read-only adapters for the original Assetto Corsa."""

from assetto_corsa_rl.ac_integration.shared_memory import (
    AssettoCorsaSharedMemoryReader,
    AssettoCorsaTelemetry,
    AssettoCorsaTelemetryUnavailableError,
    GraphicsPage,
    PhysicsPage,
    StaticPage,
)
from assetto_corsa_rl.ac_integration.slip_bridge import (
    AssettoCorsaSlipBridgeFileReader,
    AssettoCorsaSlipBridgeReceiver,
    SlipBridgePacket,
)
from assetto_corsa_rl.ac_integration.slip_features import (
    VANILLA_AC_SLIP_ANGLE_RAD_PER_RAW,
    VANILLA_AC_SLIP_ANGLE_RAW_UNIT,
    VANILLA_AC_SLIP_AXLE_AGGREGATION,
    VANILLA_AC_SLIP_CALIBRATION_NAME,
    AxleSlipFeatures,
    vanilla_ac_slip_features,
)

__all__ = [
    "VANILLA_AC_SLIP_ANGLE_RAD_PER_RAW",
    "VANILLA_AC_SLIP_ANGLE_RAW_UNIT",
    "VANILLA_AC_SLIP_AXLE_AGGREGATION",
    "VANILLA_AC_SLIP_CALIBRATION_NAME",
    "AssettoCorsaSharedMemoryReader",
    "AssettoCorsaSlipBridgeFileReader",
    "AssettoCorsaSlipBridgeReceiver",
    "AssettoCorsaTelemetry",
    "AssettoCorsaTelemetryUnavailableError",
    "AxleSlipFeatures",
    "GraphicsPage",
    "PhysicsPage",
    "SlipBridgePacket",
    "StaticPage",
    "vanilla_ac_slip_features",
]
