"""Machine-readable contract for every observation feature."""

from __future__ import annotations

from dataclasses import dataclass

from assetto_corsa_rl.types import REFERENCE_TIME_OFFSETS_S, TRACK_PREVIEW_DISTANCES_M


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Name, unit and semantic group of one scalar observation feature."""

    index: int
    name: str
    unit: str
    group: str


def _build_observation_schema() -> tuple[FeatureSpec, ...]:
    definitions: list[tuple[str, str, str]] = [
        ("vehicle.speed_mps", "m/s", "vehicle_state"),
        ("vehicle.acceleration_x_mps2", "m/s^2", "vehicle_state"),
        ("vehicle.acceleration_z_mps2", "m/s^2", "vehicle_state"),
        ("vehicle.front_slip_ratio", "1", "vehicle_state"),
        ("vehicle.rear_slip_ratio", "1", "vehicle_state"),
        ("vehicle.front_sideslip_angle_rad", "rad", "vehicle_state"),
        ("vehicle.rear_sideslip_angle_rad", "rad", "vehicle_state"),
    ]
    for side in ("left", "right"):
        for distance in TRACK_PREVIEW_DISTANCES_M:
            label = f"{int(distance)}m"
            definitions.extend(
                (
                    (f"track.{side}.{label}.x_m", "m", f"{side}_track_boundary"),
                    (f"track.{side}.{label}.z_m", "m", f"{side}_track_boundary"),
                )
            )
    for offset in REFERENCE_TIME_OFFSETS_S:
        label = f"{float(offset):.2f}".replace(".", "_")
        definitions.extend(
            (
                (f"reference.t_plus_{label}s.x_m", "m", "reference_line"),
                (f"reference.t_plus_{label}s.z_m", "m", "reference_line"),
            )
        )
    definitions.extend(
        (
            ("action.previous_combined_pedal", "1", "previous_action"),
            ("action.previous_steering", "1", "previous_action"),
        )
    )
    return tuple(
        FeatureSpec(index=index, name=name, unit=unit, group=group)
        for index, (name, unit, group) in enumerate(definitions)
    )


OBSERVATION_FEATURES = _build_observation_schema()
OBSERVATION_SIZE = 77

if len(OBSERVATION_FEATURES) != OBSERVATION_SIZE:
    raise RuntimeError(
        f"observation schema contains {len(OBSERVATION_FEATURES)} entries; "
        f"expected {OBSERVATION_SIZE}"
    )

OBSERVATION_FEATURE_NAMES = tuple(feature.name for feature in OBSERVATION_FEATURES)
