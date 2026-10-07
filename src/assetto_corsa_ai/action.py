"""Represent simulator-independent vehicle control actions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentAction:
    """Represent one agent control command."""
    
    steering: float # -1.0 (full left) to 1.0 (full right)
    longitudinal: float # -1.0 (full brake) to 1.0 (full throttle)

    @staticmethod
    def map_agent_action(action: "AgentAction") -> "VehicleAction":
        return VehicleAction(
            steering=action.steering,
            throttle=max(action.longitudinal, 0.0),
            brake=max(-action.longitudinal, 0.0),
        )
    


@dataclass(frozen=True)
class VehicleAction:
    """Represent one vehicle control command."""

    steering: float # -1.0 (full left) to 1.0 (full right)
    throttle: float # 0.0 (no throttle) to 1.0 (full throttle)
    brake: float # 0.0 (no brake) to 1.0 (full brake)

    def __post_init__(self) -> None:
        if not -1.0 <= self.steering <= 1.0:
            raise ValueError("Steering must be between -1.0 and 1.0.")

        if not 0.0 <= self.throttle <= 1.0:
            raise ValueError("Throttle must be between 0.0 and 1.0.")

        if not 0.0 <= self.brake <= 1.0:
            raise ValueError("Brake must be between 0.0 and 1.0.")
