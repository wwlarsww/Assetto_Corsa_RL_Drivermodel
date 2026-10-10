"""Represent simulator-independent vehicle control actions."""

from dataclasses import dataclass
from math import isfinite


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
class ActionConfig:
    """Per-step control limits; zero disables an individual limit."""

    max_steering_delta: float
    max_longitudinal_delta: float
    longitudinal_deadband: float

    def validate(self) -> None:
        for name in ("max_steering_delta", "max_longitudinal_delta"):
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 2.0:
                raise ValueError(f"{name} must be finite and in [0, 2].")
        if not isfinite(self.longitudinal_deadband) or not 0.0 <= self.longitudinal_deadband < 1.0:
            raise ValueError("longitudinal_deadband must be finite and in [0, 1).")

    def constrain(self, requested: AgentAction, current: AgentAction) -> AgentAction:
        """Apply slew limits and deadband to final vehicle controls."""
        values = (requested.steering, requested.longitudinal, current.steering, current.longitudinal)
        if not all(isfinite(value) for value in values):
            raise ValueError("Controls must be finite.")

        def limit(target: float, previous: float, maximum: float) -> float:
            target, previous = max(-1.0, min(1.0, target)), max(-1.0, min(1.0, previous))
            return previous + max(-maximum, min(maximum, target - previous)) if maximum else target

        steering = limit(requested.steering, current.steering, self.max_steering_delta)
        longitudinal = limit(requested.longitudinal, current.longitudinal, self.max_longitudinal_delta)
        # Snap to coast only when neutral is reachable within this step's
        # limit. Larger changes approach neutral over multiple steps.
        if abs(longitudinal) < self.longitudinal_deadband and (
            not self.max_longitudinal_delta or abs(current.longitudinal) <= self.max_longitudinal_delta
        ):
            longitudinal = 0.0
        return AgentAction(steering, longitudinal)


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
