"""Define the vehicle controller interface."""

from typing import Protocol

from assetto_corsa_ai.action import VehicleAction


class VehicleController(Protocol):
    """Interface for applying vehicle control actions."""

    def apply(self, action: VehicleAction) -> None:
        """Apply one vehicle control action."""
        ...

    def reset(self) -> None:
        """Reset the vehicle controls to a neutral state."""
        ...

    def close(self) -> None:
        """Release controller resources."""
        ...
