"""Black-box boundary between vehicle simulation and RL environment."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from assetto_corsa_rl.references.fixed_reference import ReferenceTrajectory
from assetto_corsa_rl.types import BackendTransition, TelemetryFrame


class SimulatorBackend(Protocol):
    """Minimal interface an Assetto Corsa adapter must implement.

    Implementations own transport, timing and synchronisation. The environment
    only sends absolute combined-pedal/steering commands and receives complete
    telemetry frames.
    """

    @property
    def control_frequency_hz(self) -> float:
        """Nominal environment control rate."""

        ...

    def reset(
        self,
        *,
        seed: int | None,
        reference: ReferenceTrajectory,
        initial_action: npt.NDArray[np.float32] | None = None,
    ) -> TelemetryFrame:
        """Reset the session, prime its controls and return the first frame."""

        ...

    def step(self, absolute_action: npt.NDArray[np.float32]) -> BackendTransition:
        """Apply one absolute action and wait for the next telemetry frame."""

        ...

    def close(self) -> None:
        """Release shared memory, sockets, processes or other resources."""

        ...


def validate_backend_action(action: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Shared input validation for concrete backend adapters."""

    values = np.asarray(action, dtype=np.float32)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("backend action must be a finite vector with shape (2,)")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("backend action must be inside [-1, 1]")
    return values
