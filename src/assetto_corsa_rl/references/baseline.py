"""Backwards-compatibility shim exposing the historical `baseline` module.

Older code imported `ReferenceTrajectory` from
``assetto_corsa_rl.references.baseline``. The canonical implementation now
resides in ``fixed_reference.py``; re-export the expected symbols here so that
refactoring does not break import sites.
"""

from __future__ import annotations

from assetto_corsa_rl.references.fixed_reference import (
    BaselineReferenceSampler,
    FixedReferenceSampler,
    ReferenceTrajectory,
)

__all__ = ["BaselineReferenceSampler", "FixedReferenceSampler", "ReferenceTrajectory"]
