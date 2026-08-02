"""Canonical `reference` module name re-exporting the fixed-reference API.

This file exists so that the preferred import path
``assetto_corsa_rl.references.reference`` works while preserving the full
implementation in ``fixed_reference.py``. Downstream code should prefer the
shorter name.
"""

from __future__ import annotations

from assetto_corsa_rl.references.fixed_reference import (
    BaselineReferenceSampler,
    FixedReferenceSampler,
    ReferenceSampler,
    ReferenceTrajectory,
)

__all__ = [
    "BaselineReferenceSampler",
    "FixedReferenceSampler",
    "ReferenceSampler",
    "ReferenceTrajectory",
]
