"""Reference-trajectory providers, including a compact ProMP implementation."""

from assetto_corsa_rl.references.factory import load_reference_sampler
from assetto_corsa_rl.references.projection import (
    project_points_to_closed_reference_progress,
    project_points_to_polyline_progress,
    wrapped_progress_delta,
)
from assetto_corsa_rl.references.promp import (
    FixedReferenceSampler,
    ProbabilisticMovementPrimitive,
    ReferenceSampler,
    ReferenceTrajectory,
)

__all__ = [
    "FixedReferenceSampler",
    "ProbabilisticMovementPrimitive",
    "ReferenceSampler",
    "ReferenceTrajectory",
    "load_reference_sampler",
    "project_points_to_closed_reference_progress",
    "project_points_to_polyline_progress",
    "wrapped_progress_delta",
]
