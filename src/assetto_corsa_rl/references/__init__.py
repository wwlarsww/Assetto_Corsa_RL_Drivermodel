"""Reference-trajectory providers for the fixed-reference baseline."""

from assetto_corsa_rl.references.factory import load_reference_sampler
from assetto_corsa_rl.references.projection import (
    project_points_to_closed_reference_progress,
    project_points_to_polyline_progress,
    wrapped_progress_delta,
)
from assetto_corsa_rl.references.reference import (
    FixedReferenceSampler,
    ReferenceSampler,
    ReferenceTrajectory,
)

__all__ = [
    "FixedReferenceSampler",
    "ReferenceSampler",
    "ReferenceTrajectory",
    "load_reference_sampler",
    "project_points_to_closed_reference_progress",
    "project_points_to_polyline_progress",
    "wrapped_progress_delta",
]
