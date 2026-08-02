"""Reference composition independent from the simulator backend."""

from __future__ import annotations

from assetto_corsa_rl.ac_integration.ai_spline import AssettoCorsaAISpline
from assetto_corsa_rl.ac_integration.launcher import configured_fast_lane_path
from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.references.fixed_reference import (
    FixedReferenceSampler,
    ReferenceSampler,
    ReferenceTrajectory,
)


def load_reference_sampler(config: ProjectConfig) -> ReferenceSampler:
    """Load the configured safe reference artefact."""

    if config.reference.mode != "fixed":
        raise NotImplementedError("only fixed reference artefacts are loadable at this stage")
    reference = ReferenceTrajectory.from_npz(config.reference.artifact_path)
    spline = AssettoCorsaAISpline.from_file(configured_fast_lane_path(config))
    if reference.source_track_sha256 != spline.source_sha256:
        raise ValueError(
            "reference source does not match the configured fast_lane.ai; rebuild the reference"
        )
    if abs(reference.track_length_m - spline.track_length_m) > 1e-3:
        raise ValueError("reference and fast_lane.ai track lengths differ")
    return FixedReferenceSampler(reference)
