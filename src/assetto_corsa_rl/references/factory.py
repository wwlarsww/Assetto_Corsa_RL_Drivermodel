"""Reference composition independent from the simulator backend."""

from __future__ import annotations

from assetto_corsa_rl.config import ProjectConfig
from assetto_corsa_rl.references.promp import (
    FixedReferenceSampler,
    ReferenceSampler,
    ReferenceTrajectory,
)


def load_reference_sampler(config: ProjectConfig) -> ReferenceSampler:
    """Load the configured safe reference artefact."""

    if config.reference.mode != "fixed":
        raise NotImplementedError("only fixed reference artefacts are loadable at this stage")
    reference = ReferenceTrajectory.from_npz(config.reference.artifact_path)
    expected_length = config.environment.track_length_m
    if abs(reference.track_length_m - expected_length) > 1e-3:
        raise ValueError(
            "reference and environment track lengths differ: "
            f"{reference.track_length_m} != {expected_length}"
        )
    return FixedReferenceSampler(reference)
