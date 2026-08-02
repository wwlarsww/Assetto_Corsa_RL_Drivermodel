from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from assetto_corsa_rl.ac_integration.ai_spline import (
    AISplineFormatError,
    AssettoCorsaAISpline,
)
from assetto_corsa_rl.features.track_geometry import TrackGeometry


def _fast_lane_payload(*, version: int = 7, extra_count: int = 3) -> bytes:
    positions = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 10.0),
        (10.0, 0.0, 10.0),
    )
    distances = (0.0, 10.0, 20.0)
    payload = bytearray(struct.pack("<4i", version, 3, 0, 0))
    for identifier, (position, distance) in enumerate(zip(positions, distances, strict=True)):
        payload.extend(struct.pack("<4fi", *position, distance, identifier))
    payload.extend(struct.pack("<i", extra_count))
    for _ in range(extra_count):
        values = [0.0] * 18
        values[5] = 4.0
        values[6] = 6.0
        payload.extend(struct.pack("<18f", *values))
    return bytes(payload)


def test_fast_lane_v7_supplies_length_line_and_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "fast_lane.ai"
    path.write_bytes(_fast_lane_payload())

    spline = AssettoCorsaAISpline.from_file(path)
    assert spline.track_length_m == pytest.approx(20.0 + np.sqrt(200.0))
    np.testing.assert_allclose(spline.centre_xy_m, [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    assert len(spline.source_sha256) == 64

    geometry = TrackGeometry.from_ai_spline(spline)
    assert geometry.track_length_m == pytest.approx(spline.track_length_m)
    widths = np.linalg.norm(geometry.left_xy_m - geometry.right_xy_m, axis=1)
    assert widths == pytest.approx(np.full(3, 10.0))


def test_fast_lane_reader_rejects_unknown_version_and_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "fast_lane.ai"
    path.write_bytes(_fast_lane_payload(version=6))
    with pytest.raises(AISplineFormatError, match="unsupported"):
        AssettoCorsaAISpline.from_file(path)

    path.write_bytes(_fast_lane_payload(extra_count=2))
    with pytest.raises(AISplineFormatError, match=r"ends before|mismatch"):
        AssettoCorsaAISpline.from_file(path)
