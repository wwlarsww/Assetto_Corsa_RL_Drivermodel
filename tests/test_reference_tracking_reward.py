from __future__ import annotations

import numpy as np
import pytest

from assetto_corsa_rl.rewards.reference_tracking import (
    ReferenceTrackingReward,
    SuccessorReferenceTrackingReward,
)


def test_reference_tracking_reward_uses_free_corridor_and_quadratic_penalty() -> None:
    """Distance inside the corridor is free and larger offsets cost more."""

    paths = np.zeros((4, 3, 2), dtype=np.float32)
    paths[:, 0, 0] = np.asarray([0.0, 2.5, 4.5, 6.5], dtype=np.float32)
    result = ReferenceTrackingReward(
        penalty_scale=0.1,
        free_distance_m=2.5,
        distance_scale_m=2.0,
    ).compute(paths)

    np.testing.assert_allclose(result.distance_m, [0.0, 2.5, 4.5, 6.5])
    np.testing.assert_allclose(result.reward, [0.0, 0.0, -0.1, -0.4])


def test_reference_tracking_reward_masks_invalid_steps() -> None:
    """Invalid reference projections contribute neither distance nor reward."""

    paths = np.zeros((2, 2, 2), dtype=np.float32)
    paths[:, 0, 0] = np.asarray([5.0, 5.0], dtype=np.float32)
    result = ReferenceTrackingReward(
        penalty_scale=0.1,
        free_distance_m=2.5,
        distance_scale_m=2.0,
    ).compute(paths, valid_mask=np.asarray([True, False]))

    assert result.distance_m[0] == pytest.approx(5.0)
    assert result.reward[0] < 0.0
    assert result.distance_m[1] == pytest.approx(0.0)
    assert result.reward[1] == pytest.approx(0.0)


def test_successor_reference_reward_preserves_aligned_progress() -> None:
    result = SuccessorReferenceTrackingReward(
        lateral_penalty_scale=2.0,
        free_distance_m=2.5,
        distance_scale_m=2.0,
        heading_penalty_scale=1.0,
        heading_scale_rad=np.deg2rad(10.0),
    ).compute(
        successor_lateral_offset_m=[0.0],
        successor_heading_error_rad=[0.0],
        progress_reward=[3.0],
    )

    assert result.progress_alignment[0] == pytest.approx(1.0)
    assert result.reward[0] == pytest.approx(0.0)


def test_successor_reference_reward_attributes_lateral_error_to_reached_state() -> None:
    result = SuccessorReferenceTrackingReward(
        lateral_penalty_scale=2.0,
        free_distance_m=2.5,
        distance_scale_m=2.0,
        heading_penalty_scale=0.0,
        heading_scale_rad=np.deg2rad(10.0),
    ).compute(
        successor_lateral_offset_m=[5.0],
        successor_heading_error_rad=[0.0],
        progress_reward=[3.0],
    )

    expected_alignment = np.exp(-0.5 * (2.5 / 2.0) ** 2)
    assert result.progress_alignment[0] == pytest.approx(expected_alignment)
    assert result.reward[0] == pytest.approx(
        3.0 * (expected_alignment - 1.0) - 2.0 * (2.5 / 2.0) ** 2
    )


def test_successor_reference_reward_penalises_heading_without_gating_reverse_progress() -> None:
    heading_rad = np.deg2rad(10.0)
    reward = SuccessorReferenceTrackingReward(
        lateral_penalty_scale=2.0,
        free_distance_m=2.5,
        distance_scale_m=2.0,
        heading_penalty_scale=1.0,
        heading_scale_rad=heading_rad,
    )
    result = reward.compute(
        successor_lateral_offset_m=[0.0, 0.0],
        successor_heading_error_rad=[heading_rad, 0.0],
        progress_reward=[3.0, -2.0],
    )

    assert result.reward[0] == pytest.approx(3.0 * (np.cos(heading_rad) - 1.0) - 1.0)
    assert result.reward[1] == pytest.approx(0.0)
