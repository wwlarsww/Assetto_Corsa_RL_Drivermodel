from __future__ import annotations

import numpy as np
import pytest

from assetto_corsa_rl.control.actions import BoundedActionAdapter


def test_absolute_actions_can_switch_full_range_in_one_step() -> None:
    adapter = BoundedActionAdapter()
    first = adapter.apply(np.asarray([-1.0, 1.0], dtype=np.float32))
    second = adapter.apply(np.asarray([1.0, -1.0], dtype=np.float32))

    np.testing.assert_allclose(first.absolute, [-1.0, 1.0])
    np.testing.assert_allclose(second.change, [2.0, -2.0])
    np.testing.assert_allclose(second.absolute, [1.0, -1.0])


def test_absolute_action_safety_guard_clips_both_channels() -> None:
    adapter = BoundedActionAdapter()
    applied = adapter.apply(np.asarray([2.0, -2.0], dtype=np.float32))
    np.testing.assert_allclose(applied.absolute, [1.0, -1.0])


def test_action_state_is_reset_and_copied() -> None:
    adapter = BoundedActionAdapter()
    state = adapter.reset([0.5, -0.5])
    state[:] = 0.0
    np.testing.assert_allclose(adapter.current, [0.5, -0.5])


def test_invalid_action_shape_is_rejected() -> None:
    adapter = BoundedActionAdapter()
    with pytest.raises(ValueError, match="shape"):
        adapter.apply([0.0])
