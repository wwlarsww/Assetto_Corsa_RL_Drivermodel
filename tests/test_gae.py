from __future__ import annotations

import numpy as np
import pytest

from assetto_corsa_rl.training.gae import compute_gae


def test_gae_matches_discounted_returns_when_values_are_zero() -> None:
    result = compute_gae(
        rewards=[1.0, 1.0, 1.0],
        values=[0.0, 0.0, 0.0],
        next_values=[0.0, 0.0, 0.0],
        terminated=[False, False, True],
        truncated=[False, False, False],
        gamma=1.0,
        gae_lambda=1.0,
    )
    np.testing.assert_allclose(result.advantages, [3.0, 2.0, 1.0])
    np.testing.assert_allclose(result.returns, [3.0, 2.0, 1.0])


def test_truncation_bootstraps_but_stops_trace_across_reset() -> None:
    result = compute_gae(
        rewards=[1.0, 100.0],
        values=[2.0, 20.0],
        next_values=[3.0, 30.0],
        terminated=[False, False],
        truncated=[True, False],
        gamma=0.5,
        gae_lambda=1.0,
    )
    # Step zero uses its own final-observation value: 1 + .5*3 - 2.
    # It must not absorb the advantage of the reset episode at step one.
    assert result.advantages[0] == pytest.approx(0.5)


def test_termination_does_not_bootstrap_and_invalid_inputs_fail() -> None:
    result = compute_gae(
        rewards=[1.0],
        values=[2.0],
        next_values=[100.0],
        terminated=[True],
        truncated=[False],
        gamma=0.99,
        gae_lambda=0.95,
    )
    np.testing.assert_allclose(result.advantages, [-1.0])
    with pytest.raises(ValueError, match="both"):
        compute_gae(
            rewards=[1.0],
            values=[0.0],
            next_values=[0.0],
            terminated=[True],
            truncated=[True],
            gamma=0.99,
            gae_lambda=0.95,
        )
