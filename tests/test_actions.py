from __future__ import annotations

import numpy as np
import pytest

from assetto_corsa_rl.control.actions import ActionIntegrator


def test_action_deltas_are_scaled_integrated_and_clipped() -> None:
    integrator = ActionIntegrator(np.asarray([0.1, 0.05], dtype=np.float32))
    first = integrator.integrate(np.asarray([1.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(first.delta, [0.1, -0.05])
    np.testing.assert_allclose(first.absolute, [0.1, -0.05])

    for _ in range(20):
        final = integrator.integrate(np.asarray([2.0, -2.0], dtype=np.float32))
    np.testing.assert_allclose(final.absolute, [1.0, -1.0])
    np.testing.assert_allclose(integrator.current, [1.0, -1.0])


def test_action_state_is_reset_and_copied() -> None:
    integrator = ActionIntegrator(np.asarray([0.1, 0.1], dtype=np.float32))
    state = integrator.reset([0.5, -0.5])
    state[:] = 0.0
    np.testing.assert_allclose(integrator.current, [0.5, -0.5])


def test_invalid_action_shape_is_rejected() -> None:
    integrator = ActionIntegrator(np.asarray([0.1, 0.1], dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        integrator.integrate([0.0])
