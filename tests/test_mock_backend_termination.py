import numpy as np

from assetto_corsa_rl.envs.mock_backend import MockAssettoCorsaBackend
from assetto_corsa_rl.references.fixed_reference import ReferenceTrajectory
from assetto_corsa_rl.types import TerminationReason


def test_mock_off_track_detection_reports_a_tyre_out() -> None:
    backend = MockAssettoCorsaBackend()
    backend.reset(
        seed=None,
        reference=ReferenceTrajectory.constant(track_length_m=1000.0, speed_mps=20.0),
    )
    backend._lateral_offset_m = backend.track_half_width_m + 0.1

    transition = backend.step(np.zeros(2, dtype=np.float32))

    assert transition.termination_reason is TerminationReason.OFF_TRACK
    assert transition.diagnostics is not None
    assert transition.diagnostics["number_of_tyres_out"] == 1


def test_mock_stall_is_armed_by_motion_then_debounced() -> None:
    backend = MockAssettoCorsaBackend(
        minimum_speed_mps=5.0 / 3.6,
        stall_debounce_steps=3,
    )
    backend.reset(
        seed=None,
        reference=ReferenceTrajectory.constant(track_length_m=1000.0, speed_mps=20.0),
    )
    backend.step(np.zeros(2, dtype=np.float32))
    backend._speed_mps = 0.0

    assert backend.step(np.zeros(2, dtype=np.float32)).termination_reason is None
    assert backend.step(np.zeros(2, dtype=np.float32)).termination_reason is None
    assert (
        backend.step(np.zeros(2, dtype=np.float32)).termination_reason is TerminationReason.STALLED
    )


def test_mock_uses_short_startup_fallback_if_car_never_moves() -> None:
    backend = MockAssettoCorsaBackend(
        stall_startup_grace_steps=3,
        stall_debounce_steps=1,
    )
    backend.reset(
        seed=None,
        reference=ReferenceTrajectory.constant(track_length_m=1000.0, speed_mps=20.0),
    )
    backend._stall_detection_armed = False
    backend._speed_mps = 0.0

    first = backend.step(np.zeros(2, dtype=np.float32))
    second = backend.step(np.zeros(2, dtype=np.float32))
    third = backend.step(np.zeros(2, dtype=np.float32))

    assert first.termination_reason is None
    assert second.termination_reason is None
    assert third.termination_reason is TerminationReason.STALLED
