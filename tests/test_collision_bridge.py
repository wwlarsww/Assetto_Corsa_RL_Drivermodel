from __future__ import annotations

from pathlib import Path

import pytest

from assetto_corsa_rl.ac_integration.collision_bridge import (
    AssettoCorsaCollisionBridge,
    CollisionBridgeProtocolError,
    CollisionBridgeTimeoutError,
    decode_collision_state,
)


def _payload(sequence: int = 7, track_id: str = "monza") -> str:
    return f"ACRL_CSP_STATE_V1\n{track_id}\n{sequence}\n1234\n"


def test_collision_state_decoder_validates_protocol() -> None:
    state = decode_collision_state(_payload(), received_monotonic_s=5.0)
    assert state.track_id == "monza"
    assert state.collision_sequence == 7
    assert state.simulator_frame == 1234

    with pytest.raises(CollisionBridgeProtocolError, match="header"):
        decode_collision_state("wrong\n", received_monotonic_s=5.0)
    with pytest.raises(CollisionBridgeProtocolError, match="uint32"):
        decode_collision_state(_payload(-1), received_monotonic_s=5.0)


def test_collision_bridge_latches_sequence_across_agent_steps(tmp_path: Path) -> None:
    path = tmp_path / "state.txt"
    now = 100.0
    path.write_text(_payload(10), encoding="ascii")
    bridge = AssettoCorsaCollisionBridge(
        path,
        wall_clock=lambda: path.stat().st_mtime,
        monotonic_clock=lambda: now,
        sleep=lambda _seconds: None,
    )
    bridge.synchronise(expected_track_id="monza", timeout_s=0.1)
    detected, _ = bridge.collision_since_synchronisation(timeout_s=0.1)
    assert not detected

    path.write_text(_payload(11), encoding="ascii")
    detected, state = bridge.collision_since_synchronisation(timeout_s=0.1)
    assert detected
    assert state.collision_sequence == 11


def test_collision_bridge_rejects_wrong_track_and_stale_file(tmp_path: Path) -> None:
    path = tmp_path / "state.txt"
    path.write_text(_payload(track_id="spa"), encoding="ascii")
    bridge = AssettoCorsaCollisionBridge(
        path,
        wall_clock=lambda: path.stat().st_mtime,
        monotonic_clock=lambda: 0.0,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(CollisionBridgeProtocolError, match="expected"):
        bridge.synchronise(expected_track_id="monza", timeout_s=0.1)

    stale = AssettoCorsaCollisionBridge(
        path,
        maximum_age_s=0.1,
        wall_clock=lambda: path.stat().st_mtime + 1.0,
        monotonic_clock=iter((0.0, 0.0, 1.0)).__next__,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(CollisionBridgeTimeoutError, match="stale"):
        stale.receive(timeout_s=0.5)
