from __future__ import annotations

from pathlib import Path

import pytest

from assetto_corsa_rl.ac_integration.reset_bridge import (
    HOTLAP_REFERENCE_POSE_SPEED_RESET_COMMAND_MAGIC,
    HOTLAP_REFERENCE_SPEED_RESET_COMMAND_MAGIC,
    HOTLAP_RESET_ACK_MAGIC,
    AssettoCorsaHotlapReferenceSpeedResetBridge,
    AssettoCorsaHotlapResetBridge,
    AssettoCorsaReferencePoseSpeedResetBridge,
    AssettoCorsaResetBridge,
    ResetBridgeStatus,
    ResetBridgeUnavailableError,
    decode_hotlap_reset_acknowledgement,
    decode_reset_acknowledgement,
    decode_reset_command,
    encode_hotlap_reference_pose_speed_reset_command,
    encode_hotlap_reference_speed_reset_command,
    encode_hotlap_reset_command,
    encode_reset_acknowledgement,
    encode_reset_command,
)


def test_reset_protocol_round_trip() -> None:
    assert decode_reset_command(encode_reset_command(17)) == 17
    acknowledgement = decode_reset_acknowledgement(
        encode_reset_acknowledgement(
            sequence=17,
            status=ResetBridgeStatus.SUCCESS,
            source_time_s=123.5,
        )
    )
    assert acknowledgement.sequence == 17
    assert acknowledgement.status is ResetBridgeStatus.SUCCESS
    assert acknowledgement.source_time_s == pytest.approx(123.5)


def test_hotlap_reset_protocol_round_trip() -> None:
    assert encode_hotlap_reset_command(sequence=17, track_id="monza") == (
        b"ACRL_HOTLAP_RESET_COMMAND_V1\n17\nmonza\n"
    )
    acknowledgement = decode_hotlap_reset_acknowledgement(
        f"{HOTLAP_RESET_ACK_MAGIC}\n17\n1\n1234\nok\n",
        source_time_s=42.0,
    )
    assert acknowledgement.sequence == 17
    assert acknowledgement.simulator_frame == 1234
    assert acknowledgement.method_variant == "csp_hotlap_spawn_v1"


def test_hotlap_reference_speed_command_is_explicit_and_bounded() -> None:
    assert encode_hotlap_reference_speed_reset_command(
        sequence=17,
        track_id="monza",
        start_speed_mps=70.710266,
    ) == (f"{HOTLAP_REFERENCE_SPEED_RESET_COMMAND_MAGIC}\n17\nmonza\n70.710266\n").encode("ascii")
    with pytest.raises(ValueError, match="start_speed_mps"):
        encode_hotlap_reference_speed_reset_command(
            sequence=17,
            track_id="monza",
            start_speed_mps=-1.0,
        )


def test_hotlap_reference_pose_speed_command_uses_canonical_world_pose() -> None:
    command = encode_hotlap_reference_pose_speed_reset_command(
        sequence=17,
        track_id="monza",
        start_speed_mps=70.710266,
        start_position_xy_m=(674.75, 42.96),
        start_heading_rad=-0.095,
    )
    assert command.decode("ascii").splitlines() == [
        HOTLAP_REFERENCE_POSE_SPEED_RESET_COMMAND_MAGIC,
        "17",
        "monza",
        "70.710266",
        "674.75",
        "42.96",
        "-0.095",
    ]


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _bridge_with_responder(
    tmp_path: Path,
    status: ResetBridgeStatus,
) -> AssettoCorsaResetBridge:
    command = tmp_path / "command.bin"
    acknowledgement = tmp_path / "ack.bin"
    clock = _Clock()

    def respond(seconds: float) -> None:
        clock.advance(seconds)
        if command.is_file() and not acknowledgement.is_file():
            sequence = decode_reset_command(command.read_bytes())
            acknowledgement.write_bytes(
                encode_reset_acknowledgement(
                    sequence=sequence,
                    status=status,
                    source_time_s=42.0,
                )
            )

    return AssettoCorsaResetBridge(
        command,
        acknowledgement,
        poll_interval_s=0.01,
        monotonic_clock=clock,
        sleep=respond,
    )


def test_reset_bridge_waits_for_matching_success(tmp_path: Path) -> None:
    bridge = _bridge_with_responder(tmp_path, ResetBridgeStatus.SUCCESS)
    acknowledgement = bridge.request_reset(timeout_s=1.0)
    assert acknowledgement.status is ResetBridgeStatus.SUCCESS


def test_reset_bridge_surfaces_missing_csp(tmp_path: Path) -> None:
    bridge = _bridge_with_responder(tmp_path, ResetBridgeStatus.CSP_UNAVAILABLE)
    with pytest.raises(ResetBridgeUnavailableError, match="CSP"):
        bridge.request_reset(timeout_s=1.0)


def test_hotlap_reset_bridge_waits_for_matching_success(tmp_path: Path) -> None:
    command = tmp_path / "hotlap-command.txt"
    acknowledgement = tmp_path / "hotlap-ack.txt"
    clock = _Clock()

    def respond(seconds: float) -> None:
        clock.advance(seconds)
        if command.is_file() and not acknowledgement.is_file():
            lines = command.read_text(encoding="ascii").splitlines()
            acknowledgement.write_text(
                f"{HOTLAP_RESET_ACK_MAGIC}\n{lines[1]}\n1\n99\nok\n",
                encoding="ascii",
            )

    bridge = AssettoCorsaHotlapResetBridge(
        command,
        acknowledgement,
        expected_track_id="monza",
        poll_interval_s=0.01,
        monotonic_clock=clock,
        sleep=respond,
    )
    result = bridge.request_reset(timeout_s=1.0)
    assert result.status is ResetBridgeStatus.SUCCESS
    assert result.simulator_frame == 99


def test_hotlap_reference_speed_bridge_publishes_target_speed(tmp_path: Path) -> None:
    command = tmp_path / "hotlap-command.txt"
    acknowledgement = tmp_path / "hotlap-ack.txt"
    clock = _Clock()

    def respond(seconds: float) -> None:
        clock.advance(seconds)
        if command.is_file() and not acknowledgement.is_file():
            lines = command.read_text(encoding="ascii").splitlines()
            assert lines == [
                HOTLAP_REFERENCE_SPEED_RESET_COMMAND_MAGIC,
                lines[1],
                "monza",
                "70.710266",
            ]
            acknowledgement.write_text(
                f"{HOTLAP_RESET_ACK_MAGIC}\n{lines[1]}\n1\n99\nok\n",
                encoding="ascii",
            )

    bridge = AssettoCorsaHotlapReferenceSpeedResetBridge(
        command,
        acknowledgement,
        expected_track_id="monza",
        poll_interval_s=0.01,
        monotonic_clock=clock,
        sleep=respond,
    )
    result = bridge.request_reset(timeout_s=1.0, start_speed_mps=70.710266)
    assert result.method_variant == "csp_hotlap_reference_speed_v1"


def test_reference_pose_speed_bridge_publishes_v3_target(tmp_path: Path) -> None:
    command = tmp_path / "hotlap-command.txt"
    acknowledgement = tmp_path / "hotlap-ack.txt"
    clock = _Clock()

    def respond(seconds: float) -> None:
        clock.advance(seconds)
        if command.is_file() and not acknowledgement.is_file():
            lines = command.read_text(encoding="ascii").splitlines()
            assert lines[0] == HOTLAP_REFERENCE_POSE_SPEED_RESET_COMMAND_MAGIC
            assert lines[2:] == ["monza", "70", "674.75", "42.96", "-0.095"]
            acknowledgement.write_text(
                f"{HOTLAP_RESET_ACK_MAGIC}\n{lines[1]}\n1\n99\nok\n",
                encoding="ascii",
            )

    bridge = AssettoCorsaReferencePoseSpeedResetBridge(
        command,
        acknowledgement,
        expected_track_id="monza",
        poll_interval_s=0.01,
        monotonic_clock=clock,
        sleep=respond,
    )
    result = bridge.request_reset(
        timeout_s=1.0,
        start_speed_mps=70.0,
        start_position_xy_m=(674.75, 42.96),
        start_heading_rad=-0.095,
    )
    assert result.method_variant == "csp_reference_pose_speed_v2"
