"""Verify controller commands using isolated, unnamed shared memory."""

from collections.abc import Callable, Iterator
from itertools import count
from mmap import mmap as real_mmap
from struct import unpack_from
from unittest.mock import create_autospec

import pytest

from assetto_corsa_ai.assetto_corsa import simulator_controller as module
from assetto_corsa_ai.assetto_corsa.simulator_controller import (
    AssettoCorsaSimulatorController,
    SimulatorControllerError,
)
from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.telemetry import Vector3
from tools.validation import validate_simulator_control as validation


@pytest.fixture
def memories(monkeypatch: pytest.MonkeyPatch) -> list[real_mmap]:
    buffers: list[real_mmap] = []

    def open_memory(
        fileno: int, length: int, *, tagname: str, access: int,
    ) -> real_mmap:
        memory = real_mmap(-1, length)
        buffers.append(memory)
        return memory

    def mapping_exists(name: str) -> bool:
        return True

    def skip_sleep(seconds: float) -> None:
        pass

    monkeypatch.setattr(module.mmap, "mmap", open_memory)
    monkeypatch.setattr(
        AssettoCorsaSimulatorController,
        "_windows_mapping_exists",
        staticmethod(mapping_exists),
    )
    monkeypatch.setattr(module.time, "sleep", skip_sleep)
    return buffers


@pytest.fixture
def controller(memories: list[real_mmap]) -> Iterator[AssettoCorsaSimulatorController]:
    instance = AssettoCorsaSimulatorController()
    instance.connect()
    try:
        yield instance
    finally:
        instance.close()
    assert all(memory.closed for memory in memories)


def test_connect_initializes_automatic_driving_and_is_idempotent(
    controller: AssettoCorsaSimulatorController, memories: list[real_mmap],
) -> None:
    controller.connect()
    assert len(memories) == 2
    assert len(memories[0]) == 72
    assert len(memories[1]) == 4
    assert unpack_from("=5f", memories[0]) == (0.0, 1.0, 0.0, 0.0, 0.0)
    assert memories[0][41:44] == bytes([1, 1, 1])
    assert memories[0][68] == 1  # autoshift_active


def test_controls_and_neutral_state(
    controller: AssettoCorsaSimulatorController, memories: list[real_mmap],
) -> None:
    controller.set_controls(steering=-0.5, throttle=0.75, brake=0.25)
    assert unpack_from("=5f", memories[0]) == (0.75, 0.25, 0.0, -0.5, 0.0)
    controller.neutral_controls()
    assert unpack_from("=5f", memories[0]) == (0.0,) * 5


def test_teleport_writes_pose_and_normalizes_forward(
    controller: AssettoCorsaSimulatorController, memories: list[real_mmap],
) -> None:
    controller.set_position((1.0, 2.0, 3.0), (0.0, 0.0, -4.0))
    assert unpack_from("=6f", memories[0], 44) == (1.0, 2.0, 3.0, 0.0, 0.0, -1.0)
    assert memories[0][40] == 2
    controller.clear_position_request()
    assert memories[0][40] == 0


def test_pit_teleport_remains_active_until_cleared(
    controller: AssettoCorsaSimulatorController, memories: list[real_mmap],
) -> None:
    controller.teleport_to_pits()
    assert memories[0][40] == 1
    controller.clear_position_request()
    assert memories[0][40] == 0


def test_teleport_debug_output_matches_raw_control_bytes(
    controller: AssettoCorsaSimulatorController,
    memories: list[real_mmap],
    capsys: pytest.CaptureFixture[str],
) -> None:
    controller.set_position((1.25, 2.5, 3.75), (0.0, 0.0, -4.0))
    before = memories[0][:]
    controller.debug_teleport_memory()
    controller.debug_raw_control_memory()
    output = capsys.readouterr().out
    assert "teleport_to = 2" in output
    assert "teleport_to raw: 2" in output
    assert "teleport_pos = (1.25, 2.5, 3.75)" in output
    assert "teleport_pos raw: (1.25, 2.5, 3.75)" in output
    assert "teleport_dir = (0.0, 0.0, -1.0)" in output
    assert "teleport_dir raw: (0.0, 0.0, -1.0)" in output
    assert "autoshift = 1" in output
    assert "autoshift raw: 1" in output
    assert memories[0][:] == before
    controller.clear_position_request()


def test_validation_pit_command_uses_verified_heading_correction(
    controller: AssettoCorsaSimulatorController,
    memories: list[real_mmap],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = iter([("pit", 0), ("debug", 0), ("clear", 0), ("q", 0)])
    pit_checks = 0

    def validate_pit(instance: AssettoCorsaSimulatorController, telemetry: TelemetryReader) -> None:
        nonlocal pit_checks
        pit_checks += 1
        assert instance is controller
        assert telemetry is reader

    def print_state(telemetry: TelemetryReader) -> None:
        assert telemetry is reader

    def next_command(prompt: str) -> str:
        command, expected_flag = next(commands)
        assert memories[0][40] == expected_flag
        if expected_flag == 1:
            assert unpack_from("=f", memories[0], 4)[0] == 1.0
        return command

    monkeypatch.setattr("builtins.input", next_command)
    monkeypatch.setattr(validation, "validate_pit_teleport", validate_pit)
    monkeypatch.setattr(validation, "print_telemetry", print_state)
    reader = create_autospec(TelemetryReader, instance=True)
    validation.run_menu(controller, reader)
    assert memories[0][40] == 0
    assert pit_checks == 1


def test_reset_holds_brake_until_readiness_then_releases_it(
    controller: AssettoCorsaSimulatorController, memories: list[real_mmap],
) -> None:
    checks = 0

    def is_ready() -> bool:
        nonlocal checks
        checks += 1
        assert unpack_from("=5f", memories[0]) == (0.0, 1.0, 0.0, 0.0, 0.0)
        assert memories[0][40] == 2
        return checks == 3

    controller.reset_episode((1.0, 2.0, 3.0), (0.0, 0.0, -1.0), is_ready=is_ready)
    assert checks == 3
    assert memories[0][40] == 0
    assert unpack_from("=5f", memories[0]) == (0.0,) * 5


def test_reset_timeout_leaves_brake_applied(
    controller: AssettoCorsaSimulatorController,
    memories: list[real_mmap],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = count(0.0, 0.1)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    with pytest.raises(SimulatorControllerError, match="episode-start state"):
        controller.reset_episode(
            (0.0, 0.0, 0.0), (0.0, 0.0, -1.0),
            is_ready=lambda: False, timeout=0.25,
        )
    assert unpack_from("=5f", memories[0]) == (0.0, 1.0, 0.0, 0.0, 0.0)
    assert memories[0][40] == 0


def test_reset_callback_error_leaves_brake_applied(
    controller: AssettoCorsaSimulatorController, memories: list[real_mmap],
) -> None:
    def is_ready() -> bool:
        raise RuntimeError("Telemetry unavailable")

    with pytest.raises(RuntimeError, match="Telemetry unavailable"):
        controller.reset_episode((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), is_ready=is_ready)
    assert unpack_from("=f", memories[0], 4)[0] == 1.0
    assert memories[0][40] == 0


@pytest.mark.parametrize("operation", ["reset", "restart"])
def test_interrupted_commands_clear_request_flags(
    controller: AssettoCorsaSimulatorController,
    memories: list[real_mmap],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    def interrupt(seconds: float) -> None:
        memory, offset = (memories[1], 1) if operation == "restart" else (memories[0], 40)
        assert memory[offset] != 0
        raise KeyboardInterrupt

    monkeypatch.setattr(module.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        if operation == "restart":
            controller.restart_session()
        elif operation == "reset":
            controller.reset_episode(
                (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), is_ready=lambda: False,
            )
    assert memories[0][40] == 0
    assert memories[1][1] == 0


@pytest.mark.parametrize("outcome", ["success", "timeout", "interrupt"])
def test_validation_tool_clears_persistent_teleport_request(
    controller: AssettoCorsaSimulatorController,
    memories: list[real_mmap],
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    checks = 0

    def make_check(
        reader: TelemetryReader, position: Vector3, forward: Vector3,
    ) -> Callable[[], bool]:
        def is_ready() -> bool:
            nonlocal checks
            checks += 1
            assert memories[0][40] == 2
            assert unpack_from("=f", memories[0], 4)[0] == 1.0
            return outcome == "success" and checks >= 3

        return is_ready

    def sleep_or_interrupt(seconds: float) -> None:
        if outcome == "interrupt":
            raise KeyboardInterrupt

    clock = count(0.0, 0.01)
    monkeypatch.setattr(validation, "readiness_check", make_check)
    monkeypatch.setattr(validation, "monotonic", lambda: next(clock))
    monkeypatch.setattr(validation, "sleep", sleep_or_interrupt)
    monkeypatch.setattr(validation, "RESET_TIMEOUT_SECONDS", 0.25)
    reader = create_autospec(TelemetryReader, instance=True)
    if outcome == "success":
        validation.validate_teleport(controller, reader, (1.0, 2.0, 3.0), (0.0, 0.0, -1.0))
        assert checks == 3
    else:
        error = KeyboardInterrupt if outcome == "interrupt" else SimulatorControllerError
        with pytest.raises(error):
            validation.validate_teleport(controller, reader, (1.0, 2.0, 3.0), (0.0, 0.0, -1.0))
    assert memories[0][40] == 0
    assert unpack_from("=f", memories[0], 4)[0] == 1.0


def test_interrupted_connect_releases_both_mappings(
    memories: list[real_mmap], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(name: str) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        AssettoCorsaSimulatorController, "_windows_mapping_exists", staticmethod(interrupt),
    )
    instance = AssettoCorsaSimulatorController()
    with pytest.raises(KeyboardInterrupt):
        instance.connect()
    assert not instance.is_connected
    assert all(memory.closed for memory in memories)
    instance.close()


def test_pause_and_restart_commands(
    controller: AssettoCorsaSimulatorController, memories: list[real_mmap],
) -> None:
    controller.set_paused(True)
    assert memories[1][0] == 1
    controller.set_paused(False)
    assert memories[1][0] == 0
    controller.restart_session()
    assert memories[1][1] == 0


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("parameter", ["timeout", "poll_interval"])
def test_invalid_reset_timing_does_not_change_controls(
    controller: AssettoCorsaSimulatorController,
    memories: list[real_mmap],
    duration: float,
    parameter: str,
) -> None:
    controller.set_controls(0.0, 0.5, 0.0)
    with pytest.raises(ValueError, match=parameter):
        controller.reset_episode(
            (0.0, 0.0, 0.0), (0.0, 0.0, -1.0),
            is_ready=lambda: True, **{parameter: duration},
        )
    assert unpack_from("=f", memories[0])[0] == 0.5
