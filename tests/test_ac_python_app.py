from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from assetto_corsa_rl.ac_integration.reset_bridge import (
    ResetBridgeStatus,
    decode_reset_acknowledgement,
    encode_reset_command,
)
from assetto_corsa_rl.ac_integration.slip_bridge import decode_slip_bridge_packet

APP_PATH = (
    Path(__file__).parents[1]
    / "assetto_corsa_app"
    / "assetto_corsa_rl_bridge"
    / "assetto_corsa_rl_bridge.py"
)


def test_in_game_app_avoids_starred_calls_rejected_by_embedded_python_33() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    starred_arguments = [
        argument
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for argument in node.args
        if isinstance(argument, ast.Starred)
    ]
    assert starred_arguments == []


class _FakeAc(ModuleType):
    def __init__(self) -> None:
        super().__init__("ac")
        self.texts: list[str] = []
        self.logs: list[str] = []
        self.states: dict[int, tuple[float, ...]] = {
            1: (0.1, 0.2, -0.1, -0.2),
            2: (0.01, 0.02, -0.01, -0.02),
        }

    def newApp(self, _name: str) -> int:  # noqa: N802
        return 1

    def setSize(self, _app: int, _width: int, _height: int) -> None:  # noqa: N802
        return None

    def addLabel(self, _app: int, _text: str) -> int:  # noqa: N802
        return 2

    def setPosition(self, _label: int, _x: int, _y: int) -> None:  # noqa: N802
        return None

    def setText(self, _label: int, text: str) -> None:  # noqa: N802
        self.texts.append(text)

    def log(self, text: str) -> None:
        self.logs.append(text)

    def getCarState(self, _car: int, state: int) -> tuple[float, ...]:  # noqa: N802
        return self.states[state]


class _FakeBridgeFile:
    def __init__(self) -> None:
        self.payloads: list[bytes] = []
        self.closed = False

    def seek(self, position: int) -> None:
        assert position == 0

    def write(self, payload: bytes) -> int:
        self.payloads.append(payload)
        return len(payload)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _load_app(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _FakeAc]:
    fake_ac = _FakeAc()
    fake_acsys = ModuleType("acsys")
    fake_acsys_dynamic: Any = fake_acsys
    fake_acsys_dynamic.CS = type("CS", (), {"SlipRatio": 1, "SlipAngle": 2})
    monkeypatch.setitem(sys.modules, "ac", fake_ac)
    monkeypatch.setitem(sys.modules, "acsys", fake_acsys)
    spec = importlib.util.spec_from_file_location("tested_ac_rl_bridge", APP_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fake_ac


def test_in_game_app_packet_matches_external_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, fake_ac = _load_app(monkeypatch)
    bridge_file = _FakeBridgeFile()

    def open_bridge(_path: str, mode: str, *_args: object) -> _FakeBridgeFile:
        if mode == "rb":
            raise OSError
        return bridge_file

    monkeypatch.setattr(app, "open", open_bridge, raising=False)
    monkeypatch.setattr(app.os, "remove", lambda _path: None)
    assert app.acMain("1.16.4") == "Assetto Corsa RL Bridge"
    app.acUpdate(0.02)
    assert len(bridge_file.payloads) == 1
    payload = bridge_file.payloads[0]
    packet = decode_slip_bridge_packet(payload, received_monotonic_s=1.0)
    assert packet.sequence == 0
    assert packet.slip_ratio == pytest.approx(fake_ac.states[1])
    assert packet.slip_angle_raw == pytest.approx(fake_ac.states[2])
    app.acShutdown()
    app.acShutdown()
    assert bridge_file.closed


def test_in_game_app_reports_invalid_api_shape_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, fake_ac = _load_app(monkeypatch)
    bridge_file = _FakeBridgeFile()

    def open_bridge(_path: str, mode: str, *_args: object) -> _FakeBridgeFile:
        if mode == "rb":
            raise OSError
        return bridge_file

    monkeypatch.setattr(app, "open", open_bridge, raising=False)
    monkeypatch.setattr(app.os, "remove", lambda _path: None)
    app.acMain("1.16.4")
    fake_ac.states[1] = (0.0,)
    app.acUpdate(0.02)
    app.acUpdate(0.02)
    assert not bridge_file.payloads
    assert sum("did not return FL" in entry for entry in fake_ac.logs) == 1


def test_in_game_app_acknowledges_csp_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, fake_ac = _load_app(monkeypatch)
    reset_calls: list[None] = []
    vars(fake_ac)["ext_resetCar"] = lambda: reset_calls.append(None)
    app.BRIDGE_FILE_PATH = str(tmp_path / "telemetry.bin")
    app.COMMAND_FILE_PATH = str(tmp_path / "command.bin")
    app.ACK_FILE_PATH = str(tmp_path / "ack.bin")
    Path(app.COMMAND_FILE_PATH).write_bytes(encode_reset_command(23))

    app.acMain("1.16.4")
    app.acUpdate(0.02)

    acknowledgement = decode_reset_acknowledgement(Path(app.ACK_FILE_PATH).read_bytes())
    assert acknowledgement.sequence == 23
    assert acknowledgement.status is ResetBridgeStatus.SUCCESS
    assert reset_calls == [None]
    assert not Path(app.COMMAND_FILE_PATH).exists()
    app.acUpdate(0.02)
    assert reset_calls == [None]
    app.acShutdown()


def test_in_game_app_retries_ack_replace_during_windows_reader_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _ = _load_app(monkeypatch)
    app.ACK_FILE_PATH = str(tmp_path / "ack.bin")
    original_replace = app.os.replace
    replace_attempts = 0
    sleep_calls: list[float] = []

    def sharing_sensitive_replace(source: str, destination: str) -> None:
        nonlocal replace_attempts
        replace_attempts += 1
        if replace_attempts == 1:
            raise PermissionError(5, "simulated Windows sharing violation")
        original_replace(source, destination)

    monkeypatch.setattr(app.os, "replace", sharing_sensitive_replace)
    monkeypatch.setattr(app.time, "sleep", sleep_calls.append)

    app._write_reset_ack(41, app.ACK_SUCCESS)

    acknowledgement = decode_reset_acknowledgement(Path(app.ACK_FILE_PATH).read_bytes())
    assert acknowledgement.sequence == 41
    assert replace_attempts == 2
    assert sleep_calls == [app.ACK_REPLACE_RETRY_DELAY_S]
