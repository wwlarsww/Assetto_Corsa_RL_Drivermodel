"""Verify managed launch and cleanup in the interactive validation tool."""

from unittest.mock import Mock, create_autospec

import pytest

from assetto_corsa_ai.assetto_corsa.session_manager import (
    AssettoCorsaSessionManager,
    SessionManagerError,
    SimulatorScenario,
)
from assetto_corsa_ai.assetto_corsa.simulator_controller import AssettoCorsaSimulatorController
from assetto_corsa_ai.assetto_corsa.static_info_reader import StaticInfoReader
from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from tools.validation import validate_simulator_control as validation


@pytest.mark.parametrize("outcome", ["success", "failure", "already_running", "access_denied", "connection_failure"])
def test_managed_launch_connects_after_start_and_closes_owned_resources(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], outcome: str,
) -> None:
    manager = create_autospec(AssettoCorsaSessionManager, instance=True)
    controller = create_autospec(AssettoCorsaSimulatorController, instance=True)
    controller.is_connected = outcome == "success"
    static = create_autospec(StaticInfoReader, instance=True)
    if outcome == "access_denied":
        static.read.side_effect = PermissionError("Shared memory access denied")
    elif outcome != "already_running":
        static.read.side_effect = FileNotFoundError("AC has not created shared memory yet")
    reader = create_autospec(TelemetryReader, instance=True)
    events = Mock()
    for name, mock in (("manager", manager), ("controller", controller), ("static", static), ("reader", reader)):
        events.attach_mock(mock, name)
    if outcome == "failure":
        manager.start_session.side_effect = SessionManagerError("Startup failed")
    if outcome == "connection_failure":
        controller.connect.side_effect = validation.SimulatorControllerError("CSP activation timed out")
        manager.is_running = True
    monkeypatch.setattr(validation, "AssettoCorsaSessionManager", Mock(return_value=manager))
    monkeypatch.setattr(validation, "AssettoCorsaSimulatorController", Mock(return_value=controller))
    monkeypatch.setattr(validation, "StaticInfoReader", Mock(return_value=static))
    monkeypatch.setattr(validation, "TelemetryReader", Mock(return_value=reader))
    menu = Mock()
    monkeypatch.setattr(validation, "run_menu", menu)
    def confirm_driving(prompt: str) -> str:
        return ""

    monkeypatch.setattr("builtins.input", confirm_driving)
    monkeypatch.setattr("sys.argv", ["validate_simulator_control.py", "--launch", "--car", "test_car", "--track", "test_track"])

    if outcome == "success":
        validation.main()
        manager.start_session.assert_called_once_with(SimulatorScenario(car="test_car", track="test_track"))
        controller.connect.assert_called_once()
        menu.assert_called_once_with(controller, reader)
        names = [call[0] for call in events.mock_calls]
        assert names.index("manager.start_session") < names.index("controller.connect")
        assert names.index("controller.close") < names.index("manager.stop_session")
        controller.clear_position_request.assert_called_once()
        reader.close.assert_called_once()
    else:
        with pytest.raises(SystemExit):
            validation.main()
        if outcome == "connection_failure":
            controller.connect.assert_called_once()
            reader.read.assert_not_called()
            output = capsys.readouterr().out
            assert output.index("Validation failed: CSP activation timed out") < output.index(
                "Stopping AC launched by this test script."
            )
        else:
            controller.connect.assert_not_called()
        menu.assert_not_called()
    if outcome in ("already_running", "access_denied"):
        manager.start_session.assert_not_called()
        manager.stop_session.assert_not_called()
    else:
        manager.stop_session.assert_called_once()
    static.close.assert_called_once()
    controller.close.assert_called_once()


@pytest.mark.parametrize("exited", [False, True])
def test_launch_only_does_not_connect_controls_or_open_telemetry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], exited: bool,
) -> None:
    manager = create_autospec(AssettoCorsaSessionManager, instance=True)
    manager.is_running = not exited
    static = create_autospec(StaticInfoReader, instance=True)
    static.read.side_effect = FileNotFoundError
    controller = create_autospec(AssettoCorsaSimulatorController, instance=True)
    controller.is_connected = False
    telemetry_factory = Mock()
    monkeypatch.setattr(validation, "AssettoCorsaSessionManager", Mock(return_value=manager))
    monkeypatch.setattr(validation, "StaticInfoReader", Mock(return_value=static))
    monkeypatch.setattr(validation, "AssettoCorsaSimulatorController", Mock(return_value=controller))
    monkeypatch.setattr(validation, "TelemetryReader", telemetry_factory)
    monkeypatch.setattr("sys.argv", ["validate_simulator_control.py", "--launch-only"])

    def inspect_game(prompt: str) -> str:
        # The game has launched, but the test script must leave it running
        # until the user has had a chance to inspect it independently of CSP.
        manager.start_session.assert_called_once()
        manager.stop_session.assert_not_called()
        controller.connect.assert_not_called()
        return ""

    monkeypatch.setattr("builtins.input", inspect_game)
    if exited:
        with pytest.raises(SystemExit, match="before CSP controls were connected"):
            validation.main()
    else:
        validation.main()
        assert "Stopping AC launched by this test script." in capsys.readouterr().out
    controller.connect.assert_not_called()
    telemetry_factory.assert_not_called()
    manager.stop_session.assert_called_once()
    static.close.assert_called_once()
    controller.close.assert_called_once()
