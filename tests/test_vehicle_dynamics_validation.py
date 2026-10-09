"""Check CSP connection and cleanup for the vehicle-dynamics tool."""

from unittest.mock import Mock, create_autospec

import pytest

from assetto_corsa_ai.assetto_corsa.simulator_controller import (
    AssettoCorsaSimulatorController,
    SimulatorControllerError,
)
from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from tools import validate_vehicle_dynamics as validation


@pytest.mark.parametrize("outcome", ["success", "connect_error", "telemetry_error", "interrupted", "close_error"])
def test_dynamics_tool_connects_csp_and_releases_resources_on_failure(
    monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    controller = create_autospec(AssettoCorsaSimulatorController, instance=True)
    reader = create_autospec(TelemetryReader, instance=True)
    reader_factory = Mock(return_value=reader)
    manoeuvre = Mock()
    events = Mock()
    events.attach_mock(controller, "controller")
    events.attach_mock(reader, "reader")
    events.attach_mock(reader_factory, "open_telemetry")
    events.attach_mock(manoeuvre, "manoeuvre")
    monkeypatch.setattr(validation, "AssettoCorsaSimulatorController", Mock(return_value=controller))
    monkeypatch.setattr(validation, "TelemetryReader", reader_factory)
    monkeypatch.setattr(validation, "run_test", manoeuvre)
    if outcome == "connect_error":
        controller.connect.side_effect = SimulatorControllerError("CSP unavailable")
    elif outcome == "telemetry_error":
        reader_factory.side_effect = OSError("Telemetry unavailable")
    elif outcome == "interrupted":
        manoeuvre.side_effect = KeyboardInterrupt
    elif outcome == "close_error":
        controller.close.side_effect = OSError("Controller cleanup failed")

    if outcome == "success":
        validation.main()
    else:
        error = {
            "connect_error": SimulatorControllerError,
            "telemetry_error": OSError,
            "interrupted": KeyboardInterrupt,
            "close_error": OSError,
        }[outcome]
        with pytest.raises(error):
            validation.main()

    controller.close.assert_called_once()
    if outcome == "connect_error":
        reader_factory.assert_not_called()
    else:
        reader_factory.assert_called_once()
        names = [call[0] for call in events.mock_calls]
        assert names.index("controller.connect") < names.index("open_telemetry")
    if outcome in ("connect_error", "telemetry_error"):
        manoeuvre.assert_not_called()
        reader.close.assert_not_called()
    else:
        reader.close.assert_called_once()
        names = [call[0] for call in events.mock_calls]
        assert names.index("open_telemetry") < names.index("manoeuvre")
