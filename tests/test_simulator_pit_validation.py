"""Verify pit heading correction using simulated asynchronous telemetry."""

from dataclasses import replace
from itertools import count
from math import sqrt
from unittest.mock import Mock, create_autospec

import pytest

from assetto_corsa_ai.assetto_corsa.simulator_controller import (
    AssettoCorsaSimulatorController,
    SimulatorControllerError,
)
from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.telemetry import TelemetryFrame, Vector3
from tools import validate_simulator_control as validation


def frame(position: Vector3, forward: Vector3 = (0.0, 0.0, 1.0)) -> TelemetryFrame:
    front: Vector3 = (position[0] + forward[0], position[1] + forward[1], position[2] + forward[2])
    rear: Vector3 = (position[0] - forward[0], position[1] - forward[1], position[2] - forward[2])
    return TelemetryFrame(
        speed_kmh=0.0, gas=0.0, brake=1.0, fuel=30.0, gear=1, rpm=0,
        steering_angle=0.0, acceleration_g=(0.0, 0.0, 0.0),
        wheel_slip=(0.0, 0.0, 0.0, 0.0), wheel_angular_speed=(0.0, 0.0, 0.0, 0.0),
        tyre_contact_points=(front, front, rear, rear), local_velocity=(0.0, 0.0, 0.0),
        local_angular_velocity=(0.0, 0.0, 0.0), car_coordinates=position,
        normalized_track_position=0.0, car_damage=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = count(0.0, 0.05)
    monkeypatch.setattr(validation, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(validation, "sleep", Mock())


@pytest.mark.parametrize("forward", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.6, 0.8)])
@pytest.mark.parametrize("already_in_pits", [False, True])
def test_pit_correction_waits_for_stationary_pose_and_preserves_position_and_pitch(
    monkeypatch: pytest.MonkeyPatch, clock: None, forward: Vector3, already_in_pits: bool,
) -> None:
    initial = frame((100.0, 5.0, 100.0))
    pit = frame((-10.0, 2.0, 20.0), forward)
    if already_in_pits:
        initial = frame(pit.car_coordinates, (-forward[0], forward[1], -forward[2]))
    controller = create_autospec(AssettoCorsaSimulatorController, instance=True)
    reader = create_autospec(TelemetryReader, instance=True)
    reader.read.side_effect = [initial, initial, replace(pit, speed_kmh=5.0)] + [pit] * 20
    correction = Mock()
    events = Mock()
    events.attach_mock(controller, "controller")
    events.attach_mock(correction, "correction")
    monkeypatch.setattr(validation, "validate_teleport", correction)

    validation.validate_pit_teleport(controller, reader)

    controller.set_controls.assert_called_once_with(0.0, 0.0, 1.0)
    controller.teleport_to_pits.assert_called_once()
    correction.assert_called_once()
    args = correction.call_args.args
    assert args[:2] == (controller, reader)
    assert args[2] == pit.car_coordinates
    assert args[3] == pytest.approx((-forward[0], forward[1], -forward[2]))
    names = [call[0] for call in events.mock_calls]
    assert names.index("controller.teleport_to_pits") < names.index("controller.clear_position_request")
    assert names.index("controller.clear_position_request") < names.index("correction")
    assert names[-1] == "controller.clear_position_request"


@pytest.mark.parametrize("failure", ["unchanged", "moving", "invalid_heading", "invalid_position", "interrupt", "correction"])
def test_failed_pit_validation_clears_request_and_keeps_brake_applied(
    monkeypatch: pytest.MonkeyPatch, clock: None, failure: str,
) -> None:
    initial = frame((100.0, 5.0, 100.0))
    pit = frame((-10.0, 2.0, 20.0))
    controller = create_autospec(AssettoCorsaSimulatorController, instance=True)
    reader = create_autospec(TelemetryReader, instance=True)
    correction = Mock(side_effect=SimulatorControllerError("Correction failed"))
    monkeypatch.setattr(validation, "validate_teleport", correction)
    if failure == "unchanged":
        pit = initial
    elif failure == "moving":
        pit = replace(pit, speed_kmh=5.0)
    elif failure == "invalid_heading":
        pit = replace(pit, tyre_contact_points=((0.0, 0.0, 0.0),) * 4)
    elif failure == "invalid_position":
        pit = replace(pit, car_coordinates=(float("nan"), 0.0, 0.0))
    reader.read.side_effect = [initial, KeyboardInterrupt()] if failure == "interrupt" else [initial] + [pit] * 100
    error = KeyboardInterrupt if failure == "interrupt" else SimulatorControllerError

    with pytest.raises(error):
        validation.validate_pit_teleport(controller, reader)

    controller.clear_position_request.assert_called()
    controller.set_controls.assert_called_once_with(0.0, 0.0, 1.0)
    controller.neutral_controls.assert_not_called()
    if failure != "correction":
        correction.assert_not_called()


def test_heading_check_rejects_reversed_vehicle() -> None:
    position = (10.0, 2.0, 20.0)
    forward = (0.6, 0.0, 0.8)
    telemetry = frame(position, forward)
    assert validation.vehicle_forward(telemetry) == pytest.approx(forward)
    assert validation.reset_ready(telemetry, position, forward)
    assert not validation.reset_ready(telemetry, position, (-0.6, 0.0, -0.8))
    assert sqrt(sum(value * value for value in validation.vehicle_forward(telemetry))) == pytest.approx(1.0)
