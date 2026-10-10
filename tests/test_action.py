import pytest
from dataclasses import replace

from assetto_corsa_ai.action import ActionConfig, AgentAction, VehicleAction


def test_vehicle_action_accepts_valid_values() -> None:
    action = VehicleAction(
        steering=-0.5, # Valid steering value within the range [-1.0, 1.0]
        throttle=0.7, # Valid throttle value within the range [0.0, 1.0]
        brake=0.2, # Valid brake value within the range [0.0, 1.0]
    )

    assert action.steering == -0.5
    assert action.throttle == 0.7
    assert action.brake == 0.2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steering", -1.1),
        ("steering", 1.1),
        ("throttle", -0.1),
        ("throttle", 1.1),
        ("brake", -0.1),
        ("brake", 1.1),
    ],
)
def test_vehicle_action_rejects_invalid_values(
    field: str,
    value: float,
) -> None:
    values = {
        "steering": 0.0,
        "throttle": 0.0,
        "brake": 0.0,
    }

    values[field] = value

    with pytest.raises(ValueError):
        VehicleAction(**values)


@pytest.mark.parametrize("current", [AgentAction(0.0, 0.0), AgentAction(0.95, -0.95), AgentAction(-0.4, 0.14)])
@pytest.mark.parametrize("requested", [AgentAction(1.0, -1.0), AgentAction(-1.0, 1.0), AgentAction(0.0, 0.03)])
def test_control_limits_and_deadband_never_exceed_per_step_budget(current: AgentAction, requested: AgentAction) -> None:
    config = ActionConfig(0.05, 0.10, 0.05)
    result = config.constrain(requested, current)
    assert abs(result.steering - current.steering) <= 0.05 + 1e-12
    assert abs(result.longitudinal - current.longitudinal) <= 0.10 + 1e-12
    assert -1 <= result.steering <= 1 and -1 <= result.longitudinal <= 1
    # Prediction and live execution can apply the same filter safely.
    assert config.constrain(result, current) == result


@pytest.mark.parametrize("value", [-0.04, 0.0, 0.04])
def test_longitudinal_deadband_settles_small_commands_to_coast(value: float) -> None:
    assert ActionConfig(0.05, 0.10, 0.05).constrain(AgentAction(0.0, value), AgentAction(0.0, 0.0)).longitudinal == 0


def test_deadband_approaches_neutral_without_exceeding_delta_limit() -> None:
    config = ActionConfig(0.05, 0.1, 0.2)
    current = AgentAction(0.0, 0.35)
    for _ in range(4):
        result = config.constrain(AgentAction(0.0, 0.0), current)
        assert abs(result.longitudinal - current.longitudinal) <= 0.1 + 1e-12
        current = result
    assert current.longitudinal == 0.0


def test_zero_limits_keep_legacy_unrestricted_controls() -> None:
    requested = AgentAction(1.0, -1.0)
    assert ActionConfig(0.0, 0.0, 0.0).constrain(requested, AgentAction(-1.0, 1.0)) == requested


@pytest.mark.parametrize("field,value", [
    ("max_steering_delta", -0.1), ("max_steering_delta", 2.1),
    ("max_longitudinal_delta", float("inf")), ("longitudinal_deadband", float("nan")),
    ("longitudinal_deadband", -0.1), ("longitudinal_deadband", 1.0),
])
def test_invalid_action_settings_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        replace(ActionConfig(0.05, 0.1, 0.05), **{field: value}).validate()
