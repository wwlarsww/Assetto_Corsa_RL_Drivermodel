import pytest

from assetto_corsa_ai.action import VehicleAction


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
