"""Vehicle heading and axle offsets, shared by runtime and validation tools."""

from math import hypot, isfinite

from assetto_corsa_ai.telemetry import TelemetryFrame, Vector3


def vehicle_forward(tyre_contact_points: tuple[Vector3, ...]) -> Vector3:
    """Unit rear-to-front direction in AC world coordinates, including at rest."""
    if len(tyre_contact_points) != 4:
        raise ValueError("Expected tyre contact points for exactly four wheels.")
    fl, fr, rl, rr = tyre_contact_points
    vector = tuple((fl[i] + fr[i] - rl[i] - rr[i]) / 2.0 for i in range(3))
    length = hypot(*vector)
    if not isfinite(length) or length <= 1e-9:
        raise ValueError("Cannot determine heading from invalid tyre positions.")
    return vector[0] / length, vector[1] / length, vector[2] / length


def estimate_axle_distances(telemetry: TelemetryFrame) -> tuple[float, float]:
    """Project CG-to-axle offsets onto the horizontal vehicle direction."""
    if len(telemetry.tyre_contact_points) != 4:
        raise ValueError("Expected tyre contact points for exactly four wheels.")
    fl, fr, rl, rr = telemetry.tyre_contact_points
    front = ((fl[0] + fr[0]) / 2, (fl[2] + fr[2]) / 2)
    rear = ((rl[0] + rr[0]) / 2, (rl[2] + rr[2]) / 2)
    dx, dz = front[0] - rear[0], front[1] - rear[1]
    wheelbase = hypot(dx, dz)
    if not isfinite(wheelbase) or wheelbase <= 1e-9:
        raise ValueError("Cannot estimate axle distances from invalid tyre positions.")
    x, _, z = telemetry.car_coordinates
    distances = (((front[0] - x) * dx + (front[1] - z) * dz) / wheelbase,
                 ((x - rear[0]) * dx + (z - rear[1]) * dz) / wheelbase)
    if any(not isfinite(value) or value <= 0 for value in distances):
        raise ValueError("Car origin must lie between the axles; configure explicit axle distances.")
    return distances
