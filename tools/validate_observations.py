"""Live validation of the complete RL observation."""

import matplotlib.pyplot as plt

from assetto_corsa_ai.observation import Observation


def update_plot(
    observation: Observation,
    axis,
    left_line,
    right_line,
    reference_line,
    info_text,
) -> None:
    """Update the live observation plot."""

    # Observation points are stored as:
    # (forward_distance, left_distance)
    #
    # For plotting:
    # x = lateral / left distance
    # y = forward distance

    left_x = [point[1] for point in observation.left_boundary]
    left_y = [point[0] for point in observation.left_boundary]

    right_x = [point[1] for point in observation.right_boundary]
    right_y = [point[0] for point in observation.right_boundary]

    reference_x = [point[1] for point in observation.reference_waypoints]
    reference_y = [point[0] for point in observation.reference_waypoints]

    left_line.set_data(left_x, left_y)
    right_line.set_data(right_x, right_y)
    reference_line.set_data(reference_x, reference_y)

    info_text.set_text(
        "\n".join(
            (
                f"speed:              {observation.speed:7.2f} m/s",
                f"long. acceleration: {observation.longitudinal_acceleration:+7.3f} g",
                f"lat. acceleration:  {observation.lateral_acceleration:+7.3f} g",
                f"yaw rate:           {observation.yaw_rate:+7.3f} rad/s",
                "",
                f"front slip ratio:   {observation.front_slip_ratio:+7.3f}",
                f"rear slip ratio:    {observation.rear_slip_ratio:+7.3f}",
                f"front slip angle:   {observation.front_slip_angle:+7.3f} rad",
                f"rear slip angle:    {observation.rear_slip_angle:+7.3f} rad",
                "",
                f"steering:            {observation.steering:+7.3f}",
                f"long. control:       {observation.longitudinal_control:+7.3f}",
            )
        )
    )

    # Adjust view to currently observed points.
    all_x = left_x + right_x + reference_x + [0.0]
    all_y = left_y + right_y + reference_y + [0.0]

    if all_x and all_y:
        x_margin = max(
            5.0,
            (max(all_x) - min(all_x)) * 0.15,
        )

        y_margin = max(
            10.0,
            (max(all_y) - min(all_y)) * 0.05,
        )

        axis.set_xlim(
            min(all_x) - x_margin,
            max(all_x) + x_margin,
        )

        axis.set_ylim(
            min(-5.0, min(all_y) - y_margin),
            max(all_y) + y_margin,
        )


def create_plot():
    """Create the live observation visualization."""

    plt.ion()

    figure, axis = plt.subplots(figsize=(8, 10))

    (left_line,) = axis.plot(
        [],
        [],
        "o-",
        label="Left boundary",
    )

    (right_line,) = axis.plot(
        [],
        [],
        "o-",
        label="Right boundary",
    )

    (reference_line,) = axis.plot(
        [],
        [],
        "x-",
        label="Reference trajectory",
    )

    # Vehicle remains at the origin because all observation
    # coordinates are vehicle-relative.
    axis.plot(
        0.0,
        0.0,
        marker="^",
        markersize=12,
        label="Vehicle",
    )

    # Forward-direction indicator.
    axis.plot(
        [0.0, 0.0],
        [0.0, 8.0],
        "--",
        linewidth=1.0,
    )

    axis.set_xlabel("Lateral distance [m]  ← right | left →")
    axis.set_ylabel("Forward distance [m]")

    axis.set_title("RL Observation Validation")

    axis.grid(True)
    axis.legend()

    info_text = axis.text(
        1.03,
        0.98,
        "",
        transform=axis.transAxes,
        verticalalignment="top",
        family="monospace",
    )

    figure.subplots_adjust(right=0.70)

    return (
        figure,
        axis,
        left_line,
        right_line,
        reference_line,
        info_text,
    )
