"""Inspect raw damage and per-sample changes while driving manually."""

from time import monotonic, sleep

from assetto_corsa_ai.assetto_corsa.telemetry_reader import TelemetryReader
from assetto_corsa_ai.telemetry import DamageValues

SAMPLE_INTERVAL_SECONDS = 0.05


def damage_change(
    current: DamageValues,
    previous: DamageValues | None,
) -> DamageValues:
    """Return signed channel changes; the first sample establishes a baseline."""

    if previous is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    return (
        current[0] - previous[0],
        current[1] - previous[1],
        current[2] - previous[2],
        current[3] - previous[3],
        current[4] - previous[4],
    )


def main() -> None:
    """Print damage values and retain the largest observed positive change."""

    reader = TelemetryReader()
    previous: DamageValues | None = None
    peak_increase = 0.0
    started_at = monotonic()

    try:
        print("Damage channels [0..4]: raw values, no assumed units or threshold.")
        print("Drive manually; compare normal driving, light contact and impacts.")
        print("Negative changes may indicate repair/reset. Stop with Ctrl+C.")

        while True:
            telemetry = reader.read()
            current = telemetry.car_damage
            changes = damage_change(current, previous)
            increase = max(0.0, *changes)
            peak_increase = max(peak_increase, increase)
            values_text = ", ".join(f"{value:.3f}" for value in current)
            changes_text = ", ".join(f"{value:+.3f}" for value in changes)

            print(
                f"t={monotonic() - started_at:7.2f}s | "
                f"speed={telemetry.speed_kmh:6.1f} km/h | "
                f"damage=[{values_text}] | delta=[{changes_text}] | "
                f"max increase={increase:.3f} | peak={peak_increase:.3f}"
            )

            previous = current
            sleep(SAMPLE_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print(f"\nLargest channel increase during this run: {peak_increase:.3f}")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
