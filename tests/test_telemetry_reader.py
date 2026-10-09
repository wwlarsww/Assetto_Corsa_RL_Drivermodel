"""Verify damage decoding from the simulator's binary physics page."""

from io import BytesIO
from struct import pack_into

import pytest

from assetto_corsa_ai.assetto_corsa import telemetry_reader
from assetto_corsa_ai.telemetry import DamageValues


@pytest.mark.parametrize(
    "damage",
    [
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (1.25, 2.5, 3.75, 4.0, 5.5),
    ],
)
def test_reads_all_damage_channels_in_simulator_order(
    monkeypatch: pytest.MonkeyPatch,
    damage: DamageValues,
) -> None:
    memories: list[BytesIO] = []

    def open_memory(
        fileno: int,
        length: int,
        tagname: str,
        access: int,
    ) -> BytesIO:
        raw = bytearray(length)
        if tagname == "Local\\acpmf_physics":
            # Fixed AC physics layout: speed at byte 28, carDamage at byte 224.
            # Use independent offsets to catch accidental layout changes.
            pack_into("=f", raw, 28, 72.0)
            pack_into("=5f", raw, 224, *damage)
            pack_into("=i", raw, 244, 2)  # numberOfTyresOut, after carDamage
        memory = BytesIO(raw)
        memories.append(memory)
        return memory

    monkeypatch.setattr(telemetry_reader.mmap, "mmap", open_memory)
    reader = telemetry_reader.TelemetryReader()
    try:
        frame = reader.read()
        assert frame.car_damage == damage
        assert frame.speed_kmh == 72.0
        assert reader.read().car_damage == damage
    finally:
        reader.close()

    assert all(memory.closed for memory in memories)
