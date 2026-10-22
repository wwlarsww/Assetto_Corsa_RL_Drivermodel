"""Verify damage decoding from the simulator's binary physics page."""

from io import BytesIO
from struct import pack_into
from mmap import ACCESS_WRITE, mmap as real_mmap
import os
from uuid import uuid4
from unittest.mock import Mock

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
        tagname: str,
        length: int,
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

    monkeypatch.setattr(telemetry_reader, "open_readonly_mapping", open_memory)
    reader = telemetry_reader.TelemetryReader()
    try:
        frame = reader.read()
        assert frame.car_damage == damage
        assert frame.speed_kmh == 72.0
        assert reader.read().car_damage == damage
    finally:
        reader.close()

    assert all(memory.closed for memory in memories)


@pytest.fixture
def isolated_mappings(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    if os.name != "nt":
        pytest.skip("Named shared memory requires Windows")
    names = (f"Local\\ac_test_physics_{uuid4().hex}", f"Local\\ac_test_graphics_{uuid4().hex}")
    monkeypatch.setattr(telemetry_reader, "_PHYSICS_MAPPING_NAME", names[0])
    monkeypatch.setattr(telemetry_reader, "_GRAPHICS_MAPPING_NAME", names[1])
    return names


def test_missing_telemetry_mapping_is_never_created(isolated_mappings: tuple[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    from assetto_corsa_ai.assetto_corsa import shared_memory
    open_memory = Mock()
    monkeypatch.setattr(shared_memory.mmap, "mmap", open_memory)
    with pytest.raises(FileNotFoundError):
        telemetry_reader.TelemetryReader()
    open_memory.assert_not_called()


def test_existing_telemetry_is_read_only_and_packet_id_tracks_updates(isolated_mappings: tuple[str, str]) -> None:
    physics_name, graphics_name = isolated_mappings
    with real_mmap(-1, 4096, tagname=physics_name, access=ACCESS_WRITE) as physics:
        with real_mmap(-1, 4096, tagname=graphics_name, access=ACCESS_WRITE):
            physics[0:4] = (17).to_bytes(4, "little")
            reader = telemetry_reader.TelemetryReader()
            try:
                reader.read()
                assert reader.last_packet_id == 17
                physics[0:4] = (18).to_bytes(4, "little")
                reader.read()
                assert reader.last_packet_id == 18
            finally:
                reader.close()
            physics[0:4] = (19).to_bytes(4, "little")


def test_failed_graphics_open_releases_the_physics_mapping(isolated_mappings: tuple[str, str]) -> None:
    physics_name, _ = isolated_mappings
    with real_mmap(-1, 4096, tagname=physics_name, access=ACCESS_WRITE):
        with pytest.raises(FileNotFoundError):
            telemetry_reader.TelemetryReader()
    # The failed reader must not keep the owner's physics mapping alive.
    with pytest.raises(FileNotFoundError):
        telemetry_reader.open_readonly_mapping(physics_name, 4096)
