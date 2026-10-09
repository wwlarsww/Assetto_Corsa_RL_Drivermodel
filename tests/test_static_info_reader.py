"""Check safe startup against real, isolated Windows shared-memory mappings."""

from mmap import ACCESS_WRITE, mmap as real_mmap
import os
from unittest.mock import Mock
from uuid import uuid4

import pytest

from assetto_corsa_ai.assetto_corsa import static_info_reader as module


@pytest.fixture
def mapping_name(monkeypatch: pytest.MonkeyPatch) -> str:
    if os.name != "nt":
        pytest.skip("Named shared memory requires Windows")
    name = f"Local\\ac_ai_test_{uuid4().hex}"
    monkeypatch.setattr(module, "_STATIC_MAPPING_NAME", name)
    return name


def test_missing_mapping_is_not_created(
    mapping_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_memory = Mock()
    monkeypatch.setattr(module.mmap, "mmap", open_memory)
    reader = module.StaticInfoReader()
    try:
        open_memory.assert_not_called()
        with pytest.raises(FileNotFoundError):
            reader.read()
        open_memory.assert_not_called()
    finally:
        reader.close()


def test_reader_can_wait_for_simulator_then_read_writable_memory(mapping_name: str) -> None:
    reader = module.StaticInfoReader()
    try:
        with pytest.raises(FileNotFoundError):
            reader.read()
        # The simulator owns the full mapping and its write permission.
        with real_mmap(-1, 4096, tagname=mapping_name, access=ACCESS_WRITE) as simulator:
            simulator[68:78] = "car_a".encode("utf-16-le")
            simulator[134:148] = "track_a".encode("utf-16-le")
            info = reader.read()
            assert info.car_model == "car_a"
            assert info.track_name == "track_a"
            simulator[68:78] = "car_b".encode("utf-16-le")
            assert reader.read().car_model == "car_b"
            reader.close()
            # Closing the reader must not close the simulator's handle.
            simulator[0] = 1
    finally:
        reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader.read()


def test_failed_view_does_not_keep_simulator_mapping_alive(
    mapping_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = module.StaticInfoReader()
    try:
        with real_mmap(-1, 4096, tagname=mapping_name, access=ACCESS_WRITE):
            with monkeypatch.context() as patch:
                patch.setattr(module.mmap, "mmap", Mock(side_effect=OSError("View failed")))
                with pytest.raises(OSError, match="View failed"):
                    reader.read()
        # Both the owner's handle and the temporary OpenFileMapping handle
        # must be closed, so a retry sees missing memory rather than stale data.
        with pytest.raises(FileNotFoundError):
            reader.read()
    finally:
        reader.close()


def test_closed_reader_never_opens_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    open_memory = Mock()
    monkeypatch.setattr(module.StaticInfoReader, "_open_memory", open_memory)
    reader = module.StaticInfoReader()
    reader.close()
    reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader.read()
    open_memory.assert_not_called()
