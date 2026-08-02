from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from conftest import FastLaneFactory

from assetto_corsa_rl.data import assetto_corsa_gym
from assetto_corsa_rl.data.assetto_corsa_gym import select_human_recordings
from assetto_corsa_rl.data.baseline_reference import build_baseline_reference
from assetto_corsa_rl.data.motec import _CHANNEL, _HEADER, read_motec_channels
from assetto_corsa_rl.references.fixed_reference import ReferenceTrajectory


@dataclass(frozen=True)
class _Entry:
    path: str
    size: int | None


def test_human_recording_selection_excludes_sac_and_pickle() -> None:
    root = "data_sets/monza/bmw_z4_gt3"
    entries: list[object] = [
        _Entry(f"{root}/20240229_HC/laps/stint.ld", 100),
        _Entry(f"{root}/20240229_HC/laps/stint.ldx", 10),
        _Entry(f"{root}/20240229_HC/laps/laps_info.csv", 20),
        _Entry(f"{root}/20240229_HC/laps/stint.pkl", 500),
        _Entry(f"{root}/20240311_SAC/laps/stint.ld", 600),
        _Entry("other/file.ld", 700),
        _Entry(f"{root}/folder", None),
    ]
    selected = select_human_recordings(entries)
    assert [item.path.rsplit(".", 1)[-1] for item in selected] == ["csv", "ld", "ldx"]
    assert sum(item.size for item in selected) == 130


def test_download_writes_verified_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = assetto_corsa_gym.DATASET_ROOT
    entries = [
        _Entry(f"{root}/20240229_HC/laps/stint.ld", 4),
        _Entry(f"{root}/20240229_HC/laps/stint.ldx", 5),
        _Entry(f"{root}/20240229_HC/laps/laps_info.csv", 6),
    ]
    sizes = {entry.path: cast(int, entry.size) for entry in entries}

    class _FakeApi:
        def list_repo_tree(self, *_args: object, **_kwargs: object) -> list[_Entry]:
            return entries

    def fake_snapshot_download(**kwargs: object) -> str:
        destination = cast(Path, kwargs["local_dir"])
        patterns = cast(list[str], kwargs["allow_patterns"])
        for relative in patterns:
            path = destination.joinpath(*Path(relative).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * sizes.get(relative, 4))
        return str(destination)

    def fake_source_download(destination: Path) -> list[Path]:
        paths: list[Path] = []
        for name in ("track.csv", "line.csv", "steer.csv", "brake.csv"):
            path = destination / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            paths.append(path)
        return paths

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeApi  # type: ignore[attr-defined]
    fake_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(
        assetto_corsa_gym,
        "_download_source_files",
        fake_source_download,
    )
    destination = tmp_path / "dataset"
    manifest_path = tmp_path / "manifest.json"
    manifest = assetto_corsa_gym.download_initial_dataset(
        destination=destination,
        manifest_path=manifest_path,
    )
    assert manifest["file_count"] == 8
    assert manifest["selection"] == {
        "variant": "bmw_z4_gt3_monza_v1",
        "track_id": "monza",
        "car_id": "bmw_z4_gt3",
        "sessions": 1,
        "human_recordings_only": True,
        "included_suffixes": [".csv", ".ld", ".ldx"],
        "excluded": ["SAC", "replay_buffers", "checkpoints", "pickle_telemetry"],
    }
    assert manifest_path.read_text(encoding="utf-8").endswith("\n")


def _write_motec(
    path: Path,
    channels: dict[str, tuple[npt.NDArray[np.int16], int]],
) -> None:
    metadata_pointer = _HEADER.size
    data_pointer = _HEADER.size + len(channels) * _CHANNEL.size
    header = _HEADER.pack(
        0x40,
        metadata_pointer,
        data_pointer,
        0,
        1,
        0,
        0,
        123,
        b"ADL",
        1,
        0,
        len(channels),
        b"01/01/2024",
        b"12:00",
        b"driver",
        b"bmw_z4_gt3",
        b"monza",
        0,
        b"synthetic",
    )
    metadata = bytearray()
    payload = bytearray()
    items = list(channels.items())
    for index, (name, (values, decimal_places)) in enumerate(items):
        current_pointer = metadata_pointer + index * _CHANNEL.size
        previous_pointer = current_pointer - _CHANNEL.size if index else 0
        next_pointer = current_pointer + _CHANNEL.size if index + 1 < len(items) else 0
        metadata.extend(
            _CHANNEL.pack(
                previous_pointer,
                next_pointer,
                data_pointer + len(payload),
                values.size,
                0,
                0x03,
                2,
                100,
                0,
                1,
                1,
                decimal_places,
                name.encode("ascii"),
                name[:8].encode("ascii"),
                b"",
            )
        )
        payload.extend(values.astype("<i2").tobytes())
    path.write_bytes(header + metadata + payload)


def test_motec_parser_and_baseline_reference_builder(
    tmp_path: Path,
    fast_lane_factory: FastLaneFactory,
) -> None:
    sample_count = 100
    phase_raw = np.arange(0, 1000, 10, dtype=np.int16)
    speed_kmh = np.linspace(36, 72, sample_count, dtype=np.int16)
    zero = np.zeros(sample_count, dtype=np.int16)
    recording = tmp_path / "recording.ld"
    _write_motec(
        recording,
        {
            "Car Pos Norm": (phase_raw, 3),
            "Ground Speed": (speed_kmh, 0),
            "In Pit": (zero, 0),
            "Num Tires Off Track": (zero, 0),
            "Lap Invalidated": (zero, 0),
        },
    )

    header, channels = read_motec_channels(recording, ("Car Pos Norm", "Ground Speed"))
    assert header.vehicle_id == "bmw_z4_gt3"
    np.testing.assert_allclose(channels["Car Pos Norm"][[0, -1]], [0.0, 0.99])

    installation = tmp_path / "assettocorsa"
    spline = fast_lane_factory(installation, 10.0, "monza")
    fast_lane_path = installation / "content/tracks/monza/ai/fast_lane.ai"
    output = tmp_path / "fixed_reference.npz"
    report = build_baseline_reference(
        recordings=[recording],
        fast_lane_path=fast_lane_path,
        output_path=output,
        distance_bin_m=1.0,
        speed_resolution_mps=0.5,
        smoothing_window_m=1.0,
    )
    reference = ReferenceTrajectory.from_npz(output)
    assert report.samples_seen == sample_count
    assert report.samples_used == sample_count
    assert report.empty_bins_before_interpolation == 0
    assert reference.track_distance_m.size == 10
    assert 9.0 < float(reference.speed_mps.min()) < float(reference.speed_mps.max()) < 21.0
    np.testing.assert_array_equal(reference.lateral_offset_m, np.zeros(10, dtype=np.float32))
    assert reference.world_position_xy_m is not None
    assert not np.allclose(reference.heading_rad, 0.0)
    assert reference.source_track_sha256 == spline.source_sha256
