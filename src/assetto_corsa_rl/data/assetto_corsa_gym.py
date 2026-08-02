"""Reproducible acquisition of the BMW Z4 GT3/Monza dataset variant."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

DATASET_REPOSITORY = "dasgringuen/assettoCorsaGym"
DATASET_REVISION = "ad1df12c49cce10f159ba31967744750bb9edd8b"
SOURCE_REPOSITORY = "https://github.com/dasGringuen/assetto_corsa_gym"
SOURCE_REVISION = "ef0fa502bdd257fc2affad9361c02b3dbdfe1632"
DATASET_VARIANT = "bmw_z4_gt3_monza_v1"
TRACK_ID = "monza"
CAR_ID = "bmw_z4_gt3"
DATASET_ROOT = f"data_sets/{TRACK_ID}/{CAR_ID}"
TRACK_GRID_PATH = f"AssettoCorsaConfigs/tracks/{TRACK_ID}_1.0m.pkl"
_SAFE_RECORDING_SUFFIXES = {".ld", ".ldx", ".csv"}
_SOURCE_FILES = (
    ("tracks", f"{TRACK_ID}.csv"),
    ("tracks", f"{TRACK_ID}-racing_line.csv"),
    (f"cars/{CAR_ID}", "steer_map.csv"),
    (f"cars/{CAR_ID}", "brake_map.csv"),
)


@dataclass(frozen=True, slots=True)
class DownloadFile:
    """One selected immutable remote object."""

    path: str
    size: int


@dataclass(frozen=True, slots=True)
class ManifestFile:
    """Local content identity without machine-specific absolute paths."""

    path: str
    size: int
    sha256: str
    origin: str


def select_human_recordings(entries: Sequence[object]) -> list[DownloadFile]:
    """Select raw human files while excluding SAC data and Python pickles."""

    selected: list[DownloadFile] = []
    root = PurePosixPath(DATASET_ROOT)
    for entry in entries:
        path_value = getattr(entry, "path", None)
        size_value = getattr(entry, "size", None)
        if not isinstance(path_value, str) or not isinstance(size_value, int):
            continue
        path = PurePosixPath(path_value)
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) < 2:
            continue
        session = relative.parts[0]
        if "_SAC" in session.upper() or path.suffix.lower() not in _SAFE_RECORDING_SUFFIXES:
            continue
        selected.append(DownloadFile(path=path.as_posix(), size=size_value))
    return sorted(selected, key=lambda item: item.path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _download_source_files(destination: Path) -> list[Path]:
    local_paths: list[Path] = []
    source_root = PurePosixPath("assetto_corsa_gym", "AssettoCorsaConfigs")
    for directory, name in _SOURCE_FILES:
        relative = PurePosixPath("source", "AssettoCorsaConfigs", directory, name)
        local_path = destination.joinpath(*relative.parts)
        url = (
            "https://raw.githubusercontent.com/dasGringuen/assetto_corsa_gym/"
            f"{SOURCE_REVISION}/{source_root}/{directory}/{name}"
        )
        parsed_url = urllib.parse.urlsplit(url)
        if parsed_url.scheme != "https" or parsed_url.hostname != "raw.githubusercontent.com":
            raise RuntimeError(f"refusing unexpected source URL: {url}")
        # The scheme and host are restricted above; the path uses a pinned commit.
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            _write_atomic(local_path, response.read())
        local_paths.append(local_path)
    return local_paths


def download_initial_dataset(
    *,
    destination: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Download and verify raw human data plus the official baseline line."""

    try:
        from huggingface_hub import (  # noqa: PLC0415
            HfApi,
            snapshot_download,  # pyright: ignore[reportUnknownVariableType]
        )
    except ImportError as error:
        raise RuntimeError(
            'dataset support is optional; install the project with ".[data]"'
        ) from error

    api = HfApi()
    tree = list(
        api.list_repo_tree(
            DATASET_REPOSITORY,
            repo_type="dataset",
            path_in_repo=DATASET_ROOT,
            recursive=True,
            expand=True,
            revision=DATASET_REVISION,
        )
    )
    recordings = select_human_recordings(tree)
    if not recordings:
        raise RuntimeError("the pinned dataset revision contains no matching human recordings")

    remote_paths = [item.path for item in recordings]
    remote_paths.append(TRACK_GRID_PATH)
    snapshot_download(
        repo_id=DATASET_REPOSITORY,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=destination,
        allow_patterns=remote_paths,
        max_workers=8,
    )
    source_paths = _download_source_files(destination)

    manifest_files: list[ManifestFile] = []
    expected_sizes = {item.path: item.size for item in recordings}
    for relative, expected_size in sorted(expected_sizes.items()):
        local_path = destination.joinpath(*PurePosixPath(relative).parts)
        actual_size = local_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"size mismatch for {relative}: expected {expected_size}, got {actual_size}"
            )
        manifest_files.append(
            ManifestFile(
                path=relative,
                size=actual_size,
                sha256=_sha256(local_path),
                origin="huggingface_dataset",
            )
        )

    grid_path = destination.joinpath(*PurePosixPath(TRACK_GRID_PATH).parts)
    manifest_files.append(
        ManifestFile(
            path=TRACK_GRID_PATH,
            size=grid_path.stat().st_size,
            sha256=_sha256(grid_path),
            origin="huggingface_dataset",
        )
    )
    for source_path in source_paths:
        relative = source_path.relative_to(destination).as_posix()
        manifest_files.append(
            ManifestFile(
                path=relative,
                size=source_path.stat().st_size,
                sha256=_sha256(source_path),
                origin="github_source",
            )
        )

    sessions = {PurePosixPath(item.path).relative_to(DATASET_ROOT).parts[0] for item in recordings}
    manifest: dict[str, object] = {
        "schema_version": 1,
        "selection": {
            "variant": DATASET_VARIANT,
            "track_id": TRACK_ID,
            "car_id": CAR_ID,
            "sessions": len(sessions),
            "human_recordings_only": True,
            "included_suffixes": sorted(_SAFE_RECORDING_SUFFIXES),
            "excluded": ["SAC", "replay_buffers", "checkpoints", "pickle_telemetry"],
        },
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "license": "CC-BY-4.0",
        },
        "source_code": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "license": "MIT",
        },
        "file_count": len(manifest_files),
        "total_bytes": sum(item.size for item in manifest_files),
        "files": [asdict(item) for item in sorted(manifest_files, key=lambda item: item.path)],
    }
    rendered = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _write_atomic(manifest_path, rendered)
    return manifest
