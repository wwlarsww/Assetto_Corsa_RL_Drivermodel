from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

TRAJECTORY_PATH = Path(
    r"C:\Users\larsk\OneDrive\Projekte"
    r"\Assetto_Corsa_RL_Drivermodel"
    r"\runs\ppo-assetto-corsa"
    r"\20260805T130251Z-seed-769913"
    r"\evaluations\update_000001"
    r"\trajectories.npz"
)

GEOMETRY_PATH = Path(
    r"data\generated\assetto_corsa_tracks" r"\monza\default\track_geometry.npz"
)


def scalar_text(
    archive: np.lib.npyio.NpzFile,
    key: str,
) -> str:
    if key not in archive.files:
        return "<nicht vorhanden>"

    value = np.asarray(archive[key]).reshape(-1)

    if value.size == 0:
        return "<leer>"

    return str(value[0])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def maximum_point_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)

    if first.shape != second.shape:
        return float("nan")

    return float(
        np.max(
            np.linalg.norm(
                first - second,
                axis=1,
            )
        )
    )


if not TRAJECTORY_PATH.is_file():
    raise FileNotFoundError(f"Trajektorie nicht gefunden: {TRAJECTORY_PATH}")

if not GEOMETRY_PATH.is_file():
    raise FileNotFoundError(f"Geometrie nicht gefunden: {GEOMETRY_PATH.resolve()}")


with (
    np.load(TRAJECTORY_PATH, allow_pickle=False) as trajectory,
    np.load(GEOMETRY_PATH, allow_pickle=False) as geometry,
):
    print("=" * 70)
    print("1. Koordinatensystem")
    print("=" * 70)

    print(
        "Trajektorie coordinate_frame:",
        scalar_text(trajectory, "coordinate_frame"),
    )
    print(
        "Trajektorie coordinate_transform:",
        scalar_text(trajectory, "coordinate_transform"),
    )
    print(
        "Geometrie coordinate_frame:",
        scalar_text(geometry, "coordinate_frame"),
    )
    print(
        "Geometrie coordinate_transform:",
        scalar_text(geometry, "coordinate_transform"),
    )

    print()
    print("=" * 70)
    print("2. Datei Hashes")
    print("=" * 70)

    stored_geometry_hash = scalar_text(
        trajectory,
        "track_geometry_sha256",
    )
    external_geometry_hash = file_sha256(GEOMETRY_PATH)

    print(
        "In Trajektorie gespeicherter Geometrie Hash:",
        stored_geometry_hash,
    )
    print(
        "Hash der geladenen Geometriedatei:",
        external_geometry_hash,
    )
    print(
        "Geometrie Hash stimmt überein:",
        stored_geometry_hash == external_geometry_hash,
    )

    trajectory_fast_lane_hash = scalar_text(
        trajectory,
        "source_fast_lane_sha256",
    )
    geometry_fast_lane_hash = scalar_text(
        geometry,
        "source_fast_lane_sha256",
    )

    print()
    print(
        "Trajektorie fast_lane Hash:",
        trajectory_fast_lane_hash,
    )
    print(
        "Geometrie fast_lane Hash:",
        geometry_fast_lane_hash,
    )
    print(
        "fast_lane Hash stimmt überein:",
        trajectory_fast_lane_hash == geometry_fast_lane_hash,
    )

    print()
    print("=" * 70)
    print("3. Eingebettete und externe Geometrie")
    print("=" * 70)

    embedded_center = np.asarray(
        trajectory["track_centerline_world_xy_m"],
        dtype=np.float64,
    )
    embedded_left = np.asarray(
        trajectory["track_left_boundary_world_xy_m"],
        dtype=np.float64,
    )
    embedded_right = np.asarray(
        trajectory["track_right_boundary_world_xy_m"],
        dtype=np.float64,
    )

    external_center = np.asarray(
        geometry["centerline_world_xy_m"],
        dtype=np.float64,
    )
    external_left = np.asarray(
        geometry["left_boundary_world_xy_m"],
        dtype=np.float64,
    )
    external_right = np.asarray(
        geometry["right_boundary_world_xy_m"],
        dtype=np.float64,
    )

    print("Center eingebettet:", embedded_center.shape)
    print("Center extern:", external_center.shape)
    print("Links eingebettet:", embedded_left.shape)
    print("Links extern:", external_left.shape)
    print("Rechts eingebettet:", embedded_right.shape)
    print("Rechts extern:", external_right.shape)

    print()
    print(
        "Maximale Center Abweichung:",
        maximum_point_difference(
            embedded_center,
            external_center,
        ),
        "m",
    )
    print(
        "Maximale linke Rand Abweichung:",
        maximum_point_difference(
            embedded_left,
            external_left,
        ),
        "m",
    )
    print(
        "Maximale rechte Rand Abweichung:",
        maximum_point_difference(
            embedded_right,
            external_right,
        ),
        "m",
    )

    print()
    print("=" * 70)
    print("4. Transformationstest der Evaluation")
    print("=" * 70)

    if "frame_roundtrip_error_m" in trajectory.files:
        roundtrip_error = np.asarray(
            trajectory["frame_roundtrip_error_m"],
            dtype=np.float64,
        )

        finite = roundtrip_error[np.isfinite(roundtrip_error)]

        if finite.size:
            print(
                "Median Roundtrip Fehler:",
                float(np.median(finite)),
                "m",
            )
            print(
                "Maximum Roundtrip Fehler:",
                float(np.max(finite)),
                "m",
            )
        else:
            print("Keine gültigen Roundtrip Werte vorhanden.")
    else:
        print("frame_roundtrip_error_m nicht vorhanden.")

    print()
    print("=" * 70)
    print("5. Geometrie Metadaten")
    print("=" * 70)

    metadata_text = scalar_text(
        geometry,
        "metadata_json",
    )

    if metadata_text not in {
        "<nicht vorhanden>",
        "<leer>",
    }:
        metadata = json.loads(metadata_text)

        for key in (
            "track_id",
            "track_layout",
            "boundary_definition",
            "source_fast_lane_path",
            "source_fast_lane_sha256",
            "coordinate_frame",
            "coordinate_transform",
        ):
            print(f"{key}: {metadata.get(key)}")
