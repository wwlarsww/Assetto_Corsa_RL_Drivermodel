"""Interactive MLflow evaluation viewer with explicit coordinate frames.

The viewer keeps three data domains separate:

* Assetto Corsa world XZ in metres for the top view.
* Vehicle-local model XZ, ``[left, forward]``, for the predictor contract.
* Semantic vehicle FL, ``[forward, left]``, for the agent view.

Normalised observations are never interpreted as physical coordinates.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from mlflow import MlflowClient

DEFAULT_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)
DEFAULT_REFERENCE_TIME_OFFSETS_S = np.arange(1, 21, dtype=np.float64) * 0.25
DEFAULT_BOUNDARY_PREVIEW_DISTANCES_M = np.asarray(
    [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class Projection:
    """Projection of one point onto a closed polyline."""

    progress_m: float
    point_xz_m: np.ndarray
    distance_m: float


def _client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


@st.cache_data(ttl=10, show_spinner=False)
def _list_experiments(tracking_uri: str) -> list[dict[str, str]]:
    return [
        {"experiment_id": item.experiment_id, "name": item.name}
        for item in _client(tracking_uri).search_experiments()
    ]


@st.cache_data(ttl=10, show_spinner=False)
def _list_runs(tracking_uri: str, experiment_id: str) -> list[dict[str, Any]]:
    runs = _client(tracking_uri).search_runs(
        experiment_ids=[experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    return [
        {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "run_name": run.data.tags.get("mlflow.runName", run.info.run_id),
            "tags": dict(run.data.tags),
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
        }
        for run in runs
    ]


def _training_runs(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        run
        for run in runs
        if str(run["tags"].get("run_type", "")).lower() != "evaluation"
        and not str(run["run_name"]).startswith("eval_update_")
    ]


def _evaluation_runs(
    runs: Sequence[dict[str, Any]],
    parent_run_id: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for run in runs:
        tags = run["tags"]
        parent = tags.get("parent_training_run_id") or tags.get("mlflow.parentRunId")
        if parent != parent_run_id:
            continue
        if str(tags.get("run_type", "evaluation")).lower() != "evaluation":
            continue
        item = dict(run)
        item["checkpoint_update"] = _checkpoint_update(run)
        result.append(item)
    return sorted(result, key=lambda item: item["checkpoint_update"])


def _checkpoint_update(run: Mapping[str, Any]) -> int:
    for value in (
        run.get("tags", {}).get("checkpoint_update"),
        run.get("params", {}).get("evaluation.checkpoint_update"),
    ):
        try:
            if value is not None:
                return int(float(str(value)))
        except ValueError:
            pass
    match = re.search(r"eval_update_(\d+)", str(run.get("run_name", "")))
    return int(match.group(1)) if match else -1


def _timestamp(value: int | None) -> str:
    if not value:
        return "ohne Zeitstempel"
    return datetime.fromtimestamp(value / 1000).astimezone().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _format_training(run: Mapping[str, Any]) -> str:
    return (
        f"{run['run_name']} | {run['status']} | "
        f"{_timestamp(run.get('start_time'))} | {str(run['run_id'])[:8]}"
    )


def _format_evaluation(run: Mapping[str, Any]) -> str:
    update = int(run.get("checkpoint_update", -1))
    return f"Update {update:06d} | {run['status']} | {str(run['run_id'])[:8]}"


def _artifact_paths(client: MlflowClient, run_id: str) -> list[str]:
    paths: list[str] = []
    pending = [""]
    while pending:
        current = pending.pop()
        for item in client.list_artifacts(run_id, current):
            if item.is_dir:
                pending.append(item.path)
            else:
                paths.append(item.path)
    return paths


def _choose_artifact(
    paths: Sequence[str],
    filenames: Sequence[str],
    preferred_prefix: str = "",
) -> str | None:
    priorities = {name.casefold(): index for index, name in enumerate(filenames)}
    matches = [
        path for path in paths if Path(path).name.casefold() in priorities
    ]
    if preferred_prefix:
        preferred = [
            path
            for path in matches
            if path.replace("\\", "/").startswith(
                preferred_prefix.rstrip("/") + "/"
            )
        ]
        if preferred:
            matches = preferred
    if not matches:
        return None
    return min(
        matches,
        key=lambda item: (
            priorities[Path(item).name.casefold()],
            item.count("/"),
            len(item),
        ),
    )


def _cache_directory(run_id: str) -> Path:
    directory = (
        Path(tempfile.gettempdir())
        / "assetto_corsa_evaluation_viewer"
        / run_id
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


@st.cache_data(show_spinner=False)
def _download_npz_artifact(
    tracking_uri: str,
    run_id: str,
    end_time: int | None,
    filenames: tuple[str, ...],
    preferred_prefix: str,
) -> tuple[dict[str, np.ndarray], str | None, list[str]]:
    del end_time
    client = _client(tracking_uri)
    paths = _artifact_paths(client, run_id)
    selected = _choose_artifact(paths, filenames, preferred_prefix)
    if selected is None:
        return {}, None, paths
    downloaded = Path(
        client.download_artifacts(
            run_id,
            selected,
            dst_path=str(_cache_directory(run_id)),
        )
    )
    return _load_npz(downloaded), selected, paths


@st.cache_data(show_spinner=False)
def _load_evaluation_bundle(
    tracking_uri: str,
    run_id: str,
    end_time: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]], list[str]]:
    del end_time
    client = _client(tracking_uri)
    paths = _artifact_paths(client, run_id)
    selected = _choose_artifact(
        paths,
        ("trajectories_v3.npz", "trajectories_v2.npz", "trajectories.npz"),
    )
    if selected is None:
        raise FileNotFoundError("Evaluation enthält kein Trajektorienartefakt")
    destination = _cache_directory(run_id)
    arrays = _load_npz(
        Path(client.download_artifacts(run_id, selected, dst_path=str(destination)))
    )

    summary: dict[str, Any] = {}
    summary_artifact = _choose_artifact(paths, ("summary.json",))
    if summary_artifact is not None:
        summary_path = Path(
            client.download_artifacts(
                run_id,
                summary_artifact,
                dst_path=str(destination),
            )
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    episodes: list[dict[str, Any]] = []
    episodes_artifact = _choose_artifact(paths, ("episodes.csv",))
    if episodes_artifact is not None:
        episodes_path = Path(
            client.download_artifacts(
                run_id,
                episodes_artifact,
                dst_path=str(destination),
            )
        )
        episodes = pd.read_csv(episodes_path).to_dict(orient="records")
    return arrays, summary, episodes, paths


def _uploaded_npz(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _scalar_text(arrays: Mapping[str, np.ndarray], name: str) -> str:
    value = arrays.get(name)
    if value is None:
        return ""
    array = np.asarray(value)
    return str(array.reshape(-1)[0]) if array.size == 1 else ""


def _scalar_int(arrays: Mapping[str, np.ndarray], name: str, default: int = 0) -> int:
    value = arrays.get(name)
    if value is None:
        return default
    array = np.asarray(value)
    return int(array.reshape(-1)[0]) if array.size == 1 else default


def _points(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        return None
    result = array[:, :2]
    finite = np.isfinite(result).all(axis=1)
    result = result[finite]
    return result if result.size else None


def _first_points(
    arrays: Mapping[str, np.ndarray],
    names: Sequence[str],
) -> np.ndarray | None:
    for name in names:
        result = _points(arrays.get(name))
        if result is not None:
            return result
    return None


def _step_points(
    arrays: Mapping[str, np.ndarray],
    names: Sequence[str],
    index: int,
    total_steps: int,
) -> np.ndarray | None:
    for name in names:
        if name not in arrays:
            continue
        value = np.asarray(arrays[name], dtype=np.float64)
        if value.ndim == 3 and value.shape[0] == total_steps:
            result = value[index]
        elif value.ndim == 2:
            result = value
        else:
            continue
        if result.shape[-1] < 2:
            continue
        result = result[:, :2]
        finite = np.isfinite(result).all(axis=1)
        result = result[finite]
        if result.size:
            return result
    return None


def _canonicalize_static_artifact(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], bool]:
    """Return static data in the live telemetry plane ``[AC z, AC x]``.

    Schema 4 and the original Assetto-Corsa-Gym artefacts already use this
    order.  The broken schema-3 update stored native ``[AC x, AC z]`` and is
    converted here for display only.
    """

    result = {name: np.asarray(value).copy() for name, value in arrays.items()}
    transform = _scalar_text(result, "coordinate_transform").replace(" ", "")
    metadata_text = _scalar_text(result, "metadata_json")
    if not transform and metadata_text:
        try:
            transform = str(json.loads(metadata_text).get("coordinate_transform", ""))
        except json.JSONDecodeError:
            pass
        transform = transform.replace(" ", "")
    needs_swap = "[ac_x,ac_z]" in transform
    if not needs_swap:
        return result, False

    coordinate_names = (
        "centerline_world_xy_m",
        "left_boundary_world_xy_m",
        "right_boundary_world_xy_m",
        "centerline_world_xz_m",
        "left_boundary_world_xz_m",
        "right_boundary_world_xz_m",
        "world_position_xy_m",
        "world_position_xz_m",
        "forward_vector_xy",
        "forward_vector_xz",
    )
    for name in coordinate_names:
        value = result.get(name)
        if value is not None and value.ndim == 2 and value.shape[1] >= 2:
            swapped = value.copy()
            swapped[:, :2] = value[:, [1, 0]]
            result[name] = swapped
    if "heading_rad" in result:
        heading = np.asarray(result["heading_rad"], dtype=np.float64)
        result["heading_rad"] = (0.5 * math.pi - heading).astype(np.float32)
    if "world_heading_rad" in result:
        heading = np.asarray(result["world_heading_rad"], dtype=np.float64)
        result["world_heading_rad"] = (0.5 * math.pi - heading).astype(np.float32)
    result["coordinate_frame"] = np.asarray("assetto_corsa_gym_world_xy_v1")
    result["coordinate_transform"] = np.asarray("world_xy=[ac_z,ac_x]")
    return result, True


def _local_xz_to_world_xz(
    local_xz_m: np.ndarray | None,
    vehicle_xz_m: np.ndarray,
    heading_rad: float,
) -> np.ndarray | None:
    if local_xz_m is None:
        return None
    local = np.asarray(local_xz_m, dtype=np.float64)
    forward = np.asarray([math.cos(heading_rad), math.sin(heading_rad)])
    left = np.asarray([-math.sin(heading_rad), math.cos(heading_rad)])
    return (
        vehicle_xz_m[None, :]
        + local[:, 0:1] * left[None, :]
        + local[:, 1:2] * forward[None, :]
    )


def _world_xz_to_local_xz(
    world_xz_m: np.ndarray | None,
    vehicle_xz_m: np.ndarray,
    heading_rad: float,
) -> np.ndarray | None:
    if world_xz_m is None:
        return None
    world = np.asarray(world_xz_m, dtype=np.float64)
    delta = world - vehicle_xz_m[None, :]
    forward = np.asarray([math.cos(heading_rad), math.sin(heading_rad)])
    left = np.asarray([-math.sin(heading_rad), math.cos(heading_rad)])
    return np.column_stack((delta @ left, delta @ forward))


def _local_xz_to_vehicle_fl(local_xz_m: np.ndarray | None) -> np.ndarray | None:
    if local_xz_m is None:
        return None
    local = np.asarray(local_xz_m, dtype=np.float64)
    return np.column_stack((local[:, 1], local[:, 0]))


def _vehicle_fl_plot(points_fl_m: np.ndarray | None) -> np.ndarray | None:
    """Return plot coordinates [left, forward] from stored [forward, left]."""

    if points_fl_m is None:
        return None
    points = np.asarray(points_fl_m, dtype=np.float64)
    return np.column_stack((points[:, 1], points[:, 0]))


def _circular_interpolate(
    distance_m: np.ndarray,
    values: np.ndarray,
    query_m: np.ndarray,
    track_length_m: float,
) -> np.ndarray:
    distance = np.asarray(distance_m, dtype=np.float64).reshape(-1)
    data = np.asarray(values, dtype=np.float64)
    query = np.mod(np.asarray(query_m, dtype=np.float64), track_length_m)
    extended_distance = np.concatenate((distance, [track_length_m]))
    if data.ndim == 1:
        return np.interp(
            query,
            extended_distance,
            np.concatenate((data, [data[0]])),
        )
    return np.column_stack(
        [
            np.interp(
                query,
                extended_distance,
                np.concatenate((data[:, axis], [data[0, axis]])),
            )
            for axis in range(data.shape[1])
        ]
    )


def _project_closed_path(
    path_xz_m: np.ndarray | None,
    distance_m: np.ndarray | None,
    track_length_m: float | None,
    position_xz_m: np.ndarray,
) -> Projection | None:
    if path_xz_m is None or distance_m is None or track_length_m is None:
        return None
    path = np.asarray(path_xz_m, dtype=np.float64)
    distance = np.asarray(distance_m, dtype=np.float64).reshape(-1)
    if path.shape[0] != distance.size or path.shape[0] < 2:
        return None
    start = path
    end = np.roll(path, -1, axis=0)
    vector = end - start
    length_squared = np.einsum("ij,ij->i", vector, vector)
    valid = length_squared > 1e-12
    offset = position_xz_m[None, :] - start
    fraction = np.zeros(path.shape[0], dtype=np.float64)
    fraction[valid] = np.clip(
        np.einsum("ij,ij->i", offset[valid], vector[valid])
        / length_squared[valid],
        0.0,
        1.0,
    )
    projected = start + fraction[:, None] * vector
    squared = np.einsum(
        "ij,ij->i",
        projected - position_xz_m[None, :],
        projected - position_xz_m[None, :],
    )
    squared[~valid] = np.inf
    index = int(np.argmin(squared))
    if index == path.shape[0] - 1:
        segment_progress = max(track_length_m - float(distance[index]), 0.0)
    else:
        segment_progress = max(float(distance[index + 1] - distance[index]), 0.0)
    progress = (
        float(distance[index]) + fraction[index] * segment_progress
    ) % track_length_m
    return Projection(
        progress_m=progress,
        point_xz_m=projected[index],
        distance_m=float(math.sqrt(squared[index])),
    )


def _track_length(arrays: Mapping[str, np.ndarray]) -> float | None:
    value = arrays.get("track_length_m")
    if value is None:
        return None
    return float(np.asarray(value).reshape(-1)[0])


def _reference_preview(
    reference: Mapping[str, np.ndarray],
    progress_m: float,
) -> np.ndarray | None:
    line = _first_points(
        reference,
        ("world_position_xy_m", "world_position_xz_m"),
    )
    if line is None or "track_distance_m" not in reference:
        return None
    speed = reference.get("speed_mps")
    length = _track_length(reference)
    if speed is None or length is None:
        return None
    offsets = np.asarray(
        reference.get("reference_time_offsets_s", DEFAULT_REFERENCE_TIME_OFFSETS_S),
        dtype=np.float64,
    ).reshape(-1)
    distance = np.asarray(reference["track_distance_m"], dtype=np.float64)
    speed_values = np.asarray(speed, dtype=np.float64)
    current_progress = float(progress_m)
    current_time = 0.0
    query: list[float] = []
    for target_time in offsets:
        while current_time + 1e-12 < float(target_time):
            dt = min(0.01, float(target_time) - current_time)
            current_speed = float(
                _circular_interpolate(
                    distance,
                    speed_values,
                    np.asarray([current_progress]),
                    length,
                )[0]
            )
            current_progress = (current_progress + max(current_speed, 0.0) * dt) % length
            current_time += dt
        query.append(current_progress)
    return _circular_interpolate(distance, line, np.asarray(query), length)


def _boundary_preview(
    geometry: Mapping[str, np.ndarray],
    progress_m: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    left = _first_points(
        geometry,
        ("left_boundary_world_xy_m", "left_boundary_world_xz_m"),
    )
    right = _first_points(
        geometry,
        ("right_boundary_world_xy_m", "right_boundary_world_xz_m"),
    )
    distance = geometry.get("track_distance_m")
    length = _track_length(geometry)
    if left is None or right is None or distance is None or length is None:
        return None, None
    query = float(progress_m) + DEFAULT_BOUNDARY_PREVIEW_DISTANCES_M
    return (
        _circular_interpolate(distance, left, query, length),
        _circular_interpolate(distance, right, query, length),
    )


def _wrapped_delta(
    current: float | None,
    previous: float | None,
    length: float | None,
) -> float | None:
    if current is None or previous is None or length is None:
        return None
    return float((current - previous + 0.5 * length) % length - 0.5 * length)


def _add_line(
    figure: go.Figure,
    points: np.ndarray | None,
    name: str,
    *,
    width: int = 2,
    dash: str | None = None,
) -> None:
    if points is None or points.shape[0] < 2:
        return
    line: dict[str, Any] = {"width": width}
    if dash is not None:
        line["dash"] = dash
    figure.add_trace(
        go.Scattergl(
            x=points[:, 0],
            y=points[:, 1],
            mode="lines",
            name=name,
            line=line,
            hovertemplate=(
                f"{name}<br>x=%{{x:.2f}} m<br>z=%{{y:.2f}} m<extra></extra>"
            ),
        )
    )


def _add_points(
    figure: go.Figure,
    points: np.ndarray | None,
    name: str,
    *,
    connect: bool = True,
    symbol: str = "circle",
    size: int = 7,
) -> None:
    if points is None or not points.size:
        return
    figure.add_trace(
        go.Scattergl(
            x=points[:, 0],
            y=points[:, 1],
            mode="lines+markers" if connect else "markers",
            name=name,
            marker={"size": size, "symbol": symbol},
            hovertemplate=(
                f"{name}<br>x=%{{x:.2f}} m<br>y=%{{y:.2f}} m<extra></extra>"
            ),
        )
    )


def _speed_at_step(
    positions: np.ndarray,
    times: np.ndarray | None,
    index: int,
) -> float | None:
    if times is None or times.size != positions.shape[0] or positions.shape[0] < 2:
        return None
    previous = max(0, index - 1)
    following = min(positions.shape[0] - 1, index + 1)
    dt = float(times[following] - times[previous])
    if dt <= 1e-9:
        return None
    return float(np.linalg.norm(positions[following] - positions[previous]) / dt)


def _series(
    arrays: Mapping[str, np.ndarray],
    names: str | Sequence[str],
    index: int,
) -> float | None:
    candidates = (names,) if isinstance(names, str) else names
    for name in candidates:
        if name not in arrays:
            continue
        value = np.asarray(arrays[name]).reshape(-1)
        if index >= value.size:
            continue
        result = float(value[index])
        if math.isfinite(result):
            return result
    return None


def _action_values(
    arrays: Mapping[str, np.ndarray],
    index: int,
) -> tuple[float | None, float | None]:
    actions = arrays.get("actions")
    if actions is None:
        return None, None
    value = np.asarray(actions, dtype=np.float64)
    if value.ndim != 2 or index >= value.shape[0] or value.shape[1] < 2:
        return None, None
    return float(value[index, 0]), float(value[index, 1])


def _predictor_errors_local(
    prediction_local_xz: np.ndarray | None,
    reference_local_xz: np.ndarray | None,
) -> tuple[float | None, float | None]:
    if prediction_local_xz is None or reference_local_xz is None:
        return None, None
    count = min(prediction_local_xz.shape[0], reference_local_xz.shape[0])
    if count == 0:
        return None, None
    distance = np.linalg.norm(
        prediction_local_xz[:count] - reference_local_xz[:count],
        axis=1,
    )
    return float(distance.mean()), float(distance[-1])


def _format_value(value: object) -> str:
    if value is None:
        return "nicht aufgezeichnet"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _value_table(
    *,
    arrays: Mapping[str, np.ndarray],
    positions: np.ndarray,
    times: np.ndarray | None,
    global_index: int,
    local_step: int,
    projection: Projection | None,
    track_projection: Projection | None,
    projection_progress_increment: float | None,
    prediction_local_xz: np.ndarray | None,
    reference_local_xz: np.ndarray | None,
) -> pd.DataFrame:
    speed = _speed_at_step(positions, times, global_index)
    pedal, steering = _action_values(arrays, global_index)
    ade, fde = _predictor_errors_local(prediction_local_xz, reference_local_xz)
    rows = [
        ("Zeit", "Episode step", local_step, "-", "Artefakt"),
        (
            "Zeit",
            "Simulation time",
            _series(arrays, "simulation_time_s", global_index),
            "s",
            "Artefakt",
        ),
        (
            "Progress",
            "Environment progress",
            _series(
                arrays,
                ("environment_progress_m", "reference_progress_m"),
                global_index,
            ),
            "m",
            "Environment",
        ),
        (
            "Progress",
            "Independent reference projection",
            projection.progress_m if projection is not None else None,
            "m",
            "Viewer projection",
        ),
        (
            "Progress",
            "Reference increment",
            projection_progress_increment,
            "m/step",
            "Viewer projection",
        ),
        (
            "Progress",
            "Recorded progress reward",
            _series(arrays, "progress_reward", global_index),
            "-",
            "Environment",
        ),
        (
            "Progress",
            "Diagnostic progress reward",
            _series(arrays, "progress_reward_diagnostic", global_index),
            "-",
            "Evaluation",
        ),
        (
            "Frame",
            "Vehicle to reference",
            projection.distance_m if projection is not None else None,
            "m",
            "Viewer projection",
        ),
        (
            "Frame",
            "Vehicle to AI spline",
            track_projection.distance_m if track_projection is not None else None,
            "m",
            "Viewer projection",
        ),
        (
            "Frame",
            "Roundtrip error",
            _series(arrays, "frame_roundtrip_error_m", global_index),
            "m",
            "Evaluation",
        ),
        ("Fahrzeug", "Geschwindigkeit", speed, "m/s", "Position/Zeit"),
        (
            "Fahrzeug",
            "Geschwindigkeit",
            speed * 3.6 if speed is not None else None,
            "km/h",
            "abgeleitet",
        ),
        ("Aktion", "Combined pedal", pedal, "[-1,1]", "Action"),
        ("Aktion", "Steering command", steering, "[-1,1]", "Action"),
        (
            "Reward",
            "Environment reward",
            _series(arrays, "rewards", global_index),
            "-",
            "Artefakt",
        ),
        ("Predictor", "ADE local", ade, "m", "Vehicle local XZ"),
        ("Predictor", "FDE local", fde, "m", "Vehicle local XZ"),
    ]
    return pd.DataFrame(
        [
            {
                "Gruppe": str(group),
                "Signal": str(signal),
                "Wert": _format_value(value),
                "Einheit": str(unit),
                "Quelle": str(source),
            }
            for group, signal, value, unit, source in rows
        ],
        dtype="string",
    )


def main() -> None:  # noqa: PLR0912, PLR0915
    st.set_page_config(page_title="Assetto Corsa Evaluation Viewer", layout="wide")
    st.title("Assetto Corsa Evaluation Viewer")
    st.caption("Weltansicht und fahrzeugfeste Agentenansicht mit getrennten Frames")

    tracking_uri = st.sidebar.text_input(
        "MLflow Tracking URI",
        DEFAULT_TRACKING_URI,
    ).strip()
    if st.sidebar.button("MLflow Daten aktualisieren"):
        st.cache_data.clear()
        st.rerun()

    try:
        experiments = _list_experiments(tracking_uri)
    except Exception as error:
        st.error(f"MLflow ist nicht erreichbar: {error}")
        st.stop()
    if not experiments:
        st.warning("Keine MLflow Experimente gefunden.")
        st.stop()

    experiment_names = [item["name"] for item in experiments]
    default_index = (
        experiment_names.index("Assetto Corsa PPO")
        if "Assetto Corsa PPO" in experiment_names
        else 0
    )
    experiment_name = st.sidebar.selectbox(
        "Experiment",
        experiment_names,
        index=default_index,
    )
    experiment_id = next(
        item["experiment_id"]
        for item in experiments
        if item["name"] == experiment_name
    )
    all_runs = _list_runs(tracking_uri, experiment_id)
    training_runs = _training_runs(all_runs)
    if not training_runs:
        st.warning("Kein Training Run gefunden.")
        st.stop()

    training_by_id = {run["run_id"]: run for run in training_runs}
    training_id = st.sidebar.selectbox(
        "Training Run",
        list(training_by_id),
        format_func=lambda run_id: _format_training(training_by_id[run_id]),
    )
    training_run = training_by_id[training_id]
    evaluation_runs = _evaluation_runs(all_runs, training_id)
    if not evaluation_runs:
        st.warning("Für diesen Training Run existieren keine Evaluation Runs.")
        st.stop()

    evaluation_by_id = {run["run_id"]: run for run in evaluation_runs}
    evaluation_id = st.sidebar.selectbox(
        "Evaluation Update",
        list(evaluation_by_id),
        format_func=lambda run_id: _format_evaluation(evaluation_by_id[run_id]),
    )
    evaluation_run = evaluation_by_id[evaluation_id]

    try:
        with st.spinner("Evaluation Artefakte werden geladen …"):
            arrays, summary, episode_rows, evaluation_artifacts = (
                _load_evaluation_bundle(
                    tracking_uri,
                    evaluation_id,
                    evaluation_run.get("end_time"),
                )
            )
            geometry, geometry_artifact, training_artifacts = (
                _download_npz_artifact(
                    tracking_uri,
                    training_id,
                    training_run.get("end_time"),
                    ("track_geometry_v3.npz", "track_geometry.npz"),
                    "track",
                )
            )
            reference, reference_artifact, _ = _download_npz_artifact(
                tracking_uri,
                training_id,
                training_run.get("end_time"),
                ("reference_v3.npz", "reference.npz", "fixed_reference.npz"),
                "reference",
            )
    except Exception as error:
        st.error(f"Artefakte konnten nicht geladen werden: {error}")
        st.stop()

    if not geometry or not reference:
        uploaded = st.sidebar.file_uploader(
            "Fallback: Track oder Referenz NPZ hochladen",
            type=["npz"],
            accept_multiple_files=True,
        )
        for item in uploaded or []:
            loaded = _uploaded_npz(item.getvalue())
            if any(
                key in loaded
                for key in (
                    "left_boundary_world_xz_m",
                    "left_boundary_world_xy_m",
                )
            ):
                geometry = loaded
            if any(
                key in loaded
                for key in ("world_position_xz_m", "world_position_xy_m")
            ):
                reference = loaded

    geometry, geometry_axis_fixed = _canonicalize_static_artifact(geometry)
    reference, reference_axis_fixed = _canonicalize_static_artifact(reference)

    # Schema-4 evaluations embed the exact geometry and active reference used
    # for that episode.  Prefer these arrays over parent-run artefacts so MLflow
    # caching or a generated AI reference cannot hide what the environment saw.
    if "active_reference_world_xy_m" in arrays:
        reference = {
            "world_position_xy_m": np.asarray(
                arrays["active_reference_world_xy_m"]
            ),
            "track_distance_m": np.asarray(
                arrays.get("active_reference_track_distance_m", [])
            ),
            "speed_mps": np.asarray(
                arrays.get("active_reference_speed_mps", [])
            ),
            "world_heading_rad": np.asarray(
                arrays.get("active_reference_heading_rad", [])
            ),
            "track_length_m": np.asarray(
                arrays.get("active_reference_track_length_m", np.nan)
            ),
            "coordinate_frame": np.asarray("assetto_corsa_gym_world_xy_v1"),
            "coordinate_transform": np.asarray("world_xy=[ac_z,ac_x]"),
        }
    if "track_centerline_world_xy_m" in arrays:
        geometry = {
            "centerline_world_xy_m": np.asarray(
                arrays["track_centerline_world_xy_m"]
            ),
            "left_boundary_world_xy_m": np.asarray(
                arrays.get("track_left_boundary_world_xy_m", [])
            ),
            "right_boundary_world_xy_m": np.asarray(
                arrays.get("track_right_boundary_world_xy_m", [])
            ),
            "track_distance_m": np.asarray(
                arrays.get("track_static_distance_m", [])
            ),
            "track_length_m": np.asarray(
                arrays.get("track_static_length_m", np.nan)
            ),
            "coordinate_frame": np.asarray("assetto_corsa_gym_world_xy_v1"),
            "coordinate_transform": np.asarray("world_xy=[ac_z,ac_x]"),
        }

    positions = np.asarray(
        arrays.get(
            "vehicle_world_xy_m",
            arrays.get(
                "positions_xy_m",
                arrays.get("vehicle_world_xz_m", np.empty((0, 2))),
            ),
        ),
        dtype=np.float64,
    )
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] == 0:
        st.error("Trajektorie enthält keine gültigen Fahrzeug Weltkoordinaten.")
        st.stop()
    total_steps = positions.shape[0]
    episode_ids = np.asarray(
        arrays.get("episode_ids", np.zeros(total_steps)),
        dtype=np.int64,
    ).reshape(-1)
    if episode_ids.size != total_steps:
        st.error("episode_ids und Fahrzeugpositionen haben unterschiedliche Längen.")
        st.stop()
    episodes = sorted(int(value) for value in np.unique(episode_ids))
    selected_episode = st.sidebar.selectbox("Episode", episodes)
    episode_indices = np.flatnonzero(episode_ids == selected_episode)
    episode_positions = positions[episode_indices]
    headings = np.asarray(
        arrays.get("vehicle_heading_rad", np.zeros(total_steps)),
        dtype=np.float64,
    ).reshape(-1)
    times = (
        np.asarray(arrays["simulation_time_s"], dtype=np.float64).reshape(-1)
        if "simulation_time_s" in arrays
        else None
    )

    static_reference = _first_points(
        reference,
        ("world_position_xy_m", "world_position_xz_m"),
    )
    static_centerline = _first_points(
        geometry,
        ("centerline_world_xy_m", "centerline_world_xz_m"),
    )
    static_left = _first_points(
        geometry,
        ("left_boundary_world_xy_m", "left_boundary_world_xz_m"),
    )
    static_right = _first_points(
        geometry,
        ("right_boundary_world_xy_m", "right_boundary_world_xz_m"),
    )
    reference_distance = (
        np.asarray(reference["track_distance_m"], dtype=np.float64)
        if "track_distance_m" in reference
        else None
    )
    geometry_distance = (
        np.asarray(geometry["track_distance_m"], dtype=np.float64)
        if "track_distance_m" in geometry
        else None
    )
    reference_length = _track_length(reference)
    geometry_length = _track_length(geometry)
    start_finish = (
        np.vstack((static_left[0], static_right[0]))
        if static_left is not None and static_right is not None
        else None
    )

    artifact_schema = _scalar_int(arrays, "evaluation_artifact_schema", 1)
    coordinate_frame = _scalar_text(arrays, "coordinate_frame") or str(
        summary.get("coordinate_frame", "unbekannt")
    )
    if geometry_axis_fixed or reference_axis_fixed:
        st.warning(
            "Dieser Run enthält das fehlerhafte Schema 3 in nativer Achsenfolge "
            "[AC x, AC z]. Der Viewer tauscht es nur für die Darstellung nach "
            "[AC z, AC x]. Für Training und Reward-Validierung ist Schema 4 nötig."
        )
    if artifact_schema < 4:
        st.info(
            "Altes Evaluationsschema erkannt. Lokale Referenz und Border Features "
            "werden im Viewer aus der korrigierten Weltgeometrie rekonstruiert."
        )
    if static_left is None or static_right is None:
        st.error("Track Borders fehlen im Parent Run.")
    if "observations" in arrays or "observations_normalized" in arrays:
        st.caption(
            "Observations sind normalisierte Modelleingaben und werden nie als Meter interpretiert."
        )

    view_mode = st.sidebar.radio(
        "Darstellung",
        ("Weltansicht", "Agentenansicht"),
    )
    with st.sidebar.expander("Sichtbarkeit", expanded=True):
        show = {
            "driven": st.checkbox("Fahrline", True),
            "trail": st.checkbox("Fahrline bis aktueller Punkt", True),
            "reference": st.checkbox("Referenzlinie", True),
            "centerline": st.checkbox("AI Spline", True),
            "borders": st.checkbox("Track Borders", True),
            "start": st.checkbox("Start Ziellinie", True),
            "projection": st.checkbox("Projektionsdiagnose", True),
            "reference_preview": st.checkbox("Referenzvorschau", True),
            "boundary_preview": st.checkbox("Border Vorschau", True),
            "bezier": st.checkbox("Prädiktierte Bézier Linie", True),
        }
    zoom_mode = st.sidebar.selectbox(
        "Weltansicht",
        ("Fahrzeug folgen", "Ganze Strecke"),
        disabled=view_mode != "Weltansicht",
    )
    follow_radius = st.sidebar.slider(
        "Folgeradius [m]",
        10.0,
        300.0,
        80.0,
        5.0,
        disabled=view_mode != "Weltansicht",
    )

    signature = f"{training_id}:{evaluation_id}:{selected_episode}:{view_mode}"
    if st.session_state.get("selection") != signature:
        st.session_state["selection"] = signature
        st.session_state["step"] = 0
        st.session_state["playing"] = False
    maximum_step = episode_indices.size - 1
    st.session_state["step"] = int(
        np.clip(st.session_state.get("step", 0), 0, maximum_step)
    )

    controls = st.columns([1, 1, 1, 1, 2, 2])
    if controls[0].button("⏮ Anfang", width="stretch"):
        st.session_state["step"] = 0
    if controls[1].button("◀ Zurück", width="stretch"):
        st.session_state["step"] = max(0, st.session_state["step"] - 1)
    label = "⏸ Pause" if st.session_state["playing"] else "▶ Play"
    if controls[2].button(label, width="stretch"):
        st.session_state["playing"] = not st.session_state["playing"]
    if controls[3].button("Vor ▶", width="stretch"):
        st.session_state["step"] = min(
            maximum_step,
            st.session_state["step"] + 1,
        )
    fps = controls[4].slider("Wiedergabe [Schritte/s]", 1, 20, 5)
    increment = controls[5].selectbox("Schrittweite", [1, 2, 5, 10])
    run_every = 1.0 / fps if st.session_state["playing"] else None

    @st.fragment(run_every=run_every)
    def render() -> None:
        if st.session_state["playing"]:
            local_step = int(st.session_state["step"])
            st.progress(
                local_step / max(maximum_step, 1),
                text=f"Step {local_step} / {maximum_step}",
            )
        else:
            local_step = st.slider(
                "Zeitschritt",
                0,
                maximum_step,
                value=int(st.session_state["step"]),
            )
            st.session_state["step"] = local_step

        global_index = int(episode_indices[local_step])
        current_position = positions[global_index]
        current_heading = (
            float(headings[global_index]) if global_index < headings.size else 0.0
        )

        reference_projection = _project_closed_path(
            static_reference,
            reference_distance,
            reference_length,
            current_position,
        )
        track_projection = _project_closed_path(
            static_centerline,
            geometry_distance,
            geometry_length,
            current_position,
        )
        reference_world = _step_points(
            arrays,
            ("reference_preview_world_xy_m", "reference_preview_world_xz_m"),
            global_index,
            total_steps,
        )
        if reference_world is None and reference_projection is not None:
            reference_world = _reference_preview(
                reference, reference_projection.progress_m
            )
        left_world = _step_points(
            arrays,
            ("left_boundary_preview_world_xy_m", "left_boundary_preview_world_xz_m"),
            global_index,
            total_steps,
        )
        right_world = _step_points(
            arrays,
            ("right_boundary_preview_world_xy_m", "right_boundary_preview_world_xz_m"),
            global_index,
            total_steps,
        )
        if (left_world is None or right_world is None) and track_projection is not None:
            left_world, right_world = _boundary_preview(
                geometry, track_projection.progress_m
            )

        prediction_local_xz = _step_points(
            arrays,
            ("bezier_prediction_local_xz_m", "bezier_prediction_points"),
            global_index,
            total_steps,
        )
        prediction_world = _step_points(
            arrays,
            ("bezier_prediction_world_xy_m", "bezier_prediction_world_xz_m"),
            global_index,
            total_steps,
        )
        if prediction_world is None:
            prediction_world = _local_xz_to_world_xz(
                prediction_local_xz,
                current_position,
                current_heading,
            )
        reference_local_xz = _step_points(
            arrays,
            ("reference_preview_local_xz_m",),
            global_index,
            total_steps,
        )
        if reference_local_xz is None:
            reference_local_xz = _world_xz_to_local_xz(
                reference_world, current_position, current_heading
            )
        left_local_xz = _step_points(
            arrays,
            ("left_boundary_preview_local_xz_m",),
            global_index,
            total_steps,
        )
        if left_local_xz is None:
            left_local_xz = _world_xz_to_local_xz(
                left_world, current_position, current_heading
            )
        right_local_xz = _step_points(
            arrays,
            ("right_boundary_preview_local_xz_m",),
            global_index,
            total_steps,
        )
        if right_local_xz is None:
            right_local_xz = _world_xz_to_local_xz(
                right_world, current_position, current_heading
            )
        prediction_fl = _local_xz_to_vehicle_fl(prediction_local_xz)
        reference_fl = _local_xz_to_vehicle_fl(reference_local_xz)
        left_fl = _local_xz_to_vehicle_fl(left_local_xz)
        right_fl = _local_xz_to_vehicle_fl(right_local_xz)

        previous_projection: Projection | None = None
        if local_step > 0:
            previous_position = positions[int(episode_indices[local_step - 1])]
            previous_projection = _project_closed_path(
                static_reference,
                reference_distance,
                reference_length,
                previous_position,
            )
        projection_increment = _wrapped_delta(
            reference_projection.progress_m if reference_projection else None,
            previous_projection.progress_m if previous_projection else None,
            reference_length,
        )

        figure = go.Figure()
        if view_mode == "Weltansicht":
            if show["reference"]:
                _add_line(figure, static_reference, "Reference line", width=3)
            if show["centerline"]:
                _add_line(figure, static_centerline, "AI spline", dash="dot")
            if show["borders"]:
                _add_line(figure, static_left, "Track border left")
                _add_line(figure, static_right, "Track border right")
            if show["start"]:
                _add_line(figure, start_finish, "Start / finish", width=4)
            if show["driven"]:
                _add_line(figure, episode_positions, "Driven line complete")
            if show["trail"]:
                _add_line(
                    figure,
                    episode_positions[: local_step + 1],
                    "Driven line to current step",
                    width=4,
                )
            if show["reference_preview"]:
                _add_points(figure, reference_world, "Reference preview")
            if show["boundary_preview"]:
                _add_points(figure, left_world, "Boundary preview left")
                _add_points(figure, right_world, "Boundary preview right")
            if show["bezier"]:
                _add_points(figure, prediction_world, "Predicted Bézier line")
            if show["projection"] and reference_projection is not None:
                _add_points(
                    figure,
                    reference_projection.point_xz_m.reshape(1, 2),
                    "Reference projection",
                    connect=False,
                    symbol="x",
                    size=11,
                )
                _add_line(
                    figure,
                    np.vstack((current_position, reference_projection.point_xz_m)),
                    "Vehicle to reference",
                    dash="dash",
                )
            figure.add_trace(
                go.Scattergl(
                    x=[current_position[0]],
                    y=[current_position[1]],
                    mode="markers",
                    name="Vehicle",
                    marker={"size": 14, "symbol": "diamond"},
                )
            )
            arrow = current_position + min(10.0, follow_radius * 0.15) * np.asarray(
                [math.cos(current_heading), math.sin(current_heading)]
            )
            _add_line(
                figure,
                np.vstack((current_position, arrow)),
                "Vehicle heading",
                width=4,
            )
            figure.update_layout(
                xaxis_title="World X = AC Z (telemetry axis 0) [m]",
                yaxis_title="World Y = AC X (telemetry axis 1) [m]",
            )
            if zoom_mode == "Fahrzeug folgen":
                figure.update_xaxes(
                    range=[
                        current_position[0] - follow_radius,
                        current_position[0] + follow_radius,
                    ]
                )
                figure.update_yaxes(
                    range=[
                        current_position[1] - follow_radius,
                        current_position[1] + follow_radius,
                    ]
                )
        else:
            reference_agent = _vehicle_fl_plot(reference_fl)
            left_agent = _vehicle_fl_plot(left_fl)
            right_agent = _vehicle_fl_plot(right_fl)
            prediction_agent = _vehicle_fl_plot(prediction_fl)
            if show["reference_preview"]:
                _add_points(figure, reference_agent, "Reference feature")
            if show["boundary_preview"]:
                _add_points(figure, left_agent, "Track border feature left")
                _add_points(figure, right_agent, "Track border feature right")
            if show["bezier"]:
                _add_points(figure, prediction_agent, "Predicted Bézier line")
            figure.add_trace(
                go.Scattergl(
                    x=[0.0],
                    y=[0.0],
                    mode="markers",
                    name="Vehicle origin",
                    marker={"size": 14, "symbol": "diamond"},
                )
            )
            _add_line(
                figure,
                np.asarray([[0.0, 0.0], [0.0, 10.0]]),
                "Vehicle forward",
                width=4,
            )
            figure.update_layout(
                xaxis_title="Vehicle left [m]",
                yaxis_title="Vehicle forward [m]",
            )

        simulation_time = _series(arrays, "simulation_time_s", global_index)
        update = int(evaluation_run.get("checkpoint_update", -1))
        title = (
            f"{view_mode} · Update {update:06d} · Episode {selected_episode} · "
            f"Step {local_step}/{maximum_step}"
        )
        if simulation_time is not None:
            title += f" · t={simulation_time:.2f} s"
        figure.update_layout(
            title=title,
            height=760,
            hovermode="closest",
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        )
        figure.update_yaxes(scaleanchor="x", scaleratio=1)

        plot_column, table_column = st.columns([2.2, 1.0])
        with plot_column:
            st.plotly_chart(
                figure,
                width="stretch",
                config={"scrollZoom": True, "displaylogo": False},
            )
        with table_column:
            st.subheader("Aktueller Punkt")
            st.dataframe(
                _value_table(
                    arrays=arrays,
                    positions=positions,
                    times=times,
                    global_index=global_index,
                    local_step=local_step,
                    projection=reference_projection,
                    track_projection=track_projection,
                    projection_progress_increment=projection_increment,
                    prediction_local_xz=prediction_local_xz,
                    reference_local_xz=reference_local_xz,
                ),
                hide_index=True,
                width="stretch",
                height=710,
            )

        if st.session_state["playing"]:
            next_step = min(maximum_step, local_step + int(increment))
            if next_step == local_step:
                st.session_state["playing"] = False
            else:
                st.session_state["step"] = next_step

    render()

    metrics = st.columns(5)
    metrics[0].metric("Training Run", str(training_run["run_name"]))
    metrics[1].metric(
        "Evaluation Update",
        int(evaluation_run.get("checkpoint_update", -1)),
    )
    metrics[2].metric("Episode", selected_episode)
    metrics[3].metric("Episode steps", episode_indices.size)
    metrics[4].metric("Schema", artifact_schema)

    with st.expander("Evaluation und Datenstatus"):
        st.write(f"**Koordinatenframe:** `{coordinate_frame}`")
        st.write(
            f"**Track Geometrie:** `{geometry_artifact or 'nicht gefunden'}`"
        )
        st.write(f"**Referenz:** `{reference_artifact or 'nicht gefunden'}`")
        st.write(
            "**Lokaler Modellframe:** `[lateral X = left, forward Z = forward]`"
        )
        st.write("**Agentenframe:** `[forward, left]`")
        if summary:
            st.write("**summary.json**")
            st.json(summary)
        if episode_rows:
            st.write("**episodes.csv**")
            st.dataframe(pd.DataFrame(episode_rows), hide_index=True)
        st.write("**Arrays im Trajektorienartefakt**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Name": name,
                        "Shape": " × ".join(map(str, value.shape)),
                        "Dtype": str(value.dtype),
                    }
                    for name, value in sorted(arrays.items())
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.write("**Evaluation Artefakte**")
        st.code("\n".join(evaluation_artifacts))
        st.write("**Parent Training Artefakte**")
        st.code("\n".join(training_artifacts))


if __name__ == "__main__":
    main()
