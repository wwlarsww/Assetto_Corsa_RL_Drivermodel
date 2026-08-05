"""Interactive MLflow evaluation viewer for the Assetto Corsa RL driver.

Run from the repository root with::

    streamlit run scripts/view_evaluation.py

The viewer expects nested MLflow evaluation runs containing a
``trajectories.npz`` artifact. It works with the current evaluation export and
also detects optional richer arrays automatically when they are added later.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from mlflow import MlflowClient

# Allow imports from an editable checkout even when the package was not
# reinstalled after local changes.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

DEFAULT_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)
NONE_OPTION = "(nicht anzeigen)"
AUTO_OPTION = "(automatisch erkennen)"


# ---------------------------------------------------------------------------
# MLflow access
# ---------------------------------------------------------------------------


def _client(tracking_uri: str) -> MlflowClient:
    """Create an MLflow client for one explicit tracking server."""

    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


@st.cache_data(ttl=10, show_spinner=False)
def _list_experiments(tracking_uri: str) -> list[dict[str, str]]:
    client = _client(tracking_uri)
    experiments = client.search_experiments()
    return [
        {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
        }
        for experiment in experiments
    ]


@st.cache_data(ttl=10, show_spinner=False)
def _list_runs(
    tracking_uri: str,
    experiment_id: str,
    max_results: int = 1000,
) -> list[dict[str, Any]]:
    client = _client(tracking_uri)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=max_results,
    )
    converted: list[dict[str, Any]] = []
    for run in runs:
        converted.append(
            {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "run_name": run.data.tags.get("mlflow.runName", run.info.run_id),
                "tags": dict(run.data.tags),
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
            }
        )
    return converted


def _training_runs(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for run in runs:
        tags = run["tags"]
        run_type = str(tags.get("run_type", "")).lower()
        run_name = str(run["run_name"])
        if run_type == "evaluation" or run_name.startswith("eval_update_"):
            continue
        result.append(run)
    return result


def _evaluation_runs(
    runs: Sequence[dict[str, Any]],
    parent_run_id: str,
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for run in runs:
        tags = run["tags"]
        parent = tags.get("parent_training_run_id") or tags.get("mlflow.parentRunId")
        run_type = str(tags.get("run_type", "")).lower()
        if parent != parent_run_id:
            continue
        if run_type and run_type != "evaluation":
            continue
        converted = dict(run)
        converted["checkpoint_update"] = _checkpoint_update(run)
        evaluations.append(converted)
    return sorted(
        evaluations,
        key=lambda item: (item["checkpoint_update"], item["start_time"] or 0),
    )


def _checkpoint_update(run: Mapping[str, Any]) -> int:
    tags = run.get("tags", {})
    params = run.get("params", {})
    candidates = (
        tags.get("checkpoint_update"),
        params.get("evaluation.checkpoint_update"),
        params.get("checkpoint_update"),
    )
    for value in candidates:
        try:
            if value is not None:
                return int(float(str(value)))
        except ValueError:
            pass
    match = re.search(r"eval_update_(\d+)", str(run.get("run_name", "")))
    return int(match.group(1)) if match else -1


def _format_timestamp(timestamp_ms: int | None) -> str:
    if not timestamp_ms:
        return "ohne Zeitstempel"
    return datetime.fromtimestamp(timestamp_ms / 1000).astimezone().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _format_run(run: Mapping[str, Any]) -> str:
    return (
        f"{run['run_name']} | {run['status']} | "
        f"{_format_timestamp(run.get('start_time'))} | {str(run['run_id'])[:8]}"
    )


def _format_evaluation_run(run: Mapping[str, Any]) -> str:
    update = int(run.get("checkpoint_update", -1))
    update_text = f"Update {update:06d}" if update >= 0 else "Update unbekannt"
    return f"{update_text} | {run['status']} | {str(run['run_id'])[:8]}"


def _list_artifacts_recursive(
    client: MlflowClient,
    run_id: str,
) -> list[str]:
    paths: list[str] = []
    pending = [""]
    while pending:
        current = pending.pop()
        for artifact in client.list_artifacts(run_id, current):
            if artifact.is_dir:
                pending.append(artifact.path)
            else:
                paths.append(artifact.path)
    return paths


def _choose_artifact(
    paths: Sequence[str],
    filenames: Sequence[str],
) -> str | None:
    lower_names = {name.lower() for name in filenames}
    matches = [
        path for path in paths if Path(path).name.lower() in lower_names
    ]
    if not matches:
        return None
    return min(matches, key=lambda path: (path.count("/"), len(path)))


def _artifact_cache_directory(run_id: str) -> Path:
    directory = (
        Path(tempfile.gettempdir())
        / "assetto_corsa_evaluation_viewer"
        / run_id
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@st.cache_data(show_spinner=False)
def _load_evaluation_bundle(
    tracking_uri: str,
    run_id: str,
    end_time: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]], list[str]]:
    # end_time intentionally participates in the cache key.
    del end_time
    client = _client(tracking_uri)
    paths = _list_artifacts_recursive(client, run_id)
    trajectory_artifact = _choose_artifact(paths, ["trajectories.npz"])
    if trajectory_artifact is None:
        raise FileNotFoundError(
            "Der ausgewählte Evaluierungs-Run enthält kein trajectories.npz-Artefakt."
        )

    destination = _artifact_cache_directory(run_id)
    trajectory_path = Path(
        client.download_artifacts(
            run_id,
            trajectory_artifact,
            dst_path=str(destination),
        )
    )
    with np.load(trajectory_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    summary: dict[str, Any] = {}
    summary_artifact = _choose_artifact(paths, ["summary.json"])
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
    episodes_artifact = _choose_artifact(paths, ["episodes.csv"])
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


# ---------------------------------------------------------------------------
# NPZ and geometry helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_npz_path(path_text: str, modified_ns: int) -> dict[str, np.ndarray]:
    del modified_ns
    path = Path(path_text)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _load_uploaded_npz(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _geometry_candidates(arrays: Mapping[str, np.ndarray]) -> list[str]:
    candidates: list[str] = []
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.ndim != 2:
            continue
        if min(array.shape) < 2:
            continue
        if max(array.shape) < 2:
            continue
        candidates.append(name)
    return sorted(candidates)


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _auto_geometry_key(keys: Sequence[str], aliases: Sequence[str]) -> str | None:
    normalised = {_normalise_name(key): key for key in keys}
    for alias in aliases:
        alias_normalised = _normalise_name(alias)
        if alias_normalised in normalised:
            return normalised[alias_normalised]
    for alias in aliases:
        alias_normalised = _normalise_name(alias)
        for normalised_key, original in normalised.items():
            if alias_normalised in normalised_key:
                return original
    return None


def _geometry_selector(
    label: str,
    keys: Sequence[str],
    aliases: Sequence[str],
    widget_key: str,
) -> str | None:
    auto_key = _auto_geometry_key(keys, aliases)
    options = [AUTO_OPTION, NONE_OPTION, *keys]
    choice = st.selectbox(
        label,
        options,
        key=widget_key,
        help=f"Automatische Auswahl: {auto_key or 'nichts erkannt'}",
    )
    if choice == NONE_OPTION:
        return None
    if choice == AUTO_OPTION:
        return auto_key
    return choice


def _as_xy(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        return None
    if array.shape[0] in {2, 3} and array.shape[1] > array.shape[0]:
        array = array.T
    if array.shape[1] < 2:
        return None
    if array.shape[1] >= 3:
        variance = np.nanvar(array[:, :3], axis=0)
        # AC world coordinates are often X/Y/Z with Y as the vertical axis.
        if variance[1] < 0.1 * max(variance[0] + variance[2], 1e-12):
            array = array[:, [0, 2]]
        else:
            array = array[:, :2]
    else:
        array = array[:, :2]
    finite = np.isfinite(array).all(axis=1)
    result = array[finite]
    return result if result.size else None


def _resolve_local_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (REPOSITORY_ROOT / path).resolve()


# ---------------------------------------------------------------------------
# Observation and dynamic feature helpers
# ---------------------------------------------------------------------------


def _feature_names(arrays: Mapping[str, np.ndarray], feature_count: int) -> list[str]:
    embedded = arrays.get("observation_feature_names")
    if embedded is not None:
        names = [str(value) for value in np.asarray(embedded).tolist()]
        if len(names) == feature_count:
            return names
    try:
        from assetto_corsa_rl.features.schema import OBSERVATION_FEATURE_NAMES

        names = [str(value) for value in OBSERVATION_FEATURE_NAMES]
        if len(names) == feature_count:
            return names
    except (ImportError, AttributeError):
        pass
    return [f"observation_{index:03d}" for index in range(feature_count)]


def _tokens(name: str) -> set[str]:
    return set(_normalise_name(name).split("_"))


def _coordinate_feature_groups(
    feature_names: Sequence[str],
    keywords: Sequence[str],
) -> list[tuple[str, int, int]]:
    groups: dict[str, dict[str, int]] = {}
    keyword_set = {_normalise_name(keyword) for keyword in keywords}
    for index, name in enumerate(feature_names):
        normalised = _normalise_name(name)
        if not any(keyword in normalised for keyword in keyword_set):
            continue
        parts = normalised.split("_")
        axis_position = next(
            (position for position, token in enumerate(parts) if token in {"x", "y", "z"}),
            None,
        )
        if axis_position is None:
            continue
        axis = parts[axis_position]
        base_parts = [
            token
            for position, token in enumerate(parts)
            if position != axis_position
            and token not in {"m", "meter", "metres", "coordinate", "coord"}
        ]
        base = "_".join(base_parts)
        groups.setdefault(base, {})[axis] = index

    paired: list[tuple[str, int, int]] = []
    for base, axes in groups.items():
        if "x" in axes and "z" in axes:
            paired.append((base, axes["x"], axes["z"]))
        elif "x" in axes and "y" in axes:
            paired.append((base, axes["x"], axes["y"]))

    def sort_key(item: tuple[str, int, int]) -> tuple[list[int], str]:
        numbers = [int(value) for value in re.findall(r"\d+", item[0])]
        return numbers, item[0]

    return sorted(paired, key=sort_key)


def _feature_points(
    observation: np.ndarray,
    feature_names: Sequence[str],
    keywords: Sequence[str],
) -> np.ndarray | None:
    groups = _coordinate_feature_groups(feature_names, keywords)
    if not groups:
        return None
    points = np.asarray(
        [[observation[x_index], observation[y_index]] for _, x_index, y_index in groups],
        dtype=np.float64,
    )
    finite = np.isfinite(points).all(axis=1)
    result = points[finite]
    return result if result.size else None


def _local_to_world(
    local_points: np.ndarray | None,
    position_xy: np.ndarray,
    heading_rad: float,
    *,
    forward_axis: str,
    lateral_sign: float,
    invert_heading: bool,
    heading_offset_deg: float,
) -> np.ndarray | None:
    if local_points is None:
        return None
    points = np.asarray(local_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        return None
    if forward_axis == "z":
        lateral = points[:, 0] * lateral_sign
        forward = points[:, 1]
    else:
        forward = points[:, 0]
        lateral = points[:, 1] * lateral_sign
    heading = (-heading_rad if invert_heading else heading_rad) + math.radians(
        heading_offset_deg
    )
    cosine = math.cos(heading)
    sine = math.sin(heading)
    world_x = position_xy[0] + cosine * forward - sine * lateral
    world_y = position_xy[1] + sine * forward + cosine * lateral
    return np.column_stack((world_x, world_y))


def _step_points(
    arrays: Mapping[str, np.ndarray],
    aliases: Sequence[str],
    global_index: int,
    total_steps: int,
) -> np.ndarray | None:
    key = _first_key(arrays, aliases)
    if key is None:
        return None
    value = np.asarray(arrays[key])
    if value.ndim >= 2 and value.shape[0] == total_steps:
        value = value[global_index]
    if value.ndim == 1 and value.size % 2 == 0:
        value = value.reshape(-1, 2)
    return _as_xy(value)


def _sample_bezier(control_points: np.ndarray, samples: int = 80) -> np.ndarray:
    points = np.asarray(control_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] < 2:
        return points[:, :2]
    degree = points.shape[0] - 1
    parameter = np.linspace(0.0, 1.0, samples)
    curve = np.zeros((samples, 2), dtype=np.float64)
    for index in range(degree + 1):
        basis = (
            math.comb(degree, index)
            * (1.0 - parameter) ** (degree - index)
            * parameter**index
        )
        curve += basis[:, None] * points[index, :2]
    return curve


def _first_key(arrays: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    normalised = {_normalise_name(key): key for key in arrays}
    for alias in aliases:
        match = normalised.get(_normalise_name(alias))
        if match is not None:
            return match
    return None


def _series_value(
    arrays: Mapping[str, np.ndarray],
    aliases: Sequence[str],
    global_index: int,
) -> float | None:
    key = _first_key(arrays, aliases)
    if key is None:
        return None
    array = np.asarray(arrays[key])
    if array.ndim == 0:
        return float(array)
    if global_index >= array.shape[0]:
        return None
    value = np.asarray(array[global_index]).reshape(-1)
    if value.size != 1:
        return None
    numeric = float(value[0])
    return numeric if math.isfinite(numeric) else None


def _find_feature_value(
    observation: np.ndarray,
    feature_names: Sequence[str],
    aliases: Sequence[str],
) -> float | None:
    normalised_names = [_normalise_name(name) for name in feature_names]
    for alias in aliases:
        normalised_alias = _normalise_name(alias)
        for index, name in enumerate(normalised_names):
            if name == normalised_alias:
                return float(observation[index])
    for alias in aliases:
        normalised_alias = _normalise_name(alias)
        for index, name in enumerate(normalised_names):
            if normalised_alias in name:
                return float(observation[index])
    return None


def _axle_feature_average(
    observation: np.ndarray,
    feature_names: Sequence[str],
    *,
    axle: str,
    kind: str,
) -> float | None:
    axle_tokens = {
        "front": {"front", "fl", "fr"},
        "rear": {"rear", "rl", "rr"},
    }[axle]
    values: list[float] = []
    for index, name in enumerate(feature_names):
        name_tokens = _tokens(name)
        if "slip" not in name_tokens:
            continue
        if kind == "ratio" and "ratio" not in name_tokens:
            continue
        if kind == "angle" and not ({"angle", "sideslip"} & name_tokens):
            continue
        if not (axle_tokens & name_tokens):
            continue
        value = float(observation[index])
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------------------
# Tables and plotting
# ---------------------------------------------------------------------------


def _append_table_row(
    rows: list[dict[str, str]],
    group: str,
    signal: str,
    value: float | int | None,
    unit: str,
    source: str,
    digits: int = 4,
) -> None:
    if value is None or not math.isfinite(float(value)):
        rendered = "nicht aufgezeichnet"
    elif isinstance(value, int):
        rendered = str(value)
    else:
        rendered = f"{float(value):.{digits}f}"
    rows.append(
        {
            "Gruppe": group,
            "Signal": signal,
            "Wert": rendered,
            "Einheit": unit,
            "Quelle": source,
        }
    )


def _current_value_table(
    arrays: Mapping[str, np.ndarray],
    global_index: int,
    local_step: int,
    observation: np.ndarray | None,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    time_s = _series_value(arrays, ["simulation_time_s", "time_s"], global_index)
    progress_m = _series_value(
        arrays,
        ["reference_progress_m", "progress_m"],
        global_index,
    )
    progress_previous = (
        _series_value(arrays, ["reference_progress_m", "progress_m"], global_index - 1)
        if local_step > 0
        else None
    )
    progress_increment = (
        progress_m - progress_previous
        if progress_m is not None and progress_previous is not None
        else None
    )

    _append_table_row(rows, "Zeit", "Episode step", local_step, "-", "Artefakt", 0)
    _append_table_row(rows, "Zeit", "Simulation time", time_s, "s", "Artefakt")
    _append_table_row(rows, "Strecke", "Reference progress", progress_m, "m", "Artefakt")
    _append_table_row(
        rows,
        "Strecke",
        "Progress increment",
        progress_increment,
        "m/step",
        "abgeleitet",
    )

    speed_mps: float | None = None
    steering_angle: float | None = None
    front_slip_ratio: float | None = None
    rear_slip_ratio: float | None = None
    front_slip_angle: float | None = None
    rear_slip_angle: float | None = None
    if observation is not None:
        speed_mps = _find_feature_value(
            observation,
            feature_names,
            [
                "current_speed_mps",
                "vehicle_speed_mps",
                "speed_mps",
                "absolute_velocity_mps",
                "velocity_magnitude_mps",
            ],
        )
        steering_angle = _find_feature_value(
            observation,
            feature_names,
            ["steering_angle", "steering_wheel_angle", "current_steering"],
        )
        front_slip_ratio = _axle_feature_average(
            observation,
            feature_names,
            axle="front",
            kind="ratio",
        )
        rear_slip_ratio = _axle_feature_average(
            observation,
            feature_names,
            axle="rear",
            kind="ratio",
        )
        front_slip_angle = _axle_feature_average(
            observation,
            feature_names,
            axle="front",
            kind="angle",
        )
        rear_slip_angle = _axle_feature_average(
            observation,
            feature_names,
            axle="rear",
            kind="angle",
        )

    _append_table_row(rows, "Vehicle", "Current speed", speed_mps, "m/s", "Observation")
    _append_table_row(
        rows,
        "Vehicle",
        "Current speed",
        speed_mps * 3.6 if speed_mps is not None else None,
        "km/h",
        "Observation",
        2,
    )
    _append_table_row(
        rows,
        "Vehicle",
        "Steering angle",
        steering_angle,
        "Observation unit",
        "Observation",
    )
    _append_table_row(
        rows,
        "Wheel state",
        "Slip ratio front",
        front_slip_ratio,
        "-",
        "Observation axle mean",
    )
    _append_table_row(
        rows,
        "Wheel state",
        "Slip ratio rear",
        rear_slip_ratio,
        "-",
        "Observation axle mean",
    )
    _append_table_row(
        rows,
        "Wheel state",
        "Slip angle front",
        front_slip_angle,
        "Observation unit",
        "Observation axle mean",
    )
    _append_table_row(
        rows,
        "Wheel state",
        "Slip angle rear",
        rear_slip_angle,
        "Observation unit",
        "Observation axle mean",
    )

    actions = arrays.get("actions")
    pedal: float | None = None
    steering_command: float | None = None
    if actions is not None and global_index < actions.shape[0]:
        current_action = np.asarray(actions[global_index]).reshape(-1)
        if current_action.size >= 1:
            pedal = float(current_action[0])
        if current_action.size >= 2:
            steering_command = float(current_action[1])
    _append_table_row(rows, "Action", "Combined pedal", pedal, "[-1, 1]", "Action")
    _append_table_row(
        rows,
        "Action",
        "Throttle",
        max(pedal, 0.0) if pedal is not None else None,
        "[0, 1]",
        "aus Pedal abgeleitet",
    )
    _append_table_row(
        rows,
        "Action",
        "Brake",
        max(-pedal, 0.0) if pedal is not None else None,
        "[0, 1]",
        "aus Pedal abgeleitet",
    )
    _append_table_row(
        rows,
        "Action",
        "Steering command",
        steering_command,
        "[-1, 1]",
        "Action",
    )

    _append_table_row(
        rows,
        "Reward",
        "Environment reward",
        _series_value(arrays, ["rewards", "environment_rewards"], global_index),
        "-",
        "Artefakt",
    )
    _append_table_row(
        rows,
        "Reward",
        "Progress reward",
        _series_value(
            arrays,
            ["progress_reward", "reward_progress", "progress_rewards"],
            global_index,
        ),
        "-",
        "Artefakt",
    )
    _append_table_row(
        rows,
        "Reward",
        "Predicted progress reward",
        _series_value(
            arrays,
            [
                "predicted_progress_reward",
                "reward_predicted_progress",
                "prediction_progress_reward",
            ],
            global_index,
        ),
        "-",
        "Artefakt",
    )
    _append_table_row(
        rows,
        "Reward",
        "Shaped/total reward",
        _series_value(
            arrays,
            ["shaped_rewards", "total_rewards", "reward_total"],
            global_index,
        ),
        "-",
        "Artefakt",
    )
    return pd.DataFrame(rows)


def _add_line(
    figure: go.Figure,
    points: np.ndarray | None,
    name: str,
    *,
    dash: str = "solid",
    width: float = 2.0,
    opacity: float = 1.0,
) -> None:
    if points is None or points.shape[0] < 2:
        return
    figure.add_trace(
        go.Scattergl(
            x=points[:, 0],
            y=points[:, 1],
            mode="lines",
            name=name,
            line={"dash": dash, "width": width},
            opacity=opacity,
            hovertemplate=f"{name}<br>x=%{{x:.2f}} m<br>y=%{{y:.2f}} m<extra></extra>",
        )
    )


def _add_dynamic_points(
    figure: go.Figure,
    points: np.ndarray | None,
    name: str,
    *,
    show_line: bool = True,
) -> None:
    if points is None or points.size == 0:
        return
    mode = "lines+markers" if show_line and points.shape[0] > 1 else "markers"
    figure.add_trace(
        go.Scattergl(
            x=points[:, 0],
            y=points[:, 1],
            mode=mode,
            name=name,
            marker={"size": 7},
            line={"width": 2},
            hovertemplate=f"{name}<br>x=%{{x:.2f}} m<br>y=%{{y:.2f}} m<extra></extra>",
        )
    )


def _build_figure(
    *,
    episode_positions: np.ndarray,
    current_local_step: int,
    current_heading: float,
    static_layers: Mapping[str, np.ndarray | None],
    reference_features_world: np.ndarray | None,
    left_border_features_world: np.ndarray | None,
    right_border_features_world: np.ndarray | None,
    bezier_curve_world: np.ndarray | None,
    bezier_points_world: np.ndarray | None,
    show_options: Mapping[str, bool],
    zoom_mode: str,
    follow_radius_m: float,
    title: str,
) -> go.Figure:
    figure = go.Figure()

    if show_options["track_borders"]:
        _add_line(figure, static_layers.get("left_border"), "Track border left", width=3)
        _add_line(figure, static_layers.get("right_border"), "Track border right", width=3)
    if show_options["reference_line"]:
        _add_line(
            figure,
            static_layers.get("reference_line"),
            "Reference line",
            dash="dash",
            width=2,
        )
    if show_options["racing_line"]:
        _add_line(
            figure,
            static_layers.get("racing_line"),
            "Racing line",
            dash="dot",
            width=2,
        )
    if show_options["start_finish"]:
        _add_line(
            figure,
            static_layers.get("start_finish"),
            "Start / finish",
            width=5,
        )

    if show_options["full_driven_line"]:
        _add_line(
            figure,
            episode_positions,
            "Driven line (complete episode)",
            width=2,
            opacity=0.35,
        )
    if show_options["driven_trail"]:
        _add_line(
            figure,
            episode_positions[: current_local_step + 1],
            "Driven line up to current step",
            width=4,
        )

    if show_options["reference_features"]:
        _add_dynamic_points(
            figure,
            reference_features_world,
            "Reference features",
        )
    if show_options["border_features"]:
        _add_dynamic_points(
            figure,
            left_border_features_world,
            "Track-border features left",
        )
        _add_dynamic_points(
            figure,
            right_border_features_world,
            "Track-border features right",
        )
    if show_options["bezier_curve"]:
        _add_line(
            figure,
            bezier_curve_world,
            "Predicted Bézier line",
            width=4,
        )
    if show_options["bezier_points"]:
        _add_dynamic_points(
            figure,
            bezier_points_world,
            "Bézier points",
            show_line=False,
        )

    current_position = episode_positions[current_local_step]
    figure.add_trace(
        go.Scattergl(
            x=[current_position[0]],
            y=[current_position[1]],
            mode="markers",
            name="Vehicle",
            marker={"size": 14, "symbol": "diamond"},
            hovertemplate="Vehicle<br>x=%{x:.2f} m<br>y=%{y:.2f} m<extra></extra>",
        )
    )
    arrow_length = max(3.0, min(10.0, follow_radius_m * 0.15))
    arrow_end = current_position + arrow_length * np.asarray(
        [math.cos(current_heading), math.sin(current_heading)]
    )
    _add_line(
        figure,
        np.vstack((current_position, arrow_end)),
        "Vehicle heading",
        width=4,
    )

    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    if zoom_mode == "Fahrzeug folgen":
        figure.update_xaxes(
            range=[
                current_position[0] - follow_radius_m,
                current_position[0] + follow_radius_m,
            ]
        )
        figure.update_yaxes(
            range=[
                current_position[1] - follow_radius_m,
                current_position[1] + follow_radius_m,
            ]
        )
    figure.update_layout(
        title=title,
        height=760,
        hovermode="closest",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        xaxis_title="World X [m]",
        yaxis_title="World Y [m]",
    )
    return figure


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0915 - one declarative Streamlit entry point
    st.set_page_config(
        page_title="Assetto Corsa Evaluation Viewer",
        layout="wide",
    )
    st.title("Assetto Corsa Evaluation Viewer")
    st.caption(
        "MLflow Run → Evaluation Update → Episode → Zeitschritt"
    )

    tracking_uri = st.sidebar.text_input(
        "MLflow Tracking URI",
        value=DEFAULT_TRACKING_URI,
    ).strip()
    if st.sidebar.button("MLflow-Daten aktualisieren"):
        st.cache_data.clear()
        st.rerun()

    try:
        experiments = _list_experiments(tracking_uri)
    except Exception as error:  # MLflow raises backend-specific exceptions.
        st.error(f"MLflow ist nicht erreichbar: {error}")
        st.stop()

    if not experiments:
        st.warning("Auf dem MLflow-Server wurden keine Experimente gefunden.")
        st.stop()

    experiment_names = [item["name"] for item in experiments]
    default_experiment_index = (
        experiment_names.index("Assetto Corsa PPO")
        if "Assetto Corsa PPO" in experiment_names
        else 0
    )
    selected_experiment_name = st.sidebar.selectbox(
        "Experiment",
        experiment_names,
        index=default_experiment_index,
    )
    experiment_id = next(
        item["experiment_id"]
        for item in experiments
        if item["name"] == selected_experiment_name
    )

    all_runs = _list_runs(tracking_uri, experiment_id)
    training_runs = _training_runs(all_runs)
    if not training_runs:
        st.warning("In diesem Experiment wurde kein Training-Run gefunden.")
        st.stop()

    training_by_id = {run["run_id"]: run for run in training_runs}
    selected_training_id = st.sidebar.selectbox(
        "Training-Run",
        list(training_by_id),
        format_func=lambda run_id: _format_run(training_by_id[run_id]),
    )
    training_run = training_by_id[selected_training_id]

    evaluation_runs = _evaluation_runs(all_runs, selected_training_id)
    if not evaluation_runs:
        st.warning(
            "Für den ausgewählten Training-Run wurden keine verschachtelten "
            "Evaluation-Runs gefunden."
        )
        st.stop()

    evaluation_by_id = {run["run_id"]: run for run in evaluation_runs}
    selected_evaluation_id = st.sidebar.selectbox(
        "Evaluation-Update",
        list(evaluation_by_id),
        format_func=lambda run_id: _format_evaluation_run(evaluation_by_id[run_id]),
    )
    evaluation_run = evaluation_by_id[selected_evaluation_id]

    try:
        with st.spinner("Evaluation-Artefakte werden geladen …"):
            arrays, summary, episode_rows, artifact_paths = _load_evaluation_bundle(
                tracking_uri,
                selected_evaluation_id,
                evaluation_run.get("end_time"),
            )
    except Exception as error:
        st.error(f"Evaluation konnte nicht geladen werden: {error}")
        st.stop()

    positions = _as_xy(arrays.get("positions_xy_m"))
    if positions is None:
        st.error("trajectories.npz enthält keine gültigen positions_xy_m.")
        st.stop()
    total_steps = positions.shape[0]

    episode_ids = np.asarray(
        arrays.get("episode_ids", np.zeros(total_steps, dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)
    if episode_ids.size != total_steps:
        st.error("episode_ids und positions_xy_m haben unterschiedliche Längen.")
        st.stop()
    available_episodes = sorted(int(value) for value in np.unique(episode_ids))
    selected_episode = st.sidebar.selectbox("Episode", available_episodes)
    episode_indices = np.flatnonzero(episode_ids == selected_episode)
    episode_positions = positions[episode_indices]
    if episode_positions.size == 0:
        st.error("Die ausgewählte Episode enthält keine Trajektorienpunkte.")
        st.stop()

    headings = np.asarray(
        arrays.get("vehicle_heading_rad", np.zeros(total_steps)),
        dtype=np.float64,
    ).reshape(-1)
    if headings.size != total_steps:
        headings = np.zeros(total_steps, dtype=np.float64)

    observations = arrays.get("observations")
    if observations is not None:
        observations = np.asarray(observations)
        if observations.ndim != 2 or observations.shape[0] != total_steps:
            observations = None
    names = (
        _feature_names(arrays, observations.shape[1])
        if observations is not None
        else []
    )

    # Static geometry can be loaded from the reference artifact configured in
    # the parent training run. A manual NPZ upload is available as a fallback.
    configured_geometry_path = str(
        training_run["params"].get("reference.artifact_path", "")
    )
    geometry_path_text = st.sidebar.text_input(
        "Track-/Referenz-NPZ",
        value=configured_geometry_path,
        key=f"geometry_path_{selected_training_id}",
        help="Relativpfade werden vom Repository-Root aus aufgelöst.",
    )
    uploaded_geometry = st.sidebar.file_uploader(
        "Alternativ Track-NPZ hochladen",
        type=["npz"],
        key=f"geometry_upload_{selected_training_id}",
    )

    geometry_arrays: dict[str, np.ndarray] = {}
    geometry_source = "keine"
    if uploaded_geometry is not None:
        try:
            geometry_arrays = _load_uploaded_npz(uploaded_geometry.getvalue())
            geometry_source = uploaded_geometry.name
        except Exception as error:
            st.sidebar.error(f"Track-NPZ konnte nicht gelesen werden: {error}")
    elif geometry_path_text.strip():
        geometry_path = _resolve_local_path(geometry_path_text.strip())
        if geometry_path.is_file():
            try:
                geometry_arrays = _load_npz_path(
                    str(geometry_path),
                    geometry_path.stat().st_mtime_ns,
                )
                geometry_source = str(geometry_path)
            except Exception as error:
                st.sidebar.error(f"Track-NPZ konnte nicht gelesen werden: {error}")
        else:
            st.sidebar.info(f"Track-NPZ nicht gefunden: {geometry_path}")

    # Optional geometry embedded in trajectories.npz is also accepted. A
    # separate static reference artifact overrides duplicate keys.
    combined_geometry = {
        **{
            key: value
            for key, value in arrays.items()
            if np.asarray(value).ndim == 2
            and key not in {"positions_xy_m", "observations", "actions"}
        },
        **geometry_arrays,
    }
    geometry_keys = _geometry_candidates(combined_geometry)

    with st.sidebar.expander("Statische Plot-Layer", expanded=bool(geometry_keys)):
        reference_key = _geometry_selector(
            "Reference line",
            geometry_keys,
            [
                "reference_line_xy",
                "reference_positions_xy_m",
                "centerline_xy",
                "centreline_xy",
                "positions_xy_m",
                "line_xy",
            ],
            f"reference_key_{selected_training_id}",
        )
        racing_key = _geometry_selector(
            "Racing line",
            geometry_keys,
            [
                "racing_line_xy",
                "fast_lane_xy",
                "driving_line_xy",
                "ideal_line_xy",
            ],
            f"racing_key_{selected_training_id}",
        )
        left_key = _geometry_selector(
            "Track border left",
            geometry_keys,
            [
                "left_boundary_xy",
                "left_border_xy",
                "track_boundary_left_xy",
                "boundary_left_xy",
            ],
            f"left_key_{selected_training_id}",
        )
        right_key = _geometry_selector(
            "Track border right",
            geometry_keys,
            [
                "right_boundary_xy",
                "right_border_xy",
                "track_boundary_right_xy",
                "boundary_right_xy",
            ],
            f"right_key_{selected_training_id}",
        )
        start_finish_key = _geometry_selector(
            "Start / finish line",
            geometry_keys,
            [
                "start_finish_line_xy",
                "start_line_xy",
                "finish_line_xy",
            ],
            f"start_finish_key_{selected_training_id}",
        )

    static_layers: dict[str, np.ndarray | None] = {
        "reference_line": _as_xy(
            combined_geometry.get(reference_key) if reference_key else None
        ),
        "racing_line": _as_xy(
            combined_geometry.get(racing_key) if racing_key else None
        ),
        "left_border": _as_xy(
            combined_geometry.get(left_key) if left_key else None
        ),
        "right_border": _as_xy(
            combined_geometry.get(right_key) if right_key else None
        ),
        "start_finish": _as_xy(
            combined_geometry.get(start_finish_key) if start_finish_key else None
        ),
    }
    if (
        static_layers["start_finish"] is None
        and static_layers["left_border"] is not None
        and static_layers["right_border"] is not None
    ):
        static_layers["start_finish"] = np.vstack(
            (static_layers["left_border"][0], static_layers["right_border"][0])
        )

    with st.sidebar.expander("Sichtbarkeit", expanded=True):
        show_options = {
            "full_driven_line": st.checkbox("Komplette Fahrline", value=True),
            "driven_trail": st.checkbox("Fahrline bis zum aktuellen Punkt", value=True),
            "reference_line": st.checkbox("Reference line", value=True),
            "racing_line": st.checkbox("Racing line", value=True),
            "track_borders": st.checkbox("Track borders", value=True),
            "start_finish": st.checkbox("Start-/Ziellinie", value=True),
            "reference_features": st.checkbox("Reference features", value=True),
            "border_features": st.checkbox("Track-border features", value=True),
            "bezier_curve": st.checkbox("Prädiktierte Bézier-Linie", value=True),
            "bezier_points": st.checkbox("Bézier-Punkte", value=True),
        }

    with st.sidebar.expander("Koordinatentransformation", expanded=False):
        forward_axis = st.selectbox(
            "Lokale Vorwärtsachse",
            ["z", "x"],
            help="Standard: lokale x-Achse lateral, lokale z-Achse vorwärts.",
        )
        lateral_sign = st.selectbox(
            "Vorzeichen der lokalen Querachse",
            [1.0, -1.0],
            format_func=lambda value: "+1" if value > 0 else "-1",
        )
        invert_heading = st.checkbox("Heading-Vorzeichen invertieren", value=False)
        heading_offset_deg = st.number_input(
            "Heading-Offset [deg]",
            min_value=-180.0,
            max_value=180.0,
            value=0.0,
            step=1.0,
        )

    zoom_mode = st.sidebar.selectbox(
        "Ansicht",
        ["Ganze Strecke", "Fahrzeug folgen"],
    )
    follow_radius_m = st.sidebar.slider(
        "Radius der Folgeansicht [m]",
        min_value=10.0,
        max_value=300.0,
        value=80.0,
        step=5.0,
        disabled=zoom_mode != "Fahrzeug folgen",
    )

    selection_signature = (
        f"{selected_training_id}:{selected_evaluation_id}:{selected_episode}"
    )
    signature_key = "viewer_selection_signature"
    step_key = "viewer_current_step"
    playing_key = "viewer_playing"
    slider_key = f"viewer_slider_{abs(hash(selection_signature))}"
    if st.session_state.get(signature_key) != selection_signature:
        st.session_state[signature_key] = selection_signature
        st.session_state[step_key] = 0
        st.session_state[playing_key] = False
        st.session_state[slider_key] = 0

    maximum_step = episode_positions.shape[0] - 1
    st.session_state[step_key] = int(
        np.clip(st.session_state.get(step_key, 0), 0, maximum_step)
    )

    control_columns = st.columns([1, 1, 1, 1, 2, 2])
    if control_columns[0].button("⏮ Anfang", use_container_width=True):
        st.session_state[step_key] = 0
        st.session_state[slider_key] = 0
    if control_columns[1].button("◀ Zurück", use_container_width=True):
        step = max(0, st.session_state[step_key] - 1)
        st.session_state[step_key] = step
        st.session_state[slider_key] = step
    play_label = "⏸ Pause" if st.session_state[playing_key] else "▶ Play"
    if control_columns[2].button(play_label, use_container_width=True):
        st.session_state[playing_key] = not st.session_state[playing_key]
        if not st.session_state[playing_key]:
            st.session_state[slider_key] = st.session_state[step_key]
    if control_columns[3].button("Vor ▶", use_container_width=True):
        step = min(maximum_step, st.session_state[step_key] + 1)
        st.session_state[step_key] = step
        st.session_state[slider_key] = step
    playback_fps = control_columns[4].slider(
        "Wiedergabe [Schritte/s]",
        min_value=1,
        max_value=20,
        value=5,
    )
    playback_increment = control_columns[5].selectbox(
        "Schrittweite",
        [1, 2, 5, 10],
        index=0,
    )

    run_every: float | None = (
        1.0 / playback_fps if st.session_state[playing_key] else None
    )

    @st.fragment(run_every=run_every)
    def render_dynamic_view() -> None:
        if st.session_state[playing_key]:
            local_step = int(st.session_state[step_key])
            st.progress(
                local_step / max(maximum_step, 1),
                text=f"Step {local_step} / {maximum_step}",
            )
        else:
            local_step = st.slider(
                "Zeitschritt",
                min_value=0,
                max_value=maximum_step,
                key=slider_key,
            )
            st.session_state[step_key] = int(local_step)

        global_index = int(episode_indices[local_step])
        current_position = positions[global_index]
        current_heading = float(headings[global_index])
        current_observation = (
            np.asarray(observations[global_index], dtype=np.float64)
            if observations is not None
            else None
        )

        reference_local = (
            _feature_points(
                current_observation,
                names,
                ["reference", "ref_line", "reference_line"],
            )
            if current_observation is not None
            else None
        )
        left_border_local = (
            _feature_points(
                current_observation,
                names,
                ["left_boundary", "left_border", "boundary_left", "border_left"],
            )
            if current_observation is not None
            else None
        )
        right_border_local = (
            _feature_points(
                current_observation,
                names,
                ["right_boundary", "right_border", "boundary_right", "border_right"],
            )
            if current_observation is not None
            else None
        )

        transform_arguments = {
            "forward_axis": forward_axis,
            "lateral_sign": lateral_sign,
            "invert_heading": invert_heading,
            "heading_offset_deg": heading_offset_deg,
        }
        reference_world = _local_to_world(
            reference_local,
            current_position,
            current_heading,
            **transform_arguments,
        )
        left_border_world = _local_to_world(
            left_border_local,
            current_position,
            current_heading,
            **transform_arguments,
        )
        right_border_world = _local_to_world(
            right_border_local,
            current_position,
            current_heading,
            **transform_arguments,
        )

        bezier_samples_local = _step_points(
            arrays,
            [
                "bezier_prediction_points",
                "predicted_bezier_points",
                "bezier_curve_points",
            ],
            global_index,
            total_steps,
        )
        bezier_control_local = _step_points(
            arrays,
            ["bezier_control_points", "predicted_bezier_control_points"],
            global_index,
            total_steps,
        )
        if bezier_control_local is not None:
            bezier_curve_local = _sample_bezier(bezier_control_local)
            displayed_bezier_points = bezier_control_local
        else:
            bezier_curve_local = bezier_samples_local
            displayed_bezier_points = bezier_samples_local
        bezier_curve_world = _local_to_world(
            bezier_curve_local,
            current_position,
            current_heading,
            **transform_arguments,
        )
        bezier_points_world = _local_to_world(
            displayed_bezier_points,
            current_position,
            current_heading,
            **transform_arguments,
        )

        simulation_time = _series_value(
            arrays,
            ["simulation_time_s", "time_s"],
            global_index,
        )
        update = evaluation_run.get("checkpoint_update", -1)
        plot_title = (
            f"Update {update:06d} · Episode {selected_episode} · "
            f"Step {local_step}/{maximum_step}"
        )
        if simulation_time is not None:
            plot_title += f" · t={simulation_time:.2f} s"

        figure = _build_figure(
            episode_positions=episode_positions,
            current_local_step=local_step,
            current_heading=(
                -current_heading if invert_heading else current_heading
            )
            + math.radians(heading_offset_deg),
            static_layers=static_layers,
            reference_features_world=reference_world,
            left_border_features_world=left_border_world,
            right_border_features_world=right_border_world,
            bezier_curve_world=bezier_curve_world,
            bezier_points_world=bezier_points_world,
            show_options=show_options,
            zoom_mode=zoom_mode,
            follow_radius_m=follow_radius_m,
            title=plot_title,
        )
        table = _current_value_table(
            arrays,
            global_index,
            local_step,
            current_observation,
            names,
        )

        plot_column, table_column = st.columns([2.2, 1.0])
        with plot_column:
            st.plotly_chart(
                figure,
                use_container_width=True,
                config={"scrollZoom": True, "displaylogo": False},
                key=f"trajectory_plot_{selection_signature}_{local_step}",
            )
        with table_column:
            st.subheader("Aktueller Punkt")
            st.dataframe(
                table,
                hide_index=True,
                use_container_width=True,
                height=710,
            )

        if current_observation is not None:
            with st.expander("Alle Observation Features am aktuellen Punkt"):
                observation_table = pd.DataFrame(
                    {
                        "Index": np.arange(current_observation.size),
                        "Feature": names,
                        "Wert": current_observation,
                    }
                )
                st.dataframe(
                    observation_table,
                    hide_index=True,
                    use_container_width=True,
                )

        if st.session_state[playing_key]:
            next_step = min(
                maximum_step,
                local_step + int(playback_increment),
            )
            if next_step == local_step:
                st.session_state[playing_key] = False
                st.session_state[slider_key] = local_step
                st.rerun()
            else:
                st.session_state[step_key] = next_step

    render_dynamic_view()

    metadata_columns = st.columns(4)
    metadata_columns[0].metric("Training-Run", str(training_run["run_name"]))
    metadata_columns[1].metric(
        "Evaluation Update",
        int(evaluation_run.get("checkpoint_update", -1)),
    )
    metadata_columns[2].metric("Episode", selected_episode)
    metadata_columns[3].metric("Episode steps", episode_positions.shape[0])

    with st.expander("Evaluation- und Datenstatus"):
        st.write(f"**Geometriequelle:** `{geometry_source}`")
        if summary:
            st.write("**summary.json**")
            st.json(summary)
        if episode_rows:
            st.write("**episodes.csv**")
            st.dataframe(pd.DataFrame(episode_rows), hide_index=True)
        st.write("**Arrays in trajectories.npz**")
        array_table = pd.DataFrame(
            [
                {
                    "Name": name,
                    "Shape": " × ".join(str(value) for value in array.shape),
                    "Dtype": str(array.dtype),
                }
                for name, array in sorted(arrays.items())
            ]
        )
        st.dataframe(array_table, hide_index=True, use_container_width=True)
        st.write("**MLflow-Artefakte im Evaluation-Run**")
        st.code("\n".join(artifact_paths))


if __name__ == "__main__":
    main()
