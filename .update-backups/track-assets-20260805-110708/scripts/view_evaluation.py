"""Interactive MLflow evaluation viewer for the Assetto Corsa RL driver.

Run from the repository root with::

    python -m streamlit run scripts/view_evaluation.py

Static track geometry is loaded from the parent training run's MLflow artefacts:
``track/track_geometry.npz`` and ``reference/reference.npz``. Step-level
reference, boundary and Bézier previews are read as unnormalised world-space
arrays from ``trajectories.npz``. Normalised observations are never interpreted
as physical geometry.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
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
    names = {name.casefold() for name in filenames}
    matches = [path for path in paths if Path(path).name.casefold() in names]
    if preferred_prefix:
        preferred = [
            path
            for path in matches
            if path.replace("\\", "/").startswith(preferred_prefix.rstrip("/") + "/")
        ]
        if preferred:
            matches = preferred
    return min(matches, key=lambda item: (item.count("/"), len(item))) if matches else None


def _cache_directory(run_id: str) -> Path:
    directory = Path(tempfile.gettempdir()) / "assetto_corsa_evaluation_viewer" / run_id
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
    selected = _choose_artifact(paths, ("trajectories.npz",))
    if selected is None:
        raise FileNotFoundError("Evaluation enthält kein trajectories.npz-Artefakt")
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


def _as_xy(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        return None
    if array.shape[0] in {2, 3} and array.shape[1] > array.shape[0]:
        array = array.T
    if array.shape[1] >= 3:
        array = array[:, [0, 2]]
    elif array.shape[1] == 2:
        array = array[:, :2]
    else:
        return None
    finite = np.isfinite(array).all(axis=1)
    result = array[finite]
    return result if result.size else None


def _step_points(
    arrays: Mapping[str, np.ndarray],
    name: str,
    index: int,
    total_steps: int,
) -> np.ndarray | None:
    if name not in arrays:
        return None
    value = np.asarray(arrays[name], dtype=np.float64)
    if value.ndim == 3 and value.shape[0] == total_steps:
        result = value[index]
    elif value.ndim == 2:
        result = value
    else:
        return None
    if result.shape[-1] < 2:
        return None
    result = result[:, :2]
    finite = np.isfinite(result).all(axis=1)
    result = result[finite]
    return result if result.size else None


def _local_xz_to_world(
    local: np.ndarray | None,
    position: np.ndarray,
    heading: float,
) -> np.ndarray | None:
    if local is None:
        return None
    forward = np.asarray([math.cos(heading), math.sin(heading)])
    left = np.asarray([-math.sin(heading), math.cos(heading)])
    return position + local[:, 0:1] * left + local[:, 1:2] * forward


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
            hovertemplate=f"{name}<br>x=%{{x:.2f}} m<br>y=%{{y:.2f}} m<extra></extra>",
        )
    )


def _add_points(
    figure: go.Figure,
    points: np.ndarray | None,
    name: str,
    *,
    connect: bool = True,
) -> None:
    if points is None or not points.size:
        return
    figure.add_trace(
        go.Scattergl(
            x=points[:, 0],
            y=points[:, 1],
            mode="lines+markers" if connect else "markers",
            name=name,
            marker={"size": 7},
            hovertemplate=f"{name}<br>x=%{{x:.2f}} m<br>y=%{{y:.2f}} m<extra></extra>",
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


def _series(arrays: Mapping[str, np.ndarray], name: str, index: int) -> float | None:
    if name not in arrays:
        return None
    value = np.asarray(arrays[name]).reshape(-1)
    if index >= value.size:
        return None
    result = float(value[index])
    return result if math.isfinite(result) else None


def _action_values(arrays: Mapping[str, np.ndarray], index: int) -> tuple[float | None, float | None]:
    actions = arrays.get("actions")
    if actions is None:
        return None, None
    value = np.asarray(actions, dtype=np.float64)
    if value.ndim != 2 or index >= value.shape[0] or value.shape[1] < 2:
        return None, None
    return float(value[index, 0]), float(value[index, 1])


def _predictor_errors(
    prediction: np.ndarray | None,
    reference: np.ndarray | None,
) -> tuple[float | None, float | None]:
    if prediction is None or reference is None:
        return None, None
    count = min(prediction.shape[0], reference.shape[0])
    if count == 0:
        return None, None
    distance = np.linalg.norm(prediction[:count] - reference[:count], axis=1)
    return float(distance.mean()), float(distance[-1])


def _value_table(
    arrays: Mapping[str, np.ndarray],
    positions: np.ndarray,
    times: np.ndarray | None,
    global_index: int,
    local_step: int,
    prediction: np.ndarray | None,
    reference: np.ndarray | None,
) -> pd.DataFrame:
    speed = _speed_at_step(positions, times, global_index)
    pedal, steering = _action_values(arrays, global_index)
    ade, fde = _predictor_errors(prediction, reference)
    rows = [
        ("Zeit", "Episode step", local_step, "-", "Artefakt"),
        ("Zeit", "Simulation time", _series(arrays, "simulation_time_s", global_index), "s", "Artefakt"),
        ("Strecke", "Reference progress", _series(arrays, "reference_progress_m", global_index), "m", "Artefakt"),
        ("Fahrzeug", "Geschwindigkeit", speed, "m/s", "aus Position/Zeit"),
        ("Fahrzeug", "Geschwindigkeit", speed * 3.6 if speed is not None else None, "km/h", "abgeleitet"),
        ("Aktion", "Combined pedal", pedal, "[-1,1]", "Action"),
        ("Aktion", "Steering command", steering, "[-1,1]", "Action"),
        ("Reward", "Environment reward", _series(arrays, "rewards", global_index), "-", "Artefakt"),
        ("Predictor", "ADE", ade, "m", "Weltkoordinaten"),
        ("Predictor", "FDE", fde, "m", "Weltkoordinaten"),
    ]
    def format_value(value: object) -> str:
        if value is None:
            return "nicht aufgezeichnet"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    return pd.DataFrame(
        [
            {
                "Gruppe": str(group),
                "Signal": str(signal),
                "Wert": format_value(value),
                "Einheit": str(unit),
                "Quelle": str(source),
            }
            for group, signal, value, unit, source in rows
        ],
        dtype="string",
    )


def main() -> None:  # noqa: PLR0915
    st.set_page_config(page_title="Assetto Corsa Evaluation Viewer", layout="wide")
    st.title("Assetto Corsa Evaluation Viewer")
    st.caption("MLflow Run → Evaluation Update → Episode → Zeitschritt")

    tracking_uri = st.sidebar.text_input("MLflow Tracking URI", DEFAULT_TRACKING_URI).strip()
    if st.sidebar.button("MLflow-Daten aktualisieren"):
        st.cache_data.clear()
        st.rerun()

    try:
        experiments = _list_experiments(tracking_uri)
    except Exception as error:
        st.error(f"MLflow ist nicht erreichbar: {error}")
        st.stop()
    if not experiments:
        st.warning("Keine MLflow-Experimente gefunden.")
        st.stop()

    experiment_names = [item["name"] for item in experiments]
    default_index = experiment_names.index("Assetto Corsa PPO") if "Assetto Corsa PPO" in experiment_names else 0
    experiment_name = st.sidebar.selectbox("Experiment", experiment_names, index=default_index)
    experiment_id = next(item["experiment_id"] for item in experiments if item["name"] == experiment_name)
    all_runs = _list_runs(tracking_uri, experiment_id)
    training_runs = _training_runs(all_runs)
    if not training_runs:
        st.warning("Kein Training-Run gefunden.")
        st.stop()

    training_by_id = {run["run_id"]: run for run in training_runs}
    training_id = st.sidebar.selectbox(
        "Training-Run",
        list(training_by_id),
        format_func=lambda run_id: _format_training(training_by_id[run_id]),
    )
    training_run = training_by_id[training_id]
    evaluation_runs = _evaluation_runs(all_runs, training_id)
    if not evaluation_runs:
        st.warning("Für diesen Training-Run existieren keine Evaluation-Runs.")
        st.stop()

    evaluation_by_id = {run["run_id"]: run for run in evaluation_runs}
    evaluation_id = st.sidebar.selectbox(
        "Evaluation-Update",
        list(evaluation_by_id),
        format_func=lambda run_id: _format_evaluation(evaluation_by_id[run_id]),
    )
    evaluation_run = evaluation_by_id[evaluation_id]

    try:
        with st.spinner("Evaluation-Artefakte werden geladen …"):
            arrays, summary, episode_rows, evaluation_artifacts = _load_evaluation_bundle(
                tracking_uri,
                evaluation_id,
                evaluation_run.get("end_time"),
            )
            geometry, geometry_artifact, training_artifacts = _download_npz_artifact(
                tracking_uri,
                training_id,
                training_run.get("end_time"),
                ("track_geometry.npz",),
                "track",
            )
            reference, reference_artifact, _ = _download_npz_artifact(
                tracking_uri,
                training_id,
                training_run.get("end_time"),
                ("reference.npz", "fixed_reference.npz"),
                "reference",
            )
    except Exception as error:
        st.error(f"Artefakte konnten nicht geladen werden: {error}")
        st.stop()

    if not geometry or not reference:
        uploaded = st.sidebar.file_uploader(
            "Fallback: Track-/Referenz-NPZ hochladen",
            type=["npz"],
            accept_multiple_files=True,
        )
        for item in uploaded or []:
            loaded = _uploaded_npz(item.getvalue())
            if "left_boundary_world_xy_m" in loaded:
                geometry = loaded
            if "world_position_xy_m" in loaded:
                reference = loaded

    positions = _as_xy(arrays.get("positions_xy_m"))
    if positions is None:
        st.error("trajectories.npz enthält keine gültigen positions_xy_m.")
        st.stop()
    total_steps = positions.shape[0]
    episode_ids = np.asarray(arrays.get("episode_ids", np.zeros(total_steps)), dtype=np.int64).reshape(-1)
    if episode_ids.size != total_steps:
        st.error("episode_ids und positions_xy_m haben unterschiedliche Längen.")
        st.stop()
    episodes = sorted(int(value) for value in np.unique(episode_ids))
    selected_episode = st.sidebar.selectbox("Episode", episodes)
    episode_indices = np.flatnonzero(episode_ids == selected_episode)
    episode_positions = positions[episode_indices]
    headings = np.asarray(arrays.get("vehicle_heading_rad", np.zeros(total_steps)), dtype=np.float64).reshape(-1)
    times = np.asarray(arrays["simulation_time_s"], dtype=np.float64).reshape(-1) if "simulation_time_s" in arrays else None

    static_reference = _as_xy(reference.get("world_position_xy_m"))
    static_centerline = _as_xy(geometry.get("centerline_world_xy_m"))
    static_left = _as_xy(geometry.get("left_boundary_world_xy_m"))
    static_right = _as_xy(geometry.get("right_boundary_world_xy_m"))
    start_finish = (
        np.vstack((static_left[0], static_right[0]))
        if static_left is not None and static_right is not None
        else None
    )

    if static_left is None or static_right is None:
        st.warning(
            "Der Parent-Run enthält keine vollständigen Track-Borders. "
            "Neue Runs müssen track/track_geometry.npz loggen."
        )
    if "observations" in arrays:
        st.caption(
            "Observations sind normalisierte Modell-Eingaben und werden nicht als Meter oder Telemetrie interpretiert."
        )

    with st.sidebar.expander("Sichtbarkeit", expanded=True):
        show = {
            "driven": st.checkbox("Fahrline", True),
            "trail": st.checkbox("Fahrline bis aktueller Punkt", True),
            "reference": st.checkbox("Referenzlinie", True),
            "centerline": st.checkbox("AI-Spline/Centreline", True),
            "borders": st.checkbox("Track-Borders", True),
            "start": st.checkbox("Start-/Ziellinie", True),
            "reference_preview": st.checkbox("Referenzvorschau", True),
            "boundary_preview": st.checkbox("Border-Vorschau", True),
            "bezier": st.checkbox("Prädiktierte Bézier-Linie", True),
        }
    zoom_mode = st.sidebar.selectbox("Ansicht", ["Fahrzeug folgen", "Ganze Strecke"])
    follow_radius = st.sidebar.slider("Folgeradius [m]", 10.0, 300.0, 80.0, 5.0)

    signature = f"{training_id}:{evaluation_id}:{selected_episode}"
    if st.session_state.get("selection") != signature:
        st.session_state["selection"] = signature
        st.session_state["step"] = 0
        st.session_state["playing"] = False
    maximum_step = episode_indices.size - 1
    st.session_state["step"] = int(np.clip(st.session_state.get("step", 0), 0, maximum_step))

    controls = st.columns([1, 1, 1, 1, 2, 2])
    if controls[0].button("⏮ Anfang", width="stretch"):
        st.session_state["step"] = 0
    if controls[1].button("◀ Zurück", width="stretch"):
        st.session_state["step"] = max(0, st.session_state["step"] - 1)
    label = "⏸ Pause" if st.session_state["playing"] else "▶ Play"
    if controls[2].button(label, width="stretch"):
        st.session_state["playing"] = not st.session_state["playing"]
    if controls[3].button("Vor ▶", width="stretch"):
        st.session_state["step"] = min(maximum_step, st.session_state["step"] + 1)
    fps = controls[4].slider("Wiedergabe [Schritte/s]", 1, 20, 5)
    increment = controls[5].selectbox("Schrittweite", [1, 2, 5, 10])

    run_every = 1.0 / fps if st.session_state["playing"] else None

    @st.fragment(run_every=run_every)
    def render() -> None:
        if st.session_state["playing"]:
            local_step = int(st.session_state["step"])
            st.progress(local_step / max(maximum_step, 1), text=f"Step {local_step} / {maximum_step}")
        else:
            local_step = st.slider("Zeitschritt", 0, maximum_step, value=int(st.session_state["step"]))
            st.session_state["step"] = local_step

        global_index = int(episode_indices[local_step])
        current_position = positions[global_index]
        current_heading = float(headings[global_index]) if global_index < headings.size else 0.0

        reference_preview = _step_points(arrays, "reference_preview_world_xy_m", global_index, total_steps)
        left_preview = _step_points(arrays, "left_boundary_preview_world_xy_m", global_index, total_steps)
        right_preview = _step_points(arrays, "right_boundary_preview_world_xy_m", global_index, total_steps)
        bezier = _step_points(arrays, "bezier_prediction_world_xy_m", global_index, total_steps)
        if bezier is None:
            local_bezier = _step_points(arrays, "bezier_prediction_local_xz_m", global_index, total_steps)
            if local_bezier is None:
                local_bezier = _step_points(arrays, "bezier_prediction_points", global_index, total_steps)
            bezier = _local_xz_to_world(local_bezier, current_position, current_heading)

        figure = go.Figure()
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
            _add_line(figure, episode_positions, "Driven line (complete)")
        if show["trail"]:
            _add_line(figure, episode_positions[: local_step + 1], "Driven line to current step", width=4)
        if show["reference_preview"]:
            _add_points(figure, reference_preview, "Reference preview")
        if show["boundary_preview"]:
            _add_points(figure, left_preview, "Boundary preview left")
            _add_points(figure, right_preview, "Boundary preview right")
        if show["bezier"]:
            _add_points(figure, bezier, "Predicted Bézier line")

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
        _add_line(figure, np.vstack((current_position, arrow)), "Vehicle heading", width=4)

        simulation_time = _series(arrays, "simulation_time_s", global_index)
        update = int(evaluation_run.get("checkpoint_update", -1))
        title = f"Update {update:06d} · Episode {selected_episode} · Step {local_step}/{maximum_step}"
        if simulation_time is not None:
            title += f" · t={simulation_time:.2f} s"
        figure.update_layout(
            title=title,
            height=760,
            hovermode="closest",
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
            xaxis_title="World X [m]",
            yaxis_title="World Y [m]",
        )
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
        if zoom_mode == "Fahrzeug folgen":
            figure.update_xaxes(range=[current_position[0] - follow_radius, current_position[0] + follow_radius])
            figure.update_yaxes(range=[current_position[1] - follow_radius, current_position[1] + follow_radius])

        plot_column, table_column = st.columns([2.2, 1.0])
        with plot_column:
            st.plotly_chart(figure, width="stretch", config={"scrollZoom": True, "displaylogo": False})
        with table_column:
            st.subheader("Aktueller Punkt")
            st.dataframe(
                _value_table(
                    arrays,
                    positions,
                    times,
                    global_index,
                    local_step,
                    bezier,
                    reference_preview,
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

    metrics = st.columns(4)
    metrics[0].metric("Training-Run", str(training_run["run_name"]))
    metrics[1].metric("Evaluation Update", int(evaluation_run.get("checkpoint_update", -1)))
    metrics[2].metric("Episode", selected_episode)
    metrics[3].metric("Episode steps", episode_indices.size)

    with st.expander("Evaluation- und Datenstatus"):
        st.write(f"**Track-Geometrie:** `{geometry_artifact or 'nicht im Parent-Run gefunden'}`")
        st.write(f"**Referenz:** `{reference_artifact or 'nicht im Parent-Run gefunden'}`")
        if summary:
            st.write("**summary.json**")
            st.json(summary)
        if episode_rows:
            st.write("**episodes.csv**")
            st.dataframe(pd.DataFrame(episode_rows), hide_index=True)
        st.write("**Arrays in trajectories.npz**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Name": name, "Shape": " × ".join(map(str, value.shape)), "Dtype": str(value.dtype)}
                    for name, value in sorted(arrays.items())
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.write("**Evaluation-Artefakte**")
        st.code("\n".join(evaluation_artifacts))
        st.write("**Parent-Training-Artefakte**")
        st.code("\n".join(training_artifacts))


if __name__ == "__main__":
    main()
