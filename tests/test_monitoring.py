from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from assetto_corsa_monitoring import factory
from assetto_corsa_monitoring.config import MonitoringConfig
from assetto_corsa_monitoring.mlflow_tracker import MLflowTracker
from assetto_corsa_monitoring.null_tracker import NullTracker


def _fake_mlflow() -> SimpleNamespace:
    return SimpleNamespace(
        set_tracking_uri=lambda _uri: None,
        set_experiment=lambda *args, **kwargs: None,
    )


def test_monitoring_defaults_to_local_fail_open_mlflow_store() -> None:
    config = MonitoringConfig()
    assert config.tracking_uri == "sqlite:///mlflow.db"
    assert config.artifact_root == "mlartifacts"
    assert config.fail_on_error is False


def test_factory_passes_local_artifact_root_and_prepares_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []

    class _Tracker:
        def __init__(self, **kwargs: object) -> None:
            events.append(("created", kwargs))

        def prepare(self) -> bool:
            events.append(("prepared", True))
            return True

    monkeypatch.setattr(factory, "MLflowTracker", _Tracker)
    tracker = factory.create_tracker(MonitoringConfig(enabled=True, backend="mlflow"))

    assert isinstance(tracker, _Tracker)
    assert events[0][0] == "created"
    created_arguments = events[0][1]
    assert isinstance(created_arguments, dict)
    assert created_arguments["artifact_root"] == "mlartifacts"
    assert events[1] == ("prepared", True)


def test_factory_falls_back_to_null_tracker_when_mlflow_prepare_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Tracker:
        def __init__(self, **_kwargs: object) -> None:
            pass

        @staticmethod
        def prepare() -> bool:
            return False

    monkeypatch.setattr(factory, "MLflowTracker", _Tracker)
    tracker = factory.create_tracker(
        MonitoringConfig(enabled=True, backend="mlflow", fail_on_error=False)
    )

    assert isinstance(tracker, NullTracker)


def test_local_sqlite_experiment_is_created_with_file_artifact_uri(tmp_path: Path) -> None:
    created: dict[str, str] = {}
    activated: dict[str, str] = {}
    client = SimpleNamespace(
        get_experiment_by_name=lambda _name: None,
        create_experiment=lambda name, *, artifact_location: (
            created.update(name=name, artifact_location=artifact_location) or "7"
        ),
    )
    mlflow = _fake_mlflow()
    mlflow.set_experiment = lambda *, experiment_id: activated.update(experiment_id=experiment_id)
    tracker = MLflowTracker(
        tracking_uri="sqlite:///mlflow.db",
        artifact_root=tmp_path / "artifacts",
        experiment_name="test",
    )
    tracker._mlflow = mlflow  # pyright: ignore[reportPrivateUsage]
    tracker._client = client  # pyright: ignore[reportPrivateUsage]

    assert tracker.prepare() is True
    assert created == {
        "name": "test",
        "artifact_location": (tmp_path / "artifacts").resolve().as_uri(),
    }
    assert activated == {"experiment_id": "7"}


def test_local_sqlite_migrates_legacy_mlflow_artifacts_experiment(tmp_path: Path) -> None:
    old = SimpleNamespace(experiment_id="1", artifact_location="mlflow-artifacts:/1")
    experiments = {"test": old}
    renamed: list[tuple[str, str]] = []
    created: dict[str, str] = {}

    def get_experiment(name: str) -> SimpleNamespace | None:
        return experiments.get(name)

    def rename_experiment(experiment_id: str, new_name: str) -> None:
        renamed.append((experiment_id, new_name))
        experiments.pop("test", None)
        experiments[new_name] = old

    def create_experiment(name: str, *, artifact_location: str) -> str:
        created.update(name=name, artifact_location=artifact_location)
        return "2"

    client = SimpleNamespace(
        get_experiment_by_name=get_experiment,
        rename_experiment=rename_experiment,
        create_experiment=create_experiment,
    )
    mlflow = _fake_mlflow()
    tracker = MLflowTracker(
        tracking_uri="sqlite:///mlflow.db",
        artifact_root=tmp_path / "artifacts",
        experiment_name="test",
    )
    tracker._mlflow = mlflow  # pyright: ignore[reportPrivateUsage]
    tracker._client = client  # pyright: ignore[reportPrivateUsage]

    assert tracker.prepare() is True
    assert renamed == [("1", "test [legacy artifacts 1]")]
    assert created["artifact_location"] == (tmp_path / "artifacts").resolve().as_uri()


def test_existing_matching_local_experiment_is_reused(tmp_path: Path) -> None:
    artifact_uri = (tmp_path / "artifacts").resolve().as_uri()
    existing = SimpleNamespace(experiment_id="4", artifact_location=artifact_uri)
    client = SimpleNamespace(get_experiment_by_name=lambda _name: existing)
    activated: dict[str, str] = {}
    mlflow = _fake_mlflow()
    mlflow.set_experiment = lambda *, experiment_id: activated.update(experiment_id=experiment_id)
    tracker = MLflowTracker(
        tracking_uri="sqlite:///mlflow.db",
        artifact_root=tmp_path / "artifacts",
        experiment_name="test",
    )
    tracker._mlflow = mlflow  # pyright: ignore[reportPrivateUsage]
    tracker._client = client  # pyright: ignore[reportPrivateUsage]

    assert tracker.prepare() is True
    assert activated == {"experiment_id": "4"}


def test_mlflow_prepare_disables_nonfatal_monitoring_on_backend_error() -> None:
    tracker = MLflowTracker(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="test",
        fail_on_error=False,
    )
    tracker._preflight_remote_tracking_uri = lambda: None  # type: ignore[method-assign]
    mlflow = _fake_mlflow()

    def fail_set_experiment(_name: str) -> None:
        raise RuntimeError("boom")

    mlflow.set_experiment = fail_set_experiment
    tracker._mlflow = mlflow  # pyright: ignore[reportPrivateUsage]

    assert tracker.prepare() is False
    assert tracker.active is False
    assert tracker.start_run() is None


def test_mlflow_prepare_raises_in_explicit_strict_mode() -> None:
    tracker = MLflowTracker(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="test",
        fail_on_error=True,
    )
    tracker._preflight_remote_tracking_uri = lambda: None  # type: ignore[method-assign]
    mlflow = _fake_mlflow()

    def fail_set_experiment(_name: str) -> None:
        raise RuntimeError("boom")

    mlflow.set_experiment = fail_set_experiment
    tracker._mlflow = mlflow  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(RuntimeError, match="disabled"):
        tracker.prepare()
