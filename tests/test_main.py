"""Exercise application wiring and real PPO lifecycle without Assetto Corsa."""

from pathlib import Path
from dataclasses import replace
from unittest.mock import Mock

import pytest

from assetto_corsa_ai import main as application
from assetto_corsa_ai.static_info import SimulatorStaticInfo
from assetto_corsa_ai.training_config import TrainingConfig
from test_agent import MemoryRacingEnvironment
from config_helpers import replace_setting


def test_layout_does_not_fall_back_to_other_geometry(tmp_path: Path) -> None:
    default = tmp_path / "content/tracks/test/ai/fast_lane.ai"
    default.parent.mkdir(parents=True)
    default.touch()
    info = SimulatorStaticInfo("car", "test", "short", 8000, 80, (0.3, 0.3, 0.3, 0.3), 1000)
    with pytest.raises(FileNotFoundError, match="short"):
        application.track_ai_path(info, tmp_path)
    layout = default.parent.parent / "short/ai/fast_lane.ai"
    layout.parent.mkdir(parents=True)
    layout.touch()
    assert application.track_ai_path(info, tmp_path) == layout


@pytest.mark.parametrize("fails", [False, True])
def test_session_reader_always_releases_mapping(monkeypatch: pytest.MonkeyPatch, fails: bool) -> None:
    reader = Mock()
    if fails:
        reader.read.side_effect = OSError("not ready")
    monkeypatch.setattr(application, "StaticInfoReader", Mock(return_value=reader))
    if fails:
        with pytest.raises(OSError):
            application.SessionStaticInfoReader().read()
    else:
        assert application.SessionStaticInfoReader().read() is reader.read.return_value
    reader.close.assert_called_once()


def test_axle_estimate_is_rotation_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = Mock()
    reader.read.return_value.car_coordinates = (10, 2, 20)
    reader.read.return_value.tyre_contact_points = (
        (11.2, 2, 19.5), (11.2, 2, 20.5), (8.2, 2, 19.5), (8.2, 2, 20.5),
    )
    monkeypatch.setattr(application, "TelemetryReader", Mock(return_value=reader))
    assert application.estimate_axle_distances() == pytest.approx((1.2, 1.8))
    reader.close.assert_called_once()


def test_training_checkpoint_resume_and_drive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import csv
    import json

    template = (Path(__file__).resolve().parents[1] / "configs/training.toml").read_text()
    content = replace_setting(template, "device", 'device = "cpu"')
    content = replace_setting(content, "verbose", "verbose = 0")
    content = replace_setting(content, "enabled", "enabled = false", section="reward.future_prediction")
    for before, after in (
        ("n_steps = 1024", "n_steps = 4"), ("batch_size = 256", "batch_size = 2"),
        ("n_epochs = 10", "n_epochs = 1"), ("total_timesteps = 4096", "total_timesteps = 4"),
        ("checkpoint_interval_updates = 4", "checkpoint_interval_updates = 1"),
        ("feature_layers = [512, 512]", "feature_layers = [8]"),
        ("feature_dim = 512", "feature_dim = 8"),
        ("policy_layers = [128, 128]", "policy_layers = [8]"),
        ("value_layers = [128, 128]", "value_layers = [8]"),
    ):
        content = replace_setting(content, before.split("=", 1)[0].strip(), after)
    config_file = tmp_path / "training.toml"
    config_file.write_text(content)
    common = ["--config", str(config_file), "--runs-directory", str(tmp_path / "runs")]
    environment = MemoryRacingEnvironment()
    application.run_agent(environment, application.parse_arguments([
        "train", *common,
    ]))
    assert environment.close_calls == 1
    first = next((tmp_path / "runs").iterdir())
    assert {path.name for path in first.iterdir()} == {
        "config.toml", "metadata.json", "metrics.csv", "checkpoints", "final",
    }
    checkpoint = first / "checkpoints/step_000004"
    assert (checkpoint / "model.zip").is_file()
    assert (checkpoint / "vecnormalize.pkl").is_file()
    assert not (checkpoint / "ppo_config.json").exists()
    assert (first / "config.toml").read_text() == content
    assert json.loads((first / "metadata.json").read_text())["git_commit"]
    with (first / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert any(row["episode_reward"] == "22.5" for row in rows)
    assert any(row["policy_loss"] and row["value_loss"] and row["learning_rate"] for row in rows)
    assert any(row["predictor_loss"] and int(row["predictor_valid_samples"]) > 0 for row in rows)
    snapshot = {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}
    config_file.write_text(content.replace('resume_from = ""', f'resume_from = "{checkpoint.as_posix()}"'))
    resumed = MemoryRacingEnvironment()
    application.run_agent(resumed, application.parse_arguments([
        "train", *common,
    ]))
    assert resumed.step_calls == 4
    assert resumed.close_calls == 1
    second = next(path for path in (tmp_path / "runs").iterdir() if path != first)
    assert (second / "checkpoints/step_000008/model.zip").is_file()
    assert (second / "final/vecnormalize.pkl").is_file()
    assert TrainingConfig.load(second / "config.toml").training.resume_from == checkpoint.as_posix()
    assert snapshot == {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}
    driving = MemoryRacingEnvironment()
    application.run_agent(driving, application.parse_arguments([
        "drive", "--model-directory", str(second / "final"), "--episodes", "2",
    ]))
    assert driving.step_calls == 6
    assert driving.close_calls == 1


def test_missing_model_closes_environment(tmp_path: Path) -> None:
    environment = MemoryRacingEnvironment()
    with pytest.raises((FileNotFoundError, ValueError)):
        application.run_agent(environment, application.parse_arguments([
            "drive", "--model-directory", str(tmp_path),
        ]))
    assert environment.close_calls == 1


@pytest.mark.parametrize("time_limit,step_limit", [(0.0, 0), (120.0, 0), (0.0, 5)])
def test_training_configuration_is_not_overridden_by_environment_variables(
    monkeypatch: pytest.MonkeyPatch, time_limit: float, step_limit: int,
) -> None:
    config = TrainingConfig.load(Path(__file__).resolve().parents[1] / "configs/training.toml")
    config = replace(config, environment=replace(
        config.environment, front_axle_distance=1.2, rear_axle_distance=1.8,
        max_episode_time=time_limit, max_episode_steps=step_limit,
        control_interval=0.1, lookahead_distances=(10.0, 20.0), reference_distances=(5.0,),
    ))
    monkeypatch.setenv("FRONT_AXLE_DISTANCE", "9.9")
    monkeypatch.setenv("REAR_AXLE_DISTANCE", "9.9")
    constructor = Mock()
    monkeypatch.setattr(application, "RacingEnvironment", constructor)
    monkeypatch.setattr(application, "AssettoCorsaSessionManager", Mock())
    monkeypatch.setattr(application, "AssettoCorsaSimulatorController", Mock())
    dynamics = Mock()
    monkeypatch.setattr(application, "TyreDynamicsCalculator", dynamics)
    monkeypatch.setattr(application, "ObservationBuilder", Mock())
    estimator = Mock(side_effect=AssertionError("Explicit offsets need no live estimate"))
    monkeypatch.setattr(application, "estimate_axle_distances", estimator)
    application.create_training_environment(config)
    options = constructor.call_args.kwargs
    options["observation_builder_factory"](Mock(), Mock())
    dynamics.assert_called_once_with(1.2, 1.8, minimum_speed=config.environment.minimum_speed)
    assert options["control_interval"] == 0.1
    assert options["max_episode_time"] == (time_limit or None)
    assert options["max_episode_steps"] == (step_limit or None)
    assert options["observation_size"] == 20
    assert options["seed"] == config.training.seed
    estimator.assert_not_called()


def test_training_refuses_cli_hyperparameter_overrides(tmp_path: Path) -> None:
    config_file = tmp_path / "training.toml"
    config_file.write_text((Path(__file__).resolve().parents[1] / "configs/training.toml").read_text())
    with pytest.raises(SystemExit):
        application.parse_arguments(["train", "--config", str(config_file), "--seed", "23"])


@pytest.mark.parametrize("arguments", [
    ["train", "--timesteps", "0"], ["drive", "--resume"],
    ["reset", "--front-axle-distance", "nan"],
])
def test_invalid_cli_arguments(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        application.parse_arguments(arguments)
