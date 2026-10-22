"""Independent checkpoint inference and publication, without AC or Qt."""

from collections.abc import Iterator
from dataclasses import replace
import json
from pathlib import Path
import re
from threading import Event

import numpy as np
import pytest
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.running_mean_std import RunningMeanStd
import torch

from assetto_corsa_ai.agent import PPOAgent
from tools.diagnostics.model import CheckpointFollower
from assetto_corsa_ai.main import create_agent
from assetto_corsa_ai.racing_ppo import RacingPPO
from assetto_corsa_ai.training_callbacks import TrainingRunCallback
from assetto_corsa_ai.training_config import TrainingConfig
from assetto_corsa_ai.training_run import ModelSaver, NormalizerSaver, TrainingRun, create_run_directory
from assetto_corsa_ai.training_snapshots import LatestSnapshotWriter
from test_training_run import FakeModel, FakeNormalizer
from tools.validation.validate_predictor import ValidationEnvironment, synthetic_rollout


def diagnostics_config() -> TrainingConfig:
    text = Path("configs/training.toml").read_text(encoding="utf-8")
    settings = {
        "network": {"feature_layers": "[7, 9]", "feature_dim": "11", "policy_layers": "[6]", "value_layers": "[5]"},
        "predictor": {"layers": "[3, 4]"},
        "ppo": {"n_steps": "32", "batch_size": "16", "n_epochs": "1"},
        "reward.future_prediction": {"enabled": "false"},
        "training": {"device": '"cpu"', "verbose": "0"},
    }
    for table, values in settings.items():
        start = text.index(f"[{table}]\n")
        end = text.find("\n[", start + 1)
        end = len(text) if end < 0 else end
        section = text[start:end]
        for key, value in values.items():
            section = re.sub(rf"(?m)^{key} = [^\n]+", f"{key} = {value}", section)
        text = text[:start] + section + text[end:]
    return TrainingConfig.from_toml(text)


@pytest.fixture
def model_run(tmp_path: Path) -> Iterator[tuple[TrainingConfig, PPOAgent, TrainingRun]]:
    config = diagnostics_config()
    run = create_run_directory(tmp_path / "runs", track=config.environment.track, car=config.environment.car)
    run.save_config(config.toml_text)
    agent = create_agent(ValidationEnvironment(synthetic_rollout(config)), config)
    agent.vec_env.reset()
    try:
        yield config, agent, run
    finally:
        agent.close()


def test_latest_publishes_only_complete_pairs_and_keeps_two_versions(tmp_path: Path) -> None:
    run = create_run_directory(tmp_path, track="track", car="car")
    checkpoint = run.save_checkpoint(8, FakeModel(), FakeNormalizer())
    versions = [run.save_latest(step, step // 4, FakeModel(), FakeNormalizer()) for step in (4, 8, 12)]
    pointer = json.loads((run.directory / "latest/current.json").read_text())
    assert pointer["directory"] == versions[-1].name
    assert pointer["step"] == 12 and pointer["policy_updates"] == 3
    assert not versions[0].exists()
    for directory in versions[1:]:
        assert {p.name for p in directory.iterdir()} == {"model.zip", "vecnormalize.pkl"}
    assert (checkpoint / "model.zip").read_bytes() == b"model"
    assert not list((run.directory / "latest").glob("*.tmp"))


def test_failed_pair_does_not_replace_the_last_complete_version(tmp_path: Path) -> None:
    run = create_run_directory(tmp_path, track="track", car="car")
    target = run.save_latest(4, 1, FakeModel(), FakeNormalizer())
    before = (run.directory / "latest/current.json").read_bytes()

    class FailingNormalizer:
        def save(self, path: str) -> None:
            Path(path).write_bytes(b"incomplete")
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        run.save_latest(8, 2, FakeModel(), FailingNormalizer())
    assert (run.directory / "latest/current.json").read_bytes() == before
    assert list((run.directory / "latest").glob("snapshot_*")) == [target]


def test_async_writer_freezes_both_objects_before_training_mutates_them(
    model_run: tuple[TrainingConfig, PPOAgent, TrainingRun], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, agent, run = model_run
    entered, release = Event(), Event()
    save_latest = TrainingRun.save_latest

    def slow_save(self: TrainingRun, step: int, updates: int, model: ModelSaver, normalizer: NormalizerSaver) -> Path:
        entered.set()
        assert release.wait(5.0)
        return save_latest(self, step, updates, model, normalizer)

    monkeypatch.setattr(TrainingRun, "save_latest", slow_save)
    parameters = {name: value.detach().clone() for name, value in agent.model.policy.state_dict().items()}
    rms = agent.vec_env.obs_rms
    assert isinstance(rms, RunningMeanStd)
    mean = rms.mean.copy()
    writer = LatestSnapshotWriter(run)
    try:
        writer.submit(agent.model, agent.vec_env, 0)
        assert entered.wait(5.0)  # Disk writing is blocked, submit has returned.
        with torch.no_grad():
            next(agent.model.policy.parameters()).add_(10.0)
        rms.mean += 10.0
    finally:
        release.set()
        writer.close()
    assert writer.error is None
    loaded = CheckpointFollower(run.directory).poll(100.0)
    assert loaded is not None
    for name, expected in parameters.items():
        torch.testing.assert_close(loaded.model.policy.state_dict()[name], expected, rtol=0, atol=0)
    loaded_rms = loaded.normalizer.obs_rms
    assert isinstance(loaded_rms, RunningMeanStd)
    np.testing.assert_array_equal(loaded_rms.mean, mean)


def test_disk_failure_is_passive_and_does_not_abort_training(
    model_run: tuple[TrainingConfig, PPOAgent, TrainingRun], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, agent, run = model_run

    def unavailable(self: TrainingRun, step: int, updates: int, model: ModelSaver, normalizer: NormalizerSaver) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(TrainingRun, "save_latest", unavailable)
    callback = TrainingRunCallback(run, 0, publish_latest=True)
    agent.learn(64, callback=callback)
    callback.close()  # Idempotent, including the CLI's final cleanup.
    assert isinstance(agent.model, RacingPPO) and agent.model.num_timesteps == 64
    assert not (run.directory / "latest/current.json").exists()


def test_publication_preserves_actions_weights_normalization_and_torch_rng(tmp_path: Path) -> None:
    config = diagnostics_config()
    results: list[tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, np.ndarray, float, torch.Tensor]] = []
    for enabled in (False, True):
        run = create_run_directory(tmp_path, track="track", car="car")
        agent = create_agent(ValidationEnvironment(synthetic_rollout(config)), config)
        try:
            agent.learn(64, callback=TrainingRunCallback(run, 0, publish_latest=enabled))
            rms = agent.vec_env.obs_rms
            assert isinstance(rms, RunningMeanStd)
            results.append((
                {name: value.detach().clone() for name, value in agent.model.policy.state_dict().items()},
                agent.model.rollout_buffer.actions.copy(), rms.mean.copy(), rms.var.copy(), rms.count,
                torch.get_rng_state().clone(),
            ))
            if enabled:
                pointer = json.loads((run.directory / "latest/current.json").read_text())
                assert pointer["step"] == 64 and pointer["policy_updates"] == 2
            else:
                assert not (run.directory / "latest").exists()
        finally:
            agent.close()
    first, second = results
    for name, expected in first[0].items():
        torch.testing.assert_close(second[0][name], expected, rtol=0, atol=0)
    for index in (1, 2, 3):
        np.testing.assert_array_equal(first[index], second[index])
    assert first[4] == second[4]
    torch.testing.assert_close(first[5], second[5], rtol=0, atol=0)


def test_local_inference_matches_training_snapshot_without_sampling_or_stat_updates(
    model_run: tuple[TrainingConfig, PPOAgent, TrainingRun], monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, agent, run = model_run
    run.save_latest(0, 0, agent.model, agent.vec_env)
    snapshot = CheckpointFollower(run.directory).poll(100.0)
    assert snapshot is not None
    raw = np.asarray(synthetic_rollout(config).observations[0, 0], dtype=np.float64)
    raw[-2:] = (0.25, 0.75)
    expected_norm = agent.vec_env.normalize_obs(raw.astype(np.float32))
    assert isinstance(expected_norm, np.ndarray)
    tensor, _ = agent.model.policy.obs_to_tensor(expected_norm)
    assert isinstance(tensor, torch.Tensor)
    with torch.no_grad():
        distribution = agent.model.policy.get_distribution(tensor)
        assert isinstance(distribution, DiagGaussianDistribution)
        expected_mean = distribution.distribution.mean[0].numpy()
    expected_prediction = agent.predict_trajectory(raw.astype(np.float32))

    def forbid_sampling(self: DiagGaussianDistribution) -> torch.Tensor:
        raise AssertionError("Diagnostics must not resample the training action")

    monkeypatch.setattr(DiagGaussianDistribution, "sample", forbid_sampling)
    rms = snapshot.normalizer.obs_rms
    assert isinstance(rms, RunningMeanStd)
    mean, variance, count = rms.mean.copy(), rms.var.copy(), rms.count
    parameters = {name: value.detach().clone() for name, value in snapshot.model.policy.state_dict().items()}
    rng = torch.get_rng_state().clone()
    for _ in range(3):
        result = snapshot.infer(raw)
        np.testing.assert_array_equal(result.normalized, expected_norm)
        np.testing.assert_array_equal(result.normalized[-2:], raw[-2:])
        np.testing.assert_allclose(result.policy_mean, expected_mean, rtol=0, atol=0)
        assert result.trajectory is not None
        np.testing.assert_allclose(result.trajectory, expected_prediction, rtol=0, atol=0)
    np.testing.assert_array_equal(rms.mean, mean)
    np.testing.assert_array_equal(rms.var, variance)
    assert rms.count == count and not snapshot.normalizer.training
    assert snapshot.normalizer.venv is None  # No AC or even a dummy environment.
    for name, expected in parameters.items():
        torch.testing.assert_close(snapshot.model.policy.state_dict()[name], expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("broken", ["incomplete", "corrupt_zip", "bad_manifest", "mismatched_version"])
def test_follower_keeps_last_good_pair_and_retries(
    model_run: tuple[TrainingConfig, PPOAgent, TrainingRun], broken: str,
) -> None:
    _, agent, run = model_run
    run.save_latest(0, 0, agent.model, agent.vec_env)
    follower = CheckpointFollower(run.directory)
    first = follower.poll(100.0)
    assert first is not None
    next_version = run.save_latest(0, 0, agent.model, agent.vec_env)
    if broken == "incomplete":
        (next_version / "vecnormalize.pkl").unlink()
    elif broken == "corrupt_zip":
        (next_version / "model.zip").write_bytes(b"invalid zip")
    elif broken == "bad_manifest":
        (run.directory / "latest/current.json").write_text("[]")
    else:
        pointer = json.loads((run.directory / "latest/current.json").read_text())
        pointer["step"] = 99
        (run.directory / "latest/current.json").write_text(json.dumps(pointer))
    assert follower.poll(100.5) is first  # Once per second, not each frame.
    assert follower.poll(101.0) is first
    assert "will retry" in follower.status
    final_version = run.save_latest(0, 0, agent.model, agent.vec_env)
    final = follower.poll(102.0)
    assert final is not None and final.directory == final_version
    assert final is not first


def test_run_selection_and_archived_checkpoint_fallback(
    model_run: tuple[TrainingConfig, PPOAgent, TrainingRun],
) -> None:
    config, agent, first = model_run
    checkpoint = first.save_checkpoint(0, agent.model, agent.vec_env)
    follower = CheckpointFollower(first.directory)
    loaded = follower.poll(100.0)
    assert loaded is not None and loaded.directory == checkpoint
    final = first.save_final_model(agent.model, agent.vec_env)
    loaded = follower.poll(101.0)
    assert loaded is not None and loaded.directory == final
    # Give the newest run a later timestamp regardless of random IDs.
    second = create_run_directory(first.directory.parent, track="track", car="car")
    directory = second.directory.with_name("20990101_000000_track_car_abcdef")
    second.directory.rename(directory)
    second = replace(second, directory=directory)
    second.save_config(config.toml_text)
    pair = second.save_latest(0, 0, agent.model, agent.vec_env)
    automatic = CheckpointFollower(runs_directory=first.directory.parent).poll(100.0)
    assert automatic is not None and automatic.directory == pair
    follower.select_run(second.directory)
    selected = follower.poll(101.1)
    assert selected is not None and selected.directory == pair


def test_legacy_udp_config_remains_loadable_without_network_settings() -> None:
    config = diagnostics_config()
    old = config.toml_text.replace("[diagnostics]\nenabled = true", '[diagnostics]\nhost = "127.0.0.1"\nport = 9876\npublish_rate_hz = 10.0\nenabled = true')
    loaded = TrainingConfig.from_toml(old)
    assert loaded.diagnostics.enabled
    assert not hasattr(loaded.diagnostics, "port")
