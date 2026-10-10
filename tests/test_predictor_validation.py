"""Test-data validation, detached inference, plots and selected NPZ loading."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from assetto_corsa_ai.main import create_agent
from tools import validate_predictor as validation
from test_racing_policy import small_config


def test_validator_reports_masked_geometry_and_errors(capsys: pytest.CaptureFixture[str]) -> None:
    config = small_config()
    data = validation.synthetic_rollout(config)
    agent = create_agent(validation.ValidationEnvironment(data), config)
    try:
        first = validation.validate_sample(agent, data, config, 0)
        assert first.prediction.shape == first.actual.shape == (10, 2)
        assert first.mask.all()
        assert np.isfinite(first.loss)
        np.testing.assert_allclose(first.forward_error, np.abs(first.prediction[:, 0] - first.actual[:, 0]))
        last = validation.validate_sample(agent, data, config, 30)
        assert last.mask.tolist() == [True] + [False] * 9
        validation.print_result(last, config.predictor.horizons_seconds)
        output = capsys.readouterr().out
        assert "predicted trajectory" in output and "actual trajectory" in output
        assert "forward error" in output and "lateral error" in output
        assert "masked" in output and "Total predictor loss" in output
        with pytest.raises(ValueError, match="Sample index"):
            validation.validate_sample(agent, data, config, 64)
    finally:
        agent.close()


def test_npz_load_uses_raw_observations_and_no_pickled_objects(tmp_path: Path) -> None:
    config = small_config()
    data = validation.synthetic_rollout(config)
    path = tmp_path / "selected.npz"
    np.savez(path, observations=data.observations, vehicle_positions=data.positions,
             vehicle_forwards=data.forwards, next_vehicle_positions=data.next_positions,
             episode_ends=data.episode_ends)
    loaded = validation.load_rollout(path)
    loaded.validate(data.observations.shape[-1])
    np.testing.assert_array_equal(loaded.observations, data.observations)
    np.testing.assert_array_equal(loaded.episode_ends, data.episode_ends)
    unsafe = tmp_path / "unsafe.npz"
    np.savez(unsafe, observations=np.array([object()], dtype=object),
             vehicle_positions=data.positions, vehicle_forwards=data.forwards,
             next_vehicle_positions=data.next_positions, episode_ends=data.episode_ends)
    with pytest.raises(ValueError, match="Object arrays"):
        validation.load_rollout(unsafe)


def test_predictor_plot_can_be_exported_without_a_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib
    matplotlib.use("Agg")
    config = small_config()
    data = validation.synthetic_rollout(config)
    agent = create_agent(validation.ValidationEnvironment(data), config)
    try:
        result = validation.validate_sample(agent, data, config, 0)
        destination = tmp_path / "trajectory.png"
        validation.plot_result(result, destination=destination, show=False)
        assert destination.read_bytes().startswith(b"\x89PNG")
    finally:
        agent.close()


def test_default_cli_is_explicitly_untrained_and_never_launches_ac(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    manager = Mock(side_effect=AssertionError("Validator must not launch AC"))
    monkeypatch.setattr("assetto_corsa_ai.main.AssettoCorsaSessionManager", manager)
    monkeypatch.setattr(validation.TrainingConfig, "load", Mock(return_value=small_config()))
    validation.main([])
    output = capsys.readouterr().out
    assert "UNTRAINED" in output
    assert "Synthetic test data" in output
    assert "Total predictor loss" in output
    manager.assert_not_called()
