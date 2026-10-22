"""Read-only checkpoint following and deterministic inference in the monitor."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import pickle
import re
import time
from typing import Any, cast
from zipfile import BadZipFile

import numpy as np
from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.vec_env import VecNormalize
import torch

from tools.diagnostics.frame import Array
from assetto_corsa_ai.racing_policy import RacingActorCriticPolicy
from assetto_corsa_ai.racing_ppo import RacingPPO
from assetto_corsa_ai.training_config import TrainingConfig


@dataclass(frozen=True)
class ModelInference:
    normalized: Array
    policy_mean: Array
    trajectory: Array | None


@dataclass(frozen=True)
class ModelSnapshot:
    directory: Path
    config: TrainingConfig
    model: RacingPPO
    normalizer: VecNormalize
    run_id: str | None
    step: int
    updates: int
    created_at: float

    def infer(self, raw: Array) -> ModelInference:
        normalized = self.normalizer.normalize_obs(np.asarray(raw, dtype=np.float32).copy())
        if not isinstance(normalized, np.ndarray) or not np.all(np.isfinite(normalized)):
            raise ValueError("Normalized observation is invalid.")
        policy = self.model.policy
        tensor, _ = policy.obs_to_tensor(normalized)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Checkpoint requires a flat tensor observation.")
        with torch.no_grad():
            distribution = policy.get_distribution(tensor)
            if not isinstance(distribution, (DiagGaussianDistribution, StateDependentNoiseDistribution)):
                raise ValueError("Diagnostics requires a continuous Gaussian policy.")
            mean = np.asarray(distribution.distribution.mean[0].cpu().numpy(), dtype=np.float64).copy()
            prediction = None
            if isinstance(policy, RacingActorCriticPolicy) and policy.prediction_points:
                prediction = np.asarray(policy.prediction_for_reward(tensor)[0].cpu().numpy(), dtype=np.float64).copy()
        return ModelInference(np.asarray(normalized, dtype=np.float64), mean, prediction)


class CheckpointFollower:
    """Poll once per second, retaining the last complete version on read failure."""
    def __init__(self, run_directory: Path | None = None, runs_directory: Path = Path("runs")) -> None:
        self.run_directory = run_directory
        self.runs_directory = runs_directory
        self.snapshot: ModelSnapshot | None = None
        self.config: TrainingConfig | None = None
        self.run_id: str | None = None
        self.status = "No model checkpoint loaded"
        self._selected: Path | None = None
        self._next_poll = 0.0

    def select_run(self, directory: Path) -> None:
        self.run_directory = directory
        self._next_poll = 0.0

    def _run(self) -> Path | None:
        if self.run_directory is not None:
            return self.run_directory
        if not self.runs_directory.is_dir():
            return None
        directories = sorted((p for p in self.runs_directory.iterdir()
                              if p.is_dir() and (p / "config.toml").is_file()), key=lambda p: p.name, reverse=True)
        return directories[0] if directories else None

    @staticmethod
    def _pair(run: Path) -> tuple[Path, dict[str, Any]] | None:
        pointer = run / "latest" / "current.json"
        if pointer.is_file():
            decoded: object = json.loads(pointer.read_text(encoding="utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("Invalid latest snapshot manifest.")
            data = cast(dict[str, Any], decoded)
            name = data.get("directory")
            if not isinstance(name, str) or not re.fullmatch(r"snapshot_[0-9a-f]{32}", name):
                raise ValueError("Invalid latest snapshot reference.")
            directory = run / "latest" / name
            if not directory.resolve().is_relative_to((run / "latest").resolve()):
                raise ValueError("Latest snapshot escapes the run directory.")
            return directory, data
        # Existing runs remain useful without a diagnostics cache.
        candidates = [run / "final", *sorted((run / "checkpoints").glob("step_*"), reverse=True)]
        for directory in candidates:
            if (directory / "model.zip").is_file() and (directory / "vecnormalize.pkl").is_file():
                return directory, {}
        return None

    def poll(self, now: float | None = None) -> ModelSnapshot | None:
        now = time.time() if now is None else now
        if now < self._next_poll:
            return self.snapshot
        self._next_poll = now + 1.0
        try:
            run = self._run()
            if run is None:
                self.status = "No model checkpoint loaded"
                return self.snapshot
            if run != self._selected:
                config = TrainingConfig.load(run / "config.toml")
                metadata_path = run / "metadata.json"
                decoded: object = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
                if not isinstance(decoded, dict):
                    raise ValueError("Invalid run metadata.")
                metadata = cast(dict[str, Any], decoded)
                if metadata.get("run_id") is not None and not isinstance(metadata["run_id"], str):
                    raise ValueError("Invalid run ID.")
                self._selected, self.config, self.run_id = run, config, metadata.get("run_id")
                self.snapshot = None
            pair = self._pair(run)
            if pair is None:
                self.status = f"Waiting for a complete snapshot: {run.name}"
                return self.snapshot
            directory, provenance = pair
            if self.snapshot is not None and self.snapshot.directory == directory:
                return self.snapshot
            assert self.config is not None
            # VecNormalize's pickle excludes its environment. normalize_obs()
            # works without attaching any simulator or DummyVecEnv.
            with (directory / "vecnormalize.pkl").open("rb") as stream:
                normalizer: object = pickle.load(stream)
            if not isinstance(normalizer, VecNormalize):
                raise ValueError("Snapshot contains no VecNormalize statistics.")
            normalizer.training, normalizer.norm_reward = False, False
            model = RacingPPO.load(directory / "model.zip", device="cpu")  # pyright: ignore[reportUnknownMemberType]
            model.policy.set_training_mode(False)
            if isinstance(model.policy, RacingActorCriticPolicy):
                model.policy.configure_reward_capture(False)
            expected = 10 + 4 * len(self.config.environment.lookahead_distances) + 2 * len(self.config.environment.reference_distances)
            if model.observation_space.shape != (expected,) or normalizer.observation_space.shape != (expected,):  # pyright: ignore[reportUnknownMemberType]
                raise ValueError("Checkpoint observation space does not match config.toml.")
            step = int(provenance.get("step", model.num_timesteps))
            updates = int(provenance.get("policy_updates", model.completed_policy_updates))
            if step != model.num_timesteps or updates != model.completed_policy_updates:
                raise ValueError("Snapshot manifest does not match the model version.")
            created = (datetime.fromisoformat(provenance["created_at"]).timestamp() if "created_at" in provenance
                       else (directory / "model.zip").stat().st_mtime)
            self.snapshot = ModelSnapshot(directory, self.config, model, normalizer, self.run_id, step, updates, created)
            self.status = f"Checkpoint inference: {run.name} · update {updates} · step {step}"
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError, BadZipFile) as error:
            self.status = f"Snapshot unavailable (will retry): {error}"
        return self.snapshot
