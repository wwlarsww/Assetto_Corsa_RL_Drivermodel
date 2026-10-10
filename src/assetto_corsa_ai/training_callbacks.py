"""SB3 integration for run metrics and snapshots."""

from collections.abc import Mapping
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter
from stable_baselines3.common.vec_env import VecNormalize

from assetto_corsa_ai.training_run import MetricValue, TrainingRun


class RunMetricsWriter(KVWriter):
    """Keep actual PPO updates separate from per-episode event rows."""

    def __init__(self, run: TrainingRun) -> None:
        self.run = run

    def write(
        self, key_values: Mapping[str, Any],
        key_excluded: Mapping[str, tuple[str, ...]], step: int = 0,
    ) -> None:
        del key_excluded
        names = {
            "policy_loss": "train/policy_gradient_loss",
            "value_loss": "train/value_loss",
            "entropy_loss": "train/entropy_loss",
            "predictor_loss": "train/predictor_loss",
            "predictor_valid_samples": "train/predictor_valid_samples",
            "predictor_forward_mae": "train/predictor_forward_mae",
            "predictor_lateral_mae": "train/predictor_lateral_mae",
            "learning_rate": "train/learning_rate",
        }
        if not any(name in key_values for name in names.values()):
            return
        row: dict[str, MetricValue] = {"step": step}
        for column, name in names.items():
            if name in key_values:
                row[column] = int(key_values[name]) if column == "predictor_valid_samples" else float(key_values[name])
        self.run.append_metrics(row)

    def close(self) -> None:
        pass


class TrainingRunCallback(BaseCallback):
    """Record raw Monitor episode rewards and save matching model/statistics."""

    def __init__(self, run: TrainingRun, interval: int) -> None:
        super().__init__()
        self.run = run
        self.interval = interval
        self.episodes = 0
        self._last_checkpoint_step = 0

    def _on_training_start(self) -> None:
        self._last_checkpoint_step = self.num_timesteps

    def _on_rollout_start(self) -> None:
        # The previous PPO update has now finished. Flush its losses at that
        # update's step instead of labelling them with the next rollout's step.
        if "train/policy_gradient_loss" in self.logger.name_to_value:
            self.logger.dump(self.num_timesteps)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            episode = info.get("episode")
            if episode is not None:
                self.episodes += 1
                self.run.append_metrics({
                    "step": self.num_timesteps, "episode": self.episodes,
                    "episode_reward": float(episode["r"]),
                    "episode_length": int(episode["l"]),
                })
        if self.num_timesteps - self._last_checkpoint_step >= self.interval:
            normalizer = self.model.get_vec_normalize_env()
            if not isinstance(normalizer, VecNormalize):
                raise RuntimeError("Training snapshots require VecNormalize.")
            self.run.save_checkpoint(self.num_timesteps, self.model, normalizer)
            self._last_checkpoint_step = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        # SB3 logs before its PPO update. Flush the final update as well.
        self.logger.dump(self.num_timesteps)
