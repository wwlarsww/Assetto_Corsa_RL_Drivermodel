"""SB3 integration for run metrics and snapshots."""

from collections.abc import Mapping
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter
from stable_baselines3.common.vec_env import VecNormalize

from assetto_corsa_ai.training_run import MetricValue, TrainingRun
from assetto_corsa_ai.reward import Reward
from assetto_corsa_ai.racing_ppo import RacingPPO


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
            "requested_longitudinal_abs": "controls/requested_longitudinal_abs",
            "applied_throttle": "controls/applied_throttle",
            "applied_brake": "controls/applied_brake",
            "observed_longitudinal_abs": "controls/observed_longitudinal_abs",
            "policy_updates": "train/policy_updates",
            "policy_loss": "train/policy_gradient_loss",
            "value_loss": "train/value_loss",
            "entropy_loss": "train/entropy_loss",
            "predictor_loss": "train/predictor_loss",
            "predictor_valid_samples": "train/predictor_valid_samples",
            "predictor_forward_mae": "train/predictor_forward_mae",
            "predictor_lateral_mae": "train/predictor_lateral_mae",
            "learning_rate": "train/learning_rate",
            "performance_step": "reward/performance_step",
            "performance_prediction": "reward/performance_prediction",
            "style_step": "reward/style_step",
            "style_prediction": "reward/style_prediction",
            "termination_penalty": "reward/termination_penalty",
            "total_reward": "reward/total",
            "future_weight_scale": "reward/future_weight_scale",
            "mean_predicted_progress": "prediction/mean_predicted_progress",
            "mean_abs_lateral_offset": "prediction/mean_abs_lateral_offset",
        }
        if not any(name in key_values for name in names.values()):
            return
        row: dict[str, MetricValue] = {"step": step}
        for column, name in names.items():
            if name in key_values:
                row[column] = int(key_values[name]) if column in ("predictor_valid_samples", "policy_updates") else float(key_values[name])
        self.run.append_metrics(row)

    def close(self) -> None:
        pass


class TrainingRunCallback(BaseCallback):
    """Save paired checkpoints after completed PPO updates, never during collection."""

    def __init__(self, run: TrainingRun, interval_updates: int) -> None:
        if type(interval_updates) is not int or interval_updates < 0:
            raise ValueError("interval_updates must be a non-negative integer.")
        super().__init__()
        self.run = run
        self.interval_updates = interval_updates
        self.episodes = 0
        self._last_recorded_update = 0
        self._reward_samples: dict[str, list[float]] = {}
        self._pending_reward_metrics: dict[str, float] = {}

    def _on_training_start(self) -> None:
        if not isinstance(self.model, RacingPPO):
            raise ValueError("Training run checkpoints require RacingPPO's completed-update counter.")
        self._last_recorded_update = self.model.completed_policy_updates

    def _on_rollout_start(self) -> None:
        # SB3 calls this after the previous update, before collecting again.
        self._finish_policy_update()
        self._reward_samples = {}

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if all(name in info for name in ("requested_action", "applied_action", "observed_action")):
                requested = float(info["requested_action"][1])
                applied = float(info["applied_action"][1])
                observed = float(info["observed_action"][1])
                for name, value in {
                    "requested_longitudinal_abs": abs(requested),
                    "applied_throttle": max(applied, 0.0),
                    "applied_brake": max(-applied, 0.0),
                    "observed_longitudinal_abs": abs(observed),
                }.items():
                    self._reward_samples.setdefault(f"controls/{name}", []).append(value)
            reward = info.get("reward")
            if isinstance(reward, Reward):
                values = {f"reward/{name}": float(getattr(reward, name)) for name in (
                    "performance_step", "performance_prediction", "style_step",
                    "style_prediction", "termination_penalty", "total",
                )}
                values["reward/future_weight_scale"] = float(info.get("future_weight_scale", 0.0))
                for name in ("mean_predicted_progress", "mean_abs_lateral_offset"):
                    values[f"prediction/{name}"] = float(info.get(f"prediction_{name}", 0.0))
                for name, value in values.items():
                    self._reward_samples.setdefault(name, []).append(value)
            episode = info.get("episode")
            if episode is not None:
                self.episodes += 1
                self.run.append_metrics({
                    "step": self.num_timesteps, "episode": self.episodes,
                    "episode_reward": float(episode["r"]),
                    "episode_length": int(episode["l"]),
                })
        return True

    def _on_rollout_end(self) -> None:
        # Hold raw component means until the matching PPO update finishes;
        # SB3's pre-update dump must not create a separate/mislabeled row.
        self._pending_reward_metrics = {
            name: sum(values) / len(values) for name, values in self._reward_samples.items()
        }

    def _record_reward_metrics(self) -> None:
        for name, value in self._pending_reward_metrics.items():
            self.logger.record(name, value)
        self._pending_reward_metrics = {}

    def _on_training_end(self) -> None:
        self._finish_policy_update()

    def _finish_policy_update(self) -> None:
        assert isinstance(self.model, RacingPPO)
        updates = self.model.completed_policy_updates
        if updates == self._last_recorded_update:
            return
        # Flush and save the final update too. A cancelled partial rollout
        # never increments the model counter and cannot create a checkpoint.
        self._record_reward_metrics()
        self.logger.dump(self.model.num_timesteps)
        if self.interval_updates and updates % self.interval_updates == 0:
            normalizer = self.model.get_vec_normalize_env()
            if not isinstance(normalizer, VecNormalize):
                raise RuntimeError("Training snapshots require VecNormalize.")
            self.run.save_checkpoint(self.model.num_timesteps, self.model, normalizer)
        self._last_recorded_update = updates
