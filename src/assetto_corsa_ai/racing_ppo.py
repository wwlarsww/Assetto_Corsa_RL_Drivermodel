"""SB3 PPO with on-policy local-trajectory supervision.

Collection, GAE, schedules and serialization remain SB3's responsibility.
The update below follows SB3 2.9's PPO.train, adding one masked loss before
the existing backward/gradient-clipping/optimizer step.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, RolloutBufferSamples, Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize
import torch
from torch.nn import functional as F

from assetto_corsa_ai.predictor import PredictorConfig, build_trajectory_targets, horizon_indices, masked_predictor_loss
from assetto_corsa_ai.future_reward import FuturePredictionRewardConfig
from assetto_corsa_ai.racing_policy import RacingActorCriticPolicy


class RacingRolloutBuffer(RolloutBuffer):
    """Extra supervision for the current SB3 rollout, without a replay buffer."""

    pos: int

    def __init__(self, *args: Any, horizon_steps: tuple[int, ...], **kwargs: Any) -> None:
        self.horizon_steps = horizon_steps
        # SB3's buffer constructor leaves its Gymnasium Space types generic.
        super().__init__(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]

    def reset(self) -> None:
        super().reset()
        self.vehicle_positions = np.zeros((self.buffer_size, self.n_envs, 3), dtype=np.float64)
        self.vehicle_forwards = np.zeros_like(self.vehicle_positions)
        self.next_vehicle_positions = np.zeros_like(self.vehicle_positions)
        self.episode_ends = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)
        self.trajectory_targets = np.zeros((self.buffer_size, self.n_envs, len(self.horizon_steps), 2), dtype=np.float32)
        self.trajectory_mask = np.zeros(self.trajectory_targets.shape[:-1], dtype=np.bool_)
        self._targets_flattened = False
        self.batch_targets = torch.empty(0, device=self.device)
        self.batch_mask = torch.empty(0, dtype=torch.bool, device=self.device)

    def prepare_targets(self) -> None:
        self.trajectory_targets, self.trajectory_mask = build_trajectory_targets(
            self.vehicle_positions, self.vehicle_forwards, self.next_vehicle_positions,
            self.episode_ends, self.horizon_steps,
        )

    def _get_samples(self, batch_inds: NDArray[np.int64], env: VecNormalize | None = None) -> RolloutBufferSamples:
        if not self._targets_flattened:
            self.trajectory_targets = self.swap_and_flatten(self.trajectory_targets)
            self.trajectory_mask = self.swap_and_flatten(self.trajectory_mask)
            self._targets_flattened = True
        self.batch_targets = self.to_torch(self.trajectory_targets[batch_inds])
        self.batch_mask = self.to_torch(self.trajectory_mask[batch_inds])
        return super()._get_samples(batch_inds, env)


def _poses(infos: list[dict[str, Any]]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    try:
        positions = np.asarray([info["vehicle_position"] for info in infos], dtype=np.float64)
        forwards = np.asarray([info["vehicle_forward"] for info in infos], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Predictor rollouts require vehicle_position and vehicle_forward in reset/step info.") from error
    if positions.shape != (len(infos), 3) or forwards.shape != positions.shape or not np.isfinite(positions).all() or not np.isfinite(forwards).all():
        raise ValueError("Vehicle pose info must contain finite 3D vectors.")
    return positions, forwards


class _PoseCallback(BaseCallback):
    """Observe SB3's existing collection loop, preserving the caller's callback."""

    def __init__(
        self, callback: BaseCallback, env: VecEnv, buffer: RacingRolloutBuffer,
        previous: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    ) -> None:
        super().__init__()
        self.callback = callback
        self.env = env
        self.buffer = buffer
        base = env.unwrapped
        if not isinstance(base, DummyVecEnv):
            raise ValueError("Racing pose collection currently requires DummyVecEnv.")
        self.base = base
        self.current = _poses(base.reset_infos) if previous is None else previous

    def _on_rollout_start(self) -> None:
        self.callback.on_rollout_start()

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        self.callback.update_locals(locals_)

    def _on_step(self) -> bool:
        positions, forwards = _poses(self.locals["infos"])
        dones = np.asarray(self.locals["dones"], dtype=np.bool_)
        index = self.buffer.pos
        self.buffer.vehicle_positions[index], self.buffer.vehicle_forwards[index] = self.current
        # info is the terminal pose, whereas new_obs already belongs to the
        # reset episode when DummyVecEnv auto-resets. Keep those distinct.
        self.buffer.next_vehicle_positions[index] = positions
        self.buffer.episode_ends[index] = dones
        reset_infos = self.base.reset_infos
        for env_index in np.flatnonzero(dones):
            reset_position, reset_forward = _poses([reset_infos[env_index]])
            positions[env_index], forwards[env_index] = reset_position[0], reset_forward[0]
        self.current = positions, forwards
        return self.callback.on_step()

    def _on_rollout_end(self) -> None:
        self.buffer.prepare_targets()
        self.callback.on_rollout_end()


class RacingPPO(PPO):
    """PPO with masked supervision for an auxiliary trajectory predictor.

    Real future poses from each rollout supervise vehicle-relative prediction
    points. One optimizer updates the predictor and PPO heads through their
    shared features. Collected rewards remain fixed throughout optimization.
    completed_policy_updates counts finished rollout/optimization phases,
    rather than the optimization epochs counted by SB3's internal counter.
    """

    num_timesteps: int

    def __init__(
        self, policy: str | type[ActorCriticPolicy], env: GymEnv | str,
        *, predictor_config: PredictorConfig | None = None,
        future_reward_config: FuturePredictionRewardConfig | None = None,
        control_interval: float | None = None, **kwargs: Any,
    ) -> None:
        self.predictor_config = predictor_config
        # SB3's _n_updates counts epochs; this counts completed PPO phases.
        # It is serialized with model.zip and continues across resume.
        self.completed_policy_updates = 0
        # Persist reward settings with the policy, so load/resume reinstalls
        # the runtime wrapper. The auxiliary PPO update itself is unchanged.
        self.future_reward_config = future_reward_config
        self.horizon_steps: tuple[int, ...] = ()
        self._last_vehicle_poses: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        if predictor_config is not None and predictor_config.enabled:
            if control_interval is None:
                raise ValueError("Predictor training requires the configured control_interval.")
            self.horizon_steps = horizon_indices(predictor_config.horizons_seconds, control_interval)
            kwargs["rollout_buffer_class"] = RacingRolloutBuffer
            kwargs["rollout_buffer_kwargs"] = {"horizon_steps": self.horizon_steps}
        super().__init__(policy, env, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "_last_vehicle_poses"]

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int) -> bool:
        if not self.predictor_config or not self.predictor_config.enabled:
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        if not isinstance(rollout_buffer, RacingRolloutBuffer):
            raise RuntimeError("Predictor requires the racing rollout buffer.")
        previous = self._last_vehicle_poses
        if self._last_episode_starts is not None and np.all(self._last_episode_starts):
            previous = None
        observer = _PoseCallback(callback, env, rollout_buffer, previous)
        observer.init_callback(self)
        completed = super().collect_rollouts(env, observer, rollout_buffer, n_rollout_steps)
        self._last_vehicle_poses = observer.current
        return completed

    def train(self) -> None:
        if not self.predictor_config or not self.predictor_config.enabled:
            super().train()
            self._record_completed_update()
            return
        if not isinstance(self.policy, RacingActorCriticPolicy) or not isinstance(self.rollout_buffer, RacingRolloutBuffer):
            raise RuntimeError("Auxiliary PPO needs the shared racing policy and rollout buffer.")
        policy, buffer = self.policy, self.rollout_buffer
        policy.set_training_mode(True)
        self._update_learning_rate(policy.optimizer)
        # SB3 converts both scalars to schedules in _setup_model; its public
        # annotations still include the scalar forms accepted by __init__.
        clip_range = cast(Schedule, self.clip_range)(self._current_progress_remaining)
        clip_range_vf = cast(Schedule, self.clip_range_vf)(self._current_progress_remaining) if self.clip_range_vf is not None else None
        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        approx_kl_divs: list[float] = []
        auxiliary_sum = 0.0
        forward_error = 0.0
        lateral_error = 0.0
        valid_visits = 0
        continue_training = True
        last_loss = 0.0
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in buffer.get(self.batch_size):
                values, log_prob, entropy, prediction = policy.evaluate_with_prediction(
                    rollout_data.observations, rollout_data.actions,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -torch.min(
                    advantages * ratio, advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range),
                ).mean()
                values_pred = values if clip_range_vf is None else rollout_data.old_values + torch.clamp(
                    values - rollout_data.old_values, -clip_range_vf, clip_range_vf,
                )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                entropy_loss = -torch.mean(-log_prob) if entropy is None else -torch.mean(entropy)
                auxiliary_loss = masked_predictor_loss(prediction, buffer.batch_targets, buffer.batch_mask)
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                loss = loss + self.predictor_config.loss_weight * auxiliary_loss
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = float(torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu())
                    approx_kl_divs.append(approx_kl)
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl:.2f}")
                    break
                # Both objectives traverse the very same feature graph before
                # SB3's usual global gradient clipping and one optimizer step.
                policy.optimizer.zero_grad()
                torch.autograd.backward(loss)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                policy.optimizer.step()
                last_loss = loss.item()
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(torch.mean((torch.abs(ratio - 1) > clip_range).float()).item())
                valid = int(buffer.batch_mask.sum().item())
                valid_visits += valid
                auxiliary_sum += auxiliary_loss.item() * valid
                if valid:
                    errors = (prediction.detach() - buffer.batch_targets).abs()[buffer.batch_mask]
                    forward_error += errors[:, 0].sum().item()
                    lateral_error += errors[:, 1].sum().item()
            self._n_updates += 1
            if not continue_training:
                break
        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)) if entropy_losses else 0.0)
        self.logger.record("train/policy_gradient_loss", float(np.mean(pg_losses)) if pg_losses else 0.0)
        self.logger.record("train/value_loss", float(np.mean(value_losses)) if value_losses else 0.0)
        self.logger.record("train/approx_kl", float(np.mean(approx_kl_divs)) if approx_kl_divs else 0.0)
        self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)) if clip_fractions else 0.0)
        self.logger.record("train/loss", last_loss)
        self.logger.record("train/explained_variance", explained_variance(buffer.values.flatten(), buffer.returns.flatten()))
        self.logger.record("train/std", torch.exp(policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        # Counts unique valid horizon points in the rollout, not n_epochs visits.
        self.logger.record("train/predictor_valid_samples", int(buffer.trajectory_mask.sum()))
        self.logger.record("train/predictor_loss", auxiliary_sum / valid_visits if valid_visits else 0.0)
        self.logger.record("train/predictor_forward_mae", forward_error / valid_visits if valid_visits else 0.0)
        self.logger.record("train/predictor_lateral_mae", lateral_error / valid_visits if valid_visits else 0.0)
        self._record_completed_update()

    def _record_completed_update(self) -> None:
        self.completed_policy_updates += 1
        self.logger.record("train/policy_updates", self.completed_policy_updates)
