from __future__ import annotations

import copy
import json
import os
import warnings
from collections import deque

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization, MLP


def _diag_limit() -> int:
    return int(os.getenv("DDT_RSL_DIAG_MAX_LOGS_PER_KEY", "8"))


def _diag_threshold(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _obs_term_threshold(set_name: str, group_name: str) -> float:
    default_threshold = _diag_threshold("DDT_RSL_DIAG_LARGE_TERM_THRESHOLD", 50.0)

    per_group_thresholds = {
        "actions": _diag_threshold("DDT_RSL_DIAG_TERM_ACTIONS_THRESHOLD", 25.0),
        "last_action": _diag_threshold("DDT_RSL_DIAG_TERM_LAST_ACTION_THRESHOLD", 25.0),
        "joint_vel": _diag_threshold("DDT_RSL_DIAG_TERM_JOINT_VEL_THRESHOLD", 25.0),
        "base_lin_vel": _diag_threshold("DDT_RSL_DIAG_TERM_BASE_LIN_VEL_THRESHOLD", 20.0),
        "base_lin_vel_xy": _diag_threshold("DDT_RSL_DIAG_TERM_BASE_LIN_VEL_XY_THRESHOLD", 20.0),
        "base_ang_vel": _diag_threshold("DDT_RSL_DIAG_TERM_BASE_ANG_VEL_THRESHOLD", 20.0),
        "velocity_commands": _diag_threshold("DDT_RSL_DIAG_TERM_COMMAND_THRESHOLD", 20.0),
        "joint_pos": _diag_threshold("DDT_RSL_DIAG_TERM_JOINT_POS_THRESHOLD", 20.0),
        "feet_avg_contact_force": _diag_threshold("DDT_RSL_DIAG_TERM_FEET_FORCE_THRESHOLD", 25.0),
        "height_scan": _diag_threshold("DDT_RSL_DIAG_TERM_HEIGHT_SCAN_THRESHOLD", 5.0),
    }
    return per_group_thresholds.get(group_name, default_threshold)


def _rate_limited_diag(owner, key: str, message: str):
    diag_state = getattr(owner, "_ddt_diag_state", None)
    if diag_state is None:
        diag_state = {}
        setattr(owner, "_ddt_diag_state", diag_state)

    count = diag_state.get(key, 0)
    if count >= _diag_limit():
        return

    print(message)
    diag_state[key] = count + 1


def _tensor_stats(tensor: torch.Tensor) -> tuple[int, float, float, float]:
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    invalid_count = int((~finite_mask).sum().item())
    finite_values = detached[finite_mask]
    if finite_values.numel() == 0:
        return invalid_count, float("nan"), float("nan"), float("nan")
    max_abs = float(finite_values.abs().max().item())
    min_value = float(finite_values.min().item())
    max_value = float(finite_values.max().item())
    return invalid_count, max_abs, min_value, max_value


def _sanitize_tensor(owner, name: str, tensor: torch.Tensor) -> torch.Tensor:
    invalid_count, max_abs, _, _ = _tensor_stats(tensor)
    if invalid_count == 0:
        return tensor

    _rate_limited_diag(
        owner,
        f"sanitize:{name}",
        (
            "[diag][obs] "
            f"sanitized non-finite values in {name}: invalid={invalid_count} "
            f"shape={tuple(tensor.shape)} max_abs_finite={max_abs:.3e}"
        ),
    )
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _log_large_tensor(owner, name: str, tensor: torch.Tensor, threshold: float):
    invalid_count, max_abs, min_value, max_value = _tensor_stats(tensor)
    if invalid_count > 0:
        _rate_limited_diag(
            owner,
            f"invalid:{name}",
            (
                "[diag][tensor] "
                f"non-finite values in {name}: invalid={invalid_count} "
                f"shape={tuple(tensor.shape)}"
            ),
        )
        return

    if max_abs != max_abs or max_abs <= threshold:
        return

    _rate_limited_diag(
        owner,
        f"large:{name}",
        (
            "[diag][tensor] "
            f"large magnitude in {name}: max_abs={max_abs:.3e} "
            f"min={min_value:.3e} max={max_value:.3e} shape={tuple(tensor.shape)}"
        ),
    )


def _log_obs_term(owner, set_name: str, group_name: str, tensor: torch.Tensor):
    _log_large_tensor(owner, f"{set_name}:{group_name}", tensor, _obs_term_threshold(set_name, group_name))


def _log_action_std_anomaly(owner, std: torch.Tensor):
    detached = std.detach()
    non_finite = int((~torch.isfinite(detached)).sum().item())
    non_positive = int((detached <= 0).sum().item())
    if non_finite == 0 and non_positive == 0:
        return

    _, max_abs, min_value, max_value = _tensor_stats(detached)
    _rate_limited_diag(
        owner,
        "action_std",
        (
            "[diag][policy] "
            f"invalid action std detected: non_finite={non_finite} non_positive={non_positive} "
            f"min={min_value:.3e} max={max_value:.3e} max_abs={max_abs:.3e}"
        ),
    )


def _max_obs_group_abs(obs_timestep, env_idx: int, group_names: list[str]) -> tuple[str, float]:
    best_name = "n/a"
    best_abs = float("nan")
    best_invalid = -1
    for group_name in group_names:
        if group_name not in obs_timestep.keys():
            continue
        group_tensor = obs_timestep[group_name][env_idx]
        invalid_count, max_abs, _, _ = _tensor_stats(group_tensor)
        if invalid_count > 0:
            return f"{group_name}(invalid={invalid_count})", max_abs
        if best_invalid < 0 or max_abs > best_abs:
            best_name = group_name
            best_abs = max_abs
            best_invalid = 0
    return best_name, best_abs


def _resolve_velocity_target_groups(obs_groups) -> list[str]:
    if "velocity_target" in obs_groups:
        return obs_groups["velocity_target"]
    return obs_groups.get("privileged", [])


class DiagnosticActorCritic(ActorCritic):
    """Default actor-critic with light-weight observation sanitization and diagnostics."""

    def _concat_obs(self, obs, set_name: str) -> torch.Tensor:
        tensors = []
        for group_name in self.obs_groups[set_name]:
            tensor = _sanitize_tensor(self, f"{set_name}:{group_name}", obs[group_name])
            _log_obs_term(self, set_name, group_name, tensor)
            tensors.append(tensor)
        merged = torch.cat(tensors, dim=-1)
        _log_large_tensor(self, f"{set_name}_obs", merged, _diag_threshold("DDT_RSL_DIAG_LARGE_OBS_THRESHOLD", 100.0))
        return merged

    def get_actor_obs(self, obs):
        return self._concat_obs(obs, "policy")

    def get_critic_obs(self, obs):
        return self._concat_obs(obs, "critic")

    def update_distribution(self, obs):
        mean = self.actor(obs)
        _log_large_tensor(self, "action_mean", mean, _diag_threshold("DDT_RSL_DIAG_LARGE_MEAN_THRESHOLD", 100.0))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        _log_action_std_anomaly(self, std)
        self.distribution = Normal(mean, std)


class PPOWithDiagnostics(PPO):
    """Default PPO with minimal return/value diagnostics."""

    def process_env_step(self, obs, rewards, dones, extras):
        if not hasattr(self, "_ddt_raw_rewards") or self._ddt_raw_rewards.shape != self.storage.rewards.shape:
            self._ddt_raw_rewards = torch.zeros_like(self.storage.rewards)
            self._ddt_timeout_flags = torch.zeros_like(self.storage.rewards)
            self._ddt_timeout_bonus = torch.zeros_like(self.storage.rewards)

        if self.storage.step < self.storage.num_transitions_per_env:
            step = self.storage.step
            raw_rewards = rewards.view(-1, 1).detach().to(self.device)
            time_outs = extras.get("time_outs", torch.zeros_like(dones)).view(-1, 1).detach().to(self.device)
            timeout_bonus = torch.zeros_like(raw_rewards)
            if self.transition.values is not None:
                timeout_bonus = self.gamma * self.transition.values.detach() * time_outs

            self._ddt_raw_rewards[step].copy_(raw_rewards)
            self._ddt_timeout_flags[step].copy_(time_outs)
            self._ddt_timeout_bonus[step].copy_(timeout_bonus)

        super().process_env_step(obs, rewards, dones, extras)

    def _log_return_outliers(self, last_values: torch.Tensor, update_index: int):
        returns = self.storage.returns.detach().squeeze(-1)
        invalid_count, max_abs, min_value, max_value = _tensor_stats(returns)
        detail_threshold = _diag_threshold("DDT_RSL_DIAG_RETURN_DETAIL_THRESHOLD", 1000.0)
        if invalid_count == 0 and (max_abs != max_abs or max_abs <= detail_threshold):
            return

        abs_returns = returns.abs()
        if invalid_count > 0:
            abs_returns = torch.where(torch.isfinite(returns), abs_returns, torch.full_like(abs_returns, float("inf")))
        large_count = int((abs_returns > detail_threshold).sum().item())
        common_step_counter = getattr(getattr(self, "env", None), "common_step_counter", None)
        _rate_limited_diag(
            self,
            f"return_summary:{update_index}",
            (
                "[diag][returns] "
                f"update={update_index} common_step={common_step_counter} invalid={invalid_count} "
                f"large_count={large_count} max_abs={max_abs:.3e} min={min_value:.3e} max={max_value:.3e}"
            ),
        )

        top_k = min(5, abs_returns.numel())
        top_values, top_indices = torch.topk(abs_returns.reshape(-1), k=top_k)
        num_envs = self.storage.num_envs
        last_step = self.storage.num_transitions_per_env - 1
        policy_groups = getattr(self.policy, "obs_groups", {}).get("policy", [])
        critic_groups = getattr(self.policy, "obs_groups", {}).get("critic", [])

        for rank, (top_abs, flat_idx) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
            if top_abs <= detail_threshold and invalid_count == 0:
                break

            step = int(flat_idx // num_envs)
            env_idx = int(flat_idx % num_envs)
            next_value = last_values[env_idx] if step == last_step else self.storage.values[step + 1, env_idx]
            obs_timestep = self.storage.observations[step]
            policy_group_name, policy_group_abs = _max_obs_group_abs(obs_timestep, env_idx, policy_groups)
            critic_group_name, critic_group_abs = _max_obs_group_abs(obs_timestep, env_idx, critic_groups)

            raw_reward = float(self._ddt_raw_rewards[step, env_idx, 0].item()) if hasattr(self, "_ddt_raw_rewards") else float("nan")
            timeout_flag = bool(self._ddt_timeout_flags[step, env_idx, 0].item()) if hasattr(self, "_ddt_timeout_flags") else False
            timeout_bonus = (
                float(self._ddt_timeout_bonus[step, env_idx, 0].item()) if hasattr(self, "_ddt_timeout_bonus") else float("nan")
            )
            done_flag = bool(self.storage.dones[step, env_idx, 0].item())
            stored_reward = float(self.storage.rewards[step, env_idx, 0].item())
            return_value = float(self.storage.returns[step, env_idx, 0].item())
            value = float(self.storage.values[step, env_idx, 0].item())
            next_value_scalar = float(next_value[0].item())
            advantage = float(self.storage.advantages[step, env_idx, 0].item())
            action_norm = float(self.storage.actions[step, env_idx].norm().item())
            mu_norm = float(self.storage.mu[step, env_idx].norm().item()) if hasattr(self.storage, "mu") else float("nan")
            sigma_mean = float(self.storage.sigma[step, env_idx].mean().item()) if hasattr(self.storage, "sigma") else float("nan")

            _rate_limited_diag(
                self,
                f"return_detail:{update_index}:{rank}",
                (
                    "[diag][returns] "
                    f"top{rank} step={step} env={env_idx} return={return_value:.3e} advantage={advantage:.3e} "
                    f"value={value:.3e} next_value={next_value_scalar:.3e} raw_reward={raw_reward:.3e} "
                    f"stored_reward={stored_reward:.3e} timeout={int(timeout_flag)} timeout_bonus={timeout_bonus:.3e} "
                    f"done={int(done_flag)} action_norm={action_norm:.3e} mu_norm={mu_norm:.3e} "
                    f"sigma_mean={sigma_mean:.3e} policy_max={policy_group_name}:{policy_group_abs:.3e} "
                    f"critic_max={critic_group_name}:{critic_group_abs:.3e}"
                ),
            )

    def compute_returns(self, obs):
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

        update_index = getattr(self, "_ddt_update_index", 0) + 1
        self._ddt_update_index = update_index
        setattr(self.policy, "_ddt_update_index", update_index)

        _log_large_tensor(
            self,
            "rollout_values",
            self.storage.values,
            _diag_threshold("DDT_RSL_DIAG_LARGE_VALUE_THRESHOLD", 100.0),
        )
        _log_large_tensor(
            self,
            "rollout_returns",
            self.storage.returns,
            _diag_threshold("DDT_RSL_DIAG_LARGE_RETURN_THRESHOLD", 100.0),
        )
        _log_large_tensor(
            self,
            "rollout_advantages",
            self.storage.advantages,
            _diag_threshold("DDT_RSL_DIAG_LARGE_ADV_THRESHOLD", 100.0),
        )
        self._log_return_outliers(last_values, update_index)
        if hasattr(self.storage, "sigma"):
            _log_action_std_anomaly(self, self.storage.sigma)


class ActorCriticWithEstimator(nn.Module):
    """Actor-critic with a jointly trained history MLP for base velocity estimation."""

    ESTIMATOR_HISTORY_FEATURES_KEY = "__estimator_history_features"
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        estimator_hidden_dims=[256, 128],
        estimator_output_dim: int | None = None,
        num_history: int = 3,
        estimated_history_length: int | None = None,
        history_term_dims: list[int] | None = None,
        estimator_lr: float | None = None,
        estimator_target_scale: float | list[float] | tuple[float, ...] = 1.0,
        estimator_feature_scale: float | list[float] | tuple[float, ...] | None = None,
        deploy_share_policy_and_history: bool | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticWithEstimator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.num_history = num_history
        self.estimated_history_length = estimated_history_length if estimated_history_length is not None else 1
        self.history_term_dims = history_term_dims
        self.estimator_lr = estimator_lr

        self._history_groups = obs_groups.get("history", obs_groups["policy"])
        self._velocity_target_groups = _resolve_velocity_target_groups(obs_groups)

        num_actor_obs = sum(obs[group_name].shape[-1] for group_name in obs_groups["policy"])
        num_history_obs = sum(obs[group_name].shape[-1] for group_name in self._history_groups)
        num_critic_obs = sum(obs[group_name].shape[-1] for group_name in obs_groups["critic"])
        if estimator_output_dim is None:
            estimator_output_dim = sum(obs[group_name].shape[-1] for group_name in self._velocity_target_groups)
        if estimator_output_dim <= 0:
            raise ValueError("ActorCriticWithEstimator requires a non-empty velocity_target observation group.")
        if self.num_history < 1:
            raise ValueError("num_history must be greater than or equal to 1.")
        if self.estimated_history_length < 1:
            raise ValueError("estimated_history_length must be greater than or equal to 1.")
        if self.history_term_dims is not None:
            if any(term_dim <= 0 for term_dim in self.history_term_dims):
                raise ValueError("history_term_dims must contain only positive values.")
            history_term_total_dim = sum(self.history_term_dims)
            if num_history_obs % history_term_total_dim != 0:
                raise ValueError(
                    "history_term_dims do not evenly divide the flattened history observation size. "
                    f"History obs dim: {num_history_obs}, term dim sum: {history_term_total_dim}."
                )
            self.history_frame_count = num_history_obs // history_term_total_dim
            self.estimator_input_dim = history_term_total_dim * self.num_history
        else:
            if self.estimated_history_length > 1:
                raise ValueError(
                    "ActorCriticWithEstimator requires history_term_dims when estimated_history_length > 1."
                )
            self.history_frame_count = self.num_history
            self.estimator_input_dim = num_history_obs

        self.estimator = MLP(self.estimator_input_dim, estimator_output_dim, estimator_hidden_dims, activation)
        actor_estimator_obs_dim = estimator_output_dim * self.estimated_history_length
        self.actor = MLP(num_actor_obs + actor_estimator_obs_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        self.num_actor_obs = num_actor_obs
        self.num_history_obs = num_history_obs
        self.num_critic_obs = num_critic_obs
        self.estimator_output_dim = estimator_output_dim
        self.actor_estimator_obs_dim = actor_estimator_obs_dim
        self.num_actions = num_actions
        auto_share_policy_and_history = self.obs_groups["policy"] == self._history_groups and num_actor_obs == num_history_obs
        self.share_policy_and_history = (
            deploy_share_policy_and_history
            if deploy_share_policy_and_history is not None
            else auto_share_policy_and_history
        )

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs + actor_estimator_obs_dim)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        self.distribution = None
        self._last_estimated_velocity = None
        self.register_buffer(
            "_estimator_target_scale",
            self._build_scale_tensor(estimator_target_scale, estimator_output_dim, "estimator_target_scale"),
            persistent=False,
        )
        feature_scale_cfg = estimator_target_scale if estimator_feature_scale is None else estimator_feature_scale
        self.register_buffer(
            "_estimator_feature_scale",
            self._build_scale_tensor(
                feature_scale_cfg,
                estimator_output_dim,
                "estimator_feature_scale",
                allow_zero=True,
            ),
            persistent=False,
        )
        self.register_buffer("_estimated_feature_history", torch.empty(0), persistent=False)
        self.register_buffer("_estimated_feature_history_counts", torch.empty(0, dtype=torch.long), persistent=False)
        Normal.set_default_validate_args(False)

        print(f"Estimator MLP: {self.estimator}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

    def reset(self, dones=None):
        if self._estimated_feature_history.numel() == 0:
            return

        if dones is None:
            self._estimated_feature_history = self._estimated_feature_history.new_empty(0)
            self._estimated_feature_history_counts = self._estimated_feature_history_counts.new_empty(0)
            self._last_estimated_velocity = None
            return

        done_mask = dones.view(-1).to(device=self._estimated_feature_history.device, dtype=torch.bool)
        if done_mask.numel() != self._estimated_feature_history.shape[0]:
            self._estimated_feature_history = self._estimated_feature_history.new_empty(0)
            self._estimated_feature_history_counts = self._estimated_feature_history_counts.new_empty(0)
            self._last_estimated_velocity = None
            return

        self._estimated_feature_history[done_mask] = 0.0
        self._estimated_feature_history_counts[done_mask] = 0
        if self._last_estimated_velocity is not None and self._last_estimated_velocity.shape[0] == done_mask.numel():
            self._last_estimated_velocity = self._last_estimated_velocity.clone()
            self._last_estimated_velocity[done_mask] = 0.0

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def estimated_velocity(self):
        return self._last_estimated_velocity

    @property
    def velocity_target_groups(self) -> list[str]:
        return self._velocity_target_groups

    @staticmethod
    def _build_scale_tensor(
        scale: float | list[float] | tuple[float, ...],
        output_dim: int,
        name: str,
        allow_zero: bool = False,
    ) -> torch.Tensor:
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32).flatten()
        if scale_tensor.numel() == 1:
            scale_tensor = scale_tensor.repeat(output_dim)
        elif scale_tensor.numel() != output_dim:
            raise ValueError(f"{name} must be a scalar or have {output_dim} values, got {scale_tensor.numel()}.")
        if not allow_zero and torch.any(scale_tensor == 0):
            raise ValueError(f"{name} cannot contain zeros because estimator scaling divides by it.")
        return scale_tensor.view(1, output_dim)

    def _scale_estimator_features(self, estimate: torch.Tensor) -> torch.Tensor:
        # The estimator predicts physical velocity. Scale only the feature history consumed by the actor.
        return estimate * self._estimator_feature_scale.to(device=estimate.device, dtype=estimate.dtype)

    def _unscale_estimator_features(self, estimate: torch.Tensor) -> torch.Tensor:
        scale = self._estimator_feature_scale.to(device=estimate.device, dtype=estimate.dtype)
        active = scale != 0
        safe_scale = torch.where(active, scale, torch.ones_like(scale))
        unscaled = estimate / safe_scale
        return torch.where(active, unscaled, torch.zeros_like(unscaled))

    def _unscale_estimator_targets(self, target: torch.Tensor) -> torch.Tensor:
        # Keep the estimator supervised in physical velocity units even if the target obs is configured scaled.
        return target / self._estimator_target_scale.to(device=target.device, dtype=target.dtype)

    def _concat_obs(self, obs, group_names: list[str], set_name: str = "obs") -> torch.Tensor:
        tensors = []
        for group_name in group_names:
            tensor = _sanitize_tensor(self, f"{set_name}:{group_name}", obs[group_name])
            _log_obs_term(self, set_name, group_name, tensor)
            tensors.append(tensor)
        merged = torch.cat(tensors, dim=-1)
        _log_large_tensor(self, f"{set_name}_obs", merged, _diag_threshold("DDT_RSL_DIAG_LARGE_OBS_THRESHOLD", 100.0))
        return merged

    def estimate_from_history(self, obs) -> torch.Tensor:
        history_obs = self._concat_obs(obs, self._history_groups, set_name="history")
        self._last_estimated_velocity = self._estimate_current_from_history(history_obs)
        return self._last_estimated_velocity

    def _get_cached_estimator_features(self, obs) -> torch.Tensor | None:
        if self.ESTIMATOR_HISTORY_FEATURES_KEY not in obs.keys():
            return None

        cached_features = _sanitize_tensor(self, self.ESTIMATOR_HISTORY_FEATURES_KEY, obs[self.ESTIMATOR_HISTORY_FEATURES_KEY])
        if cached_features.shape[-1] != self.actor_estimator_obs_dim:
            raise ValueError(
                f"Cached estimator history features have invalid shape {cached_features.shape[-1]}; "
                f"expected {self.actor_estimator_obs_dim}."
            )
        return cached_features

    def _cache_estimator_features(self, obs, estimated_features: torch.Tensor) -> None:
        obs[self.ESTIMATOR_HISTORY_FEATURES_KEY] = estimated_features.detach().clone()

    def _history_frame_count_from_flat_obs(self, history_obs: torch.Tensor) -> int:
        if self.history_term_dims is None:
            return self.num_history

        history_term_total_dim = sum(self.history_term_dims)
        if history_obs.shape[-1] % history_term_total_dim != 0:
            raise ValueError(
                "history_term_dims do not evenly divide the flattened history observation. "
                f"History obs dim: {history_obs.shape[-1]}, term dim sum: {history_term_total_dim}."
            )
        return history_obs.shape[-1] // history_term_total_dim

    def _history_term_major_to_frame_major(self, history_obs: torch.Tensor) -> torch.Tensor:
        """Convert term-major flattened history into frame-major history."""
        if self.history_term_dims is None:
            raise ValueError("history_term_dims must be provided to convert history observations.")

        frame_count = self._history_frame_count_from_flat_obs(history_obs)
        history_chunks = []
        cursor = 0
        for term_dim in self.history_term_dims:
            block_size = term_dim * frame_count
            term_history = history_obs[:, cursor : cursor + block_size]
            term_history = term_history.reshape(history_obs.shape[0], frame_count, term_dim)
            history_chunks.append(term_history)
            cursor += block_size

        if cursor != history_obs.shape[-1]:
            raise ValueError(
                "history_term_dims do not cover the full flattened history observation. "
                f"Consumed {cursor}, total {history_obs.shape[-1]}."
            )

        return torch.cat(history_chunks, dim=-1)

    def _frame_major_to_term_major_history(self, history_frames: torch.Tensor) -> torch.Tensor:
        """Convert frame-major history back to the term-major flattened layout used by the estimator."""
        if self.history_term_dims is None:
            raise ValueError("history_term_dims must be provided to convert history observations.")

        frame_count = history_frames.shape[1]
        term_histories = []
        cursor = 0
        for term_dim in self.history_term_dims:
            term_history = history_frames[:, :, cursor : cursor + term_dim]
            term_histories.append(term_history.reshape(history_frames.shape[0], frame_count * term_dim))
            cursor += term_dim

        return torch.cat(term_histories, dim=-1)

    def _build_estimator_window(self, history_frames: torch.Tensor, end_idx: int) -> torch.Tensor:
        start_idx = max(0, end_idx - self.num_history + 1)
        window = history_frames[:, start_idx : end_idx + 1, :]
        pad_len = self.num_history - window.shape[1]
        if pad_len > 0:
            pad = history_frames[:, :1, :].expand(-1, pad_len, -1)
            window = torch.cat([pad, window], dim=1)
        return self._frame_major_to_term_major_history(window)

    def _estimate_current_from_history(self, history_obs: torch.Tensor) -> torch.Tensor:
        """Estimate only the current velocity target from the most recent num_history raw frames."""
        if self.history_term_dims is None:
            return self.estimator(history_obs)

        history_frames = self._history_term_major_to_frame_major(history_obs)
        current_window = self._build_estimator_window(history_frames, history_frames.shape[1] - 1)
        return self.estimator(current_window)

    def _ensure_estimator_feature_history(self, current_estimate: torch.Tensor) -> None:
        batch_size = current_estimate.shape[0]
        if (
            self._estimated_feature_history.numel() == 0
            or self._estimated_feature_history.dim() != 3
            or self._estimated_feature_history.shape[0] != batch_size
            or self._estimated_feature_history.shape[1] != self.estimated_history_length
            or self._estimated_feature_history.shape[2] != self.estimator_output_dim
            or self._estimated_feature_history_counts.numel() != batch_size
        ):
            self._estimated_feature_history = torch.zeros(
                batch_size,
                self.estimated_history_length,
                self.estimator_output_dim,
                device=current_estimate.device,
                dtype=current_estimate.dtype,
            )
            self._estimated_feature_history_counts = torch.zeros(
                batch_size, device=current_estimate.device, dtype=torch.long
            )

    def _append_estimator_history(self, current_estimate: torch.Tensor) -> torch.Tensor:
        self._ensure_estimator_feature_history(current_estimate)

        history = self._estimated_feature_history
        counts = self._estimated_feature_history_counts
        first_step_mask = counts == 0
        history_input = self._scale_estimator_features(current_estimate.detach())

        if first_step_mask.any():
            history[first_step_mask] = history_input[first_step_mask].unsqueeze(1).expand(
                -1, self.estimated_history_length, -1
            )

        continuing_mask = ~first_step_mask
        if continuing_mask.any():
            history[continuing_mask, :-1] = history[continuing_mask, 1:].clone()
            history[continuing_mask, -1] = history_input[continuing_mask]

        counts.add_(1)
        counts.clamp_max_(self.estimated_history_length)
        return history.reshape(current_estimate.shape[0], -1)

    def get_estimator_targets(self, obs) -> torch.Tensor:
        target_obs = self._concat_obs(obs, self._velocity_target_groups, set_name="velocity_target")
        return self._unscale_estimator_targets(target_obs)

    def get_actor_obs(
        self,
        obs,
        bootstrap_mask: torch.Tensor | None = None,
        true_velocity: torch.Tensor | None = None,
    ):
        policy_obs = self._concat_obs(obs, self.obs_groups["policy"], set_name="policy")
        estimated_features = self._get_cached_estimator_features(obs)
        if estimated_features is None:
            history_obs = self._concat_obs(obs, self._history_groups, set_name="history")
            self._last_estimated_velocity = self._estimate_current_from_history(history_obs)

            feature_velocity = self._last_estimated_velocity
            if bootstrap_mask is not None:
                if true_velocity is None:
                    true_velocity = self.get_estimator_targets(obs).detach()
                mask = bootstrap_mask.reshape(-1, 1).to(device=feature_velocity.device, dtype=torch.bool)
                feature_velocity = torch.where(mask, feature_velocity, true_velocity.to(feature_velocity.device))

            estimated_features = self._append_estimator_history(feature_velocity)
            self._cache_estimator_features(obs, estimated_features)
        else:
            self._last_estimated_velocity = self._unscale_estimator_features(
                estimated_features[:, -self.estimator_output_dim :]
            )
        return torch.cat([policy_obs, estimated_features], dim=-1)

    def get_critic_obs(self, obs):
        return self._concat_obs(obs, self.obs_groups["critic"], set_name="critic")

    def update_distribution(self, obs):
        mean = self.actor(obs)
        _log_large_tensor(self, "action_mean", mean, _diag_threshold("DDT_RSL_DIAG_LARGE_MEAN_THRESHOLD", 100.0))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        _log_action_std_anomaly(self, std)
        self.distribution = Normal(mean, std)

    def act(self, obs, bootstrap_mask: torch.Tensor | None = None, **kwargs):
        obs = self.get_actor_obs(obs, bootstrap_mask=bootstrap_mask)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.actor(obs)

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs, bootstrap_mask: torch.Tensor | None = None):
        with torch.no_grad():
            if self.actor_obs_normalization:
                self.actor_obs_normalizer.update(self.get_actor_obs(obs, bootstrap_mask=bootstrap_mask))
            if self.critic_obs_normalization:
                self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        current_state = self.state_dict()
        filtered_state = {}
        shape_mismatches = []

        for key, value in state_dict.items():
            if key in current_state and current_state[key].shape == value.shape:
                filtered_state[key] = value
            elif key in current_state:
                shape_mismatches.append(key)

        missing_keys, unexpected_keys = super().load_state_dict(filtered_state, strict=False)
        fully_loaded = not missing_keys and not unexpected_keys and not shape_mismatches and len(filtered_state) == len(
            current_state
        )

        if not fully_loaded:
            warnings.warn(
                "ActorCriticWithEstimator loaded a partial checkpoint. "
                f"Missing keys: {missing_keys}, unexpected keys: {unexpected_keys}, "
                f"shape mismatches: {shape_mismatches}",
                stacklevel=2,
            )

        return fully_loaded


class EstimatorActorDeployWrapper(nn.Module):
    """Single-engine deploy wrapper that runs estimator and actor internally."""

    def __init__(self, policy: ActorCriticWithEstimator):
        super().__init__()
        self.estimator = copy.deepcopy(policy.estimator)
        self.actor = copy.deepcopy(policy.actor)
        self.actor_obs_normalizer = copy.deepcopy(policy.actor_obs_normalizer)

        self.num_history = int(policy.num_history)
        self.estimated_history_length = int(policy.estimated_history_length)
        self.policy_obs_dim = int(policy.num_actor_obs)
        self.history_obs_dim = int(policy.num_history_obs)
        self.history_frame_count = int(getattr(policy, "history_frame_count", self.num_history))
        self.estimator_output_dim = int(policy.estimator_output_dim)
        self.num_actions = int(policy.num_actions)
        self.share_policy_and_history = bool(policy.share_policy_and_history)
        self.history_term_dims = list(policy.history_term_dims) if policy.history_term_dims is not None else None
        self.register_buffer("_estimator_target_scale", policy._estimator_target_scale.detach().clone(), persistent=False)
        self.register_buffer("_estimator_feature_scale", policy._estimator_feature_scale.detach().clone(), persistent=False)
        self.register_buffer("_estimated_feature_history", torch.empty(0), persistent=False)
        self.register_buffer("_estimated_feature_history_counts", torch.empty(0, dtype=torch.long), persistent=False)

        if self.estimated_history_length > 1 and self.history_term_dims is None:
            raise ValueError(
                "EstimatorActorDeployWrapper requires history_term_dims when estimated_history_length > 1."
            )

    @property
    def input_dim(self) -> int:
        if self.share_policy_and_history:
            return self.history_obs_dim
        return self.policy_obs_dim + self.history_obs_dim

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.share_policy_and_history:
            return obs, obs
        policy_obs = obs[:, : self.policy_obs_dim]
        history_obs = obs[:, self.policy_obs_dim :]
        return policy_obs, history_obs

    def _history_term_major_to_frame_major(self, history_obs: torch.Tensor) -> torch.Tensor:
        if self.history_term_dims is None:
            raise ValueError("history_term_dims must be provided to convert history observations.")

        history_term_total_dim = sum(self.history_term_dims)
        if history_obs.shape[-1] % history_term_total_dim != 0:
            raise ValueError(
                "history_term_dims do not evenly divide the flattened history observation. "
                f"History obs dim: {history_obs.shape[-1]}, term dim sum: {history_term_total_dim}."
            )
        frame_count = history_obs.shape[-1] // history_term_total_dim
        history_chunks = []
        cursor = 0
        for term_dim in self.history_term_dims:
            block_size = term_dim * frame_count
            term_history = history_obs[:, cursor : cursor + block_size]
            term_history = term_history.reshape(history_obs.shape[0], frame_count, term_dim)
            history_chunks.append(term_history)
            cursor += block_size

        if cursor != history_obs.shape[-1]:
            raise ValueError(
                "history_term_dims do not cover the full flattened history observation. "
                f"Consumed {cursor}, total {history_obs.shape[-1]}."
            )

        return torch.cat(history_chunks, dim=-1)

    def _frame_major_to_term_major_history(self, history_frames: torch.Tensor) -> torch.Tensor:
        if self.history_term_dims is None:
            raise ValueError("history_term_dims must be provided to convert history observations.")

        frame_count = history_frames.shape[1]
        term_histories = []
        cursor = 0
        for term_dim in self.history_term_dims:
            term_history = history_frames[:, :, cursor : cursor + term_dim]
            term_histories.append(term_history.reshape(history_frames.shape[0], frame_count * term_dim))
            cursor += term_dim
        return torch.cat(term_histories, dim=-1)

    def _build_estimator_window(self, history_frames: torch.Tensor, end_idx: int) -> torch.Tensor:
        start_idx = max(0, end_idx - self.num_history + 1)
        window = history_frames[:, start_idx : end_idx + 1, :]
        pad_len = self.num_history - window.shape[1]
        if pad_len > 0:
            pad = history_frames[:, :1, :].expand(-1, pad_len, -1)
            window = torch.cat([pad, window], dim=1)
        return self._frame_major_to_term_major_history(window)

    def reset(self):
        self._estimated_feature_history = self._estimated_feature_history.new_empty(0)
        self._estimated_feature_history_counts = self._estimated_feature_history_counts.new_empty(0)

    def _estimate_current_from_history(self, history_obs: torch.Tensor) -> torch.Tensor:
        if self.history_term_dims is None:
            return self.estimator(history_obs)

        history_frames = self._history_term_major_to_frame_major(history_obs)
        current_window = self._build_estimator_window(history_frames, history_frames.shape[1] - 1)
        return self.estimator(current_window)

    def _ensure_estimator_feature_history(self, current_estimate: torch.Tensor) -> None:
        batch_size = current_estimate.shape[0]
        if (
            self._estimated_feature_history.numel() == 0
            or self._estimated_feature_history.dim() != 3
            or self._estimated_feature_history.shape[0] != batch_size
            or self._estimated_feature_history.shape[1] != self.estimated_history_length
            or self._estimated_feature_history.shape[2] != self.estimator_output_dim
            or self._estimated_feature_history_counts.numel() != batch_size
        ):
            self._estimated_feature_history = torch.zeros(
                batch_size,
                self.estimated_history_length,
                self.estimator_output_dim,
                device=current_estimate.device,
                dtype=current_estimate.dtype,
            )
            self._estimated_feature_history_counts = torch.zeros(
                batch_size, device=current_estimate.device, dtype=torch.long
            )

    def _scale_estimator_features(self, estimate: torch.Tensor) -> torch.Tensor:
        # The estimator predicts physical velocity. Scale only the feature history consumed by the actor.
        return estimate * self._estimator_feature_scale.to(device=estimate.device, dtype=estimate.dtype)

    def _append_estimator_history(self, current_estimate: torch.Tensor) -> torch.Tensor:
        self._ensure_estimator_feature_history(current_estimate)

        history = self._estimated_feature_history
        counts = self._estimated_feature_history_counts
        first_step_mask = counts == 0
        history_input = self._scale_estimator_features(current_estimate.detach())

        if first_step_mask.any():
            history[first_step_mask] = history_input[first_step_mask].unsqueeze(1).expand(
                -1, self.estimated_history_length, -1
            )

        continuing_mask = ~first_step_mask
        if continuing_mask.any():
            history[continuing_mask, :-1] = history[continuing_mask, 1:].clone()
            history[continuing_mask, -1] = history_input[continuing_mask]

        counts.add_(1)
        counts.clamp_max_(self.estimated_history_length)
        return history.reshape(current_estimate.shape[0], -1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy_obs, history_obs = self._split_obs(obs)
        current_estimate = self._estimate_current_from_history(history_obs)
        estimated_features = self._append_estimator_history(current_estimate)
        actor_obs = torch.cat([policy_obs, estimated_features], dim=-1)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self.actor(actor_obs), current_estimate


class EstimatorOnlyDeployWrapper(nn.Module):
    """Estimator-only deploy wrapper with an explicit history-window input."""

    def __init__(self, policy: ActorCriticWithEstimator):
        super().__init__()
        self.estimator = copy.deepcopy(policy.estimator)
        self.num_history = int(policy.num_history)
        self.estimator_input_dim = int(policy.estimator_input_dim)
        self.estimator_output_dim = int(policy.estimator_output_dim)
        self.history_term_dims = list(policy.history_term_dims) if policy.history_term_dims is not None else None
        self.history_frame_dim = int(sum(self.history_term_dims)) if self.history_term_dims is not None else self.estimator_input_dim
        self.register_buffer("_estimator_target_scale", policy._estimator_target_scale.detach().clone(), persistent=False)

    @property
    def input_dim(self) -> int:
        return self.estimator_input_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.estimator(obs)


class ActorWithEstimatorHistoryDeployWrapper(nn.Module):
    """Actor-only deploy wrapper with externally maintained estimator feature history."""

    def __init__(self, policy: ActorCriticWithEstimator):
        super().__init__()
        self.actor = copy.deepcopy(policy.actor)
        self.actor_obs_normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        self.policy_obs_dim = int(policy.num_actor_obs)
        self.estimator_output_dim = int(policy.estimator_output_dim)
        self.estimated_history_length = int(policy.estimated_history_length)
        self.actor_estimator_obs_dim = int(policy.actor_estimator_obs_dim)
        self.num_actions = int(policy.num_actions)
        self.register_buffer("_estimator_feature_scale", policy._estimator_feature_scale.detach().clone(), persistent=False)

    @property
    def input_dim(self) -> int:
        return self.policy_obs_dim + self.actor_estimator_obs_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(obs)
        return self.actor(actor_obs)


def is_estimator_policy(policy: nn.Module) -> bool:
    return isinstance(policy, ActorCriticWithEstimator)


def get_estimator_deploy_metadata(policy: ActorCriticWithEstimator) -> dict[str, object]:
    deploy_wrapper = EstimatorActorDeployWrapper(policy)
    return {
        "type": "estimator_actor_deploy_wrapper",
        "input_name": "obs",
        "output_names": ["actions", "estimated_velocity"],
        "input_dim": deploy_wrapper.input_dim,
        "output_dim": deploy_wrapper.num_actions,
        "estimated_velocity_dim": deploy_wrapper.estimator_output_dim,
        "estimated_velocity_units": "unscaled_base_lin_vel_xy",
        "policy_obs_dim": deploy_wrapper.policy_obs_dim,
        "history_obs_dim": deploy_wrapper.history_obs_dim,
        "history_frame_count": deploy_wrapper.history_frame_count,
        "estimator_output_dim": deploy_wrapper.estimator_output_dim,
        "estimated_history_length": deploy_wrapper.estimated_history_length,
        "actor_estimator_obs_dim": deploy_wrapper.estimator_output_dim * deploy_wrapper.estimated_history_length,
        "share_policy_and_history": deploy_wrapper.share_policy_and_history,
        "input_layout": "history_only" if deploy_wrapper.share_policy_and_history else "policy_then_history",
        "num_history": deploy_wrapper.num_history,
        "history_term_dims": deploy_wrapper.history_term_dims,
        "estimator_target_scale": deploy_wrapper._estimator_target_scale.view(-1).tolist(),
        "estimator_feature_scale": deploy_wrapper._estimator_feature_scale.view(-1).tolist(),
        "actor_estimator_feature_units": "estimated_velocity_times_estimator_feature_scale",
    }


def get_split_estimator_deploy_metadata(policy: ActorCriticWithEstimator) -> dict[str, object]:
    estimator_wrapper = EstimatorOnlyDeployWrapper(policy)
    actor_wrapper = ActorWithEstimatorHistoryDeployWrapper(policy)
    return {
        "type": "split_estimator_actor_deploy",
        "history_term_dims": estimator_wrapper.history_term_dims,
        "policy_history_frame_count": int(getattr(policy, "history_frame_count", policy.num_history)),
        "estimator_history_length": estimator_wrapper.num_history,
        "estimated_history_length": actor_wrapper.estimated_history_length,
        "estimator_target_scale": estimator_wrapper._estimator_target_scale.view(-1).tolist(),
        "estimator_feature_scale": actor_wrapper._estimator_feature_scale.view(-1).tolist(),
        "estimated_velocity_units": "unscaled_base_lin_vel_xy",
        "estimator": {
            "input_name": "obs",
            "output_name": "estimated_velocity",
            "output_units": "unscaled_base_lin_vel_xy",
            "input_dim": estimator_wrapper.input_dim,
            "output_dim": estimator_wrapper.estimator_output_dim,
            "history_frame_dim": estimator_wrapper.history_frame_dim,
            "history_length": estimator_wrapper.num_history,
        },
        "actor": {
            "input_name": "obs",
            "output_name": "actions",
            "input_dim": actor_wrapper.input_dim,
            "output_dim": actor_wrapper.num_actions,
            "policy_obs_dim": actor_wrapper.policy_obs_dim,
            "estimator_feature_history_dim": actor_wrapper.actor_estimator_obs_dim,
            "estimator_feature_dim": actor_wrapper.estimator_output_dim,
            "estimator_feature_history_length": actor_wrapper.estimated_history_length,
            "estimator_feature_units": "estimated_velocity_times_estimator_feature_scale",
            "input_layout": "policy_then_estimator_history",
        },
    }


def export_estimator_policy_metadata(
    policy: ActorCriticWithEstimator, path: str, filename: str = "policy_metadata.json"
) -> None:
    os.makedirs(path, exist_ok=True)
    export_path = os.path.join(path, filename)
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(get_estimator_deploy_metadata(policy), f, indent=2)


def export_split_estimator_policy_metadata(
    policy: ActorCriticWithEstimator, path: str, filename: str = "policy_split_metadata.json"
) -> None:
    os.makedirs(path, exist_ok=True)
    export_path = os.path.join(path, filename)
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(get_split_estimator_deploy_metadata(policy), f, indent=2)


def export_estimator_policy_as_jit(policy: ActorCriticWithEstimator, path: str, filename: str = "policy.pt") -> None:
    os.makedirs(path, exist_ok=True)
    deploy_wrapper = EstimatorActorDeployWrapper(policy)
    deploy_wrapper.to("cpu")
    deploy_wrapper.eval()
    deploy_wrapper.reset()
    scripted_module = torch.jit.script(deploy_wrapper)
    scripted_module.save(os.path.join(path, filename))


def export_estimator_policy_as_onnx(
    policy: ActorCriticWithEstimator,
    path: str,
    filename: str = "policy.onnx",
    verbose: bool = False,
    opset_version: int = 18,
) -> None:
    os.makedirs(path, exist_ok=True)
    deploy_wrapper = EstimatorActorDeployWrapper(policy)
    deploy_wrapper.to("cpu")
    deploy_wrapper.eval()
    deploy_wrapper.reset()
    example_obs = torch.zeros(1, deploy_wrapper.input_dim)
    torch.onnx.export(
        deploy_wrapper,
        example_obs,
        os.path.join(path, filename),
        export_params=True,
        opset_version=opset_version,
        verbose=verbose,
        input_names=["obs"],
        output_names=["actions", "estimated_velocity"],
        dynamic_axes={},
    )


def export_estimator_only_policy_as_jit(
    policy: ActorCriticWithEstimator, path: str, filename: str = "estimator_policy.pt"
) -> None:
    os.makedirs(path, exist_ok=True)
    deploy_wrapper = EstimatorOnlyDeployWrapper(policy)
    deploy_wrapper.to("cpu")
    deploy_wrapper.eval()
    scripted_module = torch.jit.script(deploy_wrapper)
    scripted_module.save(os.path.join(path, filename))


def export_estimator_only_policy_as_onnx(
    policy: ActorCriticWithEstimator,
    path: str,
    filename: str = "estimator_policy.onnx",
    verbose: bool = False,
    opset_version: int = 18,
) -> None:
    os.makedirs(path, exist_ok=True)
    deploy_wrapper = EstimatorOnlyDeployWrapper(policy)
    deploy_wrapper.to("cpu")
    deploy_wrapper.eval()
    example_obs = torch.zeros(1, deploy_wrapper.input_dim)
    torch.onnx.export(
        deploy_wrapper,
        example_obs,
        os.path.join(path, filename),
        export_params=True,
        opset_version=opset_version,
        verbose=verbose,
        input_names=["obs"],
        output_names=["estimated_velocity"],
        dynamic_axes={},
    )


def export_actor_with_estimator_history_as_jit(
    policy: ActorCriticWithEstimator, path: str, filename: str = "actor_policy.pt"
) -> None:
    os.makedirs(path, exist_ok=True)
    deploy_wrapper = ActorWithEstimatorHistoryDeployWrapper(policy)
    deploy_wrapper.to("cpu")
    deploy_wrapper.eval()
    scripted_module = torch.jit.script(deploy_wrapper)
    scripted_module.save(os.path.join(path, filename))


def export_actor_with_estimator_history_as_onnx(
    policy: ActorCriticWithEstimator,
    path: str,
    filename: str = "actor_policy.onnx",
    verbose: bool = False,
    opset_version: int = 18,
) -> None:
    os.makedirs(path, exist_ok=True)
    deploy_wrapper = ActorWithEstimatorHistoryDeployWrapper(policy)
    deploy_wrapper.to("cpu")
    deploy_wrapper.eval()
    example_obs = torch.zeros(1, deploy_wrapper.input_dim)
    torch.onnx.export(
        deploy_wrapper,
        example_obs,
        os.path.join(path, filename),
        export_params=True,
        opset_version=opset_version,
        verbose=verbose,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )


class ActorCriticWithCENet(nn.Module):
    """DreamWaQ-style actor-critic with a context-aided estimator network."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        cenet_encoder_hidden_dims=[128, 64],
        cenet_decoder_hidden_dims=[64, 128],
        cenet_velocity_dim: int = 3,
        cenet_latent_dim: int = 16,
        num_history: int = 5,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticWithCENet.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        if len(cenet_encoder_hidden_dims) < 1:
            raise ValueError("cenet_encoder_hidden_dims must contain at least one layer size.")

        self.obs_groups = obs_groups
        self.num_history = num_history
        self._history_groups = obs_groups.get("history", obs_groups["policy"])
        self._velocity_target_groups = _resolve_velocity_target_groups(obs_groups)
        if not self._velocity_target_groups:
            raise ValueError("ActorCriticWithCENet requires a velocity_target observation group.")

        num_actor_obs = sum(obs[group_name].shape[-1] for group_name in obs_groups["policy"])
        num_history_obs = sum(obs[group_name].shape[-1] for group_name in self._history_groups)
        num_critic_obs = sum(obs[group_name].shape[-1] for group_name in obs_groups["critic"])
        num_velocity_target_obs = sum(obs[group_name].shape[-1] for group_name in self._velocity_target_groups)
        if num_velocity_target_obs != cenet_velocity_dim:
            raise ValueError(
                "ActorCriticWithCENet expects velocity_target observations to match cenet_velocity_dim. "
                f"Expected {cenet_velocity_dim}, got {num_velocity_target_obs}."
            )

        encoder_output_dim = int(cenet_encoder_hidden_dims[-1])
        encoder_hidden_dims = list(cenet_encoder_hidden_dims[:-1])
        self.cenet_encoder = MLP(num_history_obs, encoder_output_dim, encoder_hidden_dims, activation)
        self.cenet_mean_vel = nn.Linear(encoder_output_dim, cenet_velocity_dim)
        self.cenet_logvar_vel = nn.Linear(encoder_output_dim, cenet_velocity_dim)
        self.cenet_mean_latent = nn.Linear(encoder_output_dim, cenet_latent_dim)
        self.cenet_logvar_latent = nn.Linear(encoder_output_dim, cenet_latent_dim)
        self.cenet_decoder = MLP(
            cenet_velocity_dim + cenet_latent_dim,
            num_actor_obs,
            cenet_decoder_hidden_dims,
            activation,
        )

        self.actor = MLP(num_actor_obs + cenet_velocity_dim + cenet_latent_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        self.num_actor_obs = num_actor_obs
        self.num_history_obs = num_history_obs
        self.num_critic_obs = num_critic_obs
        self.cenet_velocity_dim = int(cenet_velocity_dim)
        self.cenet_latent_dim = int(cenet_latent_dim)
        self.cenet_output_dim = self.cenet_velocity_dim + self.cenet_latent_dim
        self.num_actions = num_actions

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs + self.cenet_output_dim)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        self.distribution = None
        self._last_estimated_velocity = None
        Normal.set_default_validate_args(False)

        print(f"CENet encoder: {self.cenet_encoder}")
        print(f"CENet decoder: {self.cenet_decoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def estimated_velocity(self):
        return self._last_estimated_velocity

    @property
    def velocity_target_groups(self) -> list[str]:
        return self._velocity_target_groups

    def cenet_parameters(self):
        yield from self.cenet_encoder.parameters()
        yield from self.cenet_mean_vel.parameters()
        yield from self.cenet_logvar_vel.parameters()
        yield from self.cenet_mean_latent.parameters()
        yield from self.cenet_logvar_latent.parameters()
        yield from self.cenet_decoder.parameters()

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def _concat_obs(self, obs, group_names: list[str], set_name: str = "obs") -> torch.Tensor:
        tensors = []
        for group_name in group_names:
            tensor = _sanitize_tensor(self, f"{set_name}:{group_name}", obs[group_name])
            _log_obs_term(self, set_name, group_name, tensor)
            tensors.append(tensor)
        merged = torch.cat(tensors, dim=-1)
        _log_large_tensor(self, f"{set_name}_obs", merged, _diag_threshold("DDT_RSL_DIAG_LARGE_OBS_THRESHOLD", 100.0))
        return merged

    def get_policy_obs(self, obs) -> torch.Tensor:
        return self._concat_obs(obs, self.obs_groups["policy"], set_name="policy")

    def get_history_obs(self, obs) -> torch.Tensor:
        return self._concat_obs(obs, self._history_groups, set_name="history")

    def get_velocity_targets(self, obs) -> torch.Tensor:
        return self._concat_obs(obs, self._velocity_target_groups, set_name="velocity_target")

    def get_critic_obs(self, obs) -> torch.Tensor:
        return self._concat_obs(obs, self.obs_groups["critic"], set_name="critic")

    @staticmethod
    def _reparameterize(mean: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode_cenet(self, latent_code: torch.Tensor, velocity_code: torch.Tensor) -> torch.Tensor:
        decoder_input = torch.cat([velocity_code, latent_code], dim=-1)
        return self.cenet_decoder(decoder_input)

    def cenet_forward(
        self,
        history_obs: torch.Tensor,
        sample: bool = True,
        reconstruction_velocity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.cenet_encoder(history_obs)
        mean_vel = self.cenet_mean_vel(encoded)
        logvar_vel = torch.clamp(self.cenet_logvar_vel(encoded), min=-10.0, max=5.0)
        mean_latent = self.cenet_mean_latent(encoded)
        logvar_latent = torch.clamp(self.cenet_logvar_latent(encoded), min=-10.0, max=5.0)
        code_vel = self._reparameterize(mean_vel, logvar_vel, sample=sample)
        code_latent = self._reparameterize(mean_latent, logvar_latent, sample=sample)
        code = torch.cat([code_vel, code_latent], dim=-1)
        decoder_velocity = (
            code_vel
            if reconstruction_velocity is None
            else reconstruction_velocity.to(device=code_latent.device, dtype=code_latent.dtype)
        )
        reconstruction = self.decode_cenet(code_latent, decoder_velocity)
        self._last_estimated_velocity = mean_vel
        return {
            "code": code,
            "code_vel": code_vel,
            "code_latent": code_latent,
            "mean_vel": mean_vel,
            "logvar_vel": logvar_vel,
            "mean_latent": mean_latent,
            "logvar_latent": logvar_latent,
            "reconstruction": reconstruction,
        }

    def get_actor_obs(
        self,
        obs,
        bootstrap_mask: torch.Tensor | None = None,
        sample_cenet: bool = False,
        true_velocity: torch.Tensor | None = None,
        detach_cenet: bool = False,
        actor_velocity_code: torch.Tensor | None = None,
    ) -> torch.Tensor:
        policy_obs = self.get_policy_obs(obs)
        history_obs = self.get_history_obs(obs)
        cenet_outputs = self.cenet_forward(history_obs, sample=sample_cenet)

        estimated_velocity = cenet_outputs["code_vel"]
        if actor_velocity_code is not None:
            velocity_code = actor_velocity_code.to(device=estimated_velocity.device, dtype=estimated_velocity.dtype)
        elif bootstrap_mask is not None:
            if true_velocity is None:
                true_velocity = self.get_velocity_targets(obs).detach()
            mask = bootstrap_mask.reshape(-1, 1).to(device=estimated_velocity.device, dtype=torch.bool)
            velocity_code = torch.where(mask, estimated_velocity, true_velocity.to(estimated_velocity.device))
        else:
            velocity_code = estimated_velocity

        latent_code = cenet_outputs["code_latent"]
        if detach_cenet:
            velocity_code = velocity_code.detach()
            latent_code = latent_code.detach()

        self._last_actor_velocity_code = velocity_code.detach()
        return torch.cat([policy_obs, velocity_code, latent_code], dim=-1)

    def update_distribution(self, obs):
        mean = self.actor(obs)
        _log_large_tensor(self, "action_mean", mean, _diag_threshold("DDT_RSL_DIAG_LARGE_MEAN_THRESHOLD", 100.0))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        _log_action_std_anomaly(self, std)
        self.distribution = Normal(mean, std)

    @property
    def last_actor_velocity_code(self) -> torch.Tensor | None:
        return getattr(self, "_last_actor_velocity_code", None)

    def act(
        self,
        obs,
        bootstrap_mask: torch.Tensor | None = None,
        detach_cenet: bool = False,
        actor_velocity_code: torch.Tensor | None = None,
        **kwargs,
    ):
        obs = self.get_actor_obs(
            obs,
            bootstrap_mask=bootstrap_mask,
            sample_cenet=False,
            detach_cenet=detach_cenet,
            actor_velocity_code=actor_velocity_code,
        )
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs, bootstrap_mask=None, sample_cenet=False)
        obs = self.actor_obs_normalizer(obs)
        return self.actor(obs)

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs, bootstrap_mask: torch.Tensor | None = None):
        with torch.no_grad():
            if self.actor_obs_normalization:
                self.actor_obs_normalizer.update(
                    self.get_actor_obs(obs, bootstrap_mask=bootstrap_mask, sample_cenet=False)
                )
            if self.critic_obs_normalization:
                self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def compute_cenet_losses(
        self,
        obs,
        next_policy_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        history_obs = self.get_history_obs(obs)
        velocity_target = self.get_velocity_targets(obs).detach()
        outputs = self.cenet_forward(history_obs, sample=True, reconstruction_velocity=velocity_target)
        velocity_loss = torch.nn.functional.mse_loss(outputs["code_vel"], velocity_target)

        valid = 1.0 - dones.reshape(-1, 1).float()
        recon_error = (outputs["reconstruction"] - next_policy_obs.detach()).pow(2)
        reconstruction_loss = (recon_error * valid).sum() / (
            valid.sum().clamp_min(1.0) * next_policy_obs.shape[-1]
        )

        mean_latent = outputs["mean_latent"]
        logvar_latent = outputs["logvar_latent"]
        kl_loss = -0.5 * torch.sum(1.0 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp(), dim=-1).mean()

        return {
            "velocity": velocity_loss,
            "reconstruction": reconstruction_loss,
            "kl": kl_loss,
        }

    def load_state_dict(self, state_dict, strict=True):
        current_state = self.state_dict()
        filtered_state = {}
        shape_mismatches = []

        for key, value in state_dict.items():
            if key in current_state and current_state[key].shape == value.shape:
                filtered_state[key] = value
            elif key in current_state:
                shape_mismatches.append(key)

        missing_keys, unexpected_keys = super().load_state_dict(filtered_state, strict=False)
        fully_loaded = not missing_keys and not unexpected_keys and not shape_mismatches and len(filtered_state) == len(
            current_state
        )

        if not fully_loaded:
            warnings.warn(
                "ActorCriticWithCENet loaded a partial checkpoint. "
                f"Missing keys: {missing_keys}, unexpected keys: {unexpected_keys}, "
                f"shape mismatches: {shape_mismatches}",
                stacklevel=2,
            )

        return fully_loaded


class CENetActorDeployWrapper(nn.Module):
    """Deploy wrapper that runs CENet and actor from policy/history observations."""

    def __init__(self, policy: ActorCriticWithCENet):
        super().__init__()
        self.cenet_encoder = copy.deepcopy(policy.cenet_encoder)
        self.cenet_mean_vel = copy.deepcopy(policy.cenet_mean_vel)
        self.cenet_mean_latent = copy.deepcopy(policy.cenet_mean_latent)
        self.actor = copy.deepcopy(policy.actor)
        self.actor_obs_normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        self.policy_obs_dim = int(policy.num_actor_obs)
        self.history_obs_dim = int(policy.num_history_obs)
        self.cenet_velocity_dim = int(policy.cenet_velocity_dim)
        self.cenet_latent_dim = int(policy.cenet_latent_dim)
        self.num_actions = int(policy.num_actions)

    @property
    def input_dim(self) -> int:
        return self.policy_obs_dim + self.history_obs_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy_obs = obs[:, : self.policy_obs_dim]
        history_obs = obs[:, self.policy_obs_dim :]
        encoded = self.cenet_encoder(history_obs)
        mean_vel = self.cenet_mean_vel(encoded)
        mean_latent = self.cenet_mean_latent(encoded)
        actor_obs = torch.cat([policy_obs, mean_vel, mean_latent], dim=-1)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self.actor(actor_obs), mean_vel


def is_cenet_policy(policy: nn.Module) -> bool:
    return isinstance(policy, ActorCriticWithCENet)


def get_cenet_deploy_metadata(policy: ActorCriticWithCENet) -> dict[str, object]:
    deploy_wrapper = CENetActorDeployWrapper(policy)
    return {
        "type": "cenet_actor_deploy_wrapper",
        "input_name": "obs",
        "output_names": ["actions", "estimated_velocity"],
        "input_dim": deploy_wrapper.input_dim,
        "output_dim": deploy_wrapper.num_actions,
        "estimated_velocity_dim": deploy_wrapper.cenet_velocity_dim,
        "estimated_velocity_units": "base_lin_vel",
        "policy_obs_dim": deploy_wrapper.policy_obs_dim,
        "history_obs_dim": deploy_wrapper.history_obs_dim,
        "cenet_velocity_dim": deploy_wrapper.cenet_velocity_dim,
        "cenet_latent_dim": deploy_wrapper.cenet_latent_dim,
        "policy_input_layout": "policy_obs_then_history_obs",
        "num_history": policy.num_history,
        "obs_groups": policy.obs_groups,
    }


def export_cenet_policy_metadata(
    policy: ActorCriticWithCENet, path: str, filename: str = "policy_metadata.json"
) -> None:
    os.makedirs(path, exist_ok=True)
    export_path = os.path.join(path, filename)
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(get_cenet_deploy_metadata(policy), f, indent=2)


def export_cenet_policy_as_jit(policy: ActorCriticWithCENet, path: str, filename: str = "policy.pt") -> None:
    os.makedirs(path, exist_ok=True)
    deploy_wrapper = CENetActorDeployWrapper(policy)
    deploy_wrapper.to("cpu")
    deploy_wrapper.eval()
    example_obs = torch.zeros(1, deploy_wrapper.input_dim)
    traced_module = torch.jit.trace(deploy_wrapper, example_obs)
    traced_module.save(os.path.join(path, filename))


def export_cenet_policy_as_onnx(
    policy: ActorCriticWithCENet,
    path: str,
    filename: str = "policy.onnx",
    verbose: bool = False,
    opset_version: int = 18,
) -> None:
    os.makedirs(path, exist_ok=True)
    deploy_wrapper = CENetActorDeployWrapper(policy)
    deploy_wrapper.to("cpu")
    deploy_wrapper.eval()
    example_obs = torch.zeros(1, deploy_wrapper.input_dim)
    torch.onnx.export(
        deploy_wrapper,
        example_obs,
        os.path.join(path, filename),
        export_params=True,
        opset_version=opset_version,
        verbose=verbose,
        input_names=["obs"],
        output_names=["actions", "estimated_velocity"],
        dynamic_axes={},
    )


def _patch_on_policy_runner_checkpointing(on_policy_runner_module) -> None:
    runner_cls = on_policy_runner_module.OnPolicyRunner
    if getattr(runner_cls, "_ddt_cenet_checkpoint_patch", False):
        return

    original_save = runner_cls.save
    original_load = runner_cls.load

    def save_with_extra_state(self, path: str, infos: dict | None = None) -> None:
        original_save(self, path, infos)
        if not hasattr(self.alg, "get_extra_checkpoint_state"):
            return
        saved_dict = torch.load(path, weights_only=False, map_location="cpu")
        saved_dict.update(self.alg.get_extra_checkpoint_state())
        torch.save(saved_dict, path)

    def load_with_extra_state(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        infos = original_load(self, path, load_optimizer=load_optimizer, map_location=map_location)
        if hasattr(self.alg, "load_extra_checkpoint_state"):
            self.alg.load_extra_checkpoint_state(loaded_dict, load_optimizer=load_optimizer)
        return infos

    runner_cls.save = save_with_extra_state
    runner_cls.load = load_with_extra_state
    runner_cls._ddt_cenet_checkpoint_patch = True


class PPOWithCENetAdaBoot(PPO):
    """PPO with DreamWaQ CENet losses and adaptive estimator bootstrapping."""

    def __init__(
        self,
        *args,
        cenet_loss_coef: float = 1.0,
        cenet_velocity_loss_coef: float = 1.0,
        cenet_reconstruction_loss_coef: float = 1.0,
        cenet_kl_loss_coef: float = 1.0,
        vae_learning_rate: float | None = None,
        num_vae_substeps: int = 1,
        rl_grad_to_cenet: bool = True,
        adaboot_enabled: bool = True,
        adaboot_reward_window: int = 128,
        adaboot_min_episodes: int = 32,
        adaboot_eps: float = 1.0e-6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cenet_loss_coef = cenet_loss_coef
        self.cenet_velocity_loss_coef = cenet_velocity_loss_coef
        self.cenet_reconstruction_loss_coef = cenet_reconstruction_loss_coef
        self.cenet_kl_loss_coef = cenet_kl_loss_coef
        self.vae_learning_rate = self.learning_rate if vae_learning_rate is None else float(vae_learning_rate)
        self.num_vae_substeps = int(num_vae_substeps)
        if self.num_vae_substeps < 0:
            raise ValueError("num_vae_substeps must be greater than or equal to zero.")
        self.rl_grad_to_cenet = bool(rl_grad_to_cenet)
        self.vae_optimizer = torch.optim.Adam(self.policy.cenet_parameters(), lr=self.vae_learning_rate)
        self.adaboot_enabled = bool(adaboot_enabled)
        self.adaboot_reward_window = int(adaboot_reward_window)
        self.adaboot_min_episodes = int(adaboot_min_episodes)
        self.adaboot_eps = float(adaboot_eps)
        self._recent_episode_returns = deque(maxlen=self.adaboot_reward_window)
        self._episode_returns = None
        self._current_bootstrap_mask = None
        self._current_actor_velocity_code = None
        self.adaboot_probability = 0.0
        self.adaboot_cv = 0.0

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        super().init_storage(training_type, num_envs, num_transitions_per_env, obs, actions_shape)
        sample_tensor = next(iter(obs.values()))
        self._cenet_next_policy_obs = torch.zeros(
            num_transitions_per_env,
            num_envs,
            self.policy.num_actor_obs,
            device=self.device,
            dtype=sample_tensor.dtype,
        )
        self._cenet_bootstrap_masks = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            dtype=torch.bool,
            device=self.device,
        )
        self._cenet_actor_velocity_codes = torch.zeros(
            num_transitions_per_env,
            num_envs,
            self.policy.cenet_velocity_dim,
            device=self.device,
            dtype=sample_tensor.dtype,
        )

    def _compute_cenet_total_loss(self, cenet_losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.cenet_loss_coef * (
            self.cenet_velocity_loss_coef * cenet_losses["velocity"]
            + self.cenet_reconstruction_loss_coef * cenet_losses["reconstruction"]
            + self.cenet_kl_loss_coef * cenet_losses["kl"]
        )

    def _reduce_cenet_parameters(self) -> None:
        params = list(self.policy.cenet_parameters())
        grads = [param.grad.view(-1) for param in params if param.grad is not None]
        if not grads:
            return

        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in params:
            if param.grad is None:
                continue
            numel = param.numel()
            param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
            offset += numel

    def get_extra_checkpoint_state(self) -> dict[str, object]:
        return {
            "vae_optimizer_state_dict": self.vae_optimizer.state_dict(),
            "vae_learning_rate": self.vae_learning_rate,
            "rl_grad_to_cenet": self.rl_grad_to_cenet,
            "num_vae_substeps": self.num_vae_substeps,
        }

    def load_extra_checkpoint_state(self, loaded_dict: dict[str, object], load_optimizer: bool = True) -> None:
        if load_optimizer and "vae_optimizer_state_dict" in loaded_dict:
            self.vae_optimizer.load_state_dict(loaded_dict["vae_optimizer_state_dict"])

    def _ensure_episode_return_buffer(self, num_envs: int, device: torch.device):
        if self._episode_returns is None or self._episode_returns.numel() != num_envs:
            self._episode_returns = torch.zeros(num_envs, device=device)

    def _compute_adaboot_probability(self) -> float:
        if len(self._recent_episode_returns) < self.adaboot_min_episodes:
            self.adaboot_cv = 0.0
            self.adaboot_probability = 0.0
            return self.adaboot_probability

        returns = torch.as_tensor(list(self._recent_episode_returns), device=self.device, dtype=torch.float32)
        mean_abs = returns.mean().abs().clamp_min(self.adaboot_eps)
        cv = returns.std(unbiased=False) / mean_abs
        probability = torch.clamp(1.0 - torch.tanh(cv), min=0.0, max=1.0)
        self.adaboot_cv = float(cv.item())
        self.adaboot_probability = float(probability.item())
        return self.adaboot_probability

    def _sample_bootstrap_mask(self, obs) -> torch.Tensor | None:
        if not self.adaboot_enabled:
            self.adaboot_cv = 0.0
            self.adaboot_probability = 0.0
            return None

        num_envs = obs.batch_size[0] if hasattr(obs, "batch_size") else obs["policy"].shape[0]
        probability = self._compute_adaboot_probability()
        if probability <= 0.0:
            return torch.zeros(num_envs, 1, dtype=torch.bool, device=self.device)
        return torch.rand(num_envs, 1, device=self.device) < probability

    def _update_episode_returns(self, rewards: torch.Tensor, dones: torch.Tensor):
        flat_rewards = rewards.reshape(-1).detach().to(self.device)
        flat_dones = dones.reshape(-1).detach().to(self.device) > 0
        self._ensure_episode_return_buffer(flat_rewards.numel(), flat_rewards.device)
        self._episode_returns += flat_rewards
        if torch.any(flat_dones):
            completed_returns = self._episode_returns[flat_dones].detach().cpu().tolist()
            self._recent_episode_returns.extend(float(value) for value in completed_returns)
            self._episode_returns[flat_dones] = 0.0

    def act(self, obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        self._current_bootstrap_mask = self._sample_bootstrap_mask(obs)
        self.transition.actions = self.policy.act(
            obs,
            bootstrap_mask=self._current_bootstrap_mask,
            detach_cenet=not self.rl_grad_to_cenet,
        ).detach()
        self._current_actor_velocity_code = self.policy.last_actor_velocity_code
        if self._current_actor_velocity_code is not None:
            self._current_actor_velocity_code = self._current_actor_velocity_code.detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        step = self.storage.step
        if step < self.storage.num_transitions_per_env:
            self._cenet_next_policy_obs[step].copy_(self.policy.get_policy_obs(obs).detach())
            if self._current_bootstrap_mask is None:
                bootstrap_mask = torch.zeros(self.storage.num_envs, 1, dtype=torch.bool, device=self.device)
            else:
                bootstrap_mask = self._current_bootstrap_mask.to(device=self.device, dtype=torch.bool)
            self._cenet_bootstrap_masks[step].copy_(bootstrap_mask)
            if self._current_actor_velocity_code is not None:
                self._cenet_actor_velocity_codes[step].copy_(self._current_actor_velocity_code.to(self.device))

        self.policy.update_normalization(obs, bootstrap_mask=self._current_bootstrap_mask)
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        self._update_episode_returns(rewards, dones)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)
        self._current_bootstrap_mask = None
        self._current_actor_velocity_code = None

    def _mini_batch_generator_with_cenet(self):
        batch_size = self.storage.num_envs * self.storage.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        indices = torch.randperm(self.num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.storage.observations.flatten(0, 1)
        actions = self.storage.actions.flatten(0, 1)
        values = self.storage.values.flatten(0, 1)
        returns = self.storage.returns.flatten(0, 1)
        old_actions_log_prob = self.storage.actions_log_prob.flatten(0, 1)
        advantages = self.storage.advantages.flatten(0, 1)
        old_mu = self.storage.mu.flatten(0, 1)
        old_sigma = self.storage.sigma.flatten(0, 1)
        next_policy_obs = self._cenet_next_policy_obs.flatten(0, 1)
        bootstrap_masks = self._cenet_bootstrap_masks.flatten(0, 1)
        actor_velocity_codes = self._cenet_actor_velocity_codes.flatten(0, 1)
        dones = self.storage.dones.flatten(0, 1)

        for _ in range(self.num_learning_epochs):
            for i in range(self.num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield (
                    observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    next_policy_obs[batch_idx],
                    bootstrap_masks[batch_idx],
                    actor_velocity_codes[batch_idx],
                    dones[batch_idx],
                )

    def update(self):  # noqa: C901
        if self.symmetry:
            raise NotImplementedError("PPOWithCENetAdaBoot does not support symmetry augmentation.")

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_cenet_velocity_loss = 0
        mean_cenet_reconstruction_loss = 0
        mean_cenet_kl_loss = 0
        mean_cenet_total_loss = 0
        mean_rnd_loss = 0 if self.rnd else None

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            next_policy_obs_batch,
            bootstrap_mask_batch,
            actor_velocity_code_batch,
            dones_batch,
        ) in self._mini_batch_generator_with_cenet():
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            bootstrap_mask_arg = bootstrap_mask_batch if self.adaboot_enabled else None
            actor_velocity_code_arg = None if self.rl_grad_to_cenet else actor_velocity_code_batch
            self.policy.act(
                obs_batch,
                bootstrap_mask=bootstrap_mask_arg,
                detach_cenet=not self.rl_grad_to_cenet,
                actor_velocity_code=actor_velocity_code_arg,
            )
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            ppo_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = torch.nn.functional.mse_loss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            ppo_loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            self.optimizer.zero_grad()
            if self.rnd_optimizer:
                self.rnd_optimizer.zero_grad()

            cenet_losses = {
                "velocity": torch.zeros((), device=self.device),
                "reconstruction": torch.zeros((), device=self.device),
                "kl": torch.zeros((), device=self.device),
            }
            cenet_total_loss = torch.zeros((), device=self.device)
            if self.num_vae_substeps > 0:
                for _ in range(self.num_vae_substeps):
                    sub_cenet_losses = self.policy.compute_cenet_losses(obs_batch, next_policy_obs_batch, dones_batch)
                    sub_cenet_total_loss = self._compute_cenet_total_loss(sub_cenet_losses)

                    self.vae_optimizer.zero_grad()
                    sub_cenet_total_loss.backward()
                    if self.is_multi_gpu:
                        self._reduce_cenet_parameters()
                    nn.utils.clip_grad_norm_(self.policy.cenet_parameters(), self.max_grad_norm)
                    self.vae_optimizer.step()

                    for key in cenet_losses:
                        cenet_losses[key] = cenet_losses[key] + sub_cenet_losses[key].detach()
                    cenet_total_loss = cenet_total_loss + sub_cenet_total_loss.detach()

                for key in cenet_losses:
                    cenet_losses[key] = cenet_losses[key] / self.num_vae_substeps
                cenet_total_loss = cenet_total_loss / self.num_vae_substeps
                self.vae_optimizer.zero_grad()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_cenet_velocity_loss += cenet_losses["velocity"].item()
            mean_cenet_reconstruction_loss += cenet_losses["reconstruction"].item()
            mean_cenet_kl_loss += cenet_losses["kl"].item()
            mean_cenet_total_loss += cenet_total_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_cenet_velocity_loss /= num_updates
        mean_cenet_reconstruction_loss /= num_updates
        mean_cenet_kl_loss /= num_updates
        mean_cenet_total_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "cenet_velocity": mean_cenet_velocity_loss,
            "cenet_reconstruction": mean_cenet_reconstruction_loss,
            "cenet_kl": mean_cenet_kl_loss,
            "cenet_total": mean_cenet_total_loss,
            "adaboot_probability": self.adaboot_probability,
            "adaboot_cv": self.adaboot_cv,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        return loss_dict


class PPOWithEstimator(PPO):
    """PPO with an auxiliary supervised loss for the history-based estimator."""

    def __init__(self, *args, estimator_loss_coef: float = 1.0, estimator_lr: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator_loss_coef = estimator_loss_coef
        self.estimator_lr = estimator_lr

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        obs_with_estimator_features = {key: value for key, value in obs.items()}
        sample_tensor = next(iter(obs_with_estimator_features.values()))
        obs_with_estimator_features[self.policy.ESTIMATOR_HISTORY_FEATURES_KEY] = torch.zeros(
            sample_tensor.shape[0],
            self.policy.actor_estimator_obs_dim,
            device=sample_tensor.device,
            dtype=sample_tensor.dtype,
        )
        super().init_storage(training_type, num_envs, num_transitions_per_env, obs_with_estimator_features, actions_shape)

    def process_env_step(self, obs, rewards, dones, extras):
        # Reset estimator history before processing the next observation so reset envs start clean.
        self.policy.reset(dones)

        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_estimator_loss = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        mse_loss = torch.nn.MSELoss()

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            estimator_loss = torch.zeros((), device=self.device)
            if self.policy.velocity_target_groups:
                predicted_velocity = self.policy.estimate_from_history(obs_batch)
                target_velocity = self.policy.get_estimator_targets(obs_batch).detach()
                estimator_loss = mse_loss(predicted_velocity, target_velocity)
                loss = loss + self.estimator_loss_coef * estimator_loss

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss = loss + self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = mse_loss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_estimator_loss += estimator_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_estimator_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimator": mean_estimator_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict


class PPOWithEstimatorAdaBoot(PPOWithEstimator):
    """PPOWithEstimator with adaptive velocity-target bootstrapping for the actor input."""

    def __init__(
        self,
        *args,
        adaboot_reward_window: int = 128,
        adaboot_min_episodes: int = 32,
        adaboot_eps: float = 1.0e-6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.adaboot_reward_window = int(adaboot_reward_window)
        self.adaboot_min_episodes = int(adaboot_min_episodes)
        self.adaboot_eps = float(adaboot_eps)
        self._recent_episode_returns = deque(maxlen=self.adaboot_reward_window)
        self._episode_returns = None
        self.adaboot_probability = 0.0
        self.adaboot_cv = 0.0

    def _has_cached_estimator_features(self, obs) -> bool:
        return self.policy.ESTIMATOR_HISTORY_FEATURES_KEY in obs.keys()

    def _ensure_episode_return_buffer(self, num_envs: int, device: torch.device):
        if self._episode_returns is None or self._episode_returns.numel() != num_envs:
            self._episode_returns = torch.zeros(num_envs, device=device)

    def _update_episode_returns(self, rewards: torch.Tensor, dones: torch.Tensor):
        flat_rewards = rewards.reshape(-1).detach().to(self.device)
        flat_dones = dones.reshape(-1).detach().to(self.device) > 0
        self._ensure_episode_return_buffer(flat_rewards.numel(), flat_rewards.device)
        self._episode_returns += flat_rewards
        if torch.any(flat_dones):
            completed_returns = self._episode_returns[flat_dones].detach().cpu().tolist()
            self._recent_episode_returns.extend(float(value) for value in completed_returns)
            self._episode_returns[flat_dones] = 0.0

    def _compute_adaboot_probability(self) -> float:
        if len(self._recent_episode_returns) < self.adaboot_min_episodes:
            self.adaboot_cv = 0.0
            self.adaboot_probability = 0.0
            return self.adaboot_probability

        returns = torch.as_tensor(list(self._recent_episode_returns), device=self.device, dtype=torch.float32)
        mean_abs = returns.mean().abs().clamp_min(self.adaboot_eps)
        cv = returns.std(unbiased=False) / mean_abs
        probability = torch.clamp(1.0 - torch.tanh(cv), min=0.0, max=1.0)
        self.adaboot_cv = float(cv.item())
        self.adaboot_probability = float(probability.item())
        return self.adaboot_probability

    def _sample_estimated_velocity_mask(self, obs) -> torch.Tensor:
        num_envs = obs.batch_size[0] if hasattr(obs, "batch_size") else obs["policy"].shape[0]
        probability = self._compute_adaboot_probability()
        if probability <= 0.0:
            return torch.zeros(num_envs, 1, dtype=torch.bool, device=self.device)
        return torch.rand(num_envs, 1, device=self.device) < probability

    def act(self, obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        estimated_velocity_mask = None
        if not self._has_cached_estimator_features(obs):
            estimated_velocity_mask = self._sample_estimated_velocity_mask(obs)

        self.transition.actions = self.policy.act(obs, bootstrap_mask=estimated_velocity_mask).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        self._update_episode_returns(rewards, dones)

        # Reset estimator history before processing the next observation so reset envs start clean.
        self.policy.reset(dones)
        next_estimated_velocity_mask = None
        if not self._has_cached_estimator_features(obs):
            next_estimated_velocity_mask = self._sample_estimated_velocity_mask(obs)

        self.policy.update_normalization(obs, bootstrap_mask=next_estimated_velocity_mask)
        if self.rnd:
            self.rnd.update_normalization(obs)
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def update(self):
        loss_dict = super().update()
        loss_dict["adaboot_probability"] = self.adaboot_probability
        loss_dict["adaboot_cv"] = self.adaboot_cv
        return loss_dict


def register_rsl_rl_estimator_extensions():
    """Register estimator-aware policy/algo classes into rsl_rl runner namespaces."""
    import rsl_rl.algorithms as rsl_algorithms
    import rsl_rl.modules as rsl_modules
    import rsl_rl.runners.on_policy_runner as on_policy_runner_module

    rsl_modules.ActorCritic = DiagnosticActorCritic
    rsl_modules.ActorCriticWithEstimator = ActorCriticWithEstimator
    rsl_modules.ActorCriticWithCENet = ActorCriticWithCENet
    rsl_algorithms.PPO = PPOWithDiagnostics
    rsl_algorithms.PPOWithEstimator = PPOWithEstimator
    rsl_algorithms.PPOWithEstimatorAdaBoot = PPOWithEstimatorAdaBoot
    rsl_algorithms.PPOWithCENetAdaBoot = PPOWithCENetAdaBoot
    on_policy_runner_module.ActorCritic = DiagnosticActorCritic
    on_policy_runner_module.ActorCriticWithEstimator = ActorCriticWithEstimator
    on_policy_runner_module.ActorCriticWithCENet = ActorCriticWithCENet
    on_policy_runner_module.PPO = PPOWithDiagnostics
    on_policy_runner_module.PPOWithEstimator = PPOWithEstimator
    on_policy_runner_module.PPOWithEstimatorAdaBoot = PPOWithEstimatorAdaBoot
    on_policy_runner_module.PPOWithCENetAdaBoot = PPOWithCENetAdaBoot
    _patch_on_policy_runner_checkpointing(on_policy_runner_module)
