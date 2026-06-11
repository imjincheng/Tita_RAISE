# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _rate_limited_env_diag(env: ManagerBasedEnv, key: str, message: str, period_steps: int = 2000, max_logs: int = 10):
    """Print a diagnostic message at most once every ``period_steps`` for a given key."""
    diag_state = getattr(env, "_ddt_obs_diag_state", None)
    if diag_state is None:
        diag_state = {}
        setattr(env, "_ddt_obs_diag_state", diag_state)

    step = int(getattr(env, "common_step_counter", 0))
    last_step, count = diag_state.get(key, (-period_steps, 0))
    if count >= max_logs or step - last_step < period_steps:
        return

    print(message)
    diag_state[key] = (step, count + 1)


def _terrain_column_names(env: ManagerBasedEnv) -> list[str] | None:
    """Return terrain names indexed by curriculum column, matching Isaac Lab's generator logic."""
    terrain = getattr(getattr(env, "scene", None), "terrain", None)
    terrain_generator_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if terrain_generator_cfg is None or terrain_generator_cfg.sub_terrains is None:
        return None

    sub_terrain_names = list(terrain_generator_cfg.sub_terrains.keys())
    proportions = [terrain_generator_cfg.sub_terrains[name].proportion for name in sub_terrain_names]
    proportion_sum = sum(proportions)
    if len(sub_terrain_names) == 0 or proportion_sum <= 0.0:
        return None

    cumulative_proportions = []
    running_proportion = 0.0
    for proportion in proportions:
        running_proportion += proportion / proportion_sum
        cumulative_proportions.append(running_proportion)
    cumulative_proportions[-1] = 1.0

    terrain_name_by_col = []
    for col in range(terrain_generator_cfg.num_cols):
        col_fraction = col / terrain_generator_cfg.num_cols + 0.001
        sub_terrain_index = next(
            index
            for index, cumulative_proportion in enumerate(cumulative_proportions)
            if col_fraction < cumulative_proportion
        )
        terrain_name_by_col.append(sub_terrain_names[sub_terrain_index])
    return terrain_name_by_col


def _sample_env_terrain_context(env: ManagerBasedEnv, sample_env: int) -> str:
    """Format terrain context for the sampled environment in diagnostic logs."""
    if sample_env < 0:
        return ""

    terrain = getattr(getattr(env, "scene", None), "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if terrain_types is None or sample_env >= terrain_types.shape[0]:
        return ""

    terrain_col = int(terrain_types[sample_env].item())
    terrain_level = int(terrain_levels[sample_env].item()) if terrain_levels is not None else -1
    terrain_name_by_col = _terrain_column_names(env)
    terrain_name = (
        terrain_name_by_col[terrain_col]
        if terrain_name_by_col is not None and 0 <= terrain_col < len(terrain_name_by_col)
        else "unknown"
    )
    return f" terrain={terrain_name} terrain_col={terrain_col} terrain_level={terrain_level}"


def _log_large_obs_term(
    env: ManagerBasedEnv,
    term_name: str,
    tensor: torch.Tensor,
    threshold: float,
    period_steps: int = 24,
    max_logs: int = 20,
):
    """Log raw observation terms when they become non-finite or unusually large."""
    detached = tensor.detach()
    invalid_mask = ~torch.isfinite(detached)
    invalid_count = int(invalid_mask.sum().item())

    finite_values = detached[~invalid_mask]
    if finite_values.numel() > 0:
        max_abs = float(finite_values.abs().max().item())
        min_value = float(finite_values.min().item())
        max_value = float(finite_values.max().item())
    else:
        max_abs = float("nan")
        min_value = float("nan")
        max_value = float("nan")

    if invalid_count == 0 and max_abs <= threshold:
        return

    if detached.ndim == 0:
        sample_env = 0
    else:
        per_env = detached.reshape(detached.shape[0], -1)
        if invalid_count > 0:
            per_env_invalid = invalid_mask.reshape(invalid_mask.shape[0], -1).any(dim=1)
            env_ids = torch.nonzero(per_env_invalid, as_tuple=False).flatten()
            sample_env = int(env_ids[0].item()) if env_ids.numel() > 0 else -1
        else:
            sample_env = int(per_env.abs().max(dim=1).values.argmax().item()) if per_env.shape[0] > 0 else -1

    terrain_context = _sample_env_terrain_context(env, sample_env)
    _rate_limited_env_diag(
        env,
        f"obs_term:{term_name}",
        (
            "[diag][obs_term] "
            f"{term_name} invalid={invalid_count} max_abs={max_abs:.3e} "
            f"min={min_value:.3e} max={max_value:.3e} sample_env={sample_env} "
            f"shape={tuple(detached.shape)}{terrain_context}"
        ),
        period_steps=period_steps,
        max_logs=max_logs,
    )


def _repair_height_scan_from_current_valid(heights: torch.Tensor, invalid_mask: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Repair invalid rays from the nearest valid ray in the same env row.

    Returns:
        The repaired tensor, number of invalid entries replaced from current valid rays,
        and the number of entries still unresolved.
    """
    repaired = heights.clone()
    replaced_from_current = 0
    unresolved = 0

    bad_rows = torch.nonzero(invalid_mask.any(dim=1), as_tuple=False).flatten()
    for row_idx in bad_rows.tolist():
        row_invalid = invalid_mask[row_idx]
        if not torch.any(row_invalid):
            continue

        valid_idx = torch.nonzero(~row_invalid, as_tuple=False).flatten()
        invalid_idx = torch.nonzero(row_invalid, as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            unresolved += int(invalid_idx.numel())
            continue

        distances = (invalid_idx[:, None] - valid_idx[None, :]).abs()
        nearest_valid_idx = valid_idx[distances.argmin(dim=1)]
        repaired[row_idx, invalid_idx] = repaired[row_idx, nearest_valid_idx]
        replaced_from_current += int(invalid_idx.numel())

    return repaired, replaced_from_current, unresolved


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos - asset.data.default_joint_pos
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    joint_pos_rel = joint_pos_rel[:, asset_cfg.joint_ids]
    return joint_pos_rel


def base_lin_vel_xy(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the base linear velocity in the body frame x-y plane."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, :2]


def diag_base_lin_vel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    tensor = isaaclab_mdp.base_lin_vel(env, asset_cfg=asset_cfg)
    _log_large_obs_term(env, "base_lin_vel", tensor, threshold=20.0)
    return tensor


def diag_base_ang_vel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    tensor = isaaclab_mdp.base_ang_vel(env, asset_cfg=asset_cfg)
    _log_large_obs_term(env, "base_ang_vel", tensor, threshold=20.0)
    return tensor


def diag_projected_gravity(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    tensor = isaaclab_mdp.projected_gravity(env, asset_cfg=asset_cfg)
    _log_large_obs_term(env, "projected_gravity", tensor, threshold=2.0)
    return tensor


def diag_joint_pos_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    tensor = isaaclab_mdp.joint_pos_rel(env, asset_cfg=asset_cfg)
    _log_large_obs_term(env, "joint_pos_rel", tensor, threshold=10.0)
    return tensor


def diag_joint_vel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    tensor = isaaclab_mdp.joint_vel(env, asset_cfg=asset_cfg)
    _log_large_obs_term(env, "joint_vel", tensor, threshold=200.0)
    return tensor


def diag_joint_vel_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    tensor = isaaclab_mdp.joint_vel_rel(env, asset_cfg=asset_cfg)
    _log_large_obs_term(env, "joint_vel_rel", tensor, threshold=200.0)
    return tensor


def diag_last_action(env: ManagerBasedEnv) -> torch.Tensor:
    tensor = isaaclab_mdp.last_action(env)
    _log_large_obs_term(env, "last_action", tensor, threshold=20.0)
    return tensor


def processed_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The processed action command after term-specific scaling, offsets, and clipping."""
    if action_name is None:
        tensors = [env.action_manager.get_term(term_name).processed_actions for term_name in env.action_manager.active_terms]
        if len(tensors) == 0:
            return torch.empty((env.num_envs, 0), device=env.device)
        return torch.cat(tensors, dim=-1)
    return env.action_manager.get_term(action_name).processed_actions


def diag_processed_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    tensor = processed_action(env, action_name=action_name)
    _log_large_obs_term(env, "processed_action", tensor, threshold=20.0)
    return tensor


def blended_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """Action observation in policy-action space after FF blending but before scale/offset/clip.

    For FF-enabled joint position terms this returns ``k_fb * raw_action + k_ff * ff_action``.
    For other action terms it falls back to the raw policy action.
    """

    def _get_term_tensor(term_name: str) -> torch.Tensor:
        term = env.action_manager.get_term(term_name)
        if hasattr(term, "blended_actions"):
            return term.blended_actions
        return term.raw_actions

    if action_name is None:
        tensors = [_get_term_tensor(term_name) for term_name in env.action_manager.active_terms]
        if len(tensors) == 0:
            return torch.empty((env.num_envs, 0), device=env.device)
        return torch.cat(tensors, dim=-1)
    return _get_term_tensor(action_name)


def diag_blended_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    tensor = blended_action(env, action_name=action_name)
    _log_large_obs_term(env, "blended_action", tensor, threshold=20.0)
    return tensor


def safe_blended_action(
    env: ManagerBasedEnv,
    action_name: str | None = None,
    threshold: float = 100.0,
) -> torch.Tensor:
    """Return blended action with per-env outlier suppression before it enters observations.

    If an env's blended action is non-finite or exceeds the threshold on any dimension, the whole
    action vector for that env is replaced by the most recent valid cached vector. During cold
    start, if no cached vector is available yet, the fallback is zeros.
    """

    tensor = blended_action(env, action_name=action_name)
    flat = tensor.detach().reshape(tensor.shape[0], -1)
    invalid_mask = ~torch.isfinite(flat)
    per_env_invalid = invalid_mask.any(dim=1)

    finite_flat = torch.where(invalid_mask, torch.zeros_like(flat), flat)
    per_env_max_abs = finite_flat.abs().max(dim=1).values
    per_env_abnormal = per_env_invalid | (per_env_max_abs > threshold)
    if not torch.any(per_env_abnormal):
        cache_store = getattr(env, "_ddt_blended_action_cache", None)
        if cache_store is None:
            cache_store = {}
            setattr(env, "_ddt_blended_action_cache", cache_store)
        cache_key = action_name if action_name is not None else "__all__"
        cache_store[cache_key] = tensor.detach().clone()
        return tensor

    repaired = tensor.clone()
    cache_store = getattr(env, "_ddt_blended_action_cache", None)
    if cache_store is None:
        cache_store = {}
        setattr(env, "_ddt_blended_action_cache", cache_store)
    cache_key = action_name if action_name is not None else "__all__"
    cached_tensor = cache_store.get(cache_key)

    reused_history = 0
    fallback_zero = 0
    abnormal_env_ids = torch.nonzero(per_env_abnormal, as_tuple=False).flatten()
    if cached_tensor is not None and cached_tensor.shape == tensor.shape and cached_tensor.device == tensor.device:
        repaired[abnormal_env_ids] = cached_tensor[abnormal_env_ids]
        reused_history = int(abnormal_env_ids.numel())
    else:
        repaired[abnormal_env_ids] = 0.0
        fallback_zero = int(abnormal_env_ids.numel())

    cache_store[cache_key] = repaired.detach().clone()
    sample_env = int(abnormal_env_ids[0].item()) if abnormal_env_ids.numel() > 0 else -1
    _rate_limited_env_diag(
        env,
        "safe_blended_action_abnormal",
        (
            "[diag][safe_blended_action] "
            f"repaired_envs={int(per_env_abnormal.sum().item())} invalid_envs={int(per_env_invalid.sum().item())} "
            f"over_threshold_envs={int((per_env_max_abs > threshold).sum().item())} threshold={threshold:.1f} "
            f"sample_env={sample_env} sample_max_abs={float(per_env_max_abs[sample_env].item()):.3e} "
            f"reused_history={reused_history} fallback_zero={fallback_zero}"
        ),
        period_steps=24,
        max_logs=50,
    )
    return repaired


def diag_safe_blended_action(
    env: ManagerBasedEnv,
    action_name: str | None = None,
    threshold: float = 100.0,
) -> torch.Tensor:
    tensor = safe_blended_action(env, action_name=action_name, threshold=threshold)
    _log_large_obs_term(env, "safe_blended_action", tensor, threshold=20.0)
    return tensor


def abnormal_blended_action_termination(
    env: ManagerBasedEnv,
    action_name: str | None = None,
    threshold: float = 100.0,
) -> torch.Tensor:
    """Terminate envs whose blended action becomes non-finite or clearly out-of-range.

    The threshold here is intentionally looser than the observation diagnostic threshold. We only
    want to cut off envs that have already left the expected training distribution, not envs that
    merely show moderately large but still recoverable actions.
    """

    tensor = blended_action(env, action_name=action_name)
    flat = tensor.detach().reshape(tensor.shape[0], -1)
    invalid_mask = ~torch.isfinite(flat)
    per_env_invalid = invalid_mask.any(dim=1)

    finite_flat = torch.where(invalid_mask, torch.zeros_like(flat), flat)
    per_env_max_abs = finite_flat.abs().max(dim=1).values
    per_env_done = per_env_invalid | (per_env_max_abs > threshold)

    if torch.any(per_env_done):
        offending_envs = torch.nonzero(per_env_done, as_tuple=False).flatten()
        sample_env = int(offending_envs[0].item())
        invalid_count = int(per_env_invalid.sum().item())
        over_threshold_count = int((per_env_max_abs > threshold).sum().item())
        _rate_limited_env_diag(
            env,
            "termination:abnormal_blended_action",
            (
                "[diag][termination] abnormal_blended_action "
                f"triggered_envs={int(per_env_done.sum().item())} invalid_envs={invalid_count} "
                f"over_threshold_envs={over_threshold_count} threshold={threshold:.1f} "
                f"sample_env={sample_env} sample_max_abs={float(per_env_max_abs[sample_env].item()):.3e}"
            ),
            period_steps=24,
            max_logs=50,
        )

    return per_env_done


def diag_base_lin_vel_xy(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    tensor = base_lin_vel_xy(env, asset_cfg=asset_cfg)
    _log_large_obs_term(env, "base_lin_vel_xy", tensor, threshold=20.0)
    return tensor


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def feet_average_contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the average world-frame contact force of the selected feet over the stored history."""
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    avg_forces = torch.mean(forces_history, dim=1)
    return avg_forces.reshape(avg_forces.shape[0], -1)


def diag_feet_average_contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    tensor = feet_average_contact_force(env, sensor_cfg=sensor_cfg)
    _log_large_obs_term(env, "feet_average_contact_force", tensor, threshold=2000.0)
    return tensor


def safe_height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan with NaN/Inf sanitization and rate-limited diagnostics.

    Invalid rays are repaired using the most recent valid value at the same env/ray position.
    During cold start, if no history is available, invalid rays are filled from the nearest valid
    ray in the same env row. Only if an entire row is invalid with no history do we fall back to 0.
    """
    heights = isaaclab_mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
    cache_store = getattr(env, "_ddt_height_scan_cache", None)
    if cache_store is None:
        cache_store = {}
        setattr(env, "_ddt_height_scan_cache", cache_store)

    cache_key = sensor_cfg.name
    cached_heights = cache_store.get(cache_key)
    if cached_heights is not None:
        if (
            cached_heights.shape != heights.shape
            or cached_heights.device != heights.device
            or cached_heights.dtype != heights.dtype
        ):
            cached_heights = None

    invalid_mask = ~torch.isfinite(heights)
    if not torch.any(invalid_mask):
        cache_store[cache_key] = heights.detach().clone()
        return heights

    repaired = heights.clone()
    invalid_count = int(invalid_mask.sum().item())
    reused_history = 0
    filled_from_current = 0
    unresolved = 0

    if cached_heights is not None:
        cached_valid_mask = torch.isfinite(cached_heights)
        reuse_mask = invalid_mask & cached_valid_mask
        if torch.any(reuse_mask):
            repaired[reuse_mask] = cached_heights[reuse_mask]
            reused_history = int(reuse_mask.sum().item())
        remaining_mask = invalid_mask & ~cached_valid_mask
    else:
        remaining_mask = invalid_mask

    if torch.any(remaining_mask):
        repaired, filled_from_current, unresolved = _repair_height_scan_from_current_valid(repaired, remaining_mask)
        if unresolved > 0:
            repaired = torch.nan_to_num(repaired, nan=0.0, posinf=0.0, neginf=0.0)

    env_ids = torch.nonzero(invalid_mask.any(dim=1), as_tuple=False)
    sample_env = int(env_ids[0].item()) if env_ids.numel() > 0 else -1
    finite_values = heights[~invalid_mask]
    max_abs_finite = float(finite_values.abs().max().item()) if finite_values.numel() > 0 else float("nan")
    step = int(getattr(env, "common_step_counter", 0))
    _rate_limited_env_diag(
        env,
        "safe_height_scan_non_finite",
        (
            "[diag][safe_height_scan] "
            f"repaired {invalid_count} non-finite entries at step={step} "
            f"sample_env={sample_env} reused_history={reused_history} "
            f"filled_from_current={filled_from_current} fallback_zero={unresolved} "
            f"max_abs_finite={max_abs_finite:.3e}"
        ),
    )
    cache_store[cache_key] = repaired.detach().clone()
    return repaired


def diag_joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    tensor = joint_pos_rel_without_wheel(env, asset_cfg=asset_cfg, wheel_asset_cfg=wheel_asset_cfg)
    _log_large_obs_term(env, "joint_pos_rel_without_wheel", tensor, threshold=10.0)
    return tensor
