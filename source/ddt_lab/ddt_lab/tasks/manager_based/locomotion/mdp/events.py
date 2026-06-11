# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_feedforward_modifiers: dict[int, object] = {}


def _resolve_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)[env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def reset_joints_to_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_positions: list[float] | tuple[float, ...],
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_soft_joint_pos_limits: bool = True,
):
    """Reset selected joints to fixed absolute positions while keeping the asset defaults unchanged."""
    asset: Articulation = env.scene[asset_cfg.name]

    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    target_joint_pos = torch.tensor(joint_positions, dtype=joint_pos.dtype, device=joint_pos.device).view(1, -1)
    if target_joint_pos.shape[-1] != joint_pos.shape[-1]:
        raise ValueError(f"Expected {joint_pos.shape[-1]} joint positions, got {target_joint_pos.shape[-1]}.")
    joint_pos = target_joint_pos.expand_as(joint_pos).clone()

    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    joint_pos_limit_source = asset.data.soft_joint_pos_limits if use_soft_joint_pos_limits else asset.data.joint_pos_limits
    joint_pos_limits = joint_pos_limit_source[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_joints_by_offset_per_joint(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_ranges: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_soft_joint_pos_limits: bool = True,
):
    """Reset selected joints around default positions using per-joint offset ranges."""
    asset: Articulation = env.scene[asset_cfg.name]

    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    if len(position_ranges) != joint_pos.shape[-1]:
        raise ValueError(f"Expected {joint_pos.shape[-1]} joint offset ranges, got {len(position_ranges)}.")

    range_lows = torch.tensor(
        [offset_range[0] for offset_range in position_ranges],
        dtype=joint_pos.dtype,
        device=joint_pos.device,
    ).view(1, -1)
    range_highs = torch.tensor(
        [offset_range[1] for offset_range in position_ranges],
        dtype=joint_pos.dtype,
        device=joint_pos.device,
    ).view(1, -1)
    joint_pos += range_lows + torch.rand_like(joint_pos) * (range_highs - range_lows)

    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    joint_pos_limit_source = asset.data.soft_joint_pos_limits if use_soft_joint_pos_limits else asset.data.joint_pos_limits
    joint_pos_limits = joint_pos_limit_source[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_joints_to_positions_by_standing_command(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    standing_joint_positions: list[float] | tuple[float, ...],
    moving_joint_positions: list[float] | tuple[float, ...],
    moving_position_range: tuple[float, float],
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_soft_joint_pos_limits: bool = True,
):
    """Reset standing-command envs to fixed defaults and other envs to a scaled moving pose."""
    env_ids_tensor = _resolve_env_ids(env, env_ids)
    if env_ids_tensor.numel() == 0:
        return

    command_term = None
    if command_name in env.command_manager.active_terms:
        command_term = env.command_manager.get_term(command_name)
        presample_for_reset = getattr(command_term, "presample_for_reset", None)
        if presample_for_reset is not None:
            presample_for_reset(env_ids_tensor)

    is_standing_env = getattr(command_term, "is_standing_env", None)
    if is_standing_env is None:
        standing_mask = torch.zeros(env_ids_tensor.shape, dtype=torch.bool, device=env_ids_tensor.device)
    else:
        standing_mask = is_standing_env[env_ids_tensor].to(device=env_ids_tensor.device)

    standing_env_ids = env_ids_tensor[standing_mask]
    moving_env_ids = env_ids_tensor[~standing_mask]

    if standing_env_ids.numel() > 0:
        reset_joints_to_positions(
            env=env,
            env_ids=standing_env_ids,
            joint_positions=standing_joint_positions,
            velocity_range=velocity_range,
            asset_cfg=asset_cfg,
            use_soft_joint_pos_limits=use_soft_joint_pos_limits,
        )
    if moving_env_ids.numel() > 0:
        reset_joints_to_positions_by_scale(
            env=env,
            env_ids=moving_env_ids,
            joint_positions=moving_joint_positions,
            position_range=moving_position_range,
            velocity_range=velocity_range,
            asset_cfg=asset_cfg,
            use_soft_joint_pos_limits=use_soft_joint_pos_limits,
        )


def reset_joints_to_positions_by_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_positions: list[float] | tuple[float, ...],
    position_range: tuple[float, float],
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_soft_joint_pos_limits: bool = True,
):
    """Reset joints like ``reset_joints_by_scale`` but use provided reference joint positions."""
    asset: Articulation = env.scene[asset_cfg.name]

    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    target_joint_pos = torch.tensor(joint_positions, dtype=joint_pos.dtype, device=joint_pos.device).view(1, -1)
    if target_joint_pos.shape[-1] != joint_pos.shape[-1]:
        raise ValueError(f"Expected {joint_pos.shape[-1]} joint positions, got {target_joint_pos.shape[-1]}.")

    joint_pos = target_joint_pos.expand_as(joint_pos).clone()
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)

    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    joint_pos_limit_source = asset.data.soft_joint_pos_limits if use_soft_joint_pos_limits else asset.data.joint_pos_limits
    joint_pos_limits = joint_pos_limit_source[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


# def randomize_rigid_body_com(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor | None,
#     com_range: dict[str, tuple[float, float]],
#     asset_cfg: SceneEntityCfg,
# ):
#     """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

#     .. note::
#         This function uses CPU tensors to assign the CoM. It is recommended to use this function
#         only during the initialization of the environment.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # resolve environment ids
#     if env_ids is None:
#         env_ids = torch.arange(env.scene.num_envs, device="cpu")
#     else:
#         env_ids = env_ids.cpu()

#     # resolve body indices
#     if asset_cfg.body_ids == slice(None):
#         body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
#     else:
#         body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

#     # sample random CoM values
#     range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
#     ranges = torch.tensor(range_list, device="cpu")
#     rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

#     # get the current com of the bodies (num_assets, num_bodies)
#     coms = asset.root_physx_view.get_coms().clone()

#     # Randomize the com in range
#     coms[:, body_ids, :3] += rand_samples

#     # Set the new coms
#     asset.root_physx_view.set_coms(coms, env_ids)


# def randomize_com_positions(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor | None,
#     asset_cfg: SceneEntityCfg,
#     com_distribution_params: tuple[float, float],
#     operation: Literal["add", "scale", "abs"],
#     distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
# ):
#     """Randomize the center of mass (COM) positions for the rigid bodies.

#     This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
#     randomized by adding, scaling, or setting random values sampled from the specified distribution.

#     .. tip::
#         This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

#     Args:
#         env (ManagerBasedEnv): The simulation environment.
#         env_ids (torch.Tensor | None): Specific environment indices to apply randomization, or None for all environments.
#         asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
#         com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
#         operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
#         distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
#     """
#     # Extract the asset (Articulation or RigidObject)
#     asset: RigidObject | Articulation = env.scene[asset_cfg.name]

#     # Resolve environment indices
#     if env_ids is None:
#         env_ids = torch.arange(env.scene.num_envs, device="cpu")
#     else:
#         env_ids = env_ids.cpu()

#     # Resolve body indices
#     if asset_cfg.body_ids == slice(None):
#         body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
#     else:
#         body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

#     # Get the current COM offsets (num_assets, num_bodies, 3)
#     com_offsets = asset.root_physx_view.get_coms()

#     for dim_idx in range(3):  # Randomize x, y, z independently
#         randomized_offset = _randomize_prop_by_op(
#             com_offsets[:, :, dim_idx],
#             com_distribution_params,
#             env_ids,
#             body_ids,
#             operation,
#             distribution,
#         )
#         com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

#     # Set the randomized COM offsets into the simulation
#     asset.root_physx_view.set_coms(com_offsets, env_ids)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def apply_feedforward_trajectory(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_sensor_name: str = "contact_forces",
    contact_body_pattern: str = ".*_leg_4",
    feedforward_joint_names: list[str] | None = None,
    feedforward_amplitude: dict[str, float] | None = None,
    feedforward_period: float = 0.6,
    k_ff: float = 0.3,
    contact_force_threshold: float = 50.0,
    followup_trigger_delay_factor: float = 0.0,
):
    """Apply a contact-triggered feedforward trajectory modifier on every step."""
    from .feedforward_modifier import FeedforwardModifier

    env_key = id(env)
    if env_key not in _feedforward_modifiers:
        _feedforward_modifiers[env_key] = FeedforwardModifier(
            env=env,
            asset_cfg=asset_cfg,
            contact_sensor_name=contact_sensor_name,
            contact_body_pattern=contact_body_pattern,
            feedforward_joint_names=feedforward_joint_names,
            feedforward_amplitude=feedforward_amplitude,
            feedforward_period=feedforward_period,
            k_ff=k_ff,
            contact_force_threshold=contact_force_threshold,
            followup_trigger_delay_factor=followup_trigger_delay_factor,
        )
    _feedforward_modifiers[env_key].update()


def reset_feedforward_modifier(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
):
    """Reset the feedforward modifier state on episode reset."""
    env_key = id(env)
    if env_key in _feedforward_modifiers:
        _feedforward_modifiers[env_key].reset(env_ids)


def get_feedforward_lifting_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the current feedforward lifting state for the two legs."""
    env_key = id(env)
    if env_key in _feedforward_modifiers:
        return _feedforward_modifiers[env_key].lifting_state
    return torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
