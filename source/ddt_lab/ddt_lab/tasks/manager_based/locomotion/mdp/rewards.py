# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_heading_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of heading targets using an exponential kernel on heading error."""
    asset: RigidObject = env.scene[asset_cfg.name]
    zeros = torch.zeros(env.num_envs, device=asset.data.heading_w.device)
    if command_name not in env.command_manager.active_terms:
        return zeros

    command_term = env.command_manager.get_term(command_name)
    heading_target = getattr(command_term, "heading_target", None)
    if heading_target is None:
        return zeros

    is_heading_env = getattr(command_term, "is_heading_env", None)
    heading_error = math_utils.wrap_to_pi(heading_target - asset.data.heading_w)
    reward = torch.exp(-torch.square(heading_error) / std**2)
    if is_heading_env is not None:
        reward *= is_heading_env.float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actual_velocity_threshold: float | None = None,
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= _zero_command_mask(
        env,
        command_name,
        command_threshold,
        actual_velocity_threshold=actual_velocity_threshold,
        asset_cfg=asset_cfg,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _zero_command_mask(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    actual_velocity_threshold: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    cmd_still = torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    if actual_velocity_threshold is None:
        return cmd_still

    asset: RigidObject = env.scene[asset_cfg.name]
    actual_still = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1) < actual_velocity_threshold
    return cmd_still & actual_still


def _upright_scale(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


def zero_command_base_ang_vel_z_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actual_velocity_threshold: float | None = None,
) -> torch.Tensor:
    """Penalize yaw rotation only when the commanded base velocity is near zero."""
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_ang_vel_b[:, 2])
    reward *= _zero_command_mask(
        env,
        command_name,
        command_threshold,
        actual_velocity_threshold=actual_velocity_threshold,
        asset_cfg=asset_cfg,
    )
    reward *= _upright_scale(env)
    return reward


def zero_command_base_lin_vel_xy_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actual_velocity_threshold: float | None = None,
) -> torch.Tensor:
    """Penalize planar base drift only when the commanded base velocity is near zero."""
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    reward *= _zero_command_mask(
        env,
        command_name,
        command_threshold,
        actual_velocity_threshold=actual_velocity_threshold,
        asset_cfg=asset_cfg,
    )
    reward *= _upright_scale(env)
    return reward


def zero_command_wheel_vel_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actual_velocity_threshold: float | None = None,
) -> torch.Tensor:
    """Penalize wheel spin only when the commanded base velocity is near zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    reward *= _zero_command_mask(
        env,
        command_name,
        command_threshold,
        actual_velocity_threshold=actual_velocity_threshold,
        asset_cfg=asset_cfg,
    )
    reward *= _upright_scale(env)
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ff_target_pos_exp(
    env: ManagerBasedRLEnv,
    std: float,
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_ff_abs: float = 2.0e-2,
) -> torch.Tensor:
    """Early shaping reward for CTBC-style FF lifting.

    This reward only tells the policy that the FF-induced early lifting posture is useful.
    It is not intended as a long-term imitation objective.

    Key design:
    - target: full FF teacher target, q0 + a_ff
    - active mask: based on raw ff_actions, not k_ff-scaled ff_contribution
    - weight: decays with k_ff / initial_k_ff
    """

    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.data.joint_pos.device
    zeros = torch.zeros(env.num_envs, device=device)

    if action_name not in env.action_manager.active_terms:
        return zeros

    action_term = env.action_manager.get_term(action_name)

    ff_target_positions = getattr(action_term, "ff_target_positions", None)
    ff_joint_local_ids = getattr(action_term, "ff_joint_local_ids", None)
    controlled_joint_ids = getattr(action_term, "controlled_joint_ids", None)
    ff_actions = getattr(action_term, "ff_actions", None)

    current_k_ff = float(getattr(action_term, "k_ff", 0.0))
    initial_k_ff = float(getattr(action_term, "initial_k_ff", 0.0))

    if (
        ff_target_positions is None
        or ff_joint_local_ids is None
        or controlled_joint_ids is None
        or ff_actions is None
        or current_k_ff <= 0.0
        or initial_k_ff <= 0.0
    ):
        return zeros

    ff_joint_local_ids = ff_joint_local_ids.to(device=device, dtype=torch.long)
    controlled_joint_ids = controlled_joint_ids.to(device=device, dtype=torch.long)

    if ff_joint_local_ids.numel() == 0:
        return zeros

    ff_joint_global_ids = controlled_joint_ids[ff_joint_local_ids]

    current_pos = asset.data.joint_pos[:, ff_joint_global_ids]
    target_pos = ff_target_positions[:, ff_joint_local_ids]

    # Use unscaled FF action as the active mask.
    # Do not use ff_contribution here, otherwise k_ff affects both mask and reward weight.
    ff_active_mask = torch.abs(ff_actions[:, ff_joint_local_ids]) > min_ff_abs

    active_joint_count = ff_active_mask.sum(dim=1)
    active_env_mask = active_joint_count > 0

    if not torch.any(active_env_mask):
        return zeros

    sq_err = torch.square(current_pos - target_pos) * ff_active_mask.float()
    mean_sq_err = sq_err.sum(dim=1) / active_joint_count.clamp(min=1).float()

    reward = torch.exp(-mean_sq_err / (std * std)) * active_env_mask.float()

    # Since FF is only an early lifting hint, this decay should stay.
    ff_weight = current_k_ff / max(initial_k_ff, 1.0e-6)
    reward *= ff_weight

    # Keep upright gate: avoid rewarding fallen states.
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    reward *= upright

    return reward


def _get_lifting_state(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    lifting_state = None
    if action_name in env.action_manager.active_terms:
        action_term = env.action_manager.get_term(action_name)
        lifting_state = getattr(action_term, "lifting_state", None)
    if lifting_state is None:
        from .events import get_feedforward_lifting_state

        lifting_state = get_feedforward_lifting_state(env)
    return lifting_state.bool()


def joint_deviation_l2_no_lift(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize default-pose deviation while moving only when no leg is in a triggered lift phase."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(joint_error), dim=1)

    command_active = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > command_threshold
    no_lift = ~_get_lifting_state(env, action_name=action_name).any(dim=1)
    reward *= command_active & no_lift
    reward *= _upright_scale(env)
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


def wheel_spin_penalty(
    env: ManagerBasedRLEnv,
    wheel_joint_cfg: SceneEntityCfg,
    foot_body_cfg: SceneEntityCfg,
    wheel_radius: float = 0.0925,
    spin_scale: float = 0.8,
    slip_deadband: float = 0.1,
) -> torch.Tensor:
    """Penalize wheel surface speed that exceeds the corresponding foot link world-frame speed."""
    asset: Articulation = env.scene[wheel_joint_cfg.name]
    foot_asset: Articulation = env.scene[foot_body_cfg.name]

    wheel_surface_speed = wheel_radius * torch.abs(asset.data.joint_vel[:, wheel_joint_cfg.joint_ids])
    foot_speed = torch.linalg.norm(foot_asset.data.body_lin_vel_w[:, foot_body_cfg.body_ids, :2], dim=2)
    slip = torch.clamp(spin_scale * wheel_surface_speed - foot_speed - slip_deadband, min=0.0)
    return torch.sum(slip, dim=1)


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def stair_joint_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Mirror penalty that is disabled while either leg is in a triggered lift phase."""
    reward = joint_mirror(env, asset_cfg, mirror_joints)
    lifting_state = _get_lifting_state(env, action_name=action_name)
    return reward.masked_fill(lifting_state.bool().any(dim=1), 0.0)


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    triggered_only: bool = False,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward_per_foot = (last_air_time - threshold) * first_contact
    if triggered_only:
        lifting_mask = _get_lifting_state(env, action_name=action_name)[:, : reward_per_foot.shape[1]]
        reward_per_foot = reward_per_foot * lifting_mask.float()
    reward = torch.sum(reward_per_foot, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _terrain_height_under_bodies(
    env: ManagerBasedRLEnv,
    body_pos_w: torch.Tensor,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Estimate the terrain height under each body from the nearest valid ray-cast hit."""
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_hits = sensor.data.ray_hits_w
    ray_xy = ray_hits[..., :2]
    ray_z = ray_hits[..., 2]

    valid_hits = torch.isfinite(ray_z)
    distances = torch.sum((body_pos_w[..., None, :2] - ray_xy[:, None, :, :]) ** 2, dim=-1)
    distances = torch.where(valid_hits[:, None, :], distances, torch.full_like(distances, float("inf")))

    nearest_ids = torch.argmin(distances, dim=-1)
    gathered_ground_z = torch.gather(ray_z, 1, nearest_ids)
    has_valid_hit = valid_hits.any(dim=1, keepdim=True).expand_as(gathered_ground_z)

    return torch.where(has_valid_hit, gathered_ground_z, body_pos_w[..., 2])


def feet_height_relative(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Penalize swinging feet that fail to reach a target height above the local terrain."""
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    ground_height = _terrain_height_under_bodies(env, feet_pos_w, sensor_cfg)
    clearance = feet_pos_w[:, :, 2] - ground_height

    foot_z_target_error = torch.square(clearance - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_band_relative(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    wheel_radius: float = 0.0925,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Reward triggered lifting feet for matching a target clearance above local terrain."""
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    ground_height = _terrain_height_under_bodies(env, feet_pos_w, sensor_cfg)
    clearance = feet_pos_w[:, :, 2] - ground_height - wheel_radius

    lifting_state = _get_lifting_state(env, action_name=action_name)[:, : clearance.shape[1]]
    lifting_mask = lifting_state.float()
    active_count = lifting_mask.sum(dim=1)
    active_env_mask = active_count > 0
    height_error = torch.square(clearance - target_height) * lifting_mask

    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    mean_sq_err = height_error.sum(dim=1) / active_count.clamp(min=1.0)
    velocity_gate = (foot_velocity_tanh * lifting_mask).sum(dim=1) / active_count.clamp(min=1.0)
    reward = torch.exp(-mean_sq_err / std**2) * velocity_gate * active_env_mask.float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        base_pos_w = asset.data.root_pos_w[:, None, :]
        ground_height = _terrain_height_under_bodies(env, base_pos_w, sensor_cfg).squeeze(1)
        adjusted_target_height = target_height + ground_height
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_number(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    action_name: str = "joint_pos",
    mismatch_penalty: float = 1.3,
    contact_threshold: float = 10.0,
) -> torch.Tensor:
    """Reward contact state matching the expected stance/swing phase."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    actual_contact = net_forces[..., 2] > contact_threshold

    lifting_state = _get_lifting_state(env, action_name=action_name)[:, : net_forces.shape[1]]
    expected_stance = ~lifting_state.bool()

    match = actual_contact == expected_stance
    mismatch = ~match
    score = match.float().sum(dim=1) - mismatch_penalty * mismatch.float().sum(dim=1)

    double_stance_match = expected_stance.all(dim=1) & actual_contact.all(dim=1)
    double_swing_match = (~expected_stance).all(dim=1) & (~actual_contact).all(dim=1)
    zero_reward_mask = double_stance_match | double_swing_match
    return torch.where(
        zero_reward_mask,
        torch.zeros(env.num_envs, device=net_forces.device),
        score,
    )


def _left_right_contact_force_local_ids(contact_sensor: ContactSensor, sensor_cfg: SceneEntityCfg) -> list[int]:
    """Return local force indices ordered as [left, right] for the selected bodies."""
    body_ids = sensor_cfg.body_ids
    if isinstance(body_ids, slice):
        selected_body_ids = list(range(len(contact_sensor.body_names)))[body_ids]
    else:
        selected_body_ids = [int(body_id) for body_id in body_ids]

    selected_names = [contact_sensor.body_names[body_id] for body_id in selected_body_ids]
    left_ids = [i for i, name in enumerate(selected_names) if "left" in name.lower()]
    right_ids = [i for i, name in enumerate(selected_names) if "right" in name.lower()]
    if len(left_ids) != 1 or len(right_ids) != 1:
        raise ValueError(
            "Expected exactly one left and one right contact body for phase matching, "
            f"got {selected_names}."
        )
    return [left_ids[0], right_ids[0]]


def feet_xy_swing_fz_stance_match(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    action_name: str = "joint_pos",
    mismatch_penalty: float = 1.3,
    swing_xy_threshold: float = 50.0,
    stance_fz_threshold: float = 50.0,
) -> torch.Tensor:
    """Reward expected swing legs by low XY force and expected stance legs by high Z force."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    left_right_ids = _left_right_contact_force_local_ids(contact_sensor, sensor_cfg)
    net_forces = net_forces[:, left_right_ids, :]

    xy_force = torch.linalg.norm(net_forces[..., :2], dim=-1)
    actual_swing = xy_force < swing_xy_threshold
    actual_stance = net_forces[..., 2] > stance_fz_threshold

    lifting_state = _get_lifting_state(env, action_name=action_name)[:, : net_forces.shape[1]]
    expected_swing = lifting_state.bool()

    match = torch.where(expected_swing, actual_swing, actual_stance)
    mismatch = ~match
    score = match.float().sum(dim=1) - mismatch_penalty * mismatch.float().sum(dim=1)

    active_lift = expected_swing.any(dim=1)
    return torch.where(
        active_lift,
        score,
        torch.zeros(env.num_envs, device=net_forces.device),
    )


def feet_swing_xy_impact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    action_name: str = "joint_pos",
    xy_force_threshold: float = 50.0,
) -> torch.Tensor:
    """Penalize triggered swing feet that keep pushing into obstacles with high XY force."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    left_right_ids = _left_right_contact_force_local_ids(contact_sensor, sensor_cfg)
    net_forces = net_forces[:, left_right_ids, :]

    lifting_state = _get_lifting_state(env, action_name=action_name)[:, : net_forces.shape[1]]
    swing_mask = lifting_state.float()

    xy_force = torch.linalg.norm(net_forces[..., :2], dim=-1)
    impact = torch.clamp(xy_force - xy_force_threshold, min=0.0)
    penalty = torch.sum(torch.square(impact) * swing_mask, dim=1)

    active_lift = lifting_state.bool().any(dim=1)
    return torch.where(
        active_lift,
        penalty,
        torch.zeros(env.num_envs, device=net_forces.device),
    )


def _get_recently_reset_mask(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Return env ids that are at the beginning of a new episode."""
    if not hasattr(env, "episode_length_buf"):
        return None
    reset_mask = env.episode_length_buf <= 1
    return reset_mask if torch.any(reset_mask) else None


def _sanitize_action_delta(delta: torch.Tensor, clamp_value: float | None = None) -> torch.Tensor:
    """Clamp and sanitize action deltas to avoid a few outliers dominating the reward."""
    if clamp_value is None:
        return delta
    delta = torch.nan_to_num(delta, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
    return torch.clamp(delta, min=-clamp_value, max=clamp_value)


def _get_action_term_slice(
    env: ManagerBasedRLEnv, action_name: str
) -> tuple[slice | None, object | None]:
    """Resolve the slice occupied by an action term inside the concatenated action tensor."""
    start = 0
    for name, dim in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
        end = start + dim
        if name == action_name:
            return slice(start, end), env.action_manager.get_term(name)
        start = end
    return None, None


def _get_action_term_history(
    env: ManagerBasedRLEnv,
    action_name: str,
    scale_with_term: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Fetch current and previous actions for a single action term.

    When ``scale_with_term`` is enabled, the returned actions are mapped into the term's processed
    command space using the term's internal scale. This is useful for wheel actions where the policy
    outputs are normalized but the deployed command magnitude is much larger.
    """
    action_slice, action_term = _get_action_term_slice(env, action_name)
    if action_slice is None or action_term is None:
        return None, None

    current = env.action_manager.action[:, action_slice]
    prev = env.action_manager.prev_action[:, action_slice]

    if scale_with_term:
        action_scale = getattr(action_term, "_scale", None)
        if action_scale is not None:
            current = current * action_scale
            prev = prev * action_scale

    return current, prev


def _prepare_action_smooth_buffers(env: ManagerBasedRLEnv, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize and reset the internal history used by the smoothness penalty."""
    if (
        not hasattr(env, "_action_smooth_prev")
        or not hasattr(env, "_action_smooth_prev_prev")
        or env._action_smooth_prev.shape != current.shape
        or env._action_smooth_prev_prev.shape != current.shape
    ):
        env._action_smooth_prev = current.clone()
        env._action_smooth_prev_prev = env.action_manager.prev_action.clone()

    reset_mask = _get_recently_reset_mask(env)
    if reset_mask is not None:
        env._action_smooth_prev[reset_mask] = current[reset_mask]
        env._action_smooth_prev_prev[reset_mask] = current[reset_mask]

    return env._action_smooth_prev, env._action_smooth_prev_prev


def _prepare_action_smooth_buffers_for_term(
    env: ManagerBasedRLEnv,
    current: torch.Tensor,
    prev: torch.Tensor,
    buffer_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize and reset term-specific history used by the smoothness penalty."""
    if not hasattr(env, "_action_smooth_prev_by_term") or not isinstance(env._action_smooth_prev_by_term, dict):
        env._action_smooth_prev_by_term = {}
    if not hasattr(env, "_action_smooth_prev_prev_by_term") or not isinstance(
        env._action_smooth_prev_prev_by_term, dict
    ):
        env._action_smooth_prev_prev_by_term = {}

    prev_buffers = env._action_smooth_prev_by_term
    prev_prev_buffers = env._action_smooth_prev_prev_by_term

    if (
        buffer_key not in prev_buffers
        or buffer_key not in prev_prev_buffers
        or prev_buffers[buffer_key].shape != current.shape
        or prev_prev_buffers[buffer_key].shape != current.shape
    ):
        prev_buffers[buffer_key] = current.clone()
        prev_prev_buffers[buffer_key] = prev.clone()

    reset_mask = _get_recently_reset_mask(env)
    if reset_mask is not None:
        prev_buffers[buffer_key][reset_mask] = current[reset_mask]
        prev_prev_buffers[buffer_key][reset_mask] = current[reset_mask]

    return prev_buffers[buffer_key], prev_prev_buffers[buffer_key]


def action_smooth(env: ManagerBasedRLEnv, clamp_value: float | None = None) -> torch.Tensor:
    """Penalize non-smooth actions with a second-order finite difference.

    The history buffers are reset for environments that just restarted to avoid mixing pre-reset
    and post-reset actions in the second-order difference.
    """
    current = env.action_manager.action

    prev, prev_prev = _prepare_action_smooth_buffers(env, current)
    second_diff = _sanitize_action_delta(current - 2 * prev + prev_prev, clamp_value=clamp_value)
    reward = torch.sum(torch.square(second_diff), dim=1)

    env._action_smooth_prev_prev = prev.clone()
    env._action_smooth_prev = current.clone()
    return reward


def action_rate_l2_safe(env: ManagerBasedRLEnv, clamp_value: float = 5.0) -> torch.Tensor:
    """Penalize action-rate changes while clipping rare spikes from dominating the batch."""
    current = env.action_manager.action
    prev = env.action_manager.prev_action

    reset_mask = _get_recently_reset_mask(env)
    if reset_mask is not None:
        prev = prev.clone()
        prev[reset_mask] = current[reset_mask]

    delta = _sanitize_action_delta(current - prev, clamp_value=clamp_value)
    return torch.sum(torch.square(delta), dim=1)


def action_smooth_safe(env: ManagerBasedRLEnv, clamp_value: float = 5.0) -> torch.Tensor:
    """Reset-safe and clipped variant of the second-order action smoothness penalty."""
    return action_smooth(env, clamp_value=clamp_value)


def action_rate_l2_for_term_safe(
    env: ManagerBasedRLEnv,
    action_name: str,
    clamp_value: float = 5.0,
    scale_with_term: bool = True,
) -> torch.Tensor:
    """Penalize action-rate changes for a single action term.

    This is useful for selectively damping wheel actions without over-regularizing the leg joints.
    """
    current, prev = _get_action_term_history(env, action_name=action_name, scale_with_term=scale_with_term)
    if current is None or prev is None:
        return torch.zeros(env.num_envs, device=env.action_manager.action.device)

    reset_mask = _get_recently_reset_mask(env)
    if reset_mask is not None:
        prev = prev.clone()
        prev[reset_mask] = current[reset_mask]

    delta = _sanitize_action_delta(current - prev, clamp_value=clamp_value)
    return torch.sum(torch.square(delta), dim=1)


def action_smooth_for_term_safe(
    env: ManagerBasedRLEnv,
    action_name: str,
    clamp_value: float = 5.0,
    scale_with_term: bool = True,
) -> torch.Tensor:
    """Reset-safe second-order smoothness penalty for a single action term."""
    current, prev = _get_action_term_history(env, action_name=action_name, scale_with_term=scale_with_term)
    if current is None or prev is None:
        return torch.zeros(env.num_envs, device=env.action_manager.action.device)

    buffer_key = f"{action_name}:{'scaled' if scale_with_term else 'raw'}"
    prev_buffer, prev_prev_buffer = _prepare_action_smooth_buffers_for_term(env, current, prev, buffer_key)
    second_diff = _sanitize_action_delta(current - 2 * prev_buffer + prev_prev_buffer, clamp_value=clamp_value)
    reward = torch.sum(torch.square(second_diff), dim=1)

    env._action_smooth_prev_prev_by_term[buffer_key] = prev_buffer.clone()
    env._action_smooth_prev_by_term[buffer_key] = current.clone()
    return reward


def action_smooth_for_term_indices_safe(
    env: ManagerBasedRLEnv,
    action_name: str,
    action_indices: Sequence[int],
    clamp_value: float = 5.0,
    scale_with_term: bool = False,
) -> torch.Tensor:
    """Reset-safe second-order smoothness penalty for selected dimensions of one action term."""
    current, prev = _get_action_term_history(env, action_name=action_name, scale_with_term=scale_with_term)
    if current is None or prev is None:
        return torch.zeros(env.num_envs, device=env.action_manager.action.device)

    indices = tuple(int(index) for index in action_indices)
    if len(indices) == 0:
        return torch.zeros(env.num_envs, device=current.device)
    if min(indices) < 0 or max(indices) >= current.shape[1]:
        raise ValueError(
            f"Action indices {indices} are out of range for action term '{action_name}' "
            f"with dimension {current.shape[1]}."
        )

    index_tensor = torch.tensor(indices, device=current.device, dtype=torch.long)
    current = current.index_select(1, index_tensor)
    prev = prev.index_select(1, index_tensor)

    buffer_key = f"{action_name}:{indices}:{'scaled' if scale_with_term else 'raw'}"
    prev_buffer, prev_prev_buffer = _prepare_action_smooth_buffers_for_term(env, current, prev, buffer_key)
    second_diff = _sanitize_action_delta(current - 2 * prev_buffer + prev_prev_buffer, clamp_value=clamp_value)
    reward = torch.sum(torch.square(second_diff), dim=1)

    env._action_smooth_prev_prev_by_term[buffer_key] = prev_buffer.clone()
    env._action_smooth_prev_by_term[buffer_key] = current.clone()
    return reward


def opposite_base_vel(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base velocity opposite to the commanded x direction."""
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_cmd = env.command_manager.get_command(command_name)[:, 0]
    vel_actual = asset.data.root_lin_vel_b[:, 0]
    return torch.clamp(-torch.sign(vel_cmd) * vel_actual, min=0.0)


def opposite_wheel_vel(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize wheel velocity opposite to the commanded x direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_cmd = env.command_manager.get_command(command_name)[:, 0]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    sign_cmd = torch.sign(vel_cmd).unsqueeze(-1)
    opposite_penalty = torch.clamp(-sign_cmd * wheel_vel, min=0.0)
    return torch.sum(opposite_penalty, dim=1)


def feet_x_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize x-axis asymmetry between the two feet in the body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    base_pos_w = asset.data.root_pos_w
    base_quat_w = asset.data.root_quat_w

    feet_pos_rel = feet_pos_w - base_pos_w.unsqueeze(1)
    feet_pos_b = torch.zeros_like(feet_pos_rel)
    for i in range(feet_pos_rel.shape[1]):
        feet_pos_b[:, i, :] = quat_apply_inverse(base_quat_w, feet_pos_rel[:, i, :])

    return torch.abs(feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0])


def feet_y_distance(
    env: ManagerBasedRLEnv,
    min_distance: float = 0.3,
    max_distance: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize stance width outside the desired [min_distance, max_distance] range."""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    base_pos_w = asset.data.root_pos_w
    base_quat_w = asset.data.root_quat_w

    feet_pos_rel = feet_pos_w - base_pos_w.unsqueeze(1)
    feet_pos_b = torch.zeros_like(feet_pos_rel)
    for i in range(feet_pos_rel.shape[1]):
        feet_pos_b[:, i, :] = quat_apply_inverse(base_quat_w, feet_pos_rel[:, i, :])

    y_dist = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    penalty_min = torch.clamp(min_distance - y_dist, min=0.0)
    penalty_max = torch.clamp(y_dist - max_distance, min=0.0)
    return penalty_min + penalty_max
