from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ResetAwareUniformVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command that can be sampled before reset events run.

    ManagerBasedRLEnv applies reset events before ``command_manager.reset()``. This command lets
    reset events pre-sample the command so reset logic can use the same standing-env mask that will
    remain active for the new episode.
    """

    cfg: "ResetAwareUniformVelocityCommandCfg"

    def __init__(self, cfg: "ResetAwareUniformVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._presampled_reset_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resolve_env_ids(self, env_ids: Sequence[int] | slice | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def presample_for_reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """Sample commands early so reset events can read ``is_standing_env`` for this episode."""
        env_ids_tensor = self._resolve_env_ids(env_ids)
        if env_ids_tensor.numel() == 0:
            return

        self.command_counter[env_ids_tensor] = 0
        self._resample(env_ids_tensor)
        self._update_command()
        self._presampled_reset_envs[env_ids_tensor] = True

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset metrics while preserving commands that were pre-sampled by reset events."""
        env_ids_tensor = self._resolve_env_ids(env_ids)
        env_ids_index = slice(None) if env_ids is None else env_ids_tensor

        extras = {}
        for metric_name, metric_value in self.metrics.items():
            extras[metric_name] = torch.mean(metric_value[env_ids_index]).item()
            metric_value[env_ids_index] = 0.0

        if env_ids_tensor.numel() == 0:
            return extras

        presampled_mask = self._presampled_reset_envs[env_ids_tensor]
        resample_env_ids = env_ids_tensor[~presampled_mask]
        if resample_env_ids.numel() > 0:
            self.command_counter[resample_env_ids] = 0
            self._resample(resample_env_ids)

        self._update_command()
        self._presampled_reset_envs[env_ids_tensor] = False
        return extras


@configclass
class ResetAwareUniformVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for reset-aware uniform velocity commands."""

    class_type: type = ResetAwareUniformVelocityCommand


class TerrainAwareUniformVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with terrain-dependent axis filtering."""

    cfg: "TerrainAwareUniformVelocityCommandCfg"

    def __init__(self, cfg: "TerrainAwareUniformVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._presampled_reset_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._restricted_terrain_types = self._resolve_restricted_terrain_types()

    def _resolve_env_ids(self, env_ids: Sequence[int] | slice | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def presample_for_reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """Sample commands early so reset events can read ``is_standing_env`` for this episode."""
        env_ids_tensor = self._resolve_env_ids(env_ids)
        if env_ids_tensor.numel() == 0:
            return

        self.command_counter[env_ids_tensor] = 0
        self._resample(env_ids_tensor)
        self._update_command()
        self._presampled_reset_envs[env_ids_tensor] = True

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset metrics while preserving commands that were pre-sampled by reset events."""
        env_ids_tensor = self._resolve_env_ids(env_ids)
        env_ids_index = slice(None) if env_ids is None else env_ids_tensor

        extras = {}
        for metric_name, metric_value in self.metrics.items():
            extras[metric_name] = torch.mean(metric_value[env_ids_index]).item()
            metric_value[env_ids_index] = 0.0

        if env_ids_tensor.numel() == 0:
            return extras

        presampled_mask = self._presampled_reset_envs[env_ids_tensor]
        resample_env_ids = env_ids_tensor[~presampled_mask]
        if resample_env_ids.numel() > 0:
            self.command_counter[resample_env_ids] = 0
            self._resample(resample_env_ids)

        self._update_command()
        self._presampled_reset_envs[env_ids_tensor] = False
        return extras

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if self.cfg.heading_command and self.cfg.align_heading_with_robot_on_reset:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            initial_resample_mask = self.command_counter[env_ids_tensor] == 0
            if torch.any(initial_resample_mask):
                initial_env_ids = env_ids_tensor[initial_resample_mask]
                self.heading_target[initial_env_ids] = self.robot.data.heading_w[initial_env_ids]

        if (
            self._restricted_terrain_types.numel() == 0
            or (self.cfg.restricted_lin_vel_x_range is None and self.cfg.restricted_heading_range is None)
        ):
            return

        terrain: TerrainImporter = self._env.scene.terrain
        terrain_types = getattr(terrain, "terrain_types", None)
        if terrain_types is None:
            return

        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        restricted_mask = torch.isin(terrain_types[env_ids_tensor], self._restricted_terrain_types)
        if not torch.any(restricted_mask):
            return

        restricted_env_ids = env_ids_tensor[restricted_mask]
        if self.cfg.restricted_lin_vel_x_range is not None:
            self.vel_command_b[restricted_env_ids, 0] = torch.empty(
                len(restricted_env_ids), device=self.device
            ).uniform_(*self.cfg.restricted_lin_vel_x_range)
        if self.cfg.heading_command and self.cfg.restricted_heading_range is not None:
            self.heading_target[restricted_env_ids] = torch.empty(
                len(restricted_env_ids), device=self.device
            ).uniform_(*self.cfg.restricted_heading_range)

    def _update_command(self):
        super()._update_command()
        self._apply_terrain_command_filter()

    def _resolve_restricted_terrain_types(self) -> torch.Tensor:
        """Map configured sub-terrain names to terrain type ids used by TerrainImporter."""
        sub_terrain_names = tuple(self.cfg.restricted_sub_terrain_names)
        if not sub_terrain_names:
            return torch.empty(0, dtype=torch.long, device=self.device)

        terrain_cfg = getattr(getattr(self._env.cfg, "scene", None), "terrain", None)
        terrain_generator_cfg = getattr(terrain_cfg, "terrain_generator", None)
        if terrain_generator_cfg is None or not getattr(terrain_generator_cfg, "curriculum", False):
            return torch.empty(0, dtype=torch.long, device=self.device)

        configured_names = list(terrain_generator_cfg.sub_terrains.keys())
        proportions = [sub_cfg.proportion for sub_cfg in terrain_generator_cfg.sub_terrains.values()]
        total_proportion = sum(proportions)
        if total_proportion <= 0.0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        normalized_proportions = [proportion / total_proportion for proportion in proportions]
        cumulative_proportions = []
        cumulative_sum = 0.0
        for proportion in normalized_proportions:
            cumulative_sum += proportion
            cumulative_proportions.append(cumulative_sum)

        restricted_types = []
        for column_id in range(terrain_generator_cfg.num_cols):
            column_ratio = column_id / terrain_generator_cfg.num_cols + 0.001
            sub_terrain_index = next(
                index for index, boundary in enumerate(cumulative_proportions) if column_ratio < boundary
            )
            if configured_names[sub_terrain_index] in sub_terrain_names:
                restricted_types.append(column_id)

        return torch.tensor(restricted_types, dtype=torch.long, device=self.device)

    def _apply_terrain_command_filter(self):
        if self._restricted_terrain_types.numel() == 0:
            return

        terrain: TerrainImporter = self._env.scene.terrain
        terrain_types = getattr(terrain, "terrain_types", None)
        if terrain_types is None:
            return

        restricted_envs = torch.isin(terrain_types, self._restricted_terrain_types)
        if not torch.any(restricted_envs):
            return

        if self.cfg.force_zero_lin_vel_y:
            self.vel_command_b[restricted_envs, 1] = 0.0
        if self.cfg.force_zero_ang_vel_z:
            self.vel_command_b[restricted_envs, 2] = 0.0
        if self.cfg.disable_heading_command and self.cfg.heading_command:
            self.is_heading_env[restricted_envs] = False
            self.heading_target[restricted_envs] = self.robot.data.heading_w[restricted_envs]


@configclass
class TerrainAwareUniformVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for terrain-aware uniform velocity commands."""

    class_type: type = TerrainAwareUniformVelocityCommand

    restricted_sub_terrain_names: tuple[str, ...] = ()
    """Sub-terrain names whose commands should be filtered after resampling."""

    restricted_lin_vel_x_range: tuple[float, float] | None = None
    """Optional x-velocity sampling range applied only to restricted terrains during command resampling."""

    restricted_heading_range: tuple[float, float] | None = None
    """Optional heading-target sampling range applied only to restricted terrains during command resampling."""

    align_heading_with_robot_on_reset: bool = False
    """Whether to initialize the heading target from the robot's current heading on the first post-reset resample."""

    force_zero_lin_vel_y: bool = True
    """Whether to clamp lateral velocity command to zero on restricted terrains."""

    force_zero_ang_vel_z: bool = True
    """Whether to clamp yaw velocity command to zero on restricted terrains."""

    disable_heading_command: bool = True
    """Whether to disable heading-driven yaw control on restricted terrains."""
