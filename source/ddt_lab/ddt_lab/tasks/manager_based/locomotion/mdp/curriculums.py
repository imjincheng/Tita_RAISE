# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import RLTaskEnv


def terrain_levels_vel(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_by_type(
    env: RLTaskEnv, env_ids: Sequence[int], terrain_names: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Report the mean terrain level for each requested terrain type without updating curriculum state."""
    terrain: TerrainImporter = env.scene.terrain
    terrain_generator_cfg = terrain.cfg.terrain_generator
    device = terrain.terrain_levels.device

    sub_terrain_names = list(terrain_generator_cfg.sub_terrains.keys())
    proportions = [terrain_generator_cfg.sub_terrains[name].proportion for name in sub_terrain_names]
    proportion_sum = sum(proportions)
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

    terrain_level_stats = {}
    for terrain_name in terrain_names:
        type_ids = [
            col for col, col_terrain_name in enumerate(terrain_name_by_col) if col_terrain_name == terrain_name
        ]
        if len(type_ids) == 0:
            terrain_level_stats[terrain_name] = torch.zeros((), device=device)
            continue

        type_ids_tensor = torch.tensor(type_ids, device=device, dtype=terrain.terrain_types.dtype)
        terrain_type_mask = (terrain.terrain_types[:, None] == type_ids_tensor[None, :]).any(dim=1)
        if terrain_type_mask.any():
            terrain_level_stats[terrain_name] = terrain.terrain_levels[terrain_type_mask].float().mean()
        else:
            terrain_level_stats[terrain_name] = torch.zeros((), device=device)

    return terrain_level_stats
