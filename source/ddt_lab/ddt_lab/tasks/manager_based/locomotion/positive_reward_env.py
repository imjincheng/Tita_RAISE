# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.envs import ManagerBasedRLEnv


class PositiveRewardManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based RL env with optional total-reward clipping."""

    def step(self, action: torch.Tensor):
        obs, rewards, terminated, time_outs, extras = super().step(action)
        if getattr(self.cfg, "only_positive_rewards", False):
            self.reward_buf.clamp_min_(0.0)
            rewards = self.reward_buf
        return obs, rewards, terminated, time_outs, extras
