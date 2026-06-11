# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, no_base_vel_env_cfg, rough_env_cfg, stair_env_cfg

##
# Register Gym environments.
##

POSITIVE_REWARD_ENV_ENTRY_POINT = "ddt_lab.tasks.manager_based.locomotion.positive_reward_env:PositiveRewardManagerBasedRLEnv"

gym.register(
    id="DDT-Velocity-Flat-Tita-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TitaFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaFlatPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-Tita-CENet-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TitaFlatCENetEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaFlatCENetAdaBootPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-Tita-CENet-Play-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TitaFlatCENetEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaFlatCENetAdaBootPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-Tita-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TitaFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaFlatPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-Tita-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.TitaRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaRoughPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-Tita-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.TitaRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaRoughPPORunnerCfg",
    },
)


gym.register(
    id="DDT-Velocity-Stair-Tita-Estimator-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stair_env_cfg.TitaStairEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaStairEstimatorPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Stair-Tita-Estimator-Play-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stair_env_cfg.TitaStairEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaStairEstimatorPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Stair-Tita-NoEstimator-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stair_env_cfg.TitaStairNoEstimatorEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaStairPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Stair-Tita-NoEstimator-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stair_env_cfg.TitaStairNoEstimatorEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaStairPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Stair-Tita-CENet-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stair_env_cfg.TitaStairCENetEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaStairCENetAdaBootPPORunnerCfg",
    },
)


gym.register(
    id="DDT-Velocity-Stair-Tita-CENet-Play-v0",
    entry_point=POSITIVE_REWARD_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stair_env_cfg.TitaStairCENetEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaStairCENetAdaBootPPORunnerCfg",
    },
)


gym.register(
    id="DDT-Velocity-Stair-Tita-NoBaseVel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stair_env_cfg.TitaStairNoBaseVelEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaStairPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Stair-Tita-NoBaseVel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stair_env_cfg.TitaStairNoBaseVelEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaStairPPORunnerCfg",
    },
)

##
# Register environments without base_lin_vel_xy observation
##

gym.register(
    id="DDT-Velocity-Flat-Tita-NoBaseVel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": no_base_vel_env_cfg.TitaFlatNoBaseVelEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaFlatNoBaseVelEstimatorPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-Tita-NoBaseVel-NoEstimator-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": no_base_vel_env_cfg.TitaFlatNoBaseVelNoEstimatorEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaFlatNoBaseVelPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-Tita-NoBaseVel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": no_base_vel_env_cfg.TitaFlatNoBaseVelEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaFlatNoBaseVelEstimatorPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-Tita-NoBaseVel-NoEstimator-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": no_base_vel_env_cfg.TitaFlatNoBaseVelNoEstimatorEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TitaFlatNoBaseVelPPORunnerCfg",
    },
)
