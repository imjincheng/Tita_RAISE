# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp

from .no_base_vel_env_cfg import ObservationsCfgWithoutBaseVel
from .rough_env_cfg import TitaRoughEnvCfg, configure_forward_only_play_commands


FLAT_CENET_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5,
            noise_range=(0.01, 0.05),
            noise_step=0.02,
            border_width=0.25,
        ),
    },
)


@configclass
class FlatCENetObservationsCfg:
    """Rough-style observations for flat/rough-flat CENet training."""

    @configclass
    class PolicyCfg(ObservationsCfgWithoutBaseVel.PolicyCfg):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    @configclass
    class HistoryCfg(PolicyCfg):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.history_length = 5

    @configclass
    class CriticCfg(ObservationsCfgWithoutBaseVel.CriticCfg):
        feet_avg_contact_force = ObsTerm(
            func=mdp.diag_feet_average_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_4"])},
            clip=(-1000.0, 1000.0),
            scale=0.01,
        )

    @configclass
    class VelocityTargetCfg(ObsGroup):
        """Velocity supervision target for CENet; critic privileged observations live in ``critic``."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    history: HistoryCfg = HistoryCfg()
    critic: CriticCfg = CriticCfg()
    velocity_target: VelocityTargetCfg = VelocityTargetCfg()


@configclass
class TitaFlatEnvCfg(TitaRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        # self.rewards.flat_orientation_l2.weight = -5.0
        # self.rewards.dof_torques_l2.weight = -2.5e-5

        # self.rewards.feet_air_time.weight = 0.5
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        # ------------------------------Events------------------------------
        # self.events.reset_base.params = {
        #     "pose_range": {
        #         "x": (-0.5, 0.5),
        #         "y": (-0.5, 0.5),
        #         "z": (0.0, 0.2),
        #         "roll": (-0.785, 0.785),
        #         "pitch": (-1.57, 1.57),
        #         "yaw": (-3.14, 3.14),
        #     },
        #     "velocity_range": {
        #         "x": (-0.5, 0.5),
        #         "y": (-0.5, 0.5),
        #         "z": (-0.5, 0.5),
        #         "roll": (-0.5, 0.5),
        #         "pitch": (-0.5, 0.5),
        #         "yaw": (-0.5, 0.5),
        #     },
        # }
        # if self.__class__.__name__ == "TitaFlatEnvCfg":
        #     self.disable_zero_weight_rewards()


class TitaFlatEnvCfg_PLAY(TitaFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        configure_forward_only_play_commands(self)

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None


@configclass
class TitaFlatCENetEnvCfg(TitaRoughEnvCfg):
    """Tita flat/rough-flat environment with CENet context estimation."""

    observations: FlatCENetObservationsCfg = FlatCENetObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = FLAT_CENET_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 1

        self.only_positive_rewards = True
        self.rewards.flat_orientation_l2.weight = -12.0
        self.rewards.base_height_l2.weight = -5.0
        self.rewards.joint_deviation_legs_l1.weight = -0.5
        self.rewards.undesired_contacts.weight = -5.0


@configclass
class TitaFlatCENetEnvCfg_PLAY(TitaFlatCENetEnvCfg):
    """Play configuration for the flat CENet environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        configure_forward_only_play_commands(self)

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.observations.policy.enable_corruption = False
        self.observations.history.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None
