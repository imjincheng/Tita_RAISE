# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tita environment configuration without base linear velocity xy observation."""

from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp

from .rough_env_cfg import TitaRoughEnvCfg, configure_forward_only_play_commands


ESTIMATOR_TARGET_BASE_LIN_VEL_XY_SCALE = [1.0, 1.0]
ESTIMATOR_POLICY_BASE_LIN_VEL_XY_SCALE = [2.0, 2.0]
ESTIMATOR_HISTORY_LENGTH = 3
ESTIMATOR_FEATURE_HISTORY_LENGTH = 10


@configclass
class ObservationsCfgWithoutBaseVel:
    """Observation specifications with history-stacked policy input and velocity estimation."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations without base_lin_vel_xy, stacked over history."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.25,
        )  # 3 dims

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )  # 3 dims

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 0.0, 0.25),
        )  # 3 dims

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_.*_leg_[123]"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )  # 6 dims

        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )  # 8 dims

        last_action = ObsTerm(func=mdp.last_action, scale=1.0)  # 8 dims

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10

    @configclass
    class HistoryCfg(PolicyCfg):
        """History stack consumed by the velocity estimator."""

        def __post_init__(self) -> None:
            super().__post_init__()
            self.history_length = 10

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group (privileged)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 0.0, 0.25),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_4"),
            },
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, scale=1.0)
        height_scan = ObsTerm(
            func=mdp.safe_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.history_length = 1

    @configclass
    class VelocityTargetCfg(ObsGroup):
        """Velocity supervision target for the estimator."""

        base_lin_vel_xy = ObsTerm(func=mdp.base_lin_vel_xy, scale=1.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    history: HistoryCfg = HistoryCfg()
    critic: CriticCfg = CriticCfg()
    velocity_target: VelocityTargetCfg = VelocityTargetCfg()


@configclass
class ObservationsCfgWithoutBaseVelNoEstimator:
    """Observation specifications without base linear velocity xy and without a velocity estimator."""

    @configclass
    class PolicyCfg(ObservationsCfgWithoutBaseVel.PolicyCfg):
        """Actor observations without base_lin_vel_xy and without estimator history."""

    @configclass
    class CriticCfg(ObservationsCfgWithoutBaseVel.CriticCfg):
        """Critic observations without estimator-only groups."""

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class TitaFlatNoBaseVelEnvCfg(TitaRoughEnvCfg):
    """Tita flat terrain environment without base_lin_vel_xy, with velocity estimation."""

    observations: ObservationsCfgWithoutBaseVel = ObservationsCfgWithoutBaseVel()

    def __post_init__(self):
        super().__post_init__()

        # Change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # No terrain curriculum
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.rel_standing_envs = 0.1

        self.rewards.feet_y_distance = RewTerm(
            func=mdp.feet_y_distance,
            weight=-2.0,
            params={
                "min_distance": 0.5,
                "max_distance": 0.6,
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_leg_4"]),
            },
        )
        self.rewards.stand_still = RewTerm(
            func=mdp.stand_still,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_.*_leg_[123]"]),
            },
        )
        self.rewards.zero_command_wheel_vel = RewTerm(
            func=mdp.zero_command_wheel_vel_l1,
            weight=-0.1,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_4"]),
            },
        )
        self.rewards.opposite_wheel_vel = RewTerm(
            func=mdp.opposite_wheel_vel,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_4"]),
            },
        )


@configclass
class TitaFlatNoBaseVelNoEstimatorEnvCfg(TitaRoughEnvCfg):
    """Tita flat terrain environment without base_lin_vel_xy and without velocity estimation."""

    observations: ObservationsCfgWithoutBaseVelNoEstimator = ObservationsCfgWithoutBaseVelNoEstimator()

    def __post_init__(self):
        super().__post_init__()

        # Change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # No terrain curriculum
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.rel_standing_envs = 0.1

        self.rewards.feet_y_distance = RewTerm(
            func=mdp.feet_y_distance,
            weight=-2.0,
            params={
                "min_distance": 0.48,
                "max_distance": 0.6,
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_leg_4"]),
            },
        )
        # self.rewards.stand_still = RewTerm(
        #     func=mdp.stand_still,
        #     weight=-0.5,
        #     params={
        #         "command_name": "base_velocity",
        #         "command_threshold": 0.15,
        #         "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_.*_leg_[123]"]),
        #     },
        # )
        # self.rewards.zero_command_base_ang_vel_z = RewTerm(
        #     func=mdp.zero_command_base_ang_vel_z_l2,
        #     weight=-2.0,
        #     params={"command_name": "base_velocity", "command_threshold": 0.15},
        # )
        # self.rewards.zero_command_base_lin_vel_xy = RewTerm(
        #     func=mdp.zero_command_base_lin_vel_xy_l2,
        #     weight=-1.0,
        #     params={"command_name": "base_velocity", "command_threshold": 0.15},
        # )
        self.rewards.zero_command_wheel_vel = RewTerm(
            func=mdp.zero_command_wheel_vel_l1,
            weight=-0.05,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_4"]),
            },
        )
        self.rewards.opposite_wheel_vel = RewTerm(
            func=mdp.opposite_wheel_vel,
            weight=-0.2,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_4"]),
            },
        )


@configclass
class TitaFlatNoBaseVelEnvCfg_PLAY(TitaFlatNoBaseVelEnvCfg):
    """Play configuration for flat terrain without base_lin_vel_xy."""

    def __post_init__(self) -> None:
        super().__post_init__()
        configure_forward_only_play_commands(self)

        # Smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable observation noise
        self.observations.policy.enable_corruption = False
        self.observations.history.enable_corruption = False

        # Remove all domain randomization events
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None


@configclass
class TitaFlatNoBaseVelNoEstimatorEnvCfg_PLAY(TitaFlatNoBaseVelNoEstimatorEnvCfg):
    """Play configuration for flat terrain without base_lin_vel_xy and without velocity estimation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        configure_forward_only_play_commands(self)

        # Smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable observation noise
        self.observations.policy.enable_corruption = False

        # Remove all domain randomization events
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None
