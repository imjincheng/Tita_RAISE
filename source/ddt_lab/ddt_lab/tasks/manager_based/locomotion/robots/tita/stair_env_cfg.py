"""Tita stair-climbing environment configuration with and without velocity estimation."""

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp

from .rough_env_cfg import CommandsCfg as RoughCommandsCfg
from .rough_env_cfg import EventCfg, RewardsCfg, TitaRoughEnvCfg, configure_forward_only_play_commands
from .rough_env_cfg import TerminationsCfg as RoughTerminationsCfg


# STAIR_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=20.0,
#     num_rows=10,
#     num_cols=10,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     difficulty_range=(0.0, 1.0),
#     use_cache=False,
#     curriculum=True,
#     sub_terrains={
#         "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.10,
#             noise_range=(0.01, 0.05),
#             noise_step=0.02,
#             border_width=0.25,
#         ),
#         "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.10,
#             slope_range=(0.0, 0.3),
#             platform_width=2.0,
#             border_width=0.25,
#         ),
#         "discrete_obstacles": terrain_gen.MeshRandomGridTerrainCfg(
#             proportion=0.20,
#             grid_width=0.45,
#             grid_height_range=(0.02, 0.10),
#             platform_width=2.0,
#         ),
#         "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#             proportion=0.50,
#             step_height_range=(0.08, 0.15),
#             step_width=0.5,
#             platform_width=2.5,
#             border_width=0.0,
#             holes=False,
#         ),
#         "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
#     },
# )
STAIR_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.25,
        ),
        "discrete_obstacles": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.20,
            grid_width=0.45,
            grid_height_range=(0.02, 0.10),
            platform_width=2.0,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.50,
            step_height_range=(0.08, 0.15),
            step_width=0.5,
            platform_width=2.5,
            border_width=0.0,
            holes=False,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
    },
)


# STAIR_PLAY_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=20.0,
#     num_rows=6,
#     num_cols=6,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     difficulty_range=(0.0, 1.0),
#     use_cache=False,
#     curriculum=False,
#     sub_terrains={
#         "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
#         "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.15,
#             slope_range=(0.0, 0.3),
#             platform_width=2.0,
#             border_width=0.25,
#         ),
#         "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.15,
#             noise_range=(0.01, 0.05),
#             noise_step=0.02,
#             border_width=0.25,
#         ),
#         "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
#             proportion=0.25,
#             step_height_range=(0.04, 0.1),
#             step_width=0.45,
#             platform_width=2.5,
#             border_width=0.0,
#             holes=False,
#         ),
#         "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#             proportion=0.15,
#             step_height_range=(0.04, 0.1),
#             step_width=0.45,
#             platform_width=2.5,
#             border_width=0.0,
#             holes=False,
#         ),
#         "discrete_obstacles": terrain_gen.MeshRandomGridTerrainCfg(
#             proportion=0.15,
#             grid_width=0.45,
#             grid_height_range=(0.02, 0.10),
#             platform_width=2.0,
#         ),
#     },
# )

STAIR_PLAY_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=6,
    num_cols=6,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.12),
            step_width=0.45,
            platform_width=2.5,
            border_width=0.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.12),
            step_width=0.45,
            platform_width=2.5,
            border_width=0.0,
            holes=False,
        ),
    },
)


@configclass
class StairObservationsCfg:
    """Observation specification for stair climbing without velocity estimation."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.diag_base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.diag_projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 0.0, 0.25),
        )
        joint_pos = ObsTerm(
            func=mdp.diag_joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_.*_leg_[123]"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.diag_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.diag_safe_blended_action, scale=1.0)
        base_lin_vel_xy = ObsTerm(
            func=mdp.diag_base_lin_vel_xy,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            scale=2.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.diag_base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.diag_base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.diag_projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 0.0, 0.25),
        )
        joint_pos = ObsTerm(
            func=mdp.diag_joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_4"),
            },
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.diag_joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.diag_safe_blended_action, scale=1.0)
        feet_avg_contact_force = ObsTerm(
            func=mdp.diag_feet_average_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_4"])},
            clip=(-1000.0, 1000.0),
            scale=0.01,
        )
        height_scan = ObsTerm(
            func=mdp.safe_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# Stair estimator settings are centralized here so future tuning only needs this file.
STAIR_ESTIMATOR_POLICY_HISTORY_LENGTH = 10
STAIR_ESTIMATOR_WINDOW_LENGTH = 3
STAIR_ESTIMATOR_OUTPUT_HISTORY_LENGTH = STAIR_ESTIMATOR_POLICY_HISTORY_LENGTH
STAIR_ESTIMATOR_HISTORY_TERM_DIMS = (3, 3, 3, 6, 8, 8)


@configclass
class StairEstimatorObservationsCfg:
    """Observation specification for stair climbing with a velocity estimator."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.diag_base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.diag_projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 0.0, 0.25),
        )
        joint_pos = ObsTerm(
            func=mdp.diag_joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_.*_leg_[123]"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.diag_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.diag_safe_blended_action, scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = STAIR_ESTIMATOR_POLICY_HISTORY_LENGTH

    @configclass
    class HistoryCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = STAIR_ESTIMATOR_POLICY_HISTORY_LENGTH

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.diag_base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.diag_base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.diag_projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 0.0, 0.25),
        )
        joint_pos = ObsTerm(
            func=mdp.diag_joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_4"),
            },
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.diag_joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.diag_safe_blended_action, scale=1.0)
        feet_avg_contact_force = ObsTerm(
            func=mdp.diag_feet_average_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_4"])},
            clip=(-1000.0, 1000.0),
            scale=0.01,
        )
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
class StairCENetObservationsCfg:
    """DreamWaQ-style observations for stair climbing with CENet context estimation."""

    @configclass
    class PolicyCfg(StairEstimatorObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 1

    @configclass
    class HistoryCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 5

    @configclass
    class CriticCfg(StairEstimatorObservationsCfg.CriticCfg):
        pass

    @configclass
    class VelocityTargetCfg(ObsGroup):
        """Velocity supervision target for CENet; critic privileged observations live in ``critic``."""

        base_lin_vel = ObsTerm(func=mdp.diag_base_lin_vel, scale=1.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    history: HistoryCfg = HistoryCfg()
    critic: CriticCfg = CriticCfg()
    velocity_target: VelocityTargetCfg = VelocityTargetCfg()


@configclass
class StairNoBaseVelObservationsCfg(StairObservationsCfg):
    """Stair observations without actor-side ``base_lin_vel_xy`` and without a velocity estimator."""

    @configclass
    class PolicyCfg(StairObservationsCfg.PolicyCfg):
        base_lin_vel_xy = None

    policy: PolicyCfg = PolicyCfg()


@configclass
class StairActionsCfg:
    """Action specification with contact-triggered feedforward trajectory."""

    joint_pos = mdp.TitaJointPositionEffortActionCfg(
        asset_name="robot",
        leg_joint_names=[
            "joint_left_leg_1",
            "joint_left_leg_2",
            "joint_left_leg_3",
            "joint_right_leg_1",
            "joint_right_leg_2",
            "joint_right_leg_3",
        ],
        wheel_joint_names=["joint_left_leg_4", "joint_right_leg_4"],
        leg_scale=(0.25, 0.5, 0.5, 0.25, 0.5, 0.5),
        wheel_scale=0.5,
        wheel_effort_gain=12.0,
        wheel_offset=0.0,
        use_default_leg_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)},
        feedforward_enabled=True,
        k_fb=1.0,
        k_ff=0.5,
        feedforward_period=0.6,
        feedforward_amplitude={
            ".*_leg_2": 0.4,
            ".*_leg_3": -0.80,
        },
        feedforward_joint_names=[
            "joint_left_leg_2",
            "joint_left_leg_3",
            "joint_right_leg_2",
            "joint_right_leg_3",
        ],
        contact_trigger_enabled=True,
        contact_sensor_name="contact_forces",
        contact_body_pattern=".*_leg_4",
        contact_force_threshold=50.0,
        followup_trigger_delay_factor=0.5,
        k_ff_anneal_enabled=True,
        k_ff_final=0.0,
        k_ff_start_iteration=20000,
        k_ff_anneal_iterations=10000,
        k_ff_steps_per_iteration=24,
    )


@configclass
class StairCommandsCfg(RoughCommandsCfg):
    """Commands for stair climbing.

    When an environment is assigned to the ``stairs_down`` terrain columns, we keep only the
    commanded forward velocity and disable lateral motion, while preserving heading control so
    the robot learns to stay aligned with the stair direction.
    """

    base_velocity = mdp.TerrainAwareUniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        align_heading_with_robot_on_reset=True,
        debug_vis=True,
        restricted_sub_terrain_names=("stairs_up", "stairs_down"),
        restricted_lin_vel_x_range=(0.0, 1.0),
        restricted_heading_range=(-math.pi / 4.0, math.pi / 4.0),
        force_zero_lin_vel_y=True,
        force_zero_ang_vel_z=False,
        disable_heading_command=False,
        ranges=mdp.TerrainAwareUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class StairEventCfg(EventCfg):
    """Events for stair climbing."""


@configclass
class StairRewardsCfg(RewardsCfg):
    """Reward terms tailored for stair climbing."""

    # ---------------------------------------------------------------------
    # Task rewards
    # ---------------------------------------------------------------------
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    track_heading_exp = RewTerm(
        func=mdp.track_heading_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )

    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_.*_leg_[123]"]),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_4"]),
            "threshold": 0.1,
            "triggered_only": True,
            "action_name": "joint_pos",
        },
    )

    feet_height = RewTerm(
        func=mdp.feet_height_band_relative,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_leg_4"]),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_height": 0.10,
            "std": 0.05,
            "tanh_mult": 2.0,
            "wheel_radius": 0.0925,
            "action_name": "joint_pos",
        },
    )

    feet_contact_number = RewTerm(
        func=mdp.feet_xy_swing_fz_stance_match,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_4"]),
            "action_name": "joint_pos",
            "mismatch_penalty": 1.3,
            "swing_xy_threshold": 50.0,
            "stance_fz_threshold": 50.0,
        },
    )

    # feet_swing_xy_impact = RewTerm(
    #     func=mdp.feet_swing_xy_impact_penalty,
    #     weight=-0.002,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_4"]),
    #         "action_name": "joint_pos",
    #         "xy_force_threshold": 50.0,
    #     },
    # )

    tracking_target_pos = RewTerm(
        func=mdp.track_ff_target_pos_exp,
        weight= 0.8,
        params={
            "action_name": "joint_pos",
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.1,
        },
    )

    # ---------------------------------------------------------------------
    # Style rewards
    # ---------------------------------------------------------------------

    joint_mirror = RewTerm(
        func=mdp.stair_joint_mirror,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["joint_left_leg_(1|2|3)", "joint_right_leg_(1|2|3)"]],
            "action_name": "joint_pos",
        },
    )
    # joint_mirror = None

    joint_deviation_leg1_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_.*_leg_1"])},
    )

    wheel_vel_penalty = RewTerm(
        func=mdp.wheel_vel_penalty,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_4"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_4"]),
            "command_name": "base_velocity",
            "velocity_threshold": 100.0,
            "command_threshold": 0.1,
        },
    )

    # wheel_spin = RewTerm(
    #     func=mdp.wheel_spin_penalty,
    #     weight=-1.0,
    #     params={
    #         "wheel_joint_cfg": SceneEntityCfg(
    #             "robot", joint_names=["joint_left_leg_4", "joint_right_leg_4"]
    #         ),
    #         "foot_body_cfg": SceneEntityCfg("robot", body_names=["left_leg_4", "right_leg_4"]),
    #         "wheel_radius": 0.0925,
    #         "spin_scale": 0.8,
    #         "slip_deadband": 0.1,
    #     },
    # )

    feet_y_distance = RewTerm(
        func=mdp.feet_y_distance,
        weight=-2.0,
        params={
            "min_distance": 0.5,
            "max_distance": 0.62,
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_leg_4"]),
        },
    )

    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-20.0,
        params={"target_height": 0.35, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-12.0)

    upward = RewTerm(func=mdp.upward, weight=1.0)

    # ---------------------------------------------------------------------
    # Regularization rewards
    # ---------------------------------------------------------------------
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_safe, weight=-0.01, params={"clamp_value": 5.0})
    action_smooth = RewTerm(
        func=mdp.action_smooth_for_term_indices_safe,
        weight=-0.01,
        params={
            "action_name": "joint_pos",
            "action_indices": [3, 7],
            "clamp_value": 5.0,
            "scale_with_term": False,
        },
    )

    zero_command_wheel_vel = RewTerm(
        func=mdp.zero_command_wheel_vel_l1,
        weight=-0.02,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.15,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_4"]),
        },
    )

    opposite_base_vel = RewTerm(
        func=mdp.opposite_base_vel,
        weight=-40.0,
        params={"command_name": "base_velocity"},
    )

    opposite_wheel_vel = RewTerm(
        func=mdp.opposite_wheel_vel,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_4"]),
        },
    )

@configclass
class StairCurriculumCfg:
    """Curriculum for stair climbing.

    We keep terrain curriculum from the rough task.

    Note:
        The k_ff linear annealing is executed inside the feedforward action term on every
        environment step, so it stays synchronized with how feedforward is actually injected.
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    terrain_level = CurrTerm(
        func=mdp.terrain_levels_by_type,
        params={
            "terrain_names": [
                "random_rough",
                "smooth_slope",
                "discrete_obstacles",
                "stairs_down",
                "flat",
            ],
        },
    )


@configclass
class StairTerminationsCfg(RoughTerminationsCfg):
    """Termination terms for stair climbing."""

    illegal_leg2_contact = None
    illegal_leg3_contact = None
    abnormal_blended_action = DoneTerm(func=mdp.abnormal_blended_action_termination, params={"threshold": 100.0})


@configclass
class TitaStairBaseEnvCfg(TitaRoughEnvCfg):
    """Common stair-climbing environment configuration."""

    commands: StairCommandsCfg = StairCommandsCfg()
    actions: StairActionsCfg = StairActionsCfg()
    rewards: StairRewardsCfg = StairRewardsCfg()
    terminations: StairTerminationsCfg = StairTerminationsCfg()
    events: StairEventCfg = StairEventCfg()
    curriculum: StairCurriculumCfg = StairCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIR_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 2

        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.align_heading_with_robot_on_reset = True
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (-math.pi / 4.0, math.pi / 4.0)
        self.commands.base_velocity.restricted_lin_vel_x_range = (0.0, 1.0)
        self.commands.base_velocity.restricted_heading_range = (-math.pi / 4.0, math.pi / 4.0)

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-math.pi / 4.0, math.pi / 4.0)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
        }
      
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        


@configclass
class TitaStairNoEstimatorEnvCfg(TitaStairBaseEnvCfg):
    """Tita stair-climbing environment without velocity estimation."""

    observations: StairObservationsCfg = StairObservationsCfg()


@configclass
class TitaStairNoEstimatorEnvCfg_PLAY(TitaStairNoEstimatorEnvCfg):
    """Play configuration for the stair-climbing environment without velocity estimation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        configure_forward_only_play_commands(self)

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIR_PLAY_TERRAINS_CFG
        self.observations.policy.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None


@configclass
class TitaStairEnvCfg(TitaStairBaseEnvCfg):
    """Tita stair-climbing environment with velocity estimation."""

    observations: StairEstimatorObservationsCfg = StairEstimatorObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.only_positive_rewards = False


@configclass
class TitaStairNoBaseVelEnvCfg(TitaStairBaseEnvCfg):
    """Tita stair-climbing environment without base_lin_vel_xy and without velocity estimation."""

    observations: StairNoBaseVelObservationsCfg = StairNoBaseVelObservationsCfg()


@configclass
class TitaStairCENetEnvCfg(TitaStairBaseEnvCfg):
    """Tita stair-climbing environment with CENet context estimation for AdaBoot."""

    observations: StairCENetObservationsCfg = StairCENetObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.only_positive_rewards = True
        self.rewards.base_height_l2.weight = -10.0
        self.rewards.flat_orientation_l2.weight = -40.0
        self.rewards.opposite_base_vel.weight = -20.0
        self.rewards.opposite_wheel_vel.weight = -1.0
        self.rewards.feet_y_distance.weight = -2.0


@configclass
class TitaStairEnvCfg_PLAY(TitaStairEnvCfg):
    """Play configuration for the stair-climbing environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        configure_forward_only_play_commands(self)

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIR_PLAY_TERRAINS_CFG
        self.observations.policy.enable_corruption = False
        self.observations.history.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None


@configclass
class TitaStairCENetEnvCfg_PLAY(TitaStairCENetEnvCfg):
    """Play configuration for the CENet stair-climbing environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        configure_forward_only_play_commands(self)

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIR_PLAY_TERRAINS_CFG
        self.observations.policy.enable_corruption = False
        self.observations.history.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None


@configclass
class TitaStairNoBaseVelEnvCfg_PLAY(TitaStairNoBaseVelEnvCfg):
    """Play configuration for the stair no-base-velocity environment without estimator."""

    def __post_init__(self) -> None:
        super().__post_init__()
        configure_forward_only_play_commands(self)

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIR_PLAY_TERRAINS_CFG
        self.observations.policy.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None
