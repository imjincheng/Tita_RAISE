# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from ..stair_env_cfg import (
    STAIR_ESTIMATOR_HISTORY_TERM_DIMS,
    STAIR_ESTIMATOR_OUTPUT_HISTORY_LENGTH,
    STAIR_ESTIMATOR_WINDOW_LENGTH,
)
from ..no_base_vel_env_cfg import (
    ESTIMATOR_FEATURE_HISTORY_LENGTH,
    ESTIMATOR_HISTORY_LENGTH,
    ESTIMATOR_POLICY_BASE_LIN_VEL_XY_SCALE,
    ESTIMATOR_TARGET_BASE_LIN_VEL_XY_SCALE,
)


@configclass
class TitaRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "tita_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TitaFlatPPORunnerCfg(TitaRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "tita_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]


def _enable_velocity_estimator(
    runner_cfg,
    experiment_name: str,
    num_history: int = ESTIMATOR_HISTORY_LENGTH,
    estimated_history_length: int = ESTIMATOR_HISTORY_LENGTH,
    estimator_target_scale: list[float] | tuple[float, ...] = ESTIMATOR_TARGET_BASE_LIN_VEL_XY_SCALE,
    estimator_feature_scale: list[float] | tuple[float, ...] = ESTIMATOR_POLICY_BASE_LIN_VEL_XY_SCALE,
) -> None:
    runner_cfg.experiment_name = experiment_name
    runner_cfg.policy.class_name = "ActorCriticWithEstimator"
    runner_cfg.algorithm.class_name = "PPOWithEstimator"
    runner_cfg.policy.estimator_hidden_dims = [256, 128]
    runner_cfg.policy.num_history = num_history
    runner_cfg.policy.estimated_history_length = estimated_history_length
    runner_cfg.policy.estimator_output_dim = 2
    runner_cfg.policy.estimator_target_scale = estimator_target_scale
    runner_cfg.policy.estimator_feature_scale = estimator_feature_scale
    runner_cfg.algorithm.estimator_loss_coef = 1.0
    runner_cfg.obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "history": ["history"],
        "velocity_target": ["velocity_target"],
    }


@configclass
class TitaFlatNoBaseVelEstimatorPPORunnerCfg(TitaFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        _enable_velocity_estimator(
            self,
            "tita_flat_no_base_vel_estimator",
            num_history=ESTIMATOR_HISTORY_LENGTH,
            estimated_history_length=ESTIMATOR_FEATURE_HISTORY_LENGTH,
        )
        self.empirical_normalization = None
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        self.policy.history_term_dims = [3, 3, 3, 6, 8, 8]
        self.policy.deploy_share_policy_and_history = True
        self.algorithm.class_name = "PPOWithEstimatorAdaBoot"
        self.algorithm.adaboot_reward_window = 128
        self.algorithm.adaboot_min_episodes = 32
        self.algorithm.adaboot_eps = 1.0e-6


@configclass
class TitaFlatNoBaseVelPPORunnerCfg(TitaFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "tita_flat_no_base_vel"


@configclass
class TitaStairEstimatorPPORunnerCfg(TitaRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 30000
        self.empirical_normalization = None
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        _enable_velocity_estimator(
            self,
            "tita_stair_estimator",
            num_history=STAIR_ESTIMATOR_WINDOW_LENGTH,
            estimated_history_length=STAIR_ESTIMATOR_OUTPUT_HISTORY_LENGTH,
        )
        self.policy.history_term_dims = list(STAIR_ESTIMATOR_HISTORY_TERM_DIMS)
        self.policy.deploy_share_policy_and_history = True
        self.algorithm.estimator_loss_coef = 1.0
        self.algorithm.class_name = "PPOWithEstimatorAdaBoot"
        self.algorithm.adaboot_reward_window = 128
        self.algorithm.adaboot_min_episodes = 32
        self.algorithm.adaboot_eps = 1.0e-6


@configclass
class TitaStairPPORunnerCfg(TitaRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 30000
        self.experiment_name = "tita_stair"
        self.empirical_normalization = None
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True


@configclass
class TitaStairCENetAdaBootPPORunnerCfg(TitaRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 30000
        self.experiment_name = "tita_stair_cenet_adaboot"
        self.empirical_normalization = None
        self.policy.class_name = "ActorCriticWithCENet"
        self.algorithm.class_name = "PPOWithCENetAdaBoot"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        self.policy.cenet_encoder_hidden_dims = [128, 64]
        self.policy.cenet_decoder_hidden_dims = [64, 128]
        self.policy.cenet_velocity_dim = 3
        self.policy.cenet_latent_dim = 16
        self.policy.num_history = 5
        self.algorithm.cenet_loss_coef = 1.0
        self.algorithm.cenet_velocity_loss_coef = 1.0
        self.algorithm.cenet_reconstruction_loss_coef = 1.0
        self.algorithm.cenet_kl_loss_coef = 0.01
        self.algorithm.vae_learning_rate = 1.0e-3
        self.algorithm.num_vae_substeps = 1
        self.algorithm.rl_grad_to_cenet = True
        self.algorithm.adaboot_enabled = True
        self.algorithm.adaboot_reward_window = 128
        self.algorithm.adaboot_min_episodes = 32
        self.obs_groups = {
            "policy": ["policy"],
            "critic": ["critic"],
            "history": ["history"],
            "velocity_target": ["velocity_target"],
        }


@configclass
class TitaFlatCENetAdaBootPPORunnerCfg(TitaStairCENetAdaBootPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "tita_flat_cenet_adaboot"
