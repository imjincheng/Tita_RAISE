# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import os
import sys


def _prefer_local_source_tree() -> None:
    repo_source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "source", "ddt_lab"))
    if repo_source_dir not in sys.path:
        sys.path.insert(0, repo_source_dir)


_prefer_local_source_tree()

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Use keyboard to drive base_velocity.")
parser.add_argument(
    "--kff",
    type=float,
    default=None,
    help="Override joint_pos feedforward gain in play mode. Use 0 to fully disable feedforward.",
)
parser.add_argument(
    "--onnx_opset",
    type=int,
    default=15,
    help="ONNX opset version used for deploy artifact export.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

_prefer_local_source_tree()

import re
import time
import math

import ddt_lab.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
import isaaclab.utils.math as math_utils
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from ddt_lab.tasks.manager_based.locomotion.agents.rsl_rl_estimator import (
    export_actor_with_estimator_history_as_jit,
    export_actor_with_estimator_history_as_onnx,
    export_cenet_policy_as_jit,
    export_cenet_policy_as_onnx,
    export_cenet_policy_metadata,
    export_estimator_only_policy_as_jit,
    export_estimator_only_policy_as_onnx,
    export_estimator_policy_as_jit,
    export_estimator_policy_as_onnx,
    export_estimator_policy_metadata,
    export_split_estimator_policy_metadata,
    is_cenet_policy,
    is_estimator_policy,
    register_rsl_rl_estimator_extensions,
)
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

register_rsl_rl_estimator_extensions()


def _resolve_checkpoint_iteration(runner: OnPolicyRunner | DistillationRunner, resume_path: str) -> int | None:
    """Resolve the training iteration stored in the loaded checkpoint."""
    current_iteration = getattr(runner, "current_learning_iteration", None)
    if isinstance(current_iteration, int):
        return current_iteration

    match = re.search(r"model_(\d+)\.pt$", os.path.basename(resume_path))
    if match is not None:
        return int(match.group(1))

    return None


def _get_base_height_debug(env) -> tuple[float, float | None, float | None]:
    """Return base world height and, when available, terrain-relative height for env 0."""
    robot = env.unwrapped.scene["robot"]
    base_pos_w = robot.data.root_pos_w[0]
    base_height_world = float(base_pos_w[2].item())

    sensors = getattr(env.unwrapped.scene, "sensors", {})
    if "height_scanner" not in sensors:
        return base_height_world, None, None

    sensor = sensors["height_scanner"]
    ray_hits = sensor.data.ray_hits_w[0]
    ray_z = ray_hits[:, 2]
    valid_hits = torch.isfinite(ray_z)
    if not torch.any(valid_hits):
        return base_height_world, None, None

    valid_ray_hits = ray_hits[valid_hits]
    distances = torch.sum(torch.square(valid_ray_hits[:, :2] - base_pos_w[:2]), dim=1)
    nearest_id = torch.argmin(distances)
    ground_height = float(valid_ray_hits[nearest_id, 2].item())
    return base_height_world, ground_height, base_height_world - ground_height


def _override_joint_pos_k_ff(joint_pos_term, k_ff_override: float) -> None:
    """Apply a manual feedforward override for play mode."""
    if not hasattr(joint_pos_term, "_ff_enabled"):
        raise AttributeError("joint_pos term does not expose feedforward controls.")

    joint_pos_term._k_ff_anneal_enabled = False
    joint_pos_term._k_ff = float(k_ff_override)
    if hasattr(joint_pos_term, "_initial_k_ff"):
        joint_pos_term._initial_k_ff = float(k_ff_override)

    if k_ff_override <= 0.0:
        joint_pos_term._ff_enabled = False
        if hasattr(joint_pos_term, "_time"):
            joint_pos_term._time.zero_()
        if hasattr(joint_pos_term, "_lifting_state"):
            joint_pos_term._lifting_state.zero_()
        if hasattr(joint_pos_term, "_first_leg"):
            joint_pos_term._first_leg.zero_()
        if hasattr(joint_pos_term, "_last_lift_signal"):
            joint_pos_term._last_lift_signal.zero_()
        if hasattr(joint_pos_term, "_last_ff_signal"):
            joint_pos_term._last_ff_signal.zero_()
        if hasattr(joint_pos_term, "_last_trigger_signal"):
            joint_pos_term._last_trigger_signal.zero_()
        if hasattr(joint_pos_term, "_last_ff_actions"):
            joint_pos_term._last_ff_actions.zero_()
        if hasattr(joint_pos_term, "_last_ff_contribution"):
            joint_pos_term._last_ff_contribution.zero_()
    else:
        joint_pos_term._ff_enabled = True


class _OnnxPolicyExporterWithOpset(torch.nn.Module):
    """Local ONNX exporter with configurable opset for play-time exports."""

    def __init__(self, policy, normalizer=None, verbose: bool = False, opset_version: int = 18):
        super().__init__()
        self.verbose = verbose
        self.opset_version = int(opset_version)
        self.is_recurrent = policy.is_recurrent
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")

        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")

        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_gru(self, x_in, h_in):
        x_in = self.normalizer(x_in)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self.actor(x), h

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path: str, filename: str):
        os.makedirs(path, exist_ok=True)
        self.to("cpu")
        self.eval()
        export_path = os.path.join(path, filename)

        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    export_path,
                    export_params=True,
                    opset_version=self.opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    export_path,
                    export_params=True,
                    opset_version=self.opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                export_path,
                export_params=True,
                opset_version=self.opset_version,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )


def _export_policy_as_onnx_with_opset(
    policy: object,
    path: str,
    normalizer: object | None = None,
    filename: str = "policy.onnx",
    verbose: bool = False,
    opset_version: int = 18,
):
    exporter = _OnnxPolicyExporterWithOpset(
        policy=policy,
        normalizer=normalizer,
        verbose=verbose,
        opset_version=opset_version,
    )
    exporter.export(path, filename)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    keyboard_interface = None
    keyboard_command_state = {
        "command": None,
        "raw_command": None,
        "desired_heading": None,
        "current_heading": None,
        "heading_error": None,
        "last_yaw_active": None,
        "yaw_mode": None,
    }
    if args_cli.keyboard:
        if getattr(args_cli, "headless", False):
            raise ValueError("--keyboard requires a live simulator window. Please run play.py without --headless.")

        env_cfg.scene.num_envs = 1
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "base_velocity"):
            raise ValueError("--keyboard requires a configured 'base_velocity' command.")

        base_velocity_cfg = env_cfg.commands.base_velocity
        base_velocity_cfg.debug_vis = False
        base_velocity_ranges = base_velocity_cfg.ranges
        keyboard_heading_kp = float(getattr(base_velocity_cfg, "heading_control_stiffness", 1.0))
        keyboard_yaw_min = float(base_velocity_ranges.ang_vel_z[0])
        keyboard_yaw_max = float(base_velocity_ranges.ang_vel_z[1])
        keyboard_yaw_deadband = 1.0e-4

        keyboard_cfg = Se2KeyboardCfg(
            v_x_sensitivity=base_velocity_ranges.lin_vel_x[1],
            v_y_sensitivity=base_velocity_ranges.lin_vel_y[1],
            omega_z_sensitivity=base_velocity_ranges.ang_vel_z[1],
            sim_device=env_cfg.sim.device,
        )
        keyboard_interface = Se2Keyboard(keyboard_cfg)

        def _keyboard_velocity_command(env):
            raw_command = keyboard_interface.advance().to(env.device)
            if raw_command.ndim == 1:
                raw_command = raw_command.unsqueeze(0)

            command = raw_command.clone()
            robot = env.scene["robot"]
            current_heading = robot.data.heading_w[: command.shape[0]]

            desired_heading = keyboard_command_state["desired_heading"]
            if (
                desired_heading is None
                or desired_heading.shape != current_heading.shape
                or desired_heading.device != current_heading.device
            ):
                desired_heading = current_heading.clone()
            else:
                desired_heading = desired_heading.clone()

            episode_length_buf = getattr(env, "episode_length_buf", None)
            if episode_length_buf is not None:
                reset_mask = episode_length_buf[: command.shape[0]].to(device=command.device) <= 1
                if torch.any(reset_mask):
                    desired_heading[reset_mask] = current_heading[reset_mask]

            raw_yaw = raw_command[:, 2]
            yaw_active = torch.abs(raw_yaw) > keyboard_yaw_deadband
            last_yaw_active = keyboard_command_state["last_yaw_active"]
            if (
                last_yaw_active is None
                or last_yaw_active.shape != yaw_active.shape
                or last_yaw_active.device != yaw_active.device
            ):
                last_yaw_active = torch.zeros_like(yaw_active)

            just_released = last_yaw_active & ~yaw_active
            refresh_mask = yaw_active | just_released
            if torch.any(refresh_mask):
                desired_heading[refresh_mask] = current_heading[refresh_mask]

            heading_error = math_utils.wrap_to_pi(desired_heading - current_heading)
            heading_hold_yaw = torch.clamp(
                keyboard_heading_kp * heading_error,
                min=keyboard_yaw_min,
                max=keyboard_yaw_max,
            )
            direct_yaw = torch.clamp(raw_yaw, min=keyboard_yaw_min, max=keyboard_yaw_max)
            command[:, 2] = torch.where(yaw_active, direct_yaw, heading_hold_yaw)

            keyboard_command_state["raw_command"] = raw_command.detach().clone()
            keyboard_command_state["command"] = command.detach().clone()
            keyboard_command_state["desired_heading"] = desired_heading.detach().clone()
            keyboard_command_state["current_heading"] = current_heading.detach().clone()
            keyboard_command_state["heading_error"] = heading_error.detach().clone()
            keyboard_command_state["last_yaw_active"] = yaw_active.detach().clone()
            keyboard_command_state["yaw_mode"] = "direct" if bool(yaw_active[0].item()) else "heading_hold"
            return command

        observations_cfg = getattr(env_cfg, "observations", None)
        if observations_cfg is not None:
            for group_name in ("policy", "history"):
                obs_group_cfg = getattr(observations_cfg, group_name, None)
                if obs_group_cfg is not None and hasattr(obs_group_cfg, "enable_corruption"):
                    obs_group_cfg.enable_corruption = False
                if obs_group_cfg is not None and hasattr(obs_group_cfg, "velocity_commands"):
                    velocity_term_cfg = getattr(obs_group_cfg, "velocity_commands")
                    setattr(
                        obs_group_cfg,
                        "velocity_commands",
                        ObsTerm(
                            func=_keyboard_velocity_command,
                            scale=getattr(velocity_term_cfg, "scale", 1.0),
                        ),
                    )

        events_cfg = getattr(env_cfg, "events", None)
        if events_cfg is not None:
            for event_name in (
                "physics_material",
                "add_base_mass",
                "add_base_inertia",
                "add_base_com",
                "base_external_force_torque",
                "randomize_actuator_gains",
                "push_robot",
            ):
                if hasattr(events_cfg, event_name):
                    setattr(events_cfg, event_name, None)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    policy_term_names = env.unwrapped.observation_manager.active_terms.get("policy", [])
    policy_term_cfgs = env.unwrapped.observation_manager._group_obs_term_cfgs.get("policy", [])
    privileged_term_names = env.unwrapped.observation_manager.active_terms.get("privileged", [])
    privileged_term_cfgs = env.unwrapped.observation_manager._group_obs_term_cfgs.get("privileged", [])
    estimator_target_scale = 1.0
    if "base_lin_vel_xy" in privileged_term_names:
        privileged_idx = privileged_term_names.index("base_lin_vel_xy")
        privileged_cfg = privileged_term_cfgs[privileged_idx]
        if isinstance(privileged_cfg.scale, (float, int)):
            estimator_target_scale = float(privileged_cfg.scale)
    track_lin_vel_xy_cfg = getattr(getattr(env_cfg, "rewards", None), "track_lin_vel_xy_exp", None)
    track_lin_vel_xy_std = 1.0
    track_lin_vel_xy_weight = 1.0
    if track_lin_vel_xy_cfg is not None:
        track_lin_vel_xy_std = float(track_lin_vel_xy_cfg.params.get("std", track_lin_vel_xy_std))
        track_lin_vel_xy_weight = float(track_lin_vel_xy_cfg.weight)

    for term_name, term_cfg in zip(policy_term_names, policy_term_cfgs):
        if term_name not in {"joint_pos", "joint_vel"}:
            continue
        asset_cfg = term_cfg.params.get("asset_cfg")
        if asset_cfg is None:
            print(f"[INFO] policy/{term_name} resolved asset_cfg: None")
            continue
        asset = env.unwrapped.scene[asset_cfg.name]
        if asset_cfg.joint_ids == slice(None):
            joint_ids = list(range(len(asset.joint_names)))
        else:
            joint_ids = list(asset_cfg.joint_ids)
        joint_names = [asset.joint_names[i] for i in joint_ids]
        print(f"[INFO] policy/{term_name} resolved joint_ids: {joint_ids}")
        print(f"[INFO] policy/{term_name} resolved joint_names: {joint_names}")
        print(f"[INFO] policy/{term_name} preserve_order: {asset_cfg.preserve_order}")

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    joint_pos_term = None
    if "joint_pos" in env.unwrapped.action_manager.active_terms:
        candidate_joint_pos_term = env.unwrapped.action_manager.get_term("joint_pos")
        if getattr(candidate_joint_pos_term.cfg, "feedforward_enabled", False):
            joint_pos_term = candidate_joint_pos_term

    if args_cli.keyboard:
        print("[INFO] Keyboard teleoperation enabled for policy velocity_commands.")
        print(keyboard_interface)
        print("[INFO] Keyboard mode: disabled observation corruption and domain-randomization events.")
        print("[INFO] Keyboard mode: using raw Se2Keyboard commands; command manager is left unchanged.")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    checkpoint_iteration = _resolve_checkpoint_iteration(runner, resume_path)
    if joint_pos_term is not None and checkpoint_iteration is not None:
        steps_per_iteration = max(1, int(getattr(joint_pos_term, "_k_ff_steps_per_iteration", 1)))
        env.unwrapped.common_step_counter = checkpoint_iteration * steps_per_iteration
        if hasattr(joint_pos_term, "_update_k_ff_schedule"):
            joint_pos_term._update_k_ff_schedule()
        if hasattr(joint_pos_term, "k_ff"):
            print(
                "[INFO] Play mode: initialized joint_pos k_ff from checkpoint "
                f"iteration {checkpoint_iteration} -> k_ff={joint_pos_term.k_ff:.4f}."
            )
    elif joint_pos_term is not None:
        print("[INFO] Play mode: could not resolve checkpoint iteration, keeping joint_pos k_ff as configured.")

    if joint_pos_term is not None and args_cli.kff is not None:
        _override_joint_pos_k_ff(joint_pos_term, args_cli.kff)
        if args_cli.kff <= 0.0:
            print("[INFO] Play mode: joint_pos feedforward disabled via --kff 0.")
        else:
            print(f"[INFO] Play mode: overriding joint_pos k_ff to {args_cli.kff:.4f} via --kff.")

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic
    velocity_label = "base_lin_vel" if is_cenet_policy(policy_nn) else "base_lin_vel_xy"

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    exported_artifacts = []

    def _attempt_export(label, export_fn, *args, **kwargs):
        try:
            export_fn(*args, **kwargs)
            exported_artifacts.append(label)
        except Exception as exc:
            print(f"[WARN] Failed to export {label}: {type(exc).__name__}: {exc}")

    if is_estimator_policy(policy_nn):
        _attempt_export(
            "split estimator TorchScript policy",
            export_estimator_only_policy_as_jit,
            policy_nn,
            path=export_model_dir,
            filename="estimator_policy.pt",
        )
        _attempt_export(
            "split estimator ONNX policy",
            export_estimator_only_policy_as_onnx,
            policy_nn,
            path=export_model_dir,
            filename="estimator_policy.onnx",
            opset_version=args_cli.onnx_opset,
        )
        _attempt_export(
            "split actor TorchScript policy",
            export_actor_with_estimator_history_as_jit,
            policy_nn,
            path=export_model_dir,
            filename="actor_policy.pt",
        )
        _attempt_export(
            "split actor ONNX policy",
            export_actor_with_estimator_history_as_onnx,
            policy_nn,
            path=export_model_dir,
            filename="actor_policy.onnx",
            opset_version=args_cli.onnx_opset,
        )
        _attempt_export(
            "split estimator policy metadata",
            export_split_estimator_policy_metadata,
            policy_nn,
            path=export_model_dir,
            filename="policy_split_metadata.json",
        )
        _attempt_export(
            "estimator TorchScript policy",
            export_estimator_policy_as_jit,
            policy_nn,
            path=export_model_dir,
            filename="policy.pt",
        )
        _attempt_export(
            "estimator ONNX policy",
            export_estimator_policy_as_onnx,
            policy_nn,
            path=export_model_dir,
            filename="policy.onnx",
            opset_version=args_cli.onnx_opset,
        )
        _attempt_export(
            "estimator policy metadata",
            export_estimator_policy_metadata,
            policy_nn,
            path=export_model_dir,
            filename="policy_metadata.json",
        )
        if exported_artifacts:
            print(f"[INFO] Exported estimator deploy artifacts (split + single-engine) to: {export_model_dir}")
        else:
            print("[WARN] Estimator policy export failed; continuing play without exported artifacts.")
    elif is_cenet_policy(policy_nn):
        _attempt_export(
            "CENet TorchScript policy",
            export_cenet_policy_as_jit,
            policy_nn,
            path=export_model_dir,
            filename="policy.pt",
        )
        _attempt_export(
            "CENet ONNX policy",
            export_cenet_policy_as_onnx,
            policy_nn,
            path=export_model_dir,
            filename="policy.onnx",
            opset_version=args_cli.onnx_opset,
        )
        _attempt_export(
            "CENet policy metadata",
            export_cenet_policy_metadata,
            policy_nn,
            path=export_model_dir,
            filename="policy_metadata.json",
        )
        if exported_artifacts:
            print(f"[INFO] Exported CENet deploy artifacts to: {export_model_dir}")
        else:
            print("[WARN] CENet policy export failed; continuing play without exported artifacts.")
    else:
        _attempt_export(
            "TorchScript policy",
            export_policy_as_jit,
            policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.pt",
        )
        _attempt_export(
            "ONNX policy",
            _export_policy_as_onnx_with_opset,
            policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.onnx",
            opset_version=args_cli.onnx_opset,
        )
        if not exported_artifacts:
            print("[WARN] Policy export failed; continuing play without exported artifacts.")

    dt = env.unwrapped.step_dt
    print_interval = int(0.5 / dt)  # print every 0.5 seconds
    print(f"[INFO] Print interval: every {print_interval} steps (0.5s)")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            obs_for_policy = obs
            keyboard_command_for_policy = None
            if args_cli.keyboard and keyboard_command_state["command"] is not None:
                keyboard_command_for_policy = keyboard_command_state["command"].detach().clone()
            actions = policy(obs_for_policy)
            
            # env stepping
            obs, _, _, _ = env.step(actions)
            
            # print every 0.5 seconds (after step to get processed actions)
            if timestep % print_interval == 0:
                # Get action info from action_manager
                action_manager = env.unwrapped.action_manager
                action_values = actions[0].cpu().numpy()

                print(f"\n[Step {timestep}, Time {timestep * dt:.1f}s]")
                if args_cli.keyboard:
                    if "policy" in obs_for_policy.keys():
                        actor_obs_tensor = obs_for_policy["policy"]
                        actor_obs = actor_obs_tensor[0].detach().cpu().numpy()
                    else:
                        actor_obs = obs_for_policy[0].detach().cpu().numpy()

                    obs_len = len(actor_obs)
                    policy_term_dims = env.unwrapped.observation_manager.group_obs_term_dim.get("policy", [])
                    print(f"  Current policy observation: total_dim={obs_len}")
                    print("  Policy terms (current frame only):")

                    idx = 0
                    for term_name, term_dim, term_cfg in zip(policy_term_names, policy_term_dims, policy_term_cfgs):
                        flattened_dim = math.prod(term_dim)
                        term_history = term_cfg.history_length if term_cfg.history_length is not None else 1
                        base_dim = flattened_dim // term_history if term_history > 1 else flattened_dim
                        total_dim = flattened_dim
                        if idx + total_dim > obs_len:
                            break
                        term_values = actor_obs[idx : idx + total_dim]
                        current_vals = term_values[-base_dim:] if term_history > 1 else term_values
                        print(f"    {term_name}: {current_vals}")
                        idx += total_dim

                estimated_velocity = None
                if hasattr(policy_nn, "estimated_velocity") and policy_nn.estimated_velocity is not None:
                    estimated_velocity = policy_nn.estimated_velocity[0].detach().cpu()

                    if "privileged" in obs_for_policy.keys():
                        true_velocity = obs_for_policy["privileged"][0].detach().cpu()
                        true_velocity = true_velocity / estimator_target_scale
                    else:
                        velocity_dim = estimated_velocity.numel()
                        true_velocity = env.unwrapped.scene["robot"].data.root_lin_vel_b[0, :velocity_dim].detach().cpu()

                    velocity_error = estimated_velocity - true_velocity
                    velocity_error_l2 = torch.linalg.vector_norm(velocity_error).item()

                    print("  Velocity estimator:")
                    print(f"    estimated {velocity_label}: {estimated_velocity.numpy()}")
                    print(f"    true {velocity_label}:      {true_velocity.numpy()}")
                    print(f"    error:                     {velocity_error.numpy()} |l2|={velocity_error_l2:.4f}")

                command_xy = None
                command_source = None
                if keyboard_command_for_policy is not None:
                    command_xy = keyboard_command_for_policy[0, :2]
                    command_source = "keyboard"
                elif "base_velocity" in env.unwrapped.command_manager.active_terms:
                    command_xy = env.unwrapped.command_manager.get_command("base_velocity")[0, :2]
                    command_source = "command_manager"

                if command_xy is not None:
                    true_vel_xy = env.unwrapped.scene["robot"].data.root_lin_vel_b[0, :2]
                    tracking_error_xy = true_vel_xy - command_xy
                    upright_factor = torch.clamp(-env.unwrapped.scene["robot"].data.projected_gravity_b[0, 2], 0.0, 0.7) / 0.7
                    track_lin_vel_xy_exp = (
                        torch.exp(-torch.sum(torch.square(command_xy - true_vel_xy)) / (track_lin_vel_xy_std**2))
                        * upright_factor
                        * track_lin_vel_xy_weight
                    )

                    print("  Tracking diagnostics:")
                    print(f"    {command_source}_command_xy: {command_xy.detach().cpu().numpy()}")
                    print(f"    true_vel_xy:         {true_vel_xy.detach().cpu().numpy()}")
                    print(f"    tracking_error_xy:   {tracking_error_xy.detach().cpu().numpy()}")
                    print(f"    track_lin_vel_xy_exp:{track_lin_vel_xy_exp.item():.4f}")
                    if estimated_velocity is not None:
                        estimated_velocity_xy = estimated_velocity[: true_vel_xy.numel()]
                        estimated_tracking_error_xy = estimated_velocity_xy - command_xy.detach().cpu()
                        estimator_error_xy = estimated_velocity_xy - true_vel_xy.detach().cpu()
                        print(f"    estimated_vel_xy:    {estimated_velocity_xy.numpy()}")
                        print(f"    estimated_track_err: {estimated_tracking_error_xy.numpy()}")
                        print(f"    estimator_error_xy:  {estimator_error_xy.numpy()}")

                base_height_world, ground_height, base_height_relative = _get_base_height_debug(env)
                print("  Robot state:")
                print(f"    base_height_world_z: {base_height_world:.4f}")
                if ground_height is not None:
                    print(f"    ground_height_z:     {ground_height:.4f}")
                if base_height_relative is not None:
                    print(f"    base_height_rel:     {base_height_relative:.4f}")

                if env.unwrapped.num_envs == 1:
                    if args_cli.keyboard:
                        print("  Keyboard command (policy observation):")
                        raw_keyboard_command = keyboard_command_state.get("raw_command")
                        if raw_keyboard_command is not None:
                            raw_cmd = raw_keyboard_command[0].detach().cpu().numpy()
                            print(f"    raw keyboard cmd: {raw_cmd}")
                        if keyboard_command_for_policy is not None:
                            policy_cmd = keyboard_command_for_policy[0].detach().cpu().numpy()
                            print(f"    policy command:   {policy_cmd}")
                            desired_heading = keyboard_command_state.get("desired_heading")
                            current_heading = keyboard_command_state.get("current_heading")
                            heading_error = keyboard_command_state.get("heading_error")
                            yaw_mode = keyboard_command_state.get("yaw_mode")
                            if desired_heading is not None and current_heading is not None and heading_error is not None:
                                print(f"    yaw_mode:         {yaw_mode}")
                                print(f"    desired_heading:  {desired_heading[0].item():.4f}")
                                print(f"    current_heading:  {current_heading[0].item():.4f}")
                                print(f"    heading_error:    {heading_error[0].item():.4f}")
                        else:
                            print("    raw keyboard cmd: <unavailable>")
                    else:
                        print("  Commands:")
                        for command_name in env.unwrapped.command_manager.active_terms:
                            try:
                                current_command = (
                                    env.unwrapped.command_manager.get_command(command_name)[0].detach().cpu().numpy()
                                )
                            except Exception as exc:
                                print(f"    {command_name}: <unavailable: {exc}>")
                                continue
                            print(f"    {command_name}: {current_command}")
                
                print(f"  Actions (synced):")
                print(f"    {'Idx':<5} {'Term->Joint':<35} {'Raw':<10} {'Scale':<8} {'Applied':<12} {'Offset/Note':<15}")
                print(f"    {'-'*80}")
                idx = 0
                ff_debug_terms = []
                for term_name, term in action_manager._terms.items():
                    processed = term.processed_actions[0].cpu().numpy()
                    joint_names = getattr(term, "_joint_names", None)
                    if joint_names is None:
                        joint_names = getattr(term, "_combined_action_names", None)
                    if joint_names is None and hasattr(term, "_leg_joint_names") and hasattr(term, "_wheel_joint_names"):
                        joint_names = [
                            term._leg_joint_names[0],
                            term._leg_joint_names[1],
                            term._leg_joint_names[2],
                            term._wheel_joint_names[0],
                            term._leg_joint_names[3],
                            term._leg_joint_names[4],
                            term._leg_joint_names[5],
                            term._wheel_joint_names[1],
                        ]

                    leg_action_indices = getattr(term, "_leg_action_indices", None)
                    wheel_action_indices = getattr(term, "_wheel_action_indices", None)
                    leg_offset = getattr(term, "_leg_offset", None)
                    offset = getattr(term, "_offset", None)

                    for i, joint_name in enumerate(joint_names):
                        raw_val = action_values[idx]
                        applied_val = processed[i]
                        scale_val = term._scale if isinstance(term._scale, float) else term._scale[0, i].item()

                        if wheel_action_indices is not None and i in wheel_action_indices.tolist():
                            note = "effort, vel=0"
                        elif leg_action_indices is not None and leg_offset is not None and i in leg_action_indices.tolist():
                            leg_local_idx = leg_action_indices.tolist().index(i)
                            offset_val = leg_offset[0, leg_local_idx].item()
                            note = f"offset={offset_val:.4f}"
                        elif hasattr(term, 'cfg') and hasattr(term.cfg, 'use_default_offset') and term.cfg.use_default_offset:
                            # Position control: target = raw * scale + default_pos
                            offset_val = offset if isinstance(offset, float) else offset[0, i].item()
                            note = f"offset={offset_val:.4f}"
                        else:
                            note = "no offset"

                        print(f"    [{idx:<3}] {term_name}->{joint_name:<25} {raw_val:>8.4f} ×{scale_val:<6.2f} ={applied_val:>10.4f}  {note:<15}")
                        idx += 1
                    if hasattr(term, "ff_actions") and hasattr(term, "trigger_signal"):
                        ff_debug_terms.append((term_name, term))

                for term_name, term in ff_debug_terms:
                    lift_signal = term.lift_signal[0].detach().cpu().numpy()
                    ff_signal = term.ff_signal[0].detach().cpu().numpy()
                    trigger_signal = term.trigger_signal[0].detach().cpu().numpy()
                    ff_actions = term.ff_actions[0].detach().cpu().numpy()
                    ff_contribution = term.ff_contribution[0].detach().cpu().numpy()

                    print(f"  FF debug ({term_name}):")
                    print(f"    k_ff:            {term.k_ff:.4f}")
                    print(f"    lift_signal:     {lift_signal}")
                    print(f"    ff_signal:       {ff_signal}")
                    print(f"    trigger_signal:  {trigger_signal}")
                    print(f"    ff raw actions:")
                    ff_joint_names = getattr(term, "_joint_names", None)
                    if hasattr(term, "_leg_joint_names"):
                        ff_joint_names = term._leg_joint_names
                    for joint_name, ff_raw, ff_scaled in zip(ff_joint_names, ff_actions, ff_contribution):
                        print(
                            f"      {joint_name:<25} raw={ff_raw:>8.4f}  contribution={ff_scaled:>8.4f}"
                        )
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
