from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions import actions_cfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_LEFT_LEG_INDEX = 0
_RIGHT_LEG_INDEX = 1
_FIRST_LEFT_LEG = 1
_FIRST_RIGHT_LEG = 2


def _leg_index_from_name(name: str) -> int:
    """Map a left/right joint or body name to the public [left, right] leg order."""
    return _RIGHT_LEG_INDEX if "right" in name.lower() else _LEFT_LEG_INDEX


class JointPositionWithFeedforwardAction(JointPositionAction):
    """Joint position action with contact-triggered feedforward trajectory injection."""

    cfg: "JointPositionWithFeedforwardActionCfg"

    def __init__(self, cfg: "JointPositionWithFeedforwardActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._ff_enabled = cfg.feedforward_enabled
        self._k_fb = cfg.k_fb
        self._k_ff = cfg.k_ff
        self._initial_k_ff = cfg.k_ff
        self._ff_period = cfg.feedforward_period
        self._contact_trigger_enabled = cfg.contact_trigger_enabled
        self._force_threshold = cfg.contact_force_threshold
        self._followup_trigger_delay = cfg.followup_trigger_delay_factor * self._ff_period
        self._k_ff_anneal_enabled = cfg.k_ff_anneal_enabled
        self._k_ff_final = cfg.k_ff_final
        self._k_ff_start_iteration = cfg.k_ff_start_iteration
        self._k_ff_anneal_iterations = cfg.k_ff_anneal_iterations
        self._k_ff_steps_per_iteration = max(1, cfg.k_ff_steps_per_iteration)

        self._ff_amplitude = torch.zeros(self._num_joints, device=self.device)
        if isinstance(cfg.feedforward_amplitude, dict):
            import re

            for i, name in enumerate(self._joint_names):
                for pattern, amplitude in cfg.feedforward_amplitude.items():
                    if re.match(pattern, name):
                        self._ff_amplitude[i] = amplitude
                        break
        else:
            self._ff_amplitude[:] = float(cfg.feedforward_amplitude)

        if cfg.feedforward_joint_names is not None:
            ff_joint_ids, ff_joint_names = self._asset.find_joints(cfg.feedforward_joint_names)
            if isinstance(self._joint_ids, slice):
                joint_ids_list = list(range(self._asset.num_joints))
            else:
                joint_ids_list = list(self._joint_ids)
            self._ff_local_ids = torch.tensor(
                [joint_ids_list.index(joint_id) for joint_id in ff_joint_ids if joint_id in joint_ids_list],
                device=self.device,
                dtype=torch.long,
            )
            self._ff_leg_mapping = torch.tensor(
                [
                    _leg_index_from_name(name)
                    for joint_id, name in zip(ff_joint_ids, ff_joint_names)
                    if joint_id in joint_ids_list
                ],
                device=self.device,
                dtype=torch.long,
            )
        else:
            self._ff_local_ids = torch.arange(self._num_joints, device=self.device, dtype=torch.long)
            self._ff_leg_mapping = torch.tensor(
                [_leg_index_from_name(name) for name in self._joint_names],
                device=self.device,
                dtype=torch.long,
            )

        self._time = torch.zeros(self.num_envs, 2, device=self.device)
        self._contact_sensor = None
        self._lifting_state = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._first_leg = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_lift_signal = torch.zeros(self.num_envs, 2, device=self.device)
        self._last_ff_signal = torch.zeros(self.num_envs, 2, device=self.device)
        self._last_trigger_signal = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._last_ff_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._last_ff_contribution = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._last_blended_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    @property
    def lifting_state(self) -> torch.Tensor:
        return self._lifting_state

    @property
    def k_ff(self) -> float:
        return float(self._k_ff)

    @property
    def initial_k_ff(self) -> float:
        return float(self._initial_k_ff)

    @property
    def lift_signal(self) -> torch.Tensor:
        return self._last_lift_signal

    @property
    def ff_signal(self) -> torch.Tensor:
        return self._last_ff_signal

    @property
    def trigger_signal(self) -> torch.Tensor:
        return self._last_trigger_signal

    @property
    def ff_actions(self) -> torch.Tensor:
        return self._last_ff_actions

    @property
    def ff_contribution(self) -> torch.Tensor:
        return self._last_ff_contribution

    @property
    def blended_actions(self) -> torch.Tensor:
        return self._last_blended_actions

    @property
    def controlled_joint_ids(self) -> torch.Tensor:
        if isinstance(self._joint_ids, slice):
            return torch.arange(self._asset.num_joints, device=self.device, dtype=torch.long)
        return torch.as_tensor(self._joint_ids, device=self.device, dtype=torch.long)

    @property
    def ff_joint_local_ids(self) -> torch.Tensor:
        return self._ff_local_ids

    @property
    def ff_target_positions(self) -> torch.Tensor:
        # Raw FF trajectory target; final command scaling is represented separately.
        return self._last_ff_actions + self._offset

    def _update_k_ff_schedule(self):
        if not (self._ff_enabled and self._k_ff_anneal_enabled):
            return

        current_iteration = self._env.common_step_counter // self._k_ff_steps_per_iteration
        if current_iteration < self._k_ff_start_iteration:
            current_k_ff = self._initial_k_ff
        else:
            progress = min(
                1.0,
                (current_iteration - self._k_ff_start_iteration) / max(1, self._k_ff_anneal_iterations),
            )
            current_k_ff = self._initial_k_ff + (self._k_ff_final - self._initial_k_ff) * progress
        self._k_ff = float(current_k_ff)

    def _clear_feedforward_state(self):
        self._time.zero_()
        self._lifting_state.zero_()
        self._first_leg.zero_()
        self._last_lift_signal.zero_()
        self._last_ff_signal.zero_()
        self._last_trigger_signal.zero_()
        self._last_ff_actions.zero_()
        self._last_ff_contribution.zero_()
        if hasattr(self, "_contact_force_history"):
            self._contact_force_history.zero_()

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._update_k_ff_schedule()
        self._last_lift_signal.zero_()
        self._last_ff_signal.zero_()
        self._last_trigger_signal.zero_()
        self._last_ff_actions.zero_()
        self._last_ff_contribution.zero_()
        ff_actions = torch.zeros_like(self._raw_actions)

        phase_enabled = self._ff_enabled and (self._contact_trigger_enabled or self._k_ff > 0.0)
        if phase_enabled:
            if self._contact_trigger_enabled:
                lift_signal = self._compute_lift_signal()
            else:
                lift_signal = torch.ones(self.num_envs, 2, device=self.device)
                self._last_trigger_signal.zero_()

            self._last_lift_signal[:] = lift_signal
            self._time += self._env.step_dt * lift_signal
            phase = 2.0 * math.pi * self._time / self._ff_period
            ff_signal = 0.5 * (1.0 - torch.cos(phase))
            self._last_ff_signal[:] = ff_signal

            if self._k_ff > 0.0:
                for i, local_id in enumerate(self._ff_local_ids):
                    leg_idx = self._ff_leg_mapping[i]
                    ff_actions[:, local_id] = (
                        ff_signal[:, leg_idx]
                        * self._ff_amplitude[local_id]
                        * lift_signal[:, leg_idx]
                    )

            self._last_ff_actions[:] = ff_actions
            self._last_ff_contribution[:] = self._k_ff * ff_actions
            blended_actions = self._k_fb * self._raw_actions + self._k_ff * ff_actions
        else:
            self._clear_feedforward_state()
            blended_actions = self._raw_actions

        self._last_blended_actions[:] = blended_actions
        self._processed_actions = blended_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._time[env_ids] = 0.0
        self._lifting_state[env_ids] = False
        self._first_leg[env_ids] = 0
        self._last_lift_signal[env_ids] = 0.0
        self._last_ff_signal[env_ids] = 0.0
        self._last_trigger_signal[env_ids] = False
        self._last_ff_actions[env_ids] = 0.0
        self._last_ff_contribution[env_ids] = 0.0
        self._last_blended_actions[env_ids] = 0.0
        if hasattr(self, "_contact_force_history"):
            self._contact_force_history[:, env_ids] = 0.0

    def _compute_lift_signal(self) -> torch.Tensor:
        if self._contact_sensor is None:
            from isaaclab.sensors import ContactSensor
            import re

            self._contact_sensor: ContactSensor = self._env.scene.sensors[self.cfg.contact_sensor_name]
            self._right_foot_id = None
            self._left_foot_id = None
            for i, name in enumerate(self._contact_sensor.body_names):
                if re.match(self.cfg.contact_body_pattern, name):
                    if "right" in name.lower():
                        self._right_foot_id = i
                    elif "left" in name.lower():
                        self._left_foot_id = i
            self._contact_force_history = torch.zeros(3, self.num_envs, 2, device=self.device)

        forces_xyz = self._contact_sensor.data.net_forces_w
        left_xy = torch.zeros(self.num_envs, device=self.device)
        right_xy = torch.zeros(self.num_envs, device=self.device)
        if self._left_foot_id is not None:
            left_force = forces_xyz[:, self._left_foot_id, :]
            left_xy = torch.sqrt(left_force[:, 0] ** 2 + left_force[:, 1] ** 2)
        if self._right_foot_id is not None:
            right_force = forces_xyz[:, self._right_foot_id, :]
            right_xy = torch.sqrt(right_force[:, 0] ** 2 + right_force[:, 1] ** 2)

        feet_xy = torch.stack([left_xy, right_xy], dim=1)
        self._contact_force_history[:-1] = self._contact_force_history[1:].clone()
        self._contact_force_history[-1] = feet_xy

        avg_force = self._contact_force_history.mean(dim=0)
        left_contact = avg_force[:, _LEFT_LEG_INDEX] > self._force_threshold
        right_contact = avg_force[:, _RIGHT_LEG_INDEX] > self._force_threshold

        stable_contact = (self._contact_force_history > self._force_threshold).all(dim=0)
        left_stable = stable_contact[:, _LEFT_LEG_INDEX]
        right_stable = stable_contact[:, _RIGHT_LEG_INDEX]

        trigger_left = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        trigger_right = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        no_leg_lifting = ~self._lifting_state[:, _LEFT_LEG_INDEX] & ~self._lifting_state[:, _RIGHT_LEG_INDEX]
        left_followup_ready = (
            (self._first_leg == _FIRST_RIGHT_LEG)
            & self._lifting_state[:, _RIGHT_LEG_INDEX]
            & ~self._lifting_state[:, _LEFT_LEG_INDEX]
            & (self._time[:, _RIGHT_LEG_INDEX] >= self._followup_trigger_delay)
        )
        right_followup_ready = (
            (self._first_leg == _FIRST_LEFT_LEG)
            & self._lifting_state[:, _LEFT_LEG_INDEX]
            & ~self._lifting_state[:, _RIGHT_LEG_INDEX]
            & (self._time[:, _LEFT_LEG_INDEX] >= self._followup_trigger_delay)
        )
        can_trigger_left = ~self._lifting_state[:, _LEFT_LEG_INDEX] & (no_leg_lifting | left_followup_ready)
        can_trigger_right = ~self._lifting_state[:, _RIGHT_LEG_INDEX] & (no_leg_lifting | right_followup_ready)

        left_only_eligible = can_trigger_left & ~can_trigger_right
        right_only_eligible = can_trigger_right & ~can_trigger_left
        trigger_left = trigger_left | (left_only_eligible & left_contact)
        trigger_right = trigger_right | (right_only_eligible & right_contact)

        both_eligible = can_trigger_right & can_trigger_left
        only_left = left_contact & ~right_contact & both_eligible
        only_right = right_contact & ~left_contact & both_eligible
        trigger_left = trigger_left | only_left
        trigger_right = trigger_right | only_right

        both_contact = right_contact & left_contact & both_eligible
        trigger_left = trigger_left | (both_contact & ~right_stable & left_stable)
        trigger_right = trigger_right | (both_contact & right_stable & ~left_stable)

        both_stable = both_contact & right_stable & left_stable
        trigger_left = trigger_left | (
            both_stable & (avg_force[:, _LEFT_LEG_INDEX] > avg_force[:, _RIGHT_LEG_INDEX])
        )
        trigger_right = trigger_right | (
            both_stable & (avg_force[:, _RIGHT_LEG_INDEX] >= avg_force[:, _LEFT_LEG_INDEX])
        )
        self._last_trigger_signal[:, _LEFT_LEG_INDEX] = trigger_left
        self._last_trigger_signal[:, _RIGHT_LEG_INDEX] = trigger_right

        self._lifting_state[:, _LEFT_LEG_INDEX] = self._lifting_state[:, _LEFT_LEG_INDEX] | trigger_left
        self._lifting_state[:, _RIGHT_LEG_INDEX] = self._lifting_state[:, _RIGHT_LEG_INDEX] | trigger_right

        self._first_leg = torch.where(
            trigger_left & (self._first_leg == 0), torch.full_like(self._first_leg, _FIRST_LEFT_LEG), self._first_leg
        )
        self._first_leg = torch.where(
            trigger_right & (self._first_leg == 0), torch.full_like(self._first_leg, _FIRST_RIGHT_LEG), self._first_leg
        )

        left_done = self._time[:, _LEFT_LEG_INDEX] >= self._ff_period
        right_done = self._time[:, _RIGHT_LEG_INDEX] >= self._ff_period
        self._time[:, _LEFT_LEG_INDEX] = torch.where(
            left_done, torch.zeros_like(self._time[:, _LEFT_LEG_INDEX]), self._time[:, _LEFT_LEG_INDEX]
        )
        self._time[:, _RIGHT_LEG_INDEX] = torch.where(
            right_done, torch.zeros_like(self._time[:, _RIGHT_LEG_INDEX]), self._time[:, _RIGHT_LEG_INDEX]
        )
        self._lifting_state[:, _LEFT_LEG_INDEX] = torch.where(
            left_done,
            torch.zeros_like(self._lifting_state[:, _LEFT_LEG_INDEX]),
            self._lifting_state[:, _LEFT_LEG_INDEX],
        )
        self._lifting_state[:, _RIGHT_LEG_INDEX] = torch.where(
            right_done,
            torch.zeros_like(self._lifting_state[:, _RIGHT_LEG_INDEX]),
            self._lifting_state[:, _RIGHT_LEG_INDEX],
        )

        no_active_lifts = ~self._lifting_state.any(dim=1)
        self._first_leg = torch.where(no_active_lifts, torch.zeros_like(self._first_leg), self._first_leg)
        return self._lifting_state.float()


@configclass
class JointPositionWithFeedforwardActionCfg(ActionTermCfg):
    """Configuration for joint position action with contact-triggered feedforward support."""

    class_type: type[ActionTerm] = JointPositionWithFeedforwardAction

    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    preserve_order: bool = False
    use_default_offset: bool = True
    clip: dict[str, tuple[float, float]] | None = None

    feedforward_enabled: bool = False
    k_fb: float = 1.0
    k_ff: float = 0.0
    feedforward_period: float = 0.6
    feedforward_amplitude: float | dict[str, float] = 0.0
    feedforward_joint_names: list[str] | None = None
    contact_trigger_enabled: bool = False
    contact_sensor_name: str = "contact_forces"
    contact_body_pattern: str = ".*_leg_4"
    contact_force_threshold: float = 10.0
    followup_trigger_delay_factor: float = 0.0
    k_ff_anneal_enabled: bool = False
    k_ff_final: float = 0.0
    k_ff_start_iteration: int = 0
    k_ff_anneal_iterations: int = 0
    k_ff_steps_per_iteration: int = 24


class TitaJointPositionEffortAction(ActionTerm):
    """Tita-specific 8D action term with sim2sim-aligned ordering.

    The external action order is:

    ``[L1, L2, L3, L4, R1, R2, R3, R4]``

    where leg joints ``1/2/3`` are interpreted as position commands and wheel joints ``4`` are
    interpreted as velocity commands with velocity-proportional feedforward:

    - legs: ``q_des = q_default + k_ff * ff_offset + k_fb * leg_scale * action``
    - wheels: ``dq_des = wheel_scale * action + wheel_offset``
    - wheels: ``tau_ff = wheel_effort_gain * dq_des``

    With a wheel actuator configured as ``stiffness=0`` and ``damping=wheel_kd``, the resulting
    torque becomes ``tau = wheel_kd * (dq_des - dq) + tau_ff``.
    """

    cfg: "TitaJointPositionEffortActionCfg"

    def __init__(self, cfg: "TitaJointPositionEffortActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._asset: Articulation
        self._leg_joint_ids, self._leg_joint_names = self._asset.find_joints(
            cfg.leg_joint_names, preserve_order=cfg.preserve_order
        )
        self._wheel_joint_ids, self._wheel_joint_names = self._asset.find_joints(
            cfg.wheel_joint_names, preserve_order=cfg.preserve_order
        )
        self._num_leg_joints = len(self._leg_joint_ids)
        self._num_wheel_joints = len(self._wheel_joint_ids)

        if self._num_leg_joints != 6 or self._num_wheel_joints != 2:
            raise ValueError(
                "TitaJointPositionEffortAction expects 6 leg joints and 2 wheel joints, "
                f"but got {self._num_leg_joints} legs and {self._num_wheel_joints} wheels."
            )

        self._leg_action_indices = torch.tensor([0, 1, 2, 4, 5, 6], device=self.device, dtype=torch.long)
        self._wheel_action_indices = torch.tensor([3, 7], device=self.device, dtype=torch.long)
        self._action_dim = 8

        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._leg_raw_actions = torch.zeros(self.num_envs, self._num_leg_joints, device=self.device)
        self._wheel_raw_actions = torch.zeros(self.num_envs, self._num_wheel_joints, device=self.device)
        self._leg_processed_actions = torch.zeros_like(self._leg_raw_actions)
        self._wheel_command_actions = torch.zeros_like(self._wheel_raw_actions)
        self._wheel_effort_actions = torch.zeros_like(self._wheel_raw_actions)

        self._combined_action_names = [
            self._leg_joint_names[0],
            self._leg_joint_names[1],
            self._leg_joint_names[2],
            self._wheel_joint_names[0],
            self._leg_joint_names[3],
            self._leg_joint_names[4],
            self._leg_joint_names[5],
            self._wheel_joint_names[1],
        ]

        self._leg_scale = self._resolve_param(cfg.leg_scale, self._leg_joint_names, default=1.0).to(self.device)
        self._wheel_scale = self._resolve_param(cfg.wheel_scale, self._wheel_joint_names, default=1.0).to(self.device)
        self._wheel_effort_gain = self._resolve_param(
            cfg.wheel_effort_gain,
            self._wheel_joint_names,
            default=1.0,
        ).to(self.device)
        self._wheel_offset = self._resolve_param(cfg.wheel_offset, self._wheel_joint_names, default=0.0).to(
            self.device
        )

        if cfg.use_default_leg_offset:
            self._leg_offset = self._asset.data.default_joint_pos[:, self._leg_joint_ids].clone()
        else:
            self._leg_offset = self._resolve_param(cfg.leg_offset, self._leg_joint_names, default=0.0).to(
                self.device
            )

        # Expose per-dimension raw->command scales for diagnostics/reward helpers.
        self._scale = torch.ones(self.num_envs, self._action_dim, device=self.device)
        self._scale[:, self._leg_action_indices] = self._leg_scale.expand(self.num_envs, -1)
        self._scale[:, self._wheel_action_indices] = self._wheel_scale.expand(self.num_envs, -1)

        self._clip = None
        if cfg.clip is not None:
            self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                self.num_envs, self._action_dim, 1
            )
            index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.clip, self._combined_action_names)
            self._clip[:, index_list] = torch.tensor(value_list, device=self.device)

        self._ff_enabled = cfg.feedforward_enabled
        self._k_fb = cfg.k_fb
        self._k_ff = cfg.k_ff
        self._initial_k_ff = cfg.k_ff
        self._ff_period = cfg.feedforward_period
        self._contact_trigger_enabled = cfg.contact_trigger_enabled
        self._force_threshold = cfg.contact_force_threshold
        self._followup_trigger_delay = cfg.followup_trigger_delay_factor * self._ff_period
        self._k_ff_anneal_enabled = cfg.k_ff_anneal_enabled
        self._k_ff_final = cfg.k_ff_final
        self._k_ff_start_iteration = cfg.k_ff_start_iteration
        self._k_ff_anneal_iterations = cfg.k_ff_anneal_iterations
        self._k_ff_steps_per_iteration = max(1, cfg.k_ff_steps_per_iteration)

        self._ff_amplitude = torch.zeros(self._num_leg_joints, device=self.device)
        if isinstance(cfg.feedforward_amplitude, dict):
            for i, name in enumerate(self._leg_joint_names):
                for pattern, amplitude in cfg.feedforward_amplitude.items():
                    if re.match(pattern, name):
                        self._ff_amplitude[i] = amplitude
                        break
        else:
            self._ff_amplitude[:] = float(cfg.feedforward_amplitude)

        if cfg.feedforward_joint_names is not None:
            ff_joint_ids, ff_joint_names = self._asset.find_joints(cfg.feedforward_joint_names)
            leg_joint_ids_list = list(self._leg_joint_ids)
            self._ff_local_ids = torch.tensor(
                [leg_joint_ids_list.index(joint_id) for joint_id in ff_joint_ids if joint_id in leg_joint_ids_list],
                device=self.device,
                dtype=torch.long,
            )
            self._ff_leg_mapping = torch.tensor(
                [
                    _leg_index_from_name(name)
                    for joint_id, name in zip(ff_joint_ids, ff_joint_names)
                    if joint_id in leg_joint_ids_list
                ],
                device=self.device,
                dtype=torch.long,
            )
        else:
            self._ff_local_ids = torch.arange(self._num_leg_joints, device=self.device, dtype=torch.long)
            self._ff_leg_mapping = torch.tensor(
                [_leg_index_from_name(name) for name in self._leg_joint_names],
                device=self.device,
                dtype=torch.long,
            )

        self._time = torch.zeros(self.num_envs, 2, device=self.device)
        self._contact_sensor = None
        self._lifting_state = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._first_leg = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_lift_signal = torch.zeros(self.num_envs, 2, device=self.device)
        self._last_ff_signal = torch.zeros(self.num_envs, 2, device=self.device)
        self._last_trigger_signal = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._last_ff_actions = torch.zeros(self.num_envs, self._num_leg_joints, device=self.device)
        self._last_ff_contribution = torch.zeros(self.num_envs, self._num_leg_joints, device=self.device)
        self._last_blended_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._action_diag_period_steps = 2000
        self._action_diag_max_logs = 50
        self._action_diag_last_step = -self._action_diag_period_steps
        self._action_diag_count = 0

    @staticmethod
    def _resolve_param(
        value: float | dict[str, float] | Sequence[float],
        joint_names: list[str],
        default: float = 0.0,
    ) -> torch.Tensor:
        if isinstance(value, (float, int)):
            return torch.full((1, len(joint_names)), float(value))
        if isinstance(value, dict):
            tensor = torch.full((1, len(joint_names)), float(default))
            index_list, _, value_list = string_utils.resolve_matching_names_values(value, joint_names)
            tensor[:, index_list] = torch.tensor(value_list).view(1, -1)
            return tensor
        if isinstance(value, Sequence):
            if len(value) != len(joint_names):
                raise ValueError(f"Expected {len(joint_names)} values but got {len(value)}.")
            return torch.tensor(list(value), dtype=torch.float32).view(1, -1)
        raise ValueError(f"Unsupported parameter type: {type(value)}")

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def lifting_state(self) -> torch.Tensor:
        return self._lifting_state

    @property
    def k_ff(self) -> float:
        return float(self._k_ff)

    @property
    def initial_k_ff(self) -> float:
        return float(self._initial_k_ff)

    @property
    def lift_signal(self) -> torch.Tensor:
        return self._last_lift_signal

    @property
    def ff_signal(self) -> torch.Tensor:
        return self._last_ff_signal

    @property
    def trigger_signal(self) -> torch.Tensor:
        return self._last_trigger_signal

    @property
    def ff_actions(self) -> torch.Tensor:
        return self._last_ff_actions

    @property
    def ff_contribution(self) -> torch.Tensor:
        return self._last_ff_contribution

    @property
    def blended_actions(self) -> torch.Tensor:
        return self._last_blended_actions

    @property
    def controlled_joint_ids(self) -> torch.Tensor:
        return torch.as_tensor(self._leg_joint_ids, device=self.device, dtype=torch.long)

    @property
    def ff_joint_local_ids(self) -> torch.Tensor:
        return self._ff_local_ids

    @property
    def ff_target_positions(self) -> torch.Tensor:
        # Raw FF trajectory target; final command scaling is represented separately.
        return self._last_ff_actions + self._leg_offset

    def _update_k_ff_schedule(self):
        if not (self._ff_enabled and self._k_ff_anneal_enabled):
            return
        current_iteration = self._env.common_step_counter // self._k_ff_steps_per_iteration
        progress = min(
            1.0,
            max(
                0.0,
                (current_iteration - self._k_ff_start_iteration) / max(1, self._k_ff_anneal_iterations),
            ),
        )
        current_k_ff = self._initial_k_ff + (self._k_ff_final - self._initial_k_ff) * progress
        self._k_ff = float(current_k_ff)

    def _clear_feedforward_state(self):
        self._time.zero_()
        self._lifting_state.zero_()
        self._first_leg.zero_()
        self._last_lift_signal.zero_()
        self._last_ff_signal.zero_()
        self._last_trigger_signal.zero_()
        self._last_ff_actions.zero_()
        self._last_ff_contribution.zero_()
        if hasattr(self, "_contact_force_history"):
            self._contact_force_history.zero_()

    def _log_action_diagnostics(
        self,
        ff_offset: torch.Tensor,
        policy_offset: torch.Tensor,
        ff_term: torch.Tensor,
        policy_term: torch.Tensor,
    ):
        if self._action_diag_count >= self._action_diag_max_logs:
            return

        step = int(getattr(self._env, "common_step_counter", 0))
        if step - self._action_diag_last_step < self._action_diag_period_steps:
            return

        local_ids = self._ff_local_ids
        if local_ids.numel() == 0:
            local_ids = torch.arange(self._num_leg_joints, device=self.device, dtype=torch.long)

        def mean_abs_on_ff_joints(tensor: torch.Tensor) -> float:
            selected = tensor[:, local_ids]
            if selected.numel() == 0:
                return 0.0
            return float(selected.abs().mean().item())

        q_delta = self._leg_processed_actions - self._leg_offset
        print(
            "[diag][tita_action] "
            f"step={step} "
            f"alpha/k_ff={self._k_ff:.4f} "
            f"mean_abs_ff_offset={mean_abs_on_ff_joints(ff_offset):.4e} "
            f"mean_abs_policy_offset={mean_abs_on_ff_joints(policy_offset):.4e} "
            f"mean_abs_ff_term={mean_abs_on_ff_joints(ff_term):.4e} "
            f"mean_abs_policy_term={mean_abs_on_ff_joints(policy_term):.4e} "
            f"mean_abs_q_des_minus_q_default={float(q_delta.abs().mean().item()):.4e}"
        )
        self._action_diag_last_step = step
        self._action_diag_count += 1

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._leg_raw_actions[:] = actions[:, self._leg_action_indices]
        self._wheel_raw_actions[:] = actions[:, self._wheel_action_indices]

        self._update_k_ff_schedule()
        self._last_lift_signal.zero_()
        self._last_ff_signal.zero_()
        self._last_trigger_signal.zero_()
        self._last_ff_actions.zero_()
        self._last_ff_contribution.zero_()
        ff_actions = torch.zeros_like(self._leg_raw_actions)

        phase_enabled = self._ff_enabled and (self._contact_trigger_enabled or self._k_ff > 0.0)
        if phase_enabled:
            if self._contact_trigger_enabled:
                lift_signal = self._compute_lift_signal()
            else:
                lift_signal = torch.ones(self.num_envs, 2, device=self.device)
                self._last_trigger_signal.zero_()

            self._last_lift_signal[:] = lift_signal
            self._time += self._env.step_dt * lift_signal

            phase = 2.0 * math.pi * self._time / self._ff_period
            ff_signal = 0.5 * (1.0 - torch.cos(phase))
            self._last_ff_signal[:] = ff_signal

            if self._k_ff > 0.0:
                for i, local_id in enumerate(self._ff_local_ids):
                    leg_idx = self._ff_leg_mapping[i]
                    ff_actions[:, local_id] = (
                        ff_signal[:, leg_idx] * self._ff_amplitude[local_id] * lift_signal[:, leg_idx]
                    )

            self._last_ff_actions[:] = ff_actions
        else:
            self._clear_feedforward_state()

        leg_scale = self._leg_scale.to(self.device)
        policy_offset = self._k_fb * self._leg_raw_actions * leg_scale
        policy_term = policy_offset
        ff_term = self._k_ff * ff_actions
        self._last_ff_contribution[:] = ff_term
        self._leg_processed_actions[:] = self._leg_offset + ff_term + policy_term
        self._wheel_command_actions[:] = (
            self._wheel_raw_actions * self._wheel_scale.to(self.device) + self._wheel_offset.to(self.device)
        )

        self._processed_actions.zero_()
        self._processed_actions[:, self._leg_action_indices] = self._leg_processed_actions
        self._processed_actions[:, self._wheel_action_indices] = self._wheel_command_actions
        if self._clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
            self._leg_processed_actions[:] = self._processed_actions[:, self._leg_action_indices]
            self._wheel_command_actions[:] = self._processed_actions[:, self._wheel_action_indices]
        self._wheel_effort_actions[:] = self._wheel_command_actions * self._wheel_effort_gain.to(self.device)

        equivalent_ff_actions = torch.where(
            torch.abs(leg_scale) > 1.0e-6,
            ff_term / leg_scale,
            torch.zeros_like(ff_term),
        )
        blended_leg_actions = self._k_fb * self._leg_raw_actions + equivalent_ff_actions
        self._last_blended_actions.zero_()
        self._last_blended_actions[:, self._leg_action_indices] = blended_leg_actions
        self._last_blended_actions[:, self._wheel_action_indices] = self._wheel_raw_actions
        self._log_action_diagnostics(ff_actions, policy_offset, ff_term, policy_term)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._leg_processed_actions, joint_ids=self._leg_joint_ids)
        self._asset.set_joint_velocity_target(self._wheel_command_actions, joint_ids=self._wheel_joint_ids)
        self._asset.set_joint_effort_target(self._wheel_effort_actions, joint_ids=self._wheel_joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._leg_raw_actions[env_ids] = 0.0
        self._wheel_raw_actions[env_ids] = 0.0
        self._leg_processed_actions[env_ids] = 0.0
        self._wheel_command_actions[env_ids] = 0.0
        self._wheel_effort_actions[env_ids] = 0.0
        self._time[env_ids] = 0.0
        self._lifting_state[env_ids] = False
        self._first_leg[env_ids] = 0
        self._last_lift_signal[env_ids] = 0.0
        self._last_ff_signal[env_ids] = 0.0
        self._last_trigger_signal[env_ids] = False
        self._last_ff_actions[env_ids] = 0.0
        self._last_ff_contribution[env_ids] = 0.0
        self._last_blended_actions[env_ids] = 0.0
        if hasattr(self, "_contact_force_history"):
            self._contact_force_history[:, env_ids] = 0.0

    def _compute_lift_signal(self) -> torch.Tensor:
        if self._contact_sensor is None:
            from isaaclab.sensors import ContactSensor

            self._contact_sensor: ContactSensor = self._env.scene.sensors[self.cfg.contact_sensor_name]
            self._right_foot_id = None
            self._left_foot_id = None
            for i, name in enumerate(self._contact_sensor.body_names):
                if re.match(self.cfg.contact_body_pattern, name):
                    if "right" in name.lower():
                        self._right_foot_id = i
                    elif "left" in name.lower():
                        self._left_foot_id = i
            self._contact_force_history = torch.zeros(3, self.num_envs, 2, device=self.device)

        forces_xyz = self._contact_sensor.data.net_forces_w
        left_xy = torch.zeros(self.num_envs, device=self.device)
        right_xy = torch.zeros(self.num_envs, device=self.device)
        if self._left_foot_id is not None:
            left_force = forces_xyz[:, self._left_foot_id, :]
            left_xy = torch.sqrt(left_force[:, 0] ** 2 + left_force[:, 1] ** 2)
        if self._right_foot_id is not None:
            right_force = forces_xyz[:, self._right_foot_id, :]
            right_xy = torch.sqrt(right_force[:, 0] ** 2 + right_force[:, 1] ** 2)

        feet_xy = torch.stack([left_xy, right_xy], dim=1)
        self._contact_force_history[:-1] = self._contact_force_history[1:].clone()
        self._contact_force_history[-1] = feet_xy

        avg_force = self._contact_force_history.mean(dim=0)
        left_contact = avg_force[:, _LEFT_LEG_INDEX] > self._force_threshold
        right_contact = avg_force[:, _RIGHT_LEG_INDEX] > self._force_threshold

        stable_contact = (self._contact_force_history > self._force_threshold).all(dim=0)
        left_stable = stable_contact[:, _LEFT_LEG_INDEX]
        right_stable = stable_contact[:, _RIGHT_LEG_INDEX]

        trigger_left = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        trigger_right = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        no_leg_lifting = ~self._lifting_state[:, _LEFT_LEG_INDEX] & ~self._lifting_state[:, _RIGHT_LEG_INDEX]
        left_followup_ready = (
            (self._first_leg == _FIRST_RIGHT_LEG)
            & self._lifting_state[:, _RIGHT_LEG_INDEX]
            & ~self._lifting_state[:, _LEFT_LEG_INDEX]
            & (self._time[:, _RIGHT_LEG_INDEX] >= self._followup_trigger_delay)
        )
        right_followup_ready = (
            (self._first_leg == _FIRST_LEFT_LEG)
            & self._lifting_state[:, _LEFT_LEG_INDEX]
            & ~self._lifting_state[:, _RIGHT_LEG_INDEX]
            & (self._time[:, _LEFT_LEG_INDEX] >= self._followup_trigger_delay)
        )
        can_trigger_left = ~self._lifting_state[:, _LEFT_LEG_INDEX] & (no_leg_lifting | left_followup_ready)
        can_trigger_right = ~self._lifting_state[:, _RIGHT_LEG_INDEX] & (no_leg_lifting | right_followup_ready)

        left_only_eligible = can_trigger_left & ~can_trigger_right
        right_only_eligible = can_trigger_right & ~can_trigger_left
        trigger_left = trigger_left | (left_only_eligible & left_contact)
        trigger_right = trigger_right | (right_only_eligible & right_contact)

        both_eligible = can_trigger_right & can_trigger_left
        only_left = left_contact & ~right_contact & both_eligible
        only_right = right_contact & ~left_contact & both_eligible
        trigger_left = trigger_left | only_left
        trigger_right = trigger_right | only_right

        both_contact = right_contact & left_contact & both_eligible
        trigger_left = trigger_left | (both_contact & ~right_stable & left_stable)
        trigger_right = trigger_right | (both_contact & right_stable & ~left_stable)

        both_stable = both_contact & right_stable & left_stable
        trigger_left = trigger_left | (
            both_stable & (avg_force[:, _LEFT_LEG_INDEX] > avg_force[:, _RIGHT_LEG_INDEX])
        )
        trigger_right = trigger_right | (
            both_stable & (avg_force[:, _RIGHT_LEG_INDEX] >= avg_force[:, _LEFT_LEG_INDEX])
        )
        self._last_trigger_signal[:, _LEFT_LEG_INDEX] = trigger_left
        self._last_trigger_signal[:, _RIGHT_LEG_INDEX] = trigger_right

        self._lifting_state[:, _LEFT_LEG_INDEX] = self._lifting_state[:, _LEFT_LEG_INDEX] | trigger_left
        self._lifting_state[:, _RIGHT_LEG_INDEX] = self._lifting_state[:, _RIGHT_LEG_INDEX] | trigger_right

        self._first_leg = torch.where(
            trigger_left & (self._first_leg == 0), torch.full_like(self._first_leg, _FIRST_LEFT_LEG), self._first_leg
        )
        self._first_leg = torch.where(
            trigger_right & (self._first_leg == 0), torch.full_like(self._first_leg, _FIRST_RIGHT_LEG), self._first_leg
        )

        left_done = self._time[:, _LEFT_LEG_INDEX] >= self._ff_period
        right_done = self._time[:, _RIGHT_LEG_INDEX] >= self._ff_period
        self._time[:, _LEFT_LEG_INDEX] = torch.where(
            left_done, torch.zeros_like(self._time[:, _LEFT_LEG_INDEX]), self._time[:, _LEFT_LEG_INDEX]
        )
        self._time[:, _RIGHT_LEG_INDEX] = torch.where(
            right_done, torch.zeros_like(self._time[:, _RIGHT_LEG_INDEX]), self._time[:, _RIGHT_LEG_INDEX]
        )
        self._lifting_state[:, _LEFT_LEG_INDEX] = torch.where(
            left_done,
            torch.zeros_like(self._lifting_state[:, _LEFT_LEG_INDEX]),
            self._lifting_state[:, _LEFT_LEG_INDEX],
        )
        self._lifting_state[:, _RIGHT_LEG_INDEX] = torch.where(
            right_done,
            torch.zeros_like(self._lifting_state[:, _RIGHT_LEG_INDEX]),
            self._lifting_state[:, _RIGHT_LEG_INDEX],
        )

        no_active_lifts = ~self._lifting_state.any(dim=1)
        self._first_leg = torch.where(no_active_lifts, torch.zeros_like(self._first_leg), self._first_leg)
        return self._lifting_state.float()


class TitaJointPositionVelocityWithFeedforwardAction(TitaJointPositionEffortAction):
    """Tita-specific 8D action term with leg position targets and wheel velocity targets.

    The external action order is:

    ``[L1, L2, L3, L4, R1, R2, R3, R4]``

    Legs ``1/2/3`` keep the contact-triggered feedforward position target logic from
    :class:`TitaJointPositionEffortAction`, while wheel joints ``4`` are interpreted as velocity commands:

    - legs: ``q_des = q_default + k_ff * ff_offset + k_fb * leg_scale * action``
    - wheels: ``dq_des = wheel_scale * action + wheel_offset``
    """

    cfg: "TitaJointPositionVelocityWithFeedforwardActionCfg"

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._leg_raw_actions[:] = actions[:, self._leg_action_indices]
        self._wheel_raw_actions[:] = actions[:, self._wheel_action_indices]

        self._update_k_ff_schedule()
        self._last_lift_signal.zero_()
        self._last_ff_signal.zero_()
        self._last_trigger_signal.zero_()
        self._last_ff_actions.zero_()
        self._last_ff_contribution.zero_()
        ff_actions = torch.zeros_like(self._leg_raw_actions)

        phase_enabled = self._ff_enabled and (self._contact_trigger_enabled or self._k_ff > 0.0)
        if phase_enabled:
            if self._contact_trigger_enabled:
                lift_signal = self._compute_lift_signal()
            else:
                lift_signal = torch.ones(self.num_envs, 2, device=self.device)
                self._last_trigger_signal.zero_()

            self._last_lift_signal[:] = lift_signal
            self._time += self._env.step_dt * lift_signal

            phase = 2.0 * math.pi * self._time / self._ff_period
            ff_signal = 0.5 * (1.0 - torch.cos(phase))
            self._last_ff_signal[:] = ff_signal

            if self._k_ff > 0.0:
                for i, local_id in enumerate(self._ff_local_ids):
                    leg_idx = self._ff_leg_mapping[i]
                    ff_actions[:, local_id] = (
                        ff_signal[:, leg_idx] * self._ff_amplitude[local_id] * lift_signal[:, leg_idx]
                    )

            self._last_ff_actions[:] = ff_actions
        else:
            self._clear_feedforward_state()

        leg_scale = self._leg_scale.to(self.device)
        policy_offset = self._k_fb * self._leg_raw_actions * leg_scale
        policy_term = policy_offset
        ff_term = self._k_ff * ff_actions
        self._last_ff_contribution[:] = ff_term
        self._leg_processed_actions[:] = self._leg_offset + ff_term + policy_term
        self._wheel_command_actions[:] = (
            self._wheel_raw_actions * self._wheel_scale.to(self.device) + self._wheel_offset.to(self.device)
        )
        self._wheel_effort_actions.zero_()

        self._processed_actions.zero_()
        self._processed_actions[:, self._leg_action_indices] = self._leg_processed_actions
        self._processed_actions[:, self._wheel_action_indices] = self._wheel_command_actions
        if self._clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
            self._leg_processed_actions[:] = self._processed_actions[:, self._leg_action_indices]
            self._wheel_command_actions[:] = self._processed_actions[:, self._wheel_action_indices]

        equivalent_ff_actions = torch.where(
            torch.abs(leg_scale) > 1.0e-6,
            ff_term / leg_scale,
            torch.zeros_like(ff_term),
        )
        blended_leg_actions = self._k_fb * self._leg_raw_actions + equivalent_ff_actions
        self._last_blended_actions.zero_()
        self._last_blended_actions[:, self._leg_action_indices] = blended_leg_actions
        self._last_blended_actions[:, self._wheel_action_indices] = self._wheel_raw_actions
        self._log_action_diagnostics(ff_actions, policy_offset, ff_term, policy_term)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._leg_processed_actions, joint_ids=self._leg_joint_ids)
        self._asset.set_joint_velocity_target(self._wheel_command_actions, joint_ids=self._wheel_joint_ids)
        self._asset.set_joint_effort_target(self._wheel_effort_actions, joint_ids=self._wheel_joint_ids)


@configclass
class TitaJointPositionEffortActionCfg(ActionTermCfg):
    """Configuration for Tita leg-position and wheel velocity-plus-feedforward action."""

    class_type: type[ActionTerm] = TitaJointPositionEffortAction

    leg_joint_names: list[str] = MISSING
    wheel_joint_names: list[str] = MISSING
    leg_scale: float | dict[str, float] | tuple[float, ...] | list[float] = 1.0
    wheel_scale: float | dict[str, float] | tuple[float, ...] | list[float] = 1.0
    wheel_effort_gain: float | dict[str, float] | tuple[float, ...] | list[float] = 1.0
    leg_offset: float | dict[str, float] | tuple[float, ...] | list[float] = 0.0
    wheel_offset: float | dict[str, float] | tuple[float, ...] | list[float] = 0.0
    preserve_order: bool = False
    use_default_leg_offset: bool = True
    clip: dict[str, tuple[float, float]] | None = None

    feedforward_enabled: bool = False
    k_fb: float = 1.0
    k_ff: float = 0.0
    feedforward_period: float = 0.6
    feedforward_amplitude: float | dict[str, float] = 0.0
    feedforward_joint_names: list[str] | None = None
    contact_trigger_enabled: bool = False
    contact_sensor_name: str = "contact_forces"
    contact_body_pattern: str = ".*_leg_4"
    contact_force_threshold: float = 10.0
    followup_trigger_delay_factor: float = 0.0
    k_ff_anneal_enabled: bool = False
    k_ff_final: float = 0.0
    k_ff_start_iteration: int = 0
    k_ff_anneal_iterations: int = 0
    k_ff_steps_per_iteration: int = 24


@configclass
class TitaJointPositionVelocityWithFeedforwardActionCfg(TitaJointPositionEffortActionCfg):
    """Configuration for Tita mixed leg-position and wheel-velocity action with feedforward support."""

    class_type: type[ActionTerm] = TitaJointPositionVelocityWithFeedforwardAction
