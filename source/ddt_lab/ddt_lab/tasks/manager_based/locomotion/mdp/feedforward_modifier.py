from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


class FeedforwardModifier:
    """Apply a contact-triggered feedforward trajectory on top of joint position targets."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        contact_sensor_name: str = "contact_forces",
        contact_body_pattern: str = ".*_leg_4",
        feedforward_joint_names: list[str] | None = None,
        feedforward_amplitude: dict[str, float] | None = None,
        feedforward_period: float = 0.6,
        k_ff: float = 0.3,
        contact_force_threshold: float = 50.0,
        followup_trigger_delay_factor: float = 0.0,
    ):
        self._env = env
        self._device = env.device
        self._num_envs = env.num_envs
        self._asset = env.scene[asset_cfg.name]

        if feedforward_joint_names is None:
            feedforward_joint_names = [
                "joint_right_leg_2",
                "joint_right_leg_3",
                "joint_left_leg_2",
                "joint_left_leg_3",
            ]
        self._ff_joint_ids, self._ff_joint_names = self._asset.find_joints(feedforward_joint_names)

        if feedforward_amplitude is None:
            feedforward_amplitude = {
                ".*_leg_2": 0.8,
                ".*_leg_3": -1.6,
            }
        self._ff_amplitude = torch.zeros(len(self._ff_joint_ids), device=self._device)
        import re

        for i, name in enumerate(self._ff_joint_names):
            for pattern, amplitude in feedforward_amplitude.items():
                if re.match(pattern, name):
                    self._ff_amplitude[i] = amplitude
                    break

        self._ff_leg_mapping = torch.tensor(
            [0 if "right" in name.lower() else 1 for name in self._ff_joint_names],
            device=self._device,
            dtype=torch.long,
        )

        self._ff_period = feedforward_period
        self._k_ff = k_ff
        self._force_threshold = contact_force_threshold
        self._followup_trigger_delay = followup_trigger_delay_factor * self._ff_period

        self._contact_sensor_name = contact_sensor_name
        self._contact_body_pattern = contact_body_pattern
        self._contact_sensor = None
        self._right_foot_id = None
        self._left_foot_id = None

        self._time = torch.zeros(self._num_envs, 2, device=self._device)
        self._contact_force_history = torch.zeros(3, self._num_envs, 2, device=self._device)
        self._lifting_state = torch.zeros(self._num_envs, 2, dtype=torch.bool, device=self._device)
        self._first_leg = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

    @property
    def lifting_state(self) -> torch.Tensor:
        return self._lifting_state

    def set_k_ff(self, k_ff: float):
        self._k_ff = k_ff

    def update(self):
        if self._k_ff <= 0.0:
            return

        lift_signal = self._compute_lift_signal()
        self._time += self._env.step_dt * lift_signal

        phase = 2.0 * math.pi * self._time / self._ff_period
        ff_signal = 0.5 * (1.0 - torch.cos(phase))

        ff_offset = torch.zeros(self._num_envs, len(self._ff_joint_ids), device=self._device)
        for i in range(len(self._ff_joint_ids)):
            leg_idx = self._ff_leg_mapping[i]
            ff_offset[:, i] = ff_signal[:, leg_idx] * self._ff_amplitude[i] * lift_signal[:, leg_idx]

        current_targets = self._asset.data.joint_pos_target[:, self._ff_joint_ids]
        new_targets = current_targets + self._k_ff * ff_offset
        self._asset.set_joint_position_target(new_targets, joint_ids=self._ff_joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._time[:] = 0.0
            self._contact_force_history[:] = 0.0
            self._lifting_state[:] = False
            self._first_leg[:] = 0
        else:
            self._time[env_ids] = 0.0
            self._contact_force_history[:, env_ids] = 0.0
            self._lifting_state[env_ids] = False
            self._first_leg[env_ids] = 0

    def _compute_lift_signal(self) -> torch.Tensor:
        if self._contact_sensor is None:
            self._init_contact_sensor()

        forces_xyz = self._contact_sensor.data.net_forces_w

        if self._right_foot_id is not None:
            right_force = forces_xyz[:, self._right_foot_id, :]
            right_xy = torch.sqrt(right_force[:, 0] ** 2 + right_force[:, 1] ** 2)
        else:
            right_xy = torch.zeros(self._num_envs, device=self._device)

        if self._left_foot_id is not None:
            left_force = forces_xyz[:, self._left_foot_id, :]
            left_xy = torch.sqrt(left_force[:, 0] ** 2 + left_force[:, 1] ** 2)
        else:
            left_xy = torch.zeros(self._num_envs, device=self._device)

        feet_xy = torch.stack([right_xy, left_xy], dim=1)
        self._contact_force_history[:-1] = self._contact_force_history[1:].clone()
        self._contact_force_history[-1] = feet_xy

        avg_force = self._contact_force_history.mean(dim=0)
        right_contact = avg_force[:, 0] > self._force_threshold
        left_contact = avg_force[:, 1] > self._force_threshold

        contact_above = self._contact_force_history > self._force_threshold
        stable_contact = contact_above.all(dim=0)
        right_stable = stable_contact[:, 0]
        left_stable = stable_contact[:, 1]

        trigger_right = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        trigger_left = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

        no_leg_lifting = ~self._lifting_state[:, 0] & ~self._lifting_state[:, 1]
        left_followup_ready = (self._first_leg == 1) & self._lifting_state[:, 0] & ~self._lifting_state[:, 1]
        right_followup_ready = (self._first_leg == 2) & self._lifting_state[:, 1] & ~self._lifting_state[:, 0]
        can_trigger_right = ~self._lifting_state[:, 0] & (no_leg_lifting | right_followup_ready)
        can_trigger_left = ~self._lifting_state[:, 1] & (no_leg_lifting | left_followup_ready)

        right_only_eligible = can_trigger_right & ~can_trigger_left
        left_only_eligible = can_trigger_left & ~can_trigger_right
        trigger_right = trigger_right | (right_only_eligible & right_contact)
        trigger_left = trigger_left | (left_only_eligible & left_contact)

        both_eligible = can_trigger_right & can_trigger_left
        only_right = right_contact & ~left_contact & both_eligible
        only_left = left_contact & ~right_contact & both_eligible
        trigger_right = trigger_right | only_right
        trigger_left = trigger_left | only_left

        both_contact = right_contact & left_contact & both_eligible
        right_stable_only = both_contact & right_stable & ~left_stable
        left_stable_only = both_contact & ~right_stable & left_stable
        trigger_right = trigger_right | right_stable_only
        trigger_left = trigger_left | left_stable_only

        both_stable = both_contact & right_stable & left_stable
        trigger_right = trigger_right | (both_stable & (avg_force[:, 0] >= avg_force[:, 1]))
        trigger_left = trigger_left | (both_stable & (avg_force[:, 0] < avg_force[:, 1]))

        self._lifting_state[:, 0] = self._lifting_state[:, 0] | trigger_right
        self._lifting_state[:, 1] = self._lifting_state[:, 1] | trigger_left

        self._first_leg = torch.where(
            trigger_right & (self._first_leg == 0), torch.ones_like(self._first_leg), self._first_leg
        )
        self._first_leg = torch.where(
            trigger_left & (self._first_leg == 0), torch.full_like(self._first_leg, 2), self._first_leg
        )

        right_done = self._time[:, 0] >= self._ff_period
        left_done = self._time[:, 1] >= self._ff_period

        self._time[:, 0] = torch.where(right_done, torch.zeros_like(self._time[:, 0]), self._time[:, 0])
        self._time[:, 1] = torch.where(left_done, torch.zeros_like(self._time[:, 1]), self._time[:, 1])
        self._lifting_state[:, 0] = torch.where(
            right_done, torch.zeros_like(self._lifting_state[:, 0]), self._lifting_state[:, 0]
        )
        self._lifting_state[:, 1] = torch.where(
            left_done, torch.zeros_like(self._lifting_state[:, 1]), self._lifting_state[:, 1]
        )

        no_active_lifts = ~self._lifting_state.any(dim=1)
        self._first_leg = torch.where(no_active_lifts, torch.zeros_like(self._first_leg), self._first_leg)
        return self._lifting_state.float()

    def _init_contact_sensor(self):
        import re

        self._contact_sensor: ContactSensor = self._env.scene.sensors[self._contact_sensor_name]
        for i, name in enumerate(self._contact_sensor.body_names):
            if re.match(self._contact_body_pattern, name):
                if "right" in name.lower():
                    self._right_foot_id = i
                elif "left" in name.lower():
                    self._left_foot_id = i
