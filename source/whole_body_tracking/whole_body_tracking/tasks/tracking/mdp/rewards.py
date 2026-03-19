from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, quat_inv, quat_apply, quat_mul

from whole_body_tracking.tasks.tracking.mdp.commands import BindedMultiMotionCommand, MotionCommand, MultiMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def vr_position_relative_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_anchor_pos = command.robot_anchor_pos_w
    robot_anchor_quat_inv = quat_inv(command.robot_anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, 3, 1)
    vr_pos_rel = command.robot_vr_3point_pos_w - robot_anchor_pos[:, None, :]
    vr_pos_rel = quat_apply(robot_anchor_quat_inv, vr_pos_rel)
    ref_anchor_pos = command.anchor_pos_w
    ref_anchor_quat_inv = quat_inv(command.anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, 3, 1)
    vr_pos_rel_ref = command.vr_3point_body_pos_w - ref_anchor_pos[:, None, :]
    vr_pos_rel_ref = quat_apply(ref_anchor_quat_inv, vr_pos_rel_ref)

    error = torch.sum(
        torch.square(vr_pos_rel_ref - vr_pos_rel), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def vr_orientation_relative_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_anchor_quat_inv = quat_inv(command.robot_anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, 3, 1)
    vr_quat_rel = quat_mul(robot_anchor_quat_inv, command.vr_3point_body_quat_w)
    ref_anchor_quat_inv = quat_inv(command.anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, 3, 1)
    vr_quat_rel_ref = quat_mul(ref_anchor_quat_inv, command.vr_3point_body_quat_w)

    error = (
        quat_error_magnitude(vr_quat_rel_ref, vr_quat_rel)
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)

def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def grasp_contact_reward(env: ManagerBasedRLEnv, command_name: str, distance_threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    env_should_grasp = command.wrist_grasp_label
    
    # virtual desired contact locations
    wrist_pos_ref = command.vr_3point_body_pos_w[:,:2]
    root_pos_ref = command.anchor_pos_w
    root_orn_ref = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, 2, 1)
    robot_root_pos = command.robot_anchor_pos_w
    robot_root_orn = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, 2, 1)
    wrist_pos_ref_l = wrist_pos_ref - root_pos_ref[:,None,:]
    wrist_pos_ref_l = quat_apply(quat_inv(root_orn_ref), wrist_pos_ref_l)
    wrist_pos_ref_wr = quat_apply(robot_root_orn, wrist_pos_ref_l) + robot_root_pos[:,None,:]

    robot_wrist_pos = command.robot_vr_3point_pos_w[:,:2]
    o2r = robot_wrist_pos - wrist_pos_ref_wr
    dist2o = torch.norm(o2r, dim=-1)
    left_correct_contact_mask = (dist2o[:,0] < distance_threshold) * env_should_grasp
    right_correct_contact_mask = (dist2o[:,1] < distance_threshold) * env_should_grasp
    reward = left_correct_contact_mask.float() + right_correct_contact_mask.float()
    return reward


def contact_label_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    contact_indices: list[int] | None = None,
) -> torch.Tensor:
    """Reward for matching contact labels: encourage correct links to make contact.

    When contact_labels say a link should be in contact, reward if it is.
    When contact_labels say a link should not be in contact, reward if it is not.
    Returns 0 when contact_labels are not loaded (e.g. base config without contact collision).

    contact_labels columns: 0=left_foot, 1=right_foot, 2=left_wrist, 3=right_wrist.
    contact_indices selects which columns to use (default [0,1,2,3] for all).
    sensor_cfg.body_names must match contact_indices order (e.g. hands_only: [2,3] with
    body_names [left_wrist_yaw_link, right_wrist_yaw_link]).
    """
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, (MultiMotionCommand, BindedMultiMotionCommand)):
        return torch.zeros(env.num_envs, device=env.device)
    cl = command.get_contact_labels_masked()
    if cl is None:
        return torch.zeros(env.num_envs, device=env.device)

    indices = contact_indices if contact_indices is not None else [0, 1, 2, 3]
    desired_contact = cl[:, indices].float()  # (num_envs, len(indices))

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    actual_contact = (torch.max(torch.norm(net_forces, dim=-1), dim=1)[0] > threshold).float()  # (num_envs, len(body_ids))

    # Reward: fraction of links where desired == actual
    match = (desired_contact == actual_contact).float()
    return match.mean(dim=-1)


def contact_label_contact_duration_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    contact_indices: list[int] | None = None,
    max_duration_s: float = 0.5,
) -> torch.Tensor:
    """Reward for sustained contact when reference expects contact. Encourages persistent contact.

    When desired_contact and actual_contact are both True, adds bonus proportional to how long
    contact has been held (current_contact_time). Full bonus when duration >= max_duration_s.
    Returns 0 when contact_labels not loaded or current_contact_time unavailable (track_air_time=False).
    """
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, (MultiMotionCommand, BindedMultiMotionCommand)):
        return torch.zeros(env.num_envs, device=env.device)
    cl = command.get_contact_labels_masked()
    if cl is None:
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_contact_time = getattr(contact_sensor.data, "current_contact_time", None)
    if current_contact_time is None:
        return torch.zeros(env.num_envs, device=env.device)

    indices = contact_indices if contact_indices is not None else [0, 1, 2, 3]
    desired_contact = cl[:, indices].float()  # (num_envs, len(indices))

    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    actual_contact = (torch.max(torch.norm(net_forces, dim=-1), dim=1)[0] > threshold).float()

    # Duration bonus: only when both desired and actual contact
    duration_raw = current_contact_time[:, sensor_cfg.body_ids]  # (num_envs, len(body_ids))
    duration_bonus = torch.clamp(duration_raw / max_duration_s, 0.0, 1.0)
    # Apply only when in contact phase and matching
    in_contact_phase = desired_contact * actual_contact
    reward = (in_contact_phase * duration_bonus).sum(dim=-1) / (in_contact_phase.sum(dim=-1).clamp(min=1e-6))
    return torch.where(in_contact_phase.sum(dim=-1) > 0, reward, torch.zeros_like(reward))