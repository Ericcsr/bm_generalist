from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from whole_body_tracking.tasks.tracking.mdp.commands import (
    BindedMultiMotionCommand,
    MotionCommand,
    MultiMotionCommand,
)
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes

# Consecutive env-step count of wrist contact vs reference mismatch (per scene)
_wrist_contact_mismatch_steps: dict[int, torch.Tensor] = {}


def bad_wrist_contact_mismatch_duration(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    contact_indices: list[int] | None = None,
    max_mismatch_steps: int = 50,
    only_object_motions: bool = False,
) -> torch.Tensor:
    """Terminate after ``max_mismatch_steps`` consecutive **control** steps of mismatch.

    Uses the same desired/actual convention as ``contact_label_contact_reward``: for each
    wrist column in ``contact_indices`` (default hands ``[2, 3]``), reference comes from
    ``get_contact_labels_masked()`` and actual contact from ``contact_forces`` on the
    corresponding ``sensor_cfg`` bodies. A step **matches** only when **both** wrists match
    (desired == actual for left and right). On any match, the running mismatch counter
    resets to 0 for that env.

    Returns False for all envs when contact labels are not loaded.

    If ``only_object_motions=True``, mismatch is checked only for envs whose bound motion
    category is object (category id 1 in ``env.cfg.motion_category_per_motion``). Terrain
    and free-space motions are bypassed.
    """
    scene_id = id(env.scene)
    num_envs = env.scene.num_envs
    device = env.device

    if scene_id not in _wrist_contact_mismatch_steps:
        _wrist_contact_mismatch_steps[scene_id] = torch.zeros(num_envs, dtype=torch.long, device=device)
    accum = _wrist_contact_mismatch_steps[scene_id]
    if accum.shape[0] != num_envs:
        accum = torch.zeros(num_envs, dtype=torch.long, device=device)
        _wrist_contact_mismatch_steps[scene_id] = accum

    command = env.command_manager.get_term(command_name)
    if not isinstance(command, (MultiMotionCommand, BindedMultiMotionCommand)):
        accum.zero_()
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    cl = command.get_contact_labels_masked()
    if cl is None:
        accum.zero_()
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    indices = contact_indices if contact_indices is not None else [2, 3]
    desired_contact = cl[:, indices].float()

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    actual_contact = (torch.max(torch.norm(net_forces, dim=-1), dim=1)[0] > threshold).float()

    active_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
    if only_object_motions:
        active_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        motion_ids = getattr(command, "motion_ids", None)
        motion_cats = getattr(env.cfg, "motion_category_per_motion", None) or []
        if motion_ids is not None and len(motion_cats) == int(getattr(command.motion, "num_motions", len(motion_cats))):
            cats_t = torch.as_tensor(motion_cats, device=device, dtype=torch.long)
            active_mask = cats_t[motion_ids] == 1
        # Always clear counters for bypassed envs.
        accum[~active_mask] = 0

    full_match = (desired_contact == actual_contact).all(dim=-1)
    mismatch = (~full_match) & active_mask

    first_step = env.episode_length_buf == 1
    accum[first_step] = 0

    accum = torch.where(mismatch, accum + 1, torch.zeros_like(accum))
    _wrist_contact_mismatch_steps[scene_id] = accum

    return (accum >= max_mismatch_steps) & active_mask


def motion_ended(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate when the current motion has ended (reached the last frame).

    Works with MotionCommand (single motion) and MultiMotionCommand/BindedMultiMotionCommand
    (multi-motion with per-env motion_ids). Uses time_steps and motion bounds from the command.

    Note: Terminations are checked before command_manager.compute(), so we check for
    time_steps >= motion_end - 1 (on last frame) since the increment happens in compute().
    """
    command = env.command_manager.get_term(command_name)
    time_steps = command.time_steps

    if hasattr(command, "motion_ids") and hasattr(command.motion, "motion_end_idx"):
        # MultiMotionCommand or BindedMultiMotionCommand: per-env motion end
        motion_end = command.motion.motion_end_idx[command.motion_ids]
        return time_steps >= motion_end - 1
    else:
        # MotionCommand: single motion
        return time_steps >= command.motion.time_step_total - 1


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


def object_dropped_to_floor(
    env: ManagerBasedRLEnv,
    floor_height_threshold: float = 0.18,
    command_name: str = "motion",
    require_wrist_contact_label: bool = True,
) -> torch.Tensor:
    """Terminate when object drops to floor during physics_phase.

    Only active when physics_phase_flag is True. When physics phase is activated
    (both wrists contacted object, magic force disabled), if the object's
    center drops below env_origin_z + floor_height_threshold, the episode terminates.

    When require_wrist_contact_label=True (default), termination only triggers if
    the reference motion's contact labels say the wrist should be in contact (i.e.
    the robot was supposed to be holding the object). If contact_labels are not
    loaded, falls back to original behavior (terminate on drop).
    """
    if not getattr(env.cfg, "physics_phase_flag", False):
        return torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)

    try:
        proxy_obj = env.scene["proxy_object"]
    except KeyError:
        return torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)

    from whole_body_tracking.tasks.tracking.mdp.events import get_physics_phase_activated

    physics_phase = get_physics_phase_activated(env)
    if physics_phase is None:
        return torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)
    physics_phase = physics_phase.to(env.device)
    obj_pos_w = proxy_obj.data.root_link_pose_w[:, :3]
    obj_z = obj_pos_w[:, 2]
    floor_z = env.scene.env_origins[:, 2] + floor_height_threshold

    dropped = physics_phase & (obj_z < floor_z)

    if require_wrist_contact_label:
        command = env.command_manager.get_term(command_name)
        if isinstance(command, (MultiMotionCommand, BindedMultiMotionCommand)):
            cl = command.get_contact_labels_masked()  # (num_envs, 4) [left_foot, right_foot, left_wrist, right_wrist]
            if cl is not None:
                wrist_contact_active = cl[:, 2] | cl[:, 3]  # either wrist should be in contact
                dropped = dropped & wrist_contact_active  # only terminate when reference expects wrist contact

    return dropped


def physics_blowup(
    env: ManagerBasedRLEnv, max_joint_vel: float = 100.0, max_root_lin_vel: float = 20.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """NaN guard (claude 2026-10-23): end episodes whose physics state exploded -- any non-finite robot state, a joint
    faster than ``max_joint_vel`` (rad/s; actuator limits are <= 37) or a root faster than ``max_root_lin_vel`` (m/s).
    Rare isolated blow-ups (mostly OMOMO object envs) produced value losses up to 1e18 and finally a NaN policy std."""
    robot: Articulation = env.scene[asset_cfg.name]
    d = robot.data
    state = torch.cat([d.joint_pos, d.joint_vel, d.root_pos_w, d.root_quat_w, d.root_lin_vel_w, d.root_ang_vel_w], dim=-1)
    non_finite = ~torch.isfinite(state).all(dim=-1)
    fast_joint = torch.nan_to_num(d.joint_vel.abs(), nan=0.0).amax(dim=-1) > max_joint_vel
    fast_root = torch.nan_to_num(d.root_lin_vel_w.norm(dim=-1), nan=0.0) > max_root_lin_vel
    return non_finite | fast_joint | fast_root
