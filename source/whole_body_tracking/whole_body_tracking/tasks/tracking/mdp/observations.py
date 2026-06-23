from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_apply, quat_inv, quat_mul

from whole_body_tracking.tasks.tracking.mdp.commands import (
    BindedMultiMotionCommand,
    MotionCommand,
    MultiMotionCommand,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def command_lookahead(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command = MultiMotionCommand = env.command_manager.get_term(command_name)
    return command.command_lookahead

def command_lower_body(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.command_lower_body

def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)

def projected_gravity(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    grav_dir = quat_apply(quat_inv(command.robot_anchor_quat_w), command.down_dir)
    return grav_dir.view(env.num_envs, -1)

def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

def vr_3point_local_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, len(command.cfg.vr_3point_body), 1)
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_3point_root = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    return ref_3point_root.view(env.num_envs, -1)

def vr_3point_local_compliant_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ext_force_disp_w = command.last_force_applied * command.eef_stiffness_buf[:,:,None]
    root_quat = command.robot_anchor_quat_w[:,None,:].repeat(1, len(command.cfg.vr_3point_body),1)
    ext_force_disp_l = quat_apply(quat_inv(root_quat), ext_force_disp_w)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, len(command.cfg.vr_3point_body), 1)
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:,None,:]
    ref_3point_root = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    ref_3point_root -= ext_force_disp_l
    return ref_3point_root.view(env.num_envs, -1)

def vr_3point_local_orn_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, len(command.cfg.vr_3point_body), 1)
    ref_3point_quat = command.vr_3point_body_quat_w
    ref_3point_root = quat_mul(quat_inv(ref_root_quat),ref_3point_quat)
    return ref_3point_root.view(env.num_envs, -1)

def wrist_grasp_label(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.wrist_grasp_label.view(env.num_envs, -1)


def motion_bucket_onehot_for_critic(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Privileged critic aux: 2-D one-hot [terrain_or_free, object] per environment.

    Uses :attr:`env.cfg.motion_category_per_motion` (0=terrain, 1=object, 2=free) indexed by
    binded per-env ``motion_ids``. Object column is 1 only when category is 1; terrain and free
    share the first column. Returns zeros when categories are missing or length-mismatched.
    """
    command = env.command_manager.get_term(command_name)
    n = env.num_envs
    device = env.device
    out = torch.zeros(n, 2, dtype=torch.float32, device=device)
    if not isinstance(command, BindedMultiMotionCommand):
        return out
    cats = getattr(env.cfg, "motion_category_per_motion", None) or []
    if len(cats) == 0:
        return out
    if len(cats) != int(command.motion.num_motions):
        return out
    motion_ids = command.motion_ids
    cats_t = torch.as_tensor(cats, device=device, dtype=torch.long)
    motion_cats = cats_t[motion_ids]
    is_obj = (motion_cats == 1).float()
    is_tf = 1.0 - is_obj
    out[:, 0] = is_tf
    out[:, 1] = is_obj
    return out


def _contact_label_feature_dim(command) -> int:
    return int(getattr(command.cfg, "contact_label_num_dims", 4))


def contact_label(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Contact labels for current frame (C booleans: feet, wrists, optional seat).

    Uses get_contact_labels_masked() so dropout is respected. Width C matches
    ``contact_label_num_dims`` on the motion command (4 or 5).
    """
    command = env.command_manager.get_term(command_name)
    nd = _contact_label_feature_dim(command)
    if not hasattr(command, "get_contact_labels_masked"):
        return torch.zeros(env.num_envs, nd, dtype=torch.float32, device=env.device)
    cl = command.get_contact_labels_masked()
    if cl is None:
        return torch.zeros(env.num_envs, nd, dtype=torch.float32, device=env.device)
    return cl.float().view(env.num_envs, -1)


def contact_label_feet_zeroed(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Same as :func:`contact_label` but left/right foot columns (0, 1) are always zero.

    Wrist columns (2, 3) and optional seat (4) are unchanged. Uses
    ``get_contact_labels_masked()`` so dropout matches the full contact label path.
    """
    command = env.command_manager.get_term(command_name)
    nd = _contact_label_feature_dim(command)
    if not hasattr(command, "get_contact_labels_masked"):
        return torch.zeros(env.num_envs, nd, dtype=torch.float32, device=env.device)
    cl = command.get_contact_labels_masked()
    if cl is None:
        return torch.zeros(env.num_envs, nd, dtype=torch.float32, device=env.device)
    out = cl.float().view(env.num_envs, -1).clone()
    if out.shape[1] >= 2:
        out[:, 0] = 0.0
        out[:, 1] = 0.0
    return out


def contact_label_env_object_2bit(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """2-bit contact label per link: [env, object] for feet and wrists; optional seat row.

    Output layout:
    - **8 dims** when reference has 4 columns (feet + hands only):
      [L_foot_env, L_foot_obj, R_foot_env, R_foot_obj, L_wrist_env, L_wrist_obj, R_wrist_env, R_wrist_obj].
    - **10 dims** when reference has a 5th column (seat / pelvis-terrain): append
      ``[seat_env, seat_obj]`` (indices 8–9). Seat is encoded as environment contact;
      ``seat_obj`` is always 0 (proxy object does not replace terrain seating).

    Source rule:
    - Terrain/free motions: all active contacts are environment contacts.
    - Object motions: foot contacts are environment contacts; wrist contacts are object contacts.
    """
    command = env.command_manager.get_term(command_name)
    nd = _contact_label_feature_dim(command)
    if not hasattr(command, "get_contact_labels_masked"):
        return torch.zeros(env.num_envs, 2 * nd, dtype=torch.float32, device=env.device)
    cl = command.get_contact_labels_masked()
    if cl is None:
        return torch.zeros(env.num_envs, 2 * nd, dtype=torch.float32, device=env.device)

    nc = int(cl.shape[1])
    n_out = 10 if max(nc, nd) >= 5 else 8

    # Base mapping from first four columns: all contacts start as environment contacts.
    out = torch.zeros(env.num_envs, n_out, dtype=torch.float32, device=env.device)
    cl4 = cl[:, :4].reshape(env.num_envs, 4).float()
    out[:, 0] = cl4[:, 0]  # L_foot env
    out[:, 2] = cl4[:, 1]  # R_foot env
    out[:, 4] = cl4[:, 2]  # L_wrist env
    out[:, 6] = cl4[:, 3]  # R_wrist env

    # For object motions (category==1), remap wrists to object-contact bit.
    motion_ids = getattr(command, "motion_ids", None)
    motion_cats = getattr(env.cfg, "motion_category_per_motion", None) or []
    if motion_ids is not None and len(motion_cats) > 0:
        if len(motion_cats) == int(getattr(command.motion, "num_motions", len(motion_cats))):
            cats_t = torch.as_tensor(motion_cats, device=env.device, dtype=torch.long)
            is_object_motion = cats_t[motion_ids] == 1
            out[is_object_motion, 5] = out[is_object_motion, 4]  # L_wrist obj
            out[is_object_motion, 7] = out[is_object_motion, 6]  # R_wrist obj
            out[is_object_motion, 4] = 0.0
            out[is_object_motion, 6] = 0.0

    if nc >= 5:
        out[:, 8] = cl[:, 4].float()
        out[:, 9] = 0.0

    return out


def contact_label_env_object_2bit_zeroed(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Zeroed contact observation with the same width as ``contact_label_env_object_2bit``."""
    command = env.command_manager.get_term(command_name)
    nd = _contact_label_feature_dim(command)
    return torch.zeros(env.num_envs, 2 * nd, dtype=torch.float32, device=env.device)


def contact_label_env_object_2bit_feet_zeroed(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """2-bit contact observation with left/right foot bits always zero."""
    out = contact_label_env_object_2bit(env, command_name)
    if out.shape[1] >= 4:
        out[:, 0:4] = 0.0
    return out


def contact_label_wrists_only(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Same layout as contact_label [L_foot, R_foot, L_wrist, R_wrist] but foot entries are always zero.

    Uses get_contact_labels_masked() so dropout matches contact_label for wrist dims.
    """
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, "get_contact_labels_masked"):
        return torch.zeros(env.num_envs, 4, dtype=torch.float32, device=env.device)
    cl = command.get_contact_labels_masked()
    if cl is None:
        return torch.zeros(env.num_envs, 4, dtype=torch.float32, device=env.device)
    cl = cl.float().view(env.num_envs, 4)
    out = torch.zeros(env.num_envs, 4, dtype=cl.dtype, device=cl.device)
    out[:, 2:4] = cl[:, 2:4]
    return out


def compliance(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.eef_stiffness_buf * 10.0


def contact_token_future(
    env: ManagerBasedEnv,
    command_name: str,
    num_future_steps: int = 5,
) -> torch.Tensor:
    """Contact labels for the next ``num_future_steps`` (C dims per step).

    Returns (num_envs, num_future_steps * C). C is ``contact_label_num_dims`` (4 or 5).
    """
    command = env.command_manager.get_term(command_name)
    contact_labels = getattr(command.motion, "contact_labels", None)
    ncols = int(contact_labels.shape[1]) if contact_labels is not None else _contact_label_feature_dim(command)
    if contact_labels is None:
        return torch.zeros(
            env.num_envs, num_future_steps * ncols, dtype=torch.float32, device=env.device
        )

    time_steps = command.time_steps
    max_idx = contact_labels.shape[0] - 1

    tokens = []
    for k in range(1, num_future_steps + 1):
        future_ts = torch.clamp(time_steps + k, 0, max_idx)
        cl = contact_labels[future_ts]  # (num_envs, C)
        tokens.append(cl.float())
    out = torch.stack(tokens, dim=1)  # (num_envs, num_future_steps, C)
    return out.reshape(num_envs, -1)


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    Local override of :func:`isaaclab.envs.mdp.height_scan` that additionally replaces missed rays
    (``inf``) and invalid values (``nan``) with ``0.0`` to avoid policy std collapse. Keeping this
    in the extension lets the Isaac Lab installation stay unmodified.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    # Replace inf/nan (missed rays) with 0.0 to avoid policy std collapse
    return torch.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0)