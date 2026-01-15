from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand, MultiMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)

def _phase_to_weight_pyramid(phase, start=0.25, end=0.75):
    weight = torch.zeros_like(phase)
    mask1 = phase < start
    mask2 = phase > end
    weight[mask1] = 1/start * phase[mask1]
    weight[mask2] = 1/(1-end) * (1 - phase[mask2])
    weight[~(mask1 | mask2)] = 1.0
    return weight    

def force_based_push(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_duration: tuple[int, int],
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    command : MotionCommand = env.command_manager.get_term(command_name)
    
    # Mask update stuff
    mask_update = (command.force_push_counter >= command.force_update_frequency)
    if command.force_config_init == False:
        command.force_duration_per_env = torch.randint(*force_duration, (command.num_envs,), dtype=torch.int, device=command.device)
        command.force_config_init = True
    mask_finish_update = (command.force_push_counter == (command.force_duration_per_env + command.force_update_frequency))
    phase = ((command.force_push_counter - command.force_update_frequency) / command.force_duration_per_env).clamp(min=0.0, max=1.0)
    forces = torch.zeros_like(command.body_force_dir_buf)
    # NOTE: all body force magnitude updated at once!!
    command.body_force_magnitude_buf[mask_finish_update] = torch.rand(sum(mask_finish_update), device=command.device)
    if (mask_update == False).all():
        command.force_push_counter[:] += 1
        return
    
    forces[mask_update] = command.body_force_magnitude_buf[mask_update].view(-1,1,1) * _phase_to_weight_pyramid(phase)[mask_update].view(-1,1,1) * command.body_force_dir_buf[mask_update] * command.max_force
    forces[:,command.non_force_push_ids_rel,:] = 0.0 # zero out non-force bodies
    command.force_push_counter[mask_finish_update] = 0
    command.force_duration_per_env[mask_finish_update] = torch.randint(*force_duration, (sum(mask_finish_update),), dtype=torch.int, device=command.device)
    command.body_force_dir_buf[mask_finish_update] = torch.randn_like(command.body_force_dir_buf[mask_finish_update])
    command.body_force_dir_buf /= torch.norm(command.body_force_dir_buf, dim=-1, keepdim=True) # normalize
    
    
    asset = env.scene[asset_cfg.name]

    # sample random forces and torques
    torques = torch.zeros_like(forces[:,command.force_push_ids_rel,:])
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    positions = command.robot.data.body_pos_w[:,command.force_push_ids] + math_utils.quat_apply(command.robot.data.body_quat_w[:,command.force_push_ids], command.force_push_body_offsets)
    command.last_force_applied = forces[:, command.force_push_ids_rel, :].clone()
    command.force_push_counter[:] += 1
    asset.set_external_force_and_torque(forces[:,command.force_push_ids_rel,:], torques, positions=positions, body_ids=command.force_push_ids, is_global=True)


def fake_contact_force(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        command_name: str,
        distance_threshold: float = 0.05,
        contact_spring_k: float = 1000.0,
        min_friction_coeff: float = 0.5,
        squeeze_force_range: tuple[float, float] = (5.0, 20.0),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    wrist_pos_ref = command.vr_3point_body_pos_w[:,:2]
    root_pos_ref = command.anchor_pos_w
    root_orn_ref = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, 2, 1)
    robot_root_pos = command.robot_anchor_pos_w
    robot_root_orn = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, 2, 1)
    wrist_pos_ref_l = wrist_pos_ref - root_pos_ref[:,None,:]
    wrist_pos_ref_l = math_utils.quat_apply(math_utils.quat_inv(root_orn_ref), wrist_pos_ref_l)
    wrist_pos_ref_wr = math_utils.quat_apply(robot_root_orn, wrist_pos_ref_l) + robot_root_pos[:,None,:]
    l_dir = wrist_pos_ref_wr[:,0] - wrist_pos_ref_wr[:,1] # shape [envs, 3]
    l_dir = l_dir / torch.norm(l_dir, dim=-1, keepdim=True)
    r_dir = - l_dir
    
    robot_wrist_pos = command.robot_vr_3point_pos_w[:,:2]
    o2r = robot_wrist_pos - wrist_pos_ref_wr
    dist2o = torch.norm(o2r, dim=-1)
    contact_mask = (dist2o[:,0] < distance_threshold) * (dist2o[:,1] < distance_threshold) * command.wrist_grasp_label
    last_non_contact_mask = command.last_force_applied[:, :2].norm(dim=-1).sum(dim=-1) < 1e-3
    magnitude_update_mask = contact_mask & last_non_contact_mask
    command.body_force_magnitude_buf[magnitude_update_mask] = torch.rand(sum(magnitude_update_mask), device=command.device) * (squeeze_force_range[1] - squeeze_force_range[0]) + squeeze_force_range[0]
    
    l_sp_dist = (o2r[:,0]*(-l_dir)).sum(dim=-1).clamp(min=0.0)
    r_sp_dist = (o2r[:,1]*(-r_dir)).sum(dim=-1).clamp(min=0.0)
    l_normal_force = l_dir * (command.body_force_magnitude_buf + l_sp_dist * contact_spring_k)[:, None]
    r_normal_force = r_dir * (command.body_force_magnitude_buf + r_sp_dist * contact_spring_k)[:, None]
    total_force = torch.stack([l_normal_force, r_normal_force], dim=1) # shape [envs, 2, 3]
    g_force = command.down_dir * min_friction_coeff * command.body_force_magnitude_buf[:, None, None]
    total_force += g_force
    total_force[~contact_mask] = 0.0 # zero out non-contact envs
    command.last_force_applied[:,:2] = total_force.clone()
    asset = env.scene[asset_cfg.name]

    asset.set_external_force_and_torque(total_force, torch.zeros_like(total_force), positions=robot_wrist_pos, body_ids=command.force_push_ids[:2], is_global=True)

