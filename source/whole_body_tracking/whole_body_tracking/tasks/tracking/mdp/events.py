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