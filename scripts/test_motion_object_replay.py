#!/usr/bin/env python3
"""Test script: replay reference motion and object trajectories from motion_dir and object_folder.

Loads robot motion (root + joints) and resets the robot to the reference state every time step
(kinematic replay). Uses the same update order as standalone_kinematic_object_robot:
  1. write_root_link_pose_to_sim (object)
  2. write_root_com_velocity_to_sim (object)
  3. robot.write_root_state_to_sim + robot.write_joint_state_to_sim (reset robot to motion state)
  4. scene.write_data_to_sim
  5. sim.step
  6. scene.update

Supports motion formats:
  - qpos (MuJoCo: root_pos[3], root_quat[4], joint_pos[29])
  - joint_pos + root_pos + root_quat (explicit)
  - joint_pos + body_pos_w + body_quat_w (root from body 0)
  - joint_pos only (root at env origin, identity quat)

Run from bm_generalist root (with IsaacLab and whole_body_tracking in PYTHONPATH):

    python scripts/test_motion_object_replay.py --motion_dir omomo_box --object_folder omomo_object/omomo_box
    python scripts/test_motion_object_replay.py --motion_dir lafan --object_folder omomo_object/omomo_box --num_envs 4
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay motion + object from motion_dir and object_folder.")
parser.add_argument("--motion_dir", type=str, required=True, help="Path to motion .npz files.")
parser.add_argument("--object_folder", type=str, required=True, help="Path to object folder (URDFs + *_object_fitted.npz).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of envs (default: number of motions).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_binded_multi_object import (
    _build_proxy_urdf_paths,
    _load_object_trajectories,
    _resolve_path,
)

# MuJoCo to IsaacLab joint order mapping for G1 robot (29 DOF)
MUJOCO_TO_ISACLAB_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
    4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]


def _resolve_paths(motion_dir: str, object_folder: str) -> tuple[Path, Path]:
    """Resolve motion_dir and object_folder to absolute paths."""
    fallback_roots = [Path.cwd(), Path(__file__).resolve().parent.parent]
    motion_path = _resolve_path(Path(motion_dir), fallback_roots)
    object_path = _resolve_path(Path(object_folder), fallback_roots)
    return motion_path, object_path


def _load_motion_single(d: dict) -> dict:
    """Load a single motion from npz dict. Supports qpos, joint_pos+root_*, body_pos_w formats."""
    fps = float(d["fps"]) if "fps" in d else 50.0

    if "qpos" in d:
        qpos = np.asarray(d["qpos"], dtype=np.float32)
        root_pos = qpos[:, 0:3]
        root_quat = qpos[:, 3:7]
        mujoco_joint = qpos[:, 7:]
        if mujoco_joint.shape[1] == 29:
            joint_pos = mujoco_joint[:, MUJOCO_TO_ISACLAB_DOF]
        else:
            joint_pos = mujoco_joint
    else:
        joint_pos = np.asarray(d["joint_pos"], dtype=np.float32)
        if "root_pos" in d and "root_quat" in d:
            root_pos = np.asarray(d["root_pos"], dtype=np.float32)
            root_quat = np.asarray(d["root_quat"], dtype=np.float32)
        elif "body_pos_w" in d and "body_quat_w" in d:
            body_pos = np.asarray(d["body_pos_w"], dtype=np.float32)
            body_quat = np.asarray(d["body_quat_w"], dtype=np.float32)
            # body_pos_w: (T, num_bodies, 3), root = body 0
            root_pos = body_pos[:, 0, :] if body_pos.ndim == 3 else body_pos[:, :3]
            root_quat = body_quat[:, 0, :] if body_quat.ndim == 3 else body_quat[:, :4]
        else:
            n = joint_pos.shape[0]
            root_pos = np.zeros((n, 3), dtype=np.float32)
            root_quat = np.zeros((n, 4), dtype=np.float32)
            root_quat[:, 0] = 1.0

    n_frames = joint_pos.shape[0]
    if root_pos.shape[0] != n_frames:
        root_pos = np.broadcast_to(root_pos[:1], (n_frames, 3)).copy()
    if root_quat.shape[0] != n_frames:
        root_quat = np.broadcast_to(root_quat[:1], (n_frames, 4)).copy()

    if "joint_vel" in d:
        joint_vel = np.asarray(d["joint_vel"], dtype=np.float32)
        if joint_vel.shape != joint_pos.shape:
            joint_vel = np.zeros_like(joint_pos)
    else:
        joint_vel = np.zeros_like(joint_pos)

    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "root_pos": root_pos,
        "root_quat": root_quat,
        "fps": fps,
    }


def _load_motions_and_objects(motion_path: Path, object_path: Path) -> tuple[list[str], dict, dict]:
    """Load motion files and object trajectories. Returns (motion_files, object_trajectories, motion_data)."""
    motion_files = sorted(str(f.resolve()) for f in motion_path.glob("*.npz"))
    if not motion_files:
        raise FileNotFoundError(f"No .npz motion files in {motion_path}")

    object_trajectories = _load_object_trajectories(str(object_path), motion_files)
    motion_bases = [Path(mf).stem for mf in motion_files]

    motion_data = {}
    for mf in motion_files:
        base = Path(mf).stem
        d = dict(np.load(mf, allow_pickle=True))
        motion_data[base] = _load_motion_single(d)

    return motion_files, object_trajectories, motion_data


def _make_scene_cfg(proxy_urdf: str, num_envs: int) -> InteractiveSceneCfg:
    """Build scene config with ground, G1 robot, kinematic proxy."""
    from whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_binded_multi_object import (
        MultiObjectSceneCfgWithProxy,
    )

    cfg = MultiObjectSceneCfgWithProxy(num_envs=num_envs, env_spacing=5.0)
    cfg.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.proxy_object.spawn.asset_path = proxy_urdf
    cfg.replicate_physics = True
    cfg.lazy_sensor_update = False
    return cfg


def main():
    motion_path, object_path = _resolve_paths(args_cli.motion_dir, args_cli.object_folder)
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion dir not found: {motion_path}")
    if not object_path.exists():
        raise FileNotFoundError(f"Object folder not found: {object_path}")

    motion_files, object_trajectories, motion_data = _load_motions_and_objects(motion_path, object_path)
    urdf_paths = _build_proxy_urdf_paths(object_path, motion_files)
    if not urdf_paths:
        raise FileNotFoundError(f"No URDF paths for motions in {object_path}")
    proxy_urdf = urdf_paths[0]
    if not Path(proxy_urdf).exists():
        raise FileNotFoundError(f"Proxy URDF not found: {proxy_urdf}")

    num_motions = len(motion_files)
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else num_motions
    motion_bases = [Path(mf).stem for mf in motion_files]

    # Build object trajectory tensors: (num_pairs, max_frames, 3/4)
    max_obj_frames = max(
        (t["trans"].shape[0] for t in object_trajectories.values()),
        default=0,
    )
    if max_obj_frames == 0:
        print("[WARN] No object trajectories found. Objects will stay at init pose.")

    all_trans = np.zeros((num_motions, max(max_obj_frames, 1), 3), dtype=np.float32)
    all_quat = np.zeros((num_motions, max(max_obj_frames, 1), 4), dtype=np.float32)
    all_quat[:, :, 0] = 1.0
    obj_frame_counts = np.zeros(num_motions, dtype=np.int64)

    for i, base in enumerate(motion_bases):
        if base in object_trajectories:
            t = object_trajectories[base]
            n = t["trans"].shape[0]
            all_trans[i, :n] = t["trans"]
            all_quat[i, :n] = t["quat_wxyz"]
            obj_frame_counts[i] = n
        else:
            obj_frame_counts[i] = 1

    scene_cfg = _make_scene_cfg(proxy_urdf, num_envs)

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view(eye=[5, 5, 3], target=[0.0, 0.0, 0.5])

    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    proxy = scene["proxy_object"]
    device = scene.device
    env_origins = scene.env_origins
    sim_dt = sim.get_physics_dt()
    decimation = 4

    # Move tensors to device
    trans_t = torch.tensor(all_trans, dtype=torch.float32, device=device)
    quat_t = torch.tensor(all_quat, dtype=torch.float32, device=device)
    frame_counts_t = torch.tensor(obj_frame_counts, dtype=torch.long, device=device)

    # Pre-load robot motion (root + joints) per motion
    joint_pos_per_motion = []
    joint_vel_per_motion = []
    root_pos_per_motion = []
    root_quat_per_motion = []
    motion_lengths = []
    for base in motion_bases:
        md = motion_data[base]
        joint_pos_per_motion.append(torch.tensor(md["joint_pos"], dtype=torch.float32, device=device))
        joint_vel_per_motion.append(torch.tensor(md["joint_vel"], dtype=torch.float32, device=device))
        root_pos_per_motion.append(torch.tensor(md["root_pos"], dtype=torch.float32, device=device))
        root_quat_per_motion.append(torch.tensor(md["root_quat"], dtype=torch.float32, device=device))
        motion_lengths.append(md["joint_pos"].shape[0])

    # Expand to (num_envs,) - each env uses motion env_id % num_motions
    pair_ids = torch.arange(num_envs, device=device) % num_motions
    motion_len_t = torch.tensor(motion_lengths, dtype=torch.long, device=device)[pair_ids]

    zero_vel = torch.zeros(num_envs, 6, device=device)
    t = 0.0
    fps = motion_data[motion_bases[0]]["fps"]
    env_dt = sim_dt * decimation  # 0.02 at 50fps = 1 frame per env step

    print(f"[INFO] Loaded {num_motions} motions, {len(object_trajectories)} object trajectories")
    print(f"[INFO] Proxy URDF: {proxy_urdf}")
    print(f"[INFO] num_envs={num_envs}, fps={fps}, env_dt={env_dt:.4f}s")
    print("[INFO] Update order: object pose -> robot reset (root+joints) -> write_data_to_sim -> sim.step -> scene.update")

    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step()
            continue

        for _ in range(decimation):
            # 1. Object poses from trajectory (same logic as update_kinematic_object_poses)
            # Map motion progress (0..1) to object frame index per env
            global_frame = int(t * fps) % 1_000_000
            local_steps_list = [
                global_frame % max(motion_lengths[pair_ids[i].item()], 1)
                for i in range(num_envs)
            ]
            local_steps_t = torch.tensor(local_steps_list, device=device, dtype=torch.long)
            motion_len_per_env = motion_len_t.float().clamp(min=1.0)
            ratio = local_steps_t.float() / motion_len_per_env
            obj_frames = (ratio * (frame_counts_t[pair_ids].float() - 1.0)).clamp(min=0).long()
            obj_frames = torch.minimum(obj_frames, frame_counts_t[pair_ids] - 1)

            obj_trans = trans_t[pair_ids, obj_frames]
            obj_quat = quat_t[pair_ids, obj_frames]
            obj_pose = torch.zeros(num_envs, 7, device=device)
            obj_pose[:, :3] = obj_trans + env_origins
            obj_pose[:, 3:7] = obj_quat

            proxy.write_root_link_pose_to_sim(obj_pose)
            proxy.write_root_com_velocity_to_sim(zero_vel)

            # 2. Reset robot to motion state (root + joints) every step
            root_states = robot.data.default_root_state.clone()
            joint_pos_batch = torch.zeros(num_envs, robot.num_joints, device=device)
            joint_vel_batch = torch.zeros(num_envs, robot.num_joints, device=device)
            for env_id in range(num_envs):
                motion_idx = pair_ids[env_id].item()
                motion_len = motion_lengths[motion_idx]
                local_frame = int(t * fps) % max(motion_len, 1)
                local_frame = min(local_frame, motion_len - 1) if motion_len > 0 else 0
                root_states[env_id, :3] = root_pos_per_motion[motion_idx][local_frame] + env_origins[env_id]
                root_states[env_id, 3:7] = root_quat_per_motion[motion_idx][local_frame]
                root_states[env_id, 7:10] = 0.0
                root_states[env_id, 10:13] = 0.0
                joint_pos_batch[env_id] = joint_pos_per_motion[motion_idx][local_frame]
                joint_vel_batch[env_id] = joint_vel_per_motion[motion_idx][local_frame]
            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(joint_pos_batch, joint_vel_batch)

            # 3-6. Same as standalone
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

        t += env_dt


if __name__ == "__main__":
    main()
    simulation_app.close()
