"""This script loads motion trajectory data from lafan_terrain and visualizes it with IsaacLab.

The motion data has schema ['qpos', 'fps'] where qpos is in MuJoCo order:
- qpos[0:3]: root position
- qpos[3:7]: root quaternion (w, x, y, z)
- qpos[7:]: joint positions (in MuJoCo joint order)

This script converts MuJoCo joint order to IsaacLab joint order and replays the motion.

.. code-block:: bash

    # Usage
    python replay_lafan_terrain.py --motion_file lafan_terrain/obstacles1_subject1/motion_omniretarget.npz --terrain_urdf lafan_terrain/obstacles1_subject1/terrain_mesh_scaled_0.75_0.75_0.75.urdf
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay lafan terrain motions.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to motion .npz/.npy file with ['qpos', 'fps'] schema")
parser.add_argument("--terrain_urdf", type=str, required=True, help="Path to terrain URDF file")
parser.add_argument("--terrain_prim_path", type=str, default="/World/Terrain", help="Prim path for terrain in IsaacLab")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create in a grid layout")
parser.add_argument("--env_spacing", type=float, default=5.0, help="Spacing between environments in the grid")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

# MuJoCo to IsaacLab joint order mapping for G1 robot (29 DOF)
# This mapping converts from MuJoCo order to IsaacLab order
# Based on the mapping found in MultiMotionCommand class
MUJOCO_TO_ISACLAB_DOF = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]


def create_scene_cfg(terrain_urdf_path: str):
    """Create scene configuration with terrain."""
    
    @configclass
    class ReplayLafanTerrainSceneCfg(InteractiveSceneCfg):
        """Configuration for a replay lafan terrain scene."""

        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )

        # articulation
        robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain as static URDF
        terrain: AssetBaseCfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Terrain",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=terrain_urdf_path,
                fix_base=True,  # Static terrain
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    retain_accelerations=False,
                ),
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
                ),
            ),
        )
    
    return ReplayLafanTerrainSceneCfg


def load_mujoco_motion(motion_file: str, device: str = "cpu"):
    """Load motion data from npz/npy file with MuJoCo format.
    
    Args:
        motion_file: Path to .npz/.npy file with ['qpos', 'fps'] schema (npz) or qpos array (npy)
        device: Device to load data on
        
    Returns:
        Dictionary with:
        - fps: frames per second
        - qpos: (T, N) array where N = 3 (pos) + 4 (quat) + num_joints
        - joint_pos: (T, num_joints) array in IsaacLab joint order
        - root_pos: (T, 3) array
        - root_quat: (T, 4) array (w, x, y, z)
    """
    motion_path = Path(motion_file)
    data = np.load(motion_file, allow_pickle=True)
    
    # Handle .npy files (which might just be the qpos array)
    if motion_path.suffix == '.npy':
        # If it's a .npy file, it might be just the qpos array
        if isinstance(data, np.ndarray):
            # Assume it's qpos array, need to get fps from elsewhere or use default
            qpos = data
            fps = 30.0  # Default fps if not available
            print(f"Warning: Loaded .npy file as qpos array. Using default fps={fps}")
        else:
            # It's a dict-like object (from .npy with allow_pickle=True)
            if 'qpos' not in data:
                raise ValueError(f"Motion file must contain 'qpos' key. Found keys: {list(data.keys()) if hasattr(data, 'keys') else 'N/A'}")
            qpos = data['qpos']
            fps = data.get('fps', 30.0)
    else:
        # Handle .npz files
        if 'qpos' not in data:
            raise ValueError(f"Motion file must contain 'qpos' key. Found keys: {list(data.keys())}")
        qpos = data['qpos']
        if 'fps' not in data:
            fps = 30.0
            print(f"Warning: 'fps' not found in motion file. Using default fps={fps}")
        else:
            fps = float(data['fps'])
    
    num_frames = qpos.shape[0]
    num_dof = qpos.shape[1]
    
    # Extract root position and quaternion
    root_pos = qpos[:, 0:3]  # (T, 3)
    root_quat = qpos[:, 3:7]  # (T, 4) - MuJoCo format: w, x, y, z
    
    # Extract joint positions (MuJoCo order)
    mujoco_joint_pos = qpos[:, 7:]  # (T, num_joints)
    num_joints = mujoco_joint_pos.shape[1]
    
    # Convert joint order from MuJoCo to IsaacLab
    if num_joints == 29:
        # Use the mapping for 29 DOF G1 robot
        isaaclab_joint_pos = mujoco_joint_pos[:, MUJOCO_TO_ISACLAB_DOF]
    else:
        print(f"Warning: Unknown number of joints ({num_joints}). Using identity mapping.")
        isaaclab_joint_pos = mujoco_joint_pos
    
    # Create zero joint velocities (not available in the data)
    joint_vel = np.zeros_like(isaaclab_joint_pos)
    return {
        'fps': fps,
        'qpos': qpos,
        'joint_pos': torch.tensor(isaaclab_joint_pos, dtype=torch.float32, device=device),
        'joint_vel': torch.tensor(joint_vel, dtype=torch.float32, device=device),
        'root_pos': torch.tensor(root_pos, dtype=torch.float32, device=device),
        'root_quat': torch.tensor(root_quat, dtype=torch.float32, device=device),
        'num_frames': num_frames,
    }


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Load motion data - resolve path relative to current working directory or as absolute
    motion_file = args_cli.motion_file
    motion_path = Path(motion_file)
    
    # Try to resolve the path
    if not motion_path.is_absolute():
        # Try relative to current working directory first
        if not motion_path.exists():
            # Try relative to bm_generalist root
            bm_generalist_root = Path(__file__).parent.parent
            alt_path = bm_generalist_root / motion_file
            if alt_path.exists():
                motion_path = alt_path
            else:
                # Try in holosoma demo_data directory
                holosoma_demo = Path("/home/ericcsr/robot_imitation/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data")
                alt_path = holosoma_demo / motion_file
                if alt_path.exists():
                    motion_path = alt_path
                else:
                    # Try relative to workspace root
                    workspace_root = Path("/home/ericcsr/robot_imitation")
                    alt_path = workspace_root / motion_file
                    if alt_path.exists():
                        motion_path = alt_path
    
    motion_file = str(motion_path.resolve())
    if not Path(motion_file).exists():
        raise FileNotFoundError(
            f"Motion file not found: {motion_file}\n"
            f"Tried: {args_cli.motion_file}\n"
            f"Current working directory: {Path.cwd()}\n"
            f"Script directory: {Path(__file__).parent}"
        )
    
    motion_data = load_mujoco_motion(motion_file, device=sim.device)
    
    fps = motion_data['fps']
    num_frames = motion_data['num_frames']
    frame_time = 1.0 / fps  # Time per frame in seconds
    
    print(f"Loaded motion with {num_frames} frames at {fps} fps")
    print(f"Frame time: {frame_time:.4f} seconds")
    print(f"Joint positions shape: {motion_data['joint_pos'].shape}")
    
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    
    # Timing control for fps matching
    last_frame_time = time.time()
    next_frame_time = last_frame_time + frame_time

    # Simulation loop
    while simulation_app.is_running():
        current_time = time.time()
        
        # Only advance frame if enough time has passed according to fps
        if current_time >= next_frame_time:
            time_steps += 1
            
            # Reset if we've reached the end
            reset_ids = time_steps >= num_frames
            time_steps[reset_ids] = 0
            
            # Update next frame time (account for any drift)
            next_frame_time = current_time + frame_time
        
        # Get current frame data
        current_frame = time_steps
        
        # Extract root state
        root_pos = motion_data['root_pos'][current_frame]  # (num_envs, 3)
        root_quat = motion_data['root_quat'][current_frame]  # (num_envs, 4)
        
        # Set root state (position + quaternion + zero velocities)
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = root_pos + scene.env_origins
        root_states[:, 3:7] = root_quat
        root_states[:, 7:10] = 0.0  # linear velocity
        root_states[:, 10:] = 0.0  # angular velocity

        # Get joint positions and velocities
        joint_pos = motion_data['joint_pos'][current_frame]  # (num_envs, num_joints)
        joint_vel = motion_data['joint_vel'][current_frame]  # (num_envs, num_joints)

        # Write states to simulator
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()  # We don't want physics (sim.step())
        scene.update(sim_dt)

        # Update camera to show grid - look at center of all environments
        if scene.num_envs > 1:
            # Calculate center of all environments
            center_pos = scene.env_origins.mean(dim=0).cpu().numpy()
            # Position camera to show all environments
            max_offset = torch.norm(scene.env_origins, dim=1).max().item()
            camera_distance = max(max_offset * 1.5, 5.0)
            sim.set_camera_view(center_pos + np.array([camera_distance, camera_distance, camera_distance * 0.5]), center_pos)
        else:
            # Single environment: follow the robot
            pos_lookat = root_states[0, :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    # Create scene config with terrain URDF path - resolve path similar to motion file
    terrain_urdf = Path(args_cli.terrain_urdf)
    
    if not terrain_urdf.is_absolute():
        # Try relative to current working directory first
        if not terrain_urdf.exists():
            # Try relative to bm_generalist root
            bm_generalist_root = Path(__file__).parent.parent
            alt_path = bm_generalist_root / args_cli.terrain_urdf
            if alt_path.exists():
                terrain_urdf = alt_path
            else:
                # Try in holosoma demo_data directory
                holosoma_demo = Path("/home/ericcsr/robot_imitation/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data")
                alt_path = holosoma_demo / args_cli.terrain_urdf
                if alt_path.exists():
                    terrain_urdf = alt_path
                else:
                    # Try relative to workspace root
                    workspace_root = Path("/home/ericcsr/robot_imitation")
                    alt_path = workspace_root / args_cli.terrain_urdf
                    if alt_path.exists():
                        terrain_urdf = alt_path
    
    terrain_urdf_path = str(terrain_urdf.resolve())
    if not Path(terrain_urdf_path).exists():
        raise FileNotFoundError(
            f"Terrain URDF file not found: {terrain_urdf_path}\n"
            f"Tried: {args_cli.terrain_urdf}\n"
            f"Current working directory: {Path.cwd()}\n"
            f"Script directory: {Path(__file__).parent}"
        )
    
    ReplayLafanTerrainSceneCfg = create_scene_cfg(terrain_urdf_path)
    num_envs = args_cli.num_envs
    env_spacing = args_cli.env_spacing
    
    # Calculate grid dimensions for better visualization
    grid_size = int(np.ceil(np.sqrt(num_envs)))
    print(f"Creating {num_envs} environments in a {grid_size}x{grid_size} grid layout")
    print(f"Environment spacing: {env_spacing} meters")
    
    scene_cfg = ReplayLafanTerrainSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
