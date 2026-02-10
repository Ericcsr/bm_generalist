"""Script to browse IsaacLab environment setup by freezing robots at initial poses.

This script creates an RL environment and freezes all robots at their initial poses,
allowing the user to drag the camera window in IsaacLab visualizer to browse
the environment setup and terrain.

.. code-block:: bash

    # Usage
    python scripts/browse_env.py --task Tracking-Flat-3pt-G1-v0 --num_envs 16 --motion_dir lafan
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Browse IsaacLab environment setup with frozen robots.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_dir", type=str, default="lafan", help="Path to the motion directory.")
parser.add_argument(
    "--terrain_folder",
    type=str,
    default=None,
    help="Path to terrain folder (URDF files). For binded multi-terrain task, pairs with motion_dir by base name.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Browse environment with frozen robots."""
    # override configurations with CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # Sync num_envs to multi-terrain spawner if present (needed for per-env terrain variety)
    if hasattr(env_cfg.scene.terrain, "spawn") and hasattr(env_cfg.scene.terrain.spawn, "num_envs"):
        env_cfg.scene.terrain.spawn.num_envs = env_cfg.scene.num_envs
    
    # set the environment seed
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # Disable camera tracking to allow free camera movement
    # The viewer is configured to track robot by default, which resets camera
    env_cfg.viewer.origin_type = "world"  # Use world origin instead of tracking robot
    env_cfg.viewer.asset_name = None  # Don't track any asset

    # Load motion files if motion_dir is provided
    if args_cli.motion_dir and os.path.exists(args_cli.motion_dir):
        if hasattr(env_cfg, "setup_motion_terrain_pairs"):
            env_cfg.motion_dir = args_cli.motion_dir
            env_cfg.terrain_folder_path = args_cli.terrain_folder or args_cli.motion_dir
            env_cfg.setup_motion_terrain_pairs()
            if args_cli.num_envs is not None:
                env_cfg.scene.num_envs = args_cli.num_envs
                env_cfg.scene.terrain.spawn.num_envs = args_cli.num_envs
        else:
            motion_files = [args_cli.motion_dir + "/" + x for x in os.listdir(args_cli.motion_dir) if x.endswith(".npz")]
            if motion_files:
                env_cfg.commands.motion.motion_files = motion_files
                print(f"[INFO]: Loaded {len(motion_files)} motion files from {args_cli.motion_dir}")
            else:
                print(f"[WARN]: No .npz files found in {args_cli.motion_dir}")
    elif args_cli.motion_dir:
        print(f"[WARN]: Motion directory not found: {args_cli.motion_dir}")

    # create isaac environment with rendering enabled
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)  # render_mode=None uses default viewer
    
    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Reset environment to get initial states
    print("[INFO]: Resetting environment to get initial poses...")
    obs, _ = env.reset()
    
    # Get access to the scene and robot
    scene = env.unwrapped.scene
    robot = scene["robot"]
    sim = env.unwrapped.sim
    
    # Disable camera tracking after environment creation
    # The viewer might be tracking the robot, which resets camera position
    if hasattr(env.unwrapped, 'viewer') and env.unwrapped.viewer is not None:
        # Try to disable automatic camera updates
        if hasattr(env.unwrapped.viewer, 'origin_type'):
            env.unwrapped.viewer.origin_type = "world"
        if hasattr(env.unwrapped.viewer, 'asset_name'):
            env.unwrapped.viewer.asset_name = None
    
    # Store initial robot states
    initial_root_states = robot.data.root_state_w.clone()
    initial_joint_pos = robot.data.joint_pos.clone()
    initial_joint_vel = torch.zeros_like(robot.data.joint_vel)
    
    print(f"[INFO]: Environment setup complete!")
    print(f"[INFO]: Number of environments: {scene.num_envs}")
    print(f"[INFO]: Environment spacing: {scene.cfg.env_spacing}")
    print(f"[INFO]: Robot initial poses frozen. You can now drag the camera window to browse environments.")
    print(f"[INFO]: Press Ctrl+C to exit.")
    
    # Freeze robots at initial poses and allow camera browsing
    try:
        import time
        last_update_time = time.time()
        update_interval = 1.0 / 30.0  # Update robot states at 30 Hz to keep them frozen
        
        # Write initial states once before the loop
        robot.write_root_state_to_sim(initial_root_states)
        robot.write_joint_state_to_sim(initial_joint_pos, initial_joint_vel)
        scene.write_data_to_sim()
        
        while simulation_app.is_running():
            current_time = time.time()
            
            # Only update robot states periodically to keep them frozen
            # This reduces interference with camera controls
            if current_time - last_update_time >= update_interval:
                # Write initial states to freeze robots (prevent physics from moving them)
                robot.write_root_state_to_sim(initial_root_states)
                robot.write_joint_state_to_sim(initial_joint_pos, initial_joint_vel)
                scene.write_data_to_sim()
                last_update_time = current_time
            
            # Render without any scene updates that might reset camera
            # This allows free camera movement
            sim.render()
            
    except KeyboardInterrupt:
        print("\n[INFO]: Exiting browser...")
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
