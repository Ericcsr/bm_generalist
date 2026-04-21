"""Script to browse IsaacLab environment setup by freezing robots at initial poses.

This script creates an RL environment and freezes all robots at their initial poses,
allowing the user to drag the camera window in IsaacLab visualizer to browse
the environment setup and terrain.

.. code-block:: bash

    # Usage
    python scripts/browse_env.py --task Tracking-Flat-3pt-G1-v0 --num_envs 16 --motion_dir lafan

    # With height map visualization (spheres at each grid point, for tasks with height_scanner)
    python scripts/browse_env.py --task Tracking-3pt-Binded-Multi-Terrain-G1-v0 --motion_dir lafan --terrain_folder lafan --visualize_height_map
    # Visualize a specific env only (omit --height_map_env to show all envs)
    python scripts/browse_env.py --task Tracking-3pt-Binded-Multi-Terrain-G1-v0 --motion_dir lafan --terrain_folder lafan --visualize_height_map --height_map_env 0

    # With object proxy (binded multi-object task)
    python scripts/browse_env.py --task Tracking-3pt-Binded-Multi-Object-G1-v0 --motion_dir omomo_box --object_folder omomo_object/omomo_box

    # Merged terrain+object folder (same --object_terrain_folder as train.py / play.py)
    python scripts/browse_env.py --task Tracking-3pt-Binded-Multi-Terrain-Object-Merged-G1-v0 \
        --motion_dir bones_box --terrain_folder lafan_terrain/box --object_folder omomo_object/box \
        --object_terrain_folder bones_box
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
parser.add_argument(
    "--object_folder",
    type=str,
    default=None,
    help="Path to object folder (proxy URDFs + fitted trajectories). For binded multi-object task, pairs with motion_dir by base name.",
)
parser.add_argument(
    "--object_terrain_folder",
    type=str,
    default=None,
    help="Merged asset folder (per-motion terrain URDF, thin-plate proxy, *_object_fitted.npz, combined "
    "*_contact_nodes.npy). Same as train.py / play.py --object_terrain_folder; merged with terrain/object dirs.",
)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=None,
    help="Spacing between terrains/robots in meters (for multi-terrain). Default: 6.0. Use 0 for clustered at origin.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--visualize_height_map",
    action="store_true",
    help="Visualize the 2m x 2m height map grid as spheres in IsaacLab (requires height_scanner sensor).",
)
parser.add_argument(
    "--height_map_env",
    type=int,
    default=None,
    help="Environment index to visualize height map for (when --visualize_height_map is set). If not set, visualizes all envs.",
)

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
import numpy as np
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


def _visualize_height_map_spheres(
    scene,
    env_indices: list[int] | None,
    sphere_radius: float = 0.05,
    update_interval_s: float = 0.5,
    debug: bool = True,
):
    """Create sphere markers at each height map grid point in IsaacLab. Returns update callback or None.

    Args:
        scene: The simulation scene.
        env_indices: List of env indices to visualize. If None, visualizes all envs.
        sphere_radius: Radius of each sphere marker in meters.
        update_interval_s: Minimum interval between updates.
        debug: If True, print diagnostic info on first update.
    """
    if "height_scanner" not in scene.sensors:
        print("[WARN]: --visualize_height_map set but scene has no height_scanner. Skipping visualization.")
        return None

    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    sensor = scene.sensors["height_scanner"]
    num_envs = sensor.data.ray_hits_w.shape[0]
    num_rays = sensor.data.ray_hits_w.shape[1]

    if env_indices is None:
        env_indices = list(range(num_envs))
    else:
        env_indices = [min(max(e, 0), num_envs - 1) for e in env_indices]

    # Sphere marker config - use /World/ for visibility in Isaac Sim viewport
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/HeightMapSpheres",
        markers={
            "grid": sim_utils.SphereCfg(
                radius=sphere_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 1.0)),
            ),
        },
    )
    marker = VisualizationMarkers(marker_cfg)
    marker.set_visibility(True)

    last_update = [0.0]
    debug_done = [False]

    def update_visualization(current_time: float):
        if current_time - last_update[0] < update_interval_s:
            return
        last_update[0] = current_time

        ray_hits_w = sensor.data.ray_hits_w
        # Filter: valid hits (no inf), for selected envs
        positions_list = []
        for ei in env_indices:
            hits = ray_hits_w[ei].cpu().numpy()
            valid = np.isfinite(hits).all(axis=1)
            positions_list.append(hits[valid])
        if not positions_list:
            return
        positions = np.vstack(positions_list)
        if positions.size == 0:
            # Fallback: ray caster only hits slot_0; env 0 is at slot 0
            if len(env_indices) > 1 and 0 not in env_indices:
                env_indices[:] = [0]
                if debug and not debug_done[0]:
                    debug_done[0] = True
                    print("[INFO]: Falling back to env 0 (ray caster only hits slot_0 terrain).")
                return  # retry next update with env 0
            if debug and not debug_done[0]:
                debug_done[0] = True
                print(
                    "[WARN]: Height map: 0 valid ray hits. Ray caster only hits mesh at slot_0; "
                    "ensure env 0's robot is over slot_0 terrain."
                )
            return
        if debug and not debug_done[0]:
            debug_done[0] = True
            print(f"[INFO]: Height map spheres: {len(positions)} points from envs {env_indices}")
        marker.visualize(translations=positions)

    return update_visualization


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Browse environment with frozen robots."""
    # override configurations with CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.env_spacing is not None and hasattr(env_cfg, "env_spacing"):
        env_cfg.env_spacing = args_cli.env_spacing
        env_cfg.scene.env_spacing = args_cli.env_spacing
        if hasattr(env_cfg.scene, "terrain") and hasattr(env_cfg.scene.terrain, "spawn") and hasattr(env_cfg.scene.terrain.spawn, "env_spacing"):
            env_cfg.scene.terrain.spawn.env_spacing = args_cli.env_spacing
    # Sync num_envs to multi-terrain spawner if present (needed for per-env terrain variety)
    if hasattr(env_cfg.scene, "terrain") and hasattr(env_cfg.scene.terrain, "spawn") and hasattr(env_cfg.scene.terrain.spawn, "num_envs"):
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
            # object_folder_path: explicit --object_folder, or fallback; empty when only --object_terrain_folder
            if hasattr(env_cfg, "object_folder_path"):
                if args_cli.object_folder:
                    env_cfg.object_folder_path = args_cli.object_folder
                elif getattr(args_cli, "object_terrain_folder", None):
                    env_cfg.object_folder_path = ""
                else:
                    env_cfg.object_folder_path = args_cli.terrain_folder or args_cli.motion_dir
            if hasattr(env_cfg, "object_terrain_folder_path"):
                env_cfg.object_terrain_folder_path = getattr(args_cli, "object_terrain_folder", None) or ""
            env_cfg.setup_motion_terrain_pairs()
            num_pairs = len(env_cfg.commands.motion.motion_files)
            # num_envs must be >= 1 (InteractiveScene requires non-empty env_prim_paths)
            if args_cli.num_envs is not None:
                env_cfg.scene.num_envs = max(1, args_cli.num_envs)
                if hasattr(env_cfg.scene, "terrain") and hasattr(env_cfg.scene.terrain, "spawn"):
                    env_cfg.scene.terrain.spawn.num_envs = max(1, args_cli.num_envs)
            else:
                # Default to num_pairs for browsing so you see exactly one terrain/object per motion
                env_cfg.scene.num_envs = max(1, num_pairs)
                if hasattr(env_cfg.scene, "terrain") and hasattr(env_cfg.scene.terrain, "spawn"):
                    env_cfg.scene.terrain.spawn.num_envs = max(1, num_pairs)
                print(f"[INFO]: num_envs={max(1, num_pairs)} for browsing (1 terrain/object per motion). Use --num_envs N to override.")
        else:
            motion_files = [args_cli.motion_dir + "/" + x for x in os.listdir(args_cli.motion_dir) if x.endswith(".npz")]
            if motion_files:
                env_cfg.commands.motion.motion_files = motion_files
                print(f"[INFO]: Loaded {len(motion_files)} motion files from {args_cli.motion_dir}")
            else:
                print(f"[WARN]: No .npz files found in {args_cli.motion_dir}")
    elif args_cli.motion_dir:
        print(f"[WARN]: Motion directory not found: {args_cli.motion_dir}")

    # MultiMotionLoader requires at least one motion file; exit early with clear message
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        motion_files = getattr(env_cfg.commands.motion, "motion_files", None)
        if motion_files is not None and len(motion_files) == 0:
            print(
                "[ERROR]: No motion files loaded. This task requires --motion_dir (and for combined "
                "terrain+object: --terrain_folder, --object_folder, optional --object_terrain_folder). "
                "Ensure paths exist and setup_motion_terrain_pairs() succeeds."
            )
            sys.exit(1)

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

    # Setup height map visualization if requested (spheres at each grid point in IsaacLab)
    height_map_update = None
    if args_cli.visualize_height_map:
        env_indices = None if args_cli.height_map_env is None else [args_cli.height_map_env]
        height_map_update = _visualize_height_map_spheres(scene, env_indices)
        if height_map_update is not None:
            if env_indices is None:
                print(f"[INFO]: Height map sphere visualization enabled for all {scene.num_envs} envs.")
            else:
                print(f"[INFO]: Height map sphere visualization enabled for env {env_indices[0]}.")

    # Initialize kinematic object poses if available (multi-object task)
    try:
        from whole_body_tracking.tasks.tracking.mdp.events import update_kinematic_object_poses
        update_kinematic_object_poses(env.unwrapped, None, command_name="motion")
        print("[INFO]: Kinematic object poses initialized.")
    except Exception as e:
        print(f"[INFO]: No kinematic objects to initialize ({e})")

    # Freeze robots at initial poses and allow camera browsing
    try:
        import time
        last_update_time = time.time()
        update_interval = 1.0 / 30.0  # Update robot states at 30 Hz to keep them frozen

        # Write initial states once before the loop
        robot.write_root_state_to_sim(initial_root_states)
        robot.write_joint_state_to_sim(initial_joint_pos, initial_joint_vel)
        scene.write_data_to_sim()

        # Step sim once to update sensors (needed for height map)
        if height_map_update is not None:
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            # Force height scanner to recompute (lazy_sensor_update may skip it)
            scene.sensors["height_scanner"].update(sim.get_physics_dt(), force_recompute=True)
            height_map_update(time.time())

        while simulation_app.is_running():
            current_time = time.time()
            
            # Only update robot states periodically to keep them frozen
            # This reduces interference with camera controls
            if current_time - last_update_time >= update_interval:
                # Write initial states to freeze robots (prevent physics from moving them)
                robot.write_root_state_to_sim(initial_root_states)
                robot.write_joint_state_to_sim(initial_joint_pos, initial_joint_vel)
                scene.write_data_to_sim()
                # Step sim to update sensors (e.g. height_scanner) when visualizing
                if height_map_update is not None:
                    sim.step(render=False)
                    scene.update(sim.get_physics_dt())
                    scene.sensors["height_scanner"].update(sim.get_physics_dt(), force_recompute=True)
                    # Re-apply frozen state after step so robots stay frozen
                    robot.write_root_state_to_sim(initial_root_states)
                    robot.write_joint_state_to_sim(initial_joint_pos, initial_joint_vel)
                    scene.write_data_to_sim()
                last_update_time = current_time

            # Update height map visualization
            if height_map_update is not None:
                height_map_update(current_time)

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
