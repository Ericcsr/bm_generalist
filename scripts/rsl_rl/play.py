"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_dir", type=str, default="lafan", help="Path to the motion dir.")
parser.add_argument(
    "--terrain_folder",
    type=str,
    default=None,
    help="Path to terrain folder (URDF files). For binded multi-terrain task, pairs with motion_dir by base name. Defaults to motion_dir if not set.",
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
    "*_contact_nodes.npy). Same as train.py; merged with --terrain_folder / --object_folder for mixed datasets.",
)
parser.add_argument(
    "--contact_label_debug_vis",
    action="store_true",
    default=False,
    help="Visualize contact labels at ankle_roll and wrist_yaw links: green=contact, red=no contact.",
)
parser.add_argument(
    "--contact_debug_zero_out",
    action="store_true",
    default=False,
    help="Debug: zero out all contact for one env so its ankle sinks (verifies contact-based collision).",
)
parser.add_argument(
    "--contact_debug_zero_out_env",
    type=int,
    default=None,
    help="Env index to disable contact for (with --contact_debug_zero_out). Default: all envs.",
)
parser.add_argument(
    "--zero_hand_contact_label",
    action="store_true",
    default=False,
    help="Zero out hand (wrist) contact labels fed to policy. Use to compare robot behavior with vs without contact labels.",
)
parser.add_argument(
    "--quiet_logs",
    action="store_true",
    default=True,
    help="Suppress USD/Imaging warning spam so [KinObj] and other messages are visible (default: True).",
)
parser.add_argument(
    "--no_quiet_logs",
    action="store_true",
    default=False,
    help="Show all USD/Imaging warnings (disables --quiet_logs).",
)
parser.add_argument(
    "--mujoco_vis",
    action="store_true",
    default=False,
    help="Launch MuJoCo passive viewer to mirror robot and proxy poses in real time (same process).",
)
parser.add_argument(
    "--mujoco_vis_env",
    type=int,
    default=0,
    help="Environment index to visualize in MuJoCo viewer (default: 0).",
)
parser.add_argument(
    "--mujoco_g1_xml",
    type=str,
    default=None,
    help="Path to G1 MuJoCo XML. Default: auto-detect from GMR/HDMI.",
)
parser.add_argument(
    "--play_debug",
    action="store_true",
    default=True,
    help="Enable all play-time debug: contact label vis, motion vis, object pose/contact prints, sanity checks (default: True).",
)
parser.add_argument(
    "--no_play_debug",
    action="store_true",
    default=False,
    help="Disable play-time debug (overrides --play_debug).",
)
parser.add_argument(
    "--physics_phase",
    action="store_true",
    default=False,
    help="When enabled, magic force disappears the first time both wrists make contact with object; object then fully controlled by robot.",
)
parser.add_argument(
    "--contact_vis",
    action="store_true",
    default=False,
    help="Visualize hand-object contact positions (spheres) and force arrows in MuJoCo viewer. Requires --mujoco_vis.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Suppress USD/Imaging warning spam via carb.settings (PointInstancer, delegate, etc.)
quiet = getattr(args_cli, "quiet_logs", True) and not getattr(args_cli, "no_quiet_logs", False)
if quiet:
    try:
        import carb.settings
        settings = carb.settings.get_settings()
        for channel in ("omni.usd", "usdImaging", "omni.hydra", "omni.hydra.scene_delegate.plugin"):
            settings.set(f"/log/channels/{channel}", "error")
    except Exception:
        pass

import gymnasium as gym
import numpy as np
import os
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Disable camera tracking to allow free camera movement with mouse
    # The viewer is configured to track robot by default, which resets camera
    env_cfg.viewer.origin_type = "world"  # Use world origin instead of tracking robot
    env_cfg.viewer.asset_name = None  # Don't track any asset

    # Start all motions from initial frame for consistent play/eval (no random mid-motion resampling)
    env_cfg.commands.motion.start_from_initial_frame = True

    # Disable contact label dropout during play for deterministic evaluation
    if hasattr(env_cfg.commands.motion, "contact_label_dropout_prob"):
        env_cfg.commands.motion.contact_label_dropout_prob = 0.0

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Play-time debug: enable all extra visualization and prints (only in play.py, not train.py)
    play_debug = getattr(args_cli, "play_debug", True) and not getattr(args_cli, "no_play_debug", False)
    contact_label_debug_vis = getattr(args_cli, "contact_label_debug_vis", False) or play_debug

    # create isaac environment
    if hasattr(env_cfg, "setup_motion_terrain_pairs"):
        # Binded multi-terrain / multi-object: pair motions with terrains/objects by base name
        env_cfg.motion_dir = args_cli.motion_dir
        env_cfg.terrain_folder_path = args_cli.terrain_folder or args_cli.motion_dir
        if hasattr(env_cfg, "object_folder_path"):
            if args_cli.object_folder:
                env_cfg.object_folder_path = args_cli.object_folder
            elif getattr(args_cli, "object_terrain_folder", None):
                env_cfg.object_folder_path = ""
            else:
                env_cfg.object_folder_path = args_cli.terrain_folder or args_cli.motion_dir
        if hasattr(env_cfg, "object_terrain_folder_path"):
            env_cfg.object_terrain_folder_path = getattr(args_cli, "object_terrain_folder", None) or ""
        if hasattr(env_cfg, "physics_phase_flag"):
            env_cfg.physics_phase_flag = getattr(args_cli, "physics_phase", False)
            if env_cfg.physics_phase_flag:
                print("[INFO]: physics_phase enabled: magic force turns off when both wrists contact object")
        # Contact-based collision is only in Tracking-3pt-Binded-Multi-Terrain-Contact-Collision-G1-v0
        env_cfg.setup_motion_terrain_pairs()
        # Log object trajectory load for multi-object / combined task debugging
        if hasattr(env_cfg, "object_folder_path") and env_cfg.object_folder_path:
            n = 0
            for mod_name in (
                "whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_binded_multi_object",
                "whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_binded_multi_terrain_object",
            ):
                try:
                    import importlib
                    mod = importlib.import_module(mod_name)
                    ot = getattr(mod, "_object_trajectories", None)
                    if ot and len(ot) > 0:
                        n = len(ot)
                        break
                except ImportError:
                    pass
            print(f"[INFO]: Object trajectories loaded for play: {n} motions")
        # Play debug: enable contact label vis, keep motion debug_vis (don't disable for clutter)
        if play_debug:
            env_cfg.commands.motion.debug_vis = True  # Keep motion skeleton vis
        elif contact_label_debug_vis:
            env_cfg.commands.motion.debug_vis = False  # Reduce clutter when only contact vis
        if hasattr(env_cfg.events, "set_contact_link_collision"):
            env_cfg.events.set_contact_link_collision.params["enable_contact_label_debug_vis"] = contact_label_debug_vis
            if contact_label_debug_vis:
                n = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
                env_cfg.events.set_contact_link_collision.params["max_contact_label_vis_envs"] = n
                print("[ContactLabels] Debug vis enabled (green=contact, red=no contact)")
        if getattr(args_cli, "contact_debug_zero_out", False):
            env_cfg.commands.motion.contact_debug_zero_out = True
            env_cfg.commands.motion.contact_debug_zero_out_env = getattr(
                args_cli, "contact_debug_zero_out_env", None
            )
            env_idx = env_cfg.commands.motion.contact_debug_zero_out_env
            msg = f"motion {env_idx}" if env_idx is not None else "all motions"
            print(f"[ContactLabels] Debug zero-out enabled: {msg} will have contact zeroed at load (ankle should sink)")
        if getattr(args_cli, "zero_hand_contact_label", False):
            env_cfg.commands.motion.zero_hand_contact_label = True
            print("[ContactLabels] Hand contact labels zeroed (policy receives no grasp labels)")
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
            if hasattr(env_cfg.scene, "terrain") and hasattr(env_cfg.scene.terrain, "spawn"):
                env_cfg.scene.terrain.spawn.num_envs = args_cli.num_envs
    else:
        motion_files = [args_cli.motion_dir + "/" + x for x in os.listdir(args_cli.motion_dir)]
        env_cfg.commands.motion.motion_files = motion_files
        if getattr(args_cli, "contact_label_debug_vis", False):
            print("[ContactLabels] WARNING: contact_label_debug_vis requires binded-multi-terrain task (setup_motion_terrain_pairs)")
    # Signal to events (e.g. update_kinematic_object_poses) to enable debug prints; train.py never sets this
    if play_debug:
        os.environ["BMG_PLAY_DEBUG"] = "1"
    else:
        os.environ.pop("BMG_PLAY_DEBUG", None)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Disable camera tracking after environment creation (viewer may override config)
    if hasattr(env.unwrapped, "viewer") and env.unwrapped.viewer is not None:
        if hasattr(env.unwrapped.viewer, "origin_type"):
            env.unwrapped.viewer.origin_type = "world"
        if hasattr(env.unwrapped.viewer, "asset_name"):
            env.unwrapped.viewer.asset_name = None

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # MuJoCo passive viewer (same process, mirrors Isaac poses)
    mujoco_vis = None
    mujoco_vis_use_terrain_origin = False  # True when terrain URDF used (position from env_origins)
    if getattr(args_cli, "mujoco_vis", False):
        try:
            from whole_body_tracking.utils.mujoco_visualizer import create_mujoco_visualizer
            from whole_body_tracking.tasks.tracking.terrain_mesh_spawner_cfg import FLAT_PLANE_MARKER

            proxy_urdf = None
            if hasattr(env_cfg.scene, "proxy_object") and hasattr(env_cfg.scene.proxy_object, "spawn"):
                spawn = env_cfg.scene.proxy_object.spawn
                if hasattr(spawn, "asset_path") and spawn.asset_path:
                    proxy_urdf = spawn.asset_path
                elif hasattr(spawn, "assets_cfg") and spawn.assets_cfg:
                    env_id = getattr(args_cli, "mujoco_vis_env", 0)
                    idx = env_id % len(spawn.assets_cfg)
                    proxy_urdf = getattr(spawn.assets_cfg[idx], "asset_path", None)
            elif hasattr(env_cfg.scene, "terrain") and hasattr(env_cfg.scene.terrain, "spawn"):
                # Multi-terrain / multi-terrain-by-exp: resolve terrain URDF for mujoco_vis_env
                spawn = env_cfg.scene.terrain.spawn
                urdf_paths = getattr(spawn, "urdf_paths", []) or []
                motion_to_slot = getattr(spawn, "motion_to_slot", None)
                motion_files = getattr(env_cfg.commands.motion, "motion_files", []) or []
                num_motions = len(motion_files)
                vis_env = getattr(args_cli, "mujoco_vis_env", 0)
                if urdf_paths:
                    if motion_to_slot is not None and num_motions > 0:
                        slot = motion_to_slot[vis_env % num_motions]
                    else:
                        slot = vis_env % len(urdf_paths)
                    if slot < len(urdf_paths) and urdf_paths[slot] != FLAT_PLANE_MARKER:
                        proxy_urdf = urdf_paths[slot]
                        mujoco_vis_use_terrain_origin = True
            mujoco_vis = create_mujoco_visualizer(
                env_id=args_cli.mujoco_vis_env,
                g1_mjcf_path=getattr(args_cli, "mujoco_g1_xml", None),
                proxy_urdf_path=proxy_urdf,
            )
            if mujoco_vis is not None:
                msg = f"[INFO] MuJoCo visualizer launched (env {args_cli.mujoco_vis_env})"
                if getattr(args_cli, "contact_vis", False):
                    msg += " [hand-object contact vis enabled]"
                if proxy_urdf:
                    msg += f" with {'terrain' if mujoco_vis_use_terrain_origin else 'proxy'} from {proxy_urdf}"
                print(msg)
        except Exception as e:
            print(f"[WARN] MuJoCo visualizer failed: {e}")
        if getattr(args_cli, "contact_vis", False) and mujoco_vis is None:
            print("[WARN] --contact_vis requires --mujoco_vis; contact visualization disabled.")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    # Play-time sanity checks (only when play_debug)
    play_debug = getattr(args_cli, "play_debug", True) and not getattr(args_cli, "no_play_debug", False)
    if play_debug:
        base_env = env.unwrapped
        n_envs = base_env.scene.num_envs
        robot = base_env.scene["robot"]
        print(f"[PlayDebug] Sanity: num_envs={n_envs}, robot_joints={robot.num_joints}")
        try:
            proxy = base_env.scene["proxy_object"]
            proxy_pose = proxy.data.root_link_pose_w[0].cpu().numpy()
            print(f"[PlayDebug] Sanity: proxy_object present, env0 pose pos={proxy_pose[:3]}")
        except KeyError:
            print("[PlayDebug] Sanity: no proxy_object in scene")
        cmd = base_env.command_manager.get_term("motion")
        if hasattr(cmd, "time_steps"):
            print(f"[PlayDebug] Sanity: motion time_steps[0]={cmd.time_steps[0].item()}")
        if hasattr(cmd, "motion") and hasattr(cmd.motion, "time_step_total"):
            print(f"[PlayDebug] Sanity: motion total_frames={cmd.motion.time_step_total}")

    def _print_hand_object_contact(scene, threshold: float = 0.01):
        """Print hand-object contact events (matches standalone_magic_force_hand_object.py _print_contact_events)."""
        l_sensor = scene.sensors.get("object_contact_left_wrist")
        r_sensor = scene.sensors.get("object_contact_right_wrist")
        if l_sensor is None or r_sensor is None:
            return
        for env_id in range(scene.num_envs):
            l_force = 0.0
            if l_sensor.data.force_matrix_w is not None:
                force_vec = l_sensor.data.force_matrix_w[env_id].sum(dim=(0, 1))
                l_force = torch.norm(force_vec).item()
            r_force = 0.0
            if r_sensor.data.force_matrix_w is not None:
                force_vec = r_sensor.data.force_matrix_w[env_id].sum(dim=(0, 1))
                r_force = torch.norm(force_vec).item()
            if l_force > threshold and env_id == args_cli.mujoco_vis_env:
                print(f"[CONTACT | Env {env_id}] ProxyObject felt hit from LEFT WRIST (Force: {l_force:.2f} N)")
            if r_force > threshold and env_id == args_cli.mujoco_vis_env:
                print(f"[CONTACT | Env {env_id}] ProxyObject felt hit from RIGHT WRIST (Force: {r_force:.2f} N)")

    def _sync_mujoco_vis():
        """Sync MuJoCo viewer with Isaac poses for selected env."""
        if mujoco_vis is None or not mujoco_vis.is_running():
            return
        base_env = env.unwrapped
        env_id = args_cli.mujoco_vis_env
        if env_id >= base_env.scene.num_envs:
            return
        robot = base_env.scene["robot"]
        rp = robot.data.root_pos_w[env_id].cpu().numpy()
        rq = robot.data.root_quat_w[env_id].cpu().numpy()  # (w,x,y,z)
        jp = robot.data.joint_pos[env_id].cpu().numpy()
        joint_names = list(robot.joint_names)  # same order as joint_pos (URDF)
        proxy_pos, proxy_quat = None, None
        try:
            proxy = base_env.scene["proxy_object"]
            pose = proxy.data.root_link_pose_w[env_id].cpu().numpy()
            proxy_pos = pose[:3]
            proxy_quat = pose[3:7]  # (w,x,y,z)
        except KeyError:
            if mujoco_vis_use_terrain_origin and hasattr(base_env.scene, "env_origins"):
                env_origins = base_env.scene.env_origins
                proxy_pos = env_origins[env_id].cpu().numpy()
                proxy_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        contact_positions, contact_forces = None, None
        if getattr(args_cli, "contact_vis", False):
            try:
                # Prefer hand_object_contact_* (wrist-ProxyObject only, excludes hand-hand)
                try:
                    left_sensor = base_env.scene.sensors["hand_object_contact_left"]
                    right_sensor = base_env.scene.sensors["hand_object_contact_right"]
                    wrist_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
                    robot_body_ids, _ = robot.find_bodies(wrist_names, preserve_order=True)
                    positions, forces = [], []
                    hand_radius = 0.03
                    for i, sensor in enumerate([left_sensor, right_sensor]):
                        rb_id = robot_body_ids[i]
                        wrist_pos = robot.data.body_pos_w[env_id, rb_id].cpu().numpy()
                        # Use force_matrix_w (wrist-ProxyObject only) to exclude hand-hand
                        fm = sensor.data.force_matrix_w
                        if fm is not None:
                            force = fm[env_id, 0, :, :].sum(dim=0).cpu().numpy()
                        else:
                            force = sensor.data.net_forces_w[env_id, 0].cpu().numpy()
                        fnorm = float(np.linalg.norm(force))
                        if fnorm < 1.0:
                            continue
                        contact_pos = wrist_pos - (force / fnorm) * hand_radius
                        positions.append(contact_pos)
                        forces.append(force)
                    if positions and forces:
                        contact_positions = positions
                        contact_forces = forces
                    else:
                        contact_positions = []
                        contact_forces = []
                except KeyError:
                    contact_sensor = base_env.scene.sensors["contact_forces"]
                    wrist_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
                    robot_body_ids, robot_body_names = robot.find_bodies(wrist_names, preserve_order=True)
                    positions, forces = [], []
                    hand_radius = 0.03
                    for i, name in enumerate(robot_body_names):
                        if name in contact_sensor.body_names:
                            rb_id = robot_body_ids[i]
                            cb_id = contact_sensor.body_names.index(name)
                            wrist_pos = robot.data.body_pos_w[env_id, rb_id].cpu().numpy()
                            force = contact_sensor.data.net_forces_w[env_id, cb_id].cpu().numpy()
                            fnorm = float(np.linalg.norm(force))
                            if fnorm < 1.0:
                                continue
                            contact_pos = wrist_pos - (force / fnorm) * hand_radius
                            positions.append(contact_pos)
                            forces.append(force)
                    if positions and forces:
                        contact_positions = positions
                        contact_forces = forces
                    else:
                        contact_positions = []
                        contact_forces = []
            except (KeyError, AttributeError, ValueError):
                pass

        mujoco_vis.update(
            robot_root_pos=rp,
            robot_root_quat=rq,
            robot_joint_pos=jp,
            robot_joint_names=joint_names,
            proxy_pos=proxy_pos,
            proxy_quat=proxy_quat,
            contact_positions=contact_positions,
            contact_forces=contact_forces,
        )

    # simulate environment
    DEBUG_PRINT_INTERVAL = 200
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        _sync_mujoco_vis()
        # Print hand-object contact when detected (matches standalone_magic_force_hand_object.py)
        if play_debug:
            base_env = env.unwrapped
            if "proxy_object" in base_env.scene.keys():
                _print_hand_object_contact(base_env.scene)

        # Play-time periodic debug prints (object pose, contact)
        if play_debug and timestep > 0 and timestep % DEBUG_PRINT_INTERVAL == 0:
            try:
                base_env = env.unwrapped
                env_id = 0
                robot = base_env.scene["robot"]
                rp = robot.data.root_pos_w[env_id].cpu().numpy()
                print(f"[PlayDebug] step={timestep} env0 robot_pos={rp}")
                try:
                    proxy = base_env.scene["proxy_object"]
                    pose = proxy.data.root_link_pose_w[env_id].cpu().numpy()
                    print(f"[PlayDebug] step={timestep} env0 proxy_pos={pose[:3]}")
                except KeyError:
                    pass
                cmd = base_env.command_manager.get_term("motion")
                cl = getattr(cmd.motion, "contact_labels", None)
                if cl is not None and hasattr(cmd, "time_steps"):
                    ts = cmd.time_steps[env_id]
                    lbl = cl[ts].cpu().numpy()
                    print(f"[PlayDebug] step={timestep} env0 contact_labels (L_foot,R_foot,L_wrist,R_wrist)={lbl}")
            except Exception as e:
                print(f"[PlayDebug] periodic print skipped: {e}")

        if (mujoco_vis is not None and not mujoco_vis.is_running()) or (
            args_cli.video and timestep >= args_cli.video_length
        ):
            break
        timestep += 1

    # close the simulator
    if mujoco_vis is not None:
        mujoco_vis.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
