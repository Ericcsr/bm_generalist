# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

script_path = os.path.realpath(__file__)
current_file_directory = os.path.dirname(script_path)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, required=False, help="The name of the wand registry.",
default="ericcsr-bm-org/wandb-registry-motions/run")
parser.add_argument("--motion_dir", type=str, required=True, default="lafan")
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
    "*_contact_nodes.npy). Merged with --terrain_folder / --object_folder for mixed datasets; contact "
    "labels prefer this folder first.",
)
parser.add_argument(
    "--contact_label_debug_vis",
    action="store_true",
    default=False,
    help="Visualize contact labels at ankle_roll and wrist_yaw links: green=contact, red=no contact.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--physics_phase",
    action="store_true",
    default=False,
    help="When enabled, magic force disappears the first time both wrists make contact with object; object then fully controlled by robot.",
)
parser.add_argument(
    "--object_free_physics",
    action="store_true",
    default=False,
    help="Disable magic force and kinematic driving entirely: object is placed at its reference initial pose "
    "on reset and then evolves purely under physics. Overrides --physics_phase.",
)
parser.add_argument(
    "--robot",
    type=str,
    default="default",
    help="G1 robot model to train with. 'default' = bare G1 (main.urdf); "
    "'rccu_camera' = G1 + RCCU + PLA backpack + ZED X camera. See G1_ROBOT_CFGS in robots/g1.py.",
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

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    print("distributed?:",args_cli.distributed)
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # load the motion file from the wandb registry
    registry_name = args_cli.registry_name
    if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
        registry_name += ":latest"
    import pathlib

    import wandb

    #api = wandb.Api()
    #artifact = api.artifact(registry_name)
    # env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    if hasattr(env_cfg, "setup_motion_terrain_pairs"):
        # Binded multi-terrain / multi-object: pair motions with terrains/objects by base name
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
        if hasattr(env_cfg, "physics_phase_flag"):
            env_cfg.physics_phase_flag = getattr(args_cli, "physics_phase", False)
        if hasattr(env_cfg, "object_free_physics_flag"):
            env_cfg.object_free_physics_flag = getattr(args_cli, "object_free_physics", False)
            if env_cfg.object_free_physics_flag:
                print(
                    "[INFO]: object_free_physics enabled: no magic force / kinematic driving; "
                    "object initialized at reference pose on reset, then purely physical"
                )
        contact_label_debug_vis = getattr(args_cli, "contact_label_debug_vis", False)
        # Contact-based collision is only in Tracking-3pt-Binded-Multi-Terrain-Contact-Collision-G1-v0
        env_cfg.setup_motion_terrain_pairs()
        if hasattr(env_cfg.events, "set_contact_link_collision"):
            env_cfg.events.set_contact_link_collision.params["enable_contact_label_debug_vis"] = contact_label_debug_vis
            if contact_label_debug_vis:
                n = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
                env_cfg.events.set_contact_link_collision.params["max_contact_label_vis_envs"] = n
        if args_cli.num_envs is not None:
            # Allow user to override num_envs (will wrap env_id % num_pairs for terrain/motion)
            env_cfg.scene.num_envs = args_cli.num_envs
            if hasattr(env_cfg.scene, "terrain") and hasattr(env_cfg.scene.terrain, "spawn"):
                env_cfg.scene.terrain.spawn.num_envs = args_cli.num_envs
    else:
        motion_files = [args_cli.motion_dir + "/" + x for x in os.listdir(args_cli.motion_dir) if x.endswith(".npz")]
        if not motion_files:
            motion_files = [args_cli.motion_dir + "/" + x for x in os.listdir(args_cli.motion_dir)]
        env_cfg.commands.motion.motion_files = motion_files

    # optionally swap the G1 robot model (applied last so it overrides any cfg default)
    from whole_body_tracking.robots.g1 import G1_ROBOT_CFGS

    robot_key = getattr(args_cli, "robot", "default")
    if robot_key != "default":
        if robot_key not in G1_ROBOT_CFGS:
            raise ValueError(f"Unknown --robot '{robot_key}'. Choices: {sorted(G1_ROBOT_CFGS)}")
        robot_cfg = getattr(getattr(env_cfg.scene, "robot", None), "prim_path", None)
        prim_path = robot_cfg if robot_cfg is not None else "{ENV_REGEX_NS}/Robot"
        env_cfg.scene.robot = G1_ROBOT_CFGS[robot_key].replace(prim_path=prim_path)
        print(
            f"[INFO]: Using G1 robot model '{robot_key}' "
            f"({G1_ROBOT_CFGS[robot_key].spawn.asset_path})",
            flush=True,
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    print("[INFO]: Creating environment (this may take a while for many envs)...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print("[INFO]: Environment created successfully.", flush=True)

    # Print clear status when contact labels exist and contact-label reward is active
    _env = env.unwrapped
    if isinstance(_env, DirectMARLEnv):
        _env = getattr(_env, "env", _env)
    if hasattr(_env, "command_manager"):
        try:
            cmd = _env.command_manager.get_term("motion")
            has_contact_labels = getattr(cmd.motion, "contact_labels", None) is not None
        except Exception:
            has_contact_labels = False
    else:
        has_contact_labels = False
    _rew = env_cfg.rewards
    has_contact_reward = (
        hasattr(_rew, "contact_label_contact")
        or hasattr(_rew, "contact_label_contact_foot")
        or hasattr(_rew, "contact_label_contact_wrist")
    )
    if has_contact_labels and has_contact_reward:
        if hasattr(_rew, "contact_label_contact"):
            w = getattr(_rew.contact_label_contact, "weight", 0.5)
            print(
                "[CONTACT MODE] Contact labels loaded and contact_label_contact reward ACTIVE "
                "(weight=%.2f). Links: L/R foot, L/R wrist." % w,
                flush=True,
            )
        else:
            wf = getattr(_rew.contact_label_contact_foot, "weight", 0.25)
            ww = getattr(_rew.contact_label_contact_wrist, "weight", 0.25)
            print(
                "[CONTACT MODE] Contact labels loaded; foot+wrist contact rewards ACTIVE "
                "(instant: foot=%.2f wrist=%.2f; see also *_duration_* terms). Links: L/R foot, L/R wrist."
                % (wf, ww),
                flush=True,
            )
    elif has_contact_reward and not has_contact_labels:
        print(
            "[CONTACT MODE] contact-label contact rewards configured but NO contact labels loaded "
            "(returns 0). Ensure object_folder / object_terrain_folder / terrain_folder has *_contact_nodes.npy.",
            flush=True,
        )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
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

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        # Curriculum terms that use env.common_step_counter (e.g. wrist mismatch schedule) otherwise reset to 0
        # on a fresh env; approximate prior control steps as iteration × rollout depth (matches RSL-RL loop).
        sim_env = env.unwrapped
        if hasattr(sim_env, "common_step_counter"):
            steps_per_iter = agent_cfg.num_steps_per_env
            restored = int(runner.current_learning_iteration * steps_per_iter)
            sim_env.common_step_counter = restored
            print(
                f"[INFO]: Synced common_step_counter={restored} for curriculum/time-based MDP "
                f"(iteration={runner.current_learning_iteration}, num_steps_per_env={steps_per_iter}).",
                flush=True,
            )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
