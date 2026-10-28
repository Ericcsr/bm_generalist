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

# Per-process USD conversion dirs. IsaacLab's default /tmp/IsaacLab/usd_<seconds>_<rand(10000)> collides when
# several processes (e.g. the 8 ranks of a distributed run) convert ~1000 URDFs in the same second, giving
# "No contact sensors added to the prim ..." / corrupted *.tmp.usd crashes.
import itertools as _itertools
import os as _os
import isaaclab.sim.converters.asset_converter_base as _acb

_usd_dir_counter = _itertools.count()
_acb_init_orig = _acb.AssetConverterBase.__init__


def _acb_init_isolated(self, cfg):
    if getattr(cfg, "usd_dir", None) is None:
        cfg.usd_dir = f"/tmp/IsaacLab/usd_pid{_os.getpid()}_{next(_usd_dir_counter)}"
    _acb_init_orig(self, cfg)


_acb.AssetConverterBase.__init__ = _acb_init_isolated

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
    # ---------------- [claude probe] which envs blow up? (stochastic policy rollout, training settings) ----------------
    import numpy as np, json, time as _time
    import whole_body_tracking.tasks.tracking.mdp.events as _EV
    u = env.unwrapped
    env_w = RslRlVecEnvWrapper(env)
    obs, _ = env_w.get_observations()
    runner = OnPolicyRunner(env_w, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.environ["PROBE_CKPT"])
    runner.eval_mode()
    pol = runner.alg.policy
    norm = runner.obs_normalizer if agent_cfg.empirical_normalization else (lambda x: x)
    act = lambda o: pol.act(norm(o))                    # stochastic, like data collection in training
    cmd = u.command_manager.get_term("motion"); mot = cmd.motion
    names = [os.path.basename(f)[:-4] for f in u.cfg.commands.motion.motion_files]
    cats = np.array(u.cfg.motion_category_per_motion)
    CAT = {0: "terrain", 1: "object", 2: "free"}
    mids = cmd.motion_ids.cpu().numpy()
    s0 = mot.motion_start_idx.cpu().numpy()
    robot = u.scene["robot"]; proxy = u.scene["proxy_object"]
    cs = u.scene.sensors["contact_forces"]
    rm = u.reward_manager; rterms = list(rm._term_names)
    jl = rterms.index("joint_limit") if "joint_limit" in rterms else None
    jnames = robot.joint_names; bnames = cs.body_names
    N = u.num_envs
    ep_t = np.zeros(N, int)
    flagged = np.zeros(N, bool)
    events, maxima = [], {c: [] for c in CAT.values()}
    ep_max_jv = np.zeros(N)
    out = os.environ.get("PROBE_OUT", "/tmp/probe_blowup.json")
    t_end = _time.time() + float(os.environ.get("PROBE_SECONDS", "3600"))
    step = 0
    while _time.time() < t_end:
        with torch.inference_mode():
            obs, rew, dones, _ = env_w.step(act(obs))
        step += 1; ep_t += 1
        jv = robot.data.joint_vel; lv = robot.data.root_lin_vel_w
        rel = robot.data.root_pos_w - u.scene.env_origins
        jv_max = torch.nan_to_num(jv.abs(), nan=1e9).max(1).values
        bad = (~torch.isfinite(jv).all(1)) | (jv_max > 100.0) | (torch.nan_to_num(lv.norm(dim=1), nan=1e9) > 20.0) \
              | (rel[:, 2] > 5.0) | (rel[:, 2] < -2.0) | (rew < -100.0) | (~torch.isfinite(rew))
        ep_max_jv = np.maximum(ep_max_jv, jv_max.cpu().numpy())
        idx = torch.nonzero(bad & ~torch.as_tensor(flagged, device=bad.device)).flatten().cpu().numpy()
        if len(idx):
            ppa = _EV.get_physics_phase_activated(u)
            f = cs.data.net_forces_w.norm(dim=-1)                       # (N, B)
            for e in idx:
                m = int(mids[e]); flagged[e] = True
                je = int(torch.nan_to_num(jv[e].abs(), nan=1e9).argmax()); be = int(torch.nan_to_num(f[e], nan=1e9).argmax())
                srow = rm._step_reward[e].cpu().numpy()
                events.append(dict(
                    step=step, env=int(e), motion=names[m], category=CAT[int(cats[m])], ep_step=int(ep_t[e]),
                    phase=int(cmd.time_steps[e].item() - s0[m]), motion_len=int(mot.motion_end_idx[m].item() - s0[m]),
                    reward=float(rew[e]), joint_limit_rew=float(srow[jl]) if jl is not None else None,
                    worst_terms=sorted(((rterms[i], float(srow[i])) for i in range(len(rterms))), key=lambda kv: kv[1])[:3],
                    max_jv=float(jv_max[e]), max_jv_joint=jnames[je], root_speed=float(lv[e].norm()),
                    root_rel=[float(x) for x in rel[e]], max_contact_N=float(f[e, be]), max_contact_body=bnames[be],
                    physics_phase=bool(ppa[e]) if ppa is not None else None,
                    proxy_speed=float(proxy.data.root_lin_vel_w[e].norm()) if proxy.num_instances > e else None))
                print(f"[probe] BLOWUP step {step} env {e} {CAT[int(cats[m])]} {names[m]} ep_step {ep_t[e]} "
                      f"jv {float(jv_max[e]):.0f} ({jnames[je]}) contact {float(f[e, be]):.0f}N on {bnames[be]} rew {float(rew[e]):.1f}", flush=True)
        d = dones.cpu().numpy().astype(bool)
        for e in np.flatnonzero(d):
            maxima[CAT[int(cats[int(mids[e])])]].append(float(ep_max_jv[e]))
            ep_max_jv[e] = 0; ep_t[e] = 0; flagged[e] = False
        if step % 500 == 0:
            json.dump(dict(steps=step, num_envs=N, events=events,
                           ep_max_jv_pct={c: (np.percentile(v, [50, 99, 99.9]).round(1).tolist() if v else None) for c, v in maxima.items()},
                           ep_counts={c: len(v) for c, v in maxima.items()}), open(out, "w"))
            print(f"[probe] step {step}: {len(events)} blow-ups so far", flush=True)
    json.dump(dict(steps=step, num_envs=N, events=events,
                   ep_max_jv_pct={c: (np.percentile(v, [50, 99, 99.9]).round(1).tolist() if v else None) for c, v in maxima.items()},
                   ep_counts={c: len(v) for c, v in maxima.items()}), open(out, "w"))
    print(f"[probe] done: {step} steps x {N} envs, {len(events)} blow-ups -> {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
