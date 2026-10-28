#!/usr/bin/env python3
"""Standalone visualizer for a G1 robot model in Isaac Sim.

Spawns a single G1 articulation on a ground plane and holds it in its default
pose so you can inspect the geometry (e.g. the RCCU + PLA backpack + ZED X camera
variant). Uses the same G1_ROBOT_CFGS registry that training uses, so --robot
selects between the available models.

Run from the bm_generalist root (with IsaacLab + whole_body_tracking installed):

    # Visualize the new RCCU + camera robot (default)
    python scripts/standalone_visualize_g1.py

    # Visualize the bare G1
    python scripts/standalone_visualize_g1.py --robot default

    # Slowly spin the camera around the robot
    python scripts/standalone_visualize_g1.py --spin

    # Let physics run (robot is PD-held at its default pose) instead of pure kinematic hold
    python scripts/standalone_visualize_g1.py --physics
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize a G1 robot model in Isaac Sim.")
parser.add_argument(
    "--robot",
    type=str,
    default="rccu_camera",
    help="G1 model to visualize (key in G1_ROBOT_CFGS): 'default' or 'rccu_camera'.",
)
parser.add_argument(
    "--physics",
    action="store_true",
    default=False,
    help="Step physics with the robot PD-held at its default pose (default: kinematic hold, no physics).",
)
parser.add_argument(
    "--spin",
    action="store_true",
    default=False,
    help="Slowly orbit the camera around the robot.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.g1 import G1_ROBOT_CFGS


def _resolve_robot_cfg() -> ArticulationCfg:
    if args_cli.robot not in G1_ROBOT_CFGS:
        raise ValueError(f"Unknown --robot '{args_cli.robot}'. Choices: {sorted(G1_ROBOT_CFGS)}")
    return G1_ROBOT_CFGS[args_cli.robot].replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class VisualizeG1SceneCfg(InteractiveSceneCfg):
    """Scene: ground plane, dome light, and one G1 robot."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = _resolve_robot_cfg()


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # Cache the default standing pose (root + joints) from the articulation cfg.
    default_root = robot.data.default_root_state.clone()
    default_root[:, :3] += scene.env_origins
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()

    # Initialize to the default pose.
    robot.write_root_state_to_sim(default_root)
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    if args_cli.physics:
        robot.set_joint_position_target(default_joint_pos)
    scene.write_data_to_sim()

    look_at = default_root[0, :3].cpu().numpy()
    radius, height = 2.5, 0.8
    print(
        f"[INFO]: Visualizing G1 model '{args_cli.robot}' "
        f"({G1_ROBOT_CFGS[args_cli.robot].spawn.asset_path})",
        flush=True,
    )

    t = 0.0
    while simulation_app.is_running():
        if args_cli.physics:
            robot.set_joint_position_target(default_joint_pos)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
        else:
            # Kinematic hold: keep the robot frozen in its default pose, render only.
            robot.write_root_state_to_sim(default_root)
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            scene.write_data_to_sim()
            sim.render()

        if args_cli.spin:
            angle = 0.3 * t
            eye = look_at + np.array([radius * np.cos(angle), radius * np.sin(angle), height])
            sim.set_camera_view(eye, look_at)
        t += sim_dt


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene = InteractiveScene(VisualizeG1SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    # Static default camera view (overridden per-frame when --spin is set).
    sim.set_camera_view(eye=[2.5, 2.5, 1.3], target=[0.0, 0.0, 0.8])

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
