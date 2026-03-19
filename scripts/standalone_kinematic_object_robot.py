#!/usr/bin/env python3
"""Standalone minimum example: kinematic object interacting with a robot.

Each env has a different object (small/medium/large box) and a different trajectory.
The robot holds a default pose. Collision between the robot and the kinematic object
is enabled so you can see interaction.

Run from bm_generalist root (with IsaacLab and whole_body_tracking in PYTHONPATH):

    python scripts/standalone_kinematic_object_robot.py

    python scripts/standalone_kinematic_object_robot.py --num_envs 6  # multiple envs
    python scripts/standalone_kinematic_object_robot.py --headless   # no GUI (AppLauncher arg)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Kinematic object + robot interaction demo.")
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# G1 robot from whole_body_tracking (requires unitree_description in assets)
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

# Object variants: small, medium, large (cycled per env)
NUM_OBJECT_TYPES = 3
FAR_POS = (100.0, 0.0, 0.0)  # hide inactive objects


def _make_kinematic_box_cfg(name: str, size: float, color: tuple[float, float, float]) -> RigidObjectCfg:
    """Create a kinematic box config."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(size, size, size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.9)),
    )


@configclass
class KinematicObjectRobotSceneCfg(InteractiveSceneCfg):
    """Scene: ground, G1 robot, 3 kinematic box variants (small/medium/large)."""

    # Ground
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        debug_vis=False,
    )

    # G1 humanoid robot
    robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Three object variants: env i uses object type (i % 3)
    box_small = _make_kinematic_box_cfg("BoxSmall", 0.10, (0.9, 0.3, 0.2))   # red
    box_medium = _make_kinematic_box_cfg("BoxMedium", 0.15, (0.2, 0.6, 0.9))  # blue
    box_large = _make_kinematic_box_cfg("BoxLarge", 0.20, (0.2, 0.8, 0.4))    # green

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


def _trajectory_pose_for_env(
    t: float, env_id: int, env_origin: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Per-env trajectory: different phase, amplitude, and pattern. Returns (7,) pose."""
    phase = env_id * 0.7  # offset phase per env
    freq_x = 1.5 + env_id * 0.3
    freq_y = 2.0 + env_id * 0.2
    amp_x = 0.08 + env_id * 0.02
    amp_y = 0.15 + env_id * 0.03
    x = 0.5 + amp_x * math.sin(freq_x * t + phase)
    y = amp_y * math.sin(freq_y * t + phase * 1.3)
    z = 0.9 + 0.04 * math.sin(t + env_id * 0.5)
    pos = torch.tensor([x, y, z], dtype=torch.float32, device=device) + env_origin
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    return torch.cat([pos, quat])


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view(eye=[3, 3, 2], target=[0.0, 0.0, 0.5])

    scene = InteractiveScene(
        KinematicObjectRobotSceneCfg(
            num_envs=args_cli.num_envs,
            env_spacing=3.0,
            replicate_physics=True,
            lazy_sensor_update=False,
        )
    )

    sim.reset()
    print("[INFO] Scene ready. Each env: different object (small/med/large) + different trajectory.")

    robot = scene["robot"]
    boxes = [scene["box_small"], scene["box_medium"], scene["box_large"]]
    sim_dt = sim.get_physics_dt()
    decimation = 4
    t = 0.0
    device = scene.device
    num_envs = args_cli.num_envs
    env_origins = scene.env_origins

    # Default joint targets for robot (standing pose)
    joint_targets = robot.data.default_joint_pos.clone()

    far_pose = torch.tensor(
        [*FAR_POS, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device
    ).unsqueeze(0).expand(num_envs, 7).clone()
    far_pose[:, :3] += env_origins

    zero_vel = torch.zeros(num_envs, 6, device=device)

    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step()
            continue

        for _ in range(decimation):
            # Per object type: only envs with that type get trajectory; others get far pose
            for obj_type in range(NUM_OBJECT_TYPES):
                poses = far_pose.clone()
                for env_id in range(num_envs):
                    if env_id % NUM_OBJECT_TYPES == obj_type:
                        poses[env_id] = _trajectory_pose_for_env(
                            t, env_id, env_origins[env_id], device
                        )
                boxes[obj_type].write_root_link_pose_to_sim(poses)
                boxes[obj_type].write_root_com_velocity_to_sim(zero_vel)

            # Robot holds default pose
            robot.set_joint_position_target(joint_targets)

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

        t += sim_dt * decimation


if __name__ == "__main__":
    main()
    simulation_app.close()
