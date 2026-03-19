#!/usr/bin/env python3
"""Standalone script to test whether specific link collision can be disabled to terrain.

Spawns a G1 robot on a ground plane, runs simulation, then disables collision on
left_ankle_roll_link and right_ankle_roll_link. If collision disable works, the
robot's feet should sink through the terrain (base height drops).

Run from bm_generalist root:
  ./isaaclab.sh -p scripts/test_link_collision_disable.py

Or with headless:
  ./isaaclab.sh -p scripts/test_link_collision_disable.py --headless
"""

"""Launch Isaac Sim first."""
import argparse
import sys
from pathlib import Path

# Add bm_generalist source to path before other imports
bm_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(bm_root / "source" / "whole_body_tracking"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test link collision disable to terrain.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaacsim.core.utils.stage import get_current_stage

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground + single G1 robot with make_instanceable=False."""

    replicate_physics: bool = False
    num_envs: int = 1
    env_spacing: float = 2.0

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=G1_CYLINDER_CFG.spawn.replace(make_instanceable=False),
    )


def _any_descendant_instanced(stage, prim_path: str) -> bool:
    """True if any descendant of prim_path is instanced."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return True
    to_visit = [prim]
    while to_visit:
        p = to_visit.pop()
        if p.IsInstance():
            return True
        to_visit.extend(p.GetChildren())
    return False


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = SimulationContext(sim_cfg)
    scene_cfg = TestSceneCfg()
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    stage = get_current_stage()
    # For num_envs=1, robot is at /World/envs/env_0/Robot
    env0 = scene.env_prim_paths[0]
    robot_prim_path = f"{env0}/Robot"
    link_names = ["left_ankle_roll_link", "right_ankle_roll_link"]

    # Check instancing; if instanced, make_uninstanceable so we can modify collision
    sample_link_path = f"{robot_prim_path}/{link_names[0]}"
    instanced = _any_descendant_instanced(stage, sample_link_path)
    print(f"\n[Test] Collision geometry instanced: {instanced}")
    if instanced:
        print("[Test] Calling make_uninstanceable on robot (required for modify_collision_properties)...")
        sim_utils.make_uninstanceable(robot_prim_path, stage=stage)
        sim.reset()  # Re-sync PhysX after stage changes
        instanced_after = _any_descendant_instanced(stage, sample_link_path)
        print(f"[Test] After make_uninstanceable: instanced={instanced_after}")

    robot = scene["robot"]
    pelvis_id = robot.body_names.index("pelvis")

    # Run a few steps to settle
    for _ in range(50):
        sim.step()
    sim.step()

    # Record base height before disabling collision
    base_height_before = robot.data.body_pos_w[0, pelvis_id, 2].item()
    print(f"\n[Test] Base (pelvis) height before disable: {base_height_before:.4f} m")

    # Disable collision on ankle links using modify_collision_properties
    print("\n[Test] Disabling collision on left_ankle_roll_link, right_ankle_roll_link...")
    for link_name in link_names:
        link_path = f"{robot_prim_path}/{link_name}"
        cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        sim_utils.modify_collision_properties(link_path, cfg, stage=stage)

    # Run simulation - if collision is disabled, feet sink and base drops
    for step in range(200):
        sim.step()
        if step % 50 == 0:
            h = robot.data.body_pos_w[0, pelvis_id, 2].item()
            print(f"  Step {step}: base height = {h:.4f} m")

    base_height_after = robot.data.body_pos_w[0, pelvis_id, 2].item()
    drop = base_height_before - base_height_after
    print(f"\n[Test] Base height after 200 steps: {base_height_after:.4f} m")
    print(f"[Test] Height drop: {drop:.4f} m")

    if drop > 0.05:
        print("\n[Test] SUCCESS: Collision was disabled. Robot sank into terrain (feet have no collision).")
    else:
        print("\n[Test] FAIL: Collision was NOT disabled. Robot did not sink (collision still active).")
        print("        Possible causes: instanced geometry, wrong prim path, or PhysX not picking up changes.")

    print("\n[Test] Done. Close the window or Ctrl+C to exit.")
    while sim.is_playing():
        sim.step()
        if simulation_app.is_exiting():
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
