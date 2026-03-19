#!/usr/bin/env python3
"""Standalone minimum example: robot + floating object with magic force.

- Proxy object: test_sub16_largebox_007_thin_plate_proxy.urdf from omomo_object/omomo_box.
- Object is dynamic, controlled by PD magic force to track a target pose.
- Object target is keyboard-controlled: W/S=X, A/D=Y, Q/E=Z (move object toward robot hands).
- Object collides ONLY with left_wrist_yaw_link and right_wrist_yaw_link (not other robot links).
- When both hands contact the object, magic force disappears and object is fully physics-driven.

Run from bm_generalist root (with IsaacLab and whole_body_tracking in PYTHONPATH):

    python scripts/standalone_magic_force_hand_object.py

    python scripts/standalone_magic_force_hand_object.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Magic force object + hand-only collision + physics phase demo."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import carb
from pathlib import Path
import omni
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import omni.usd
from pxr import UsdPhysics
import omni.physx
from pxr import UsdPhysics, PhysxSchema, PhysicsSchemaTools

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

# FIX 3: Ensure these links actually have collision meshes in the G1 USD!
# If they are just dummy kinematic frames, the object will still pass through. 
# You may need to change these to e.g., "left_hand_link" or "left_palm".
PROXY_CONTACT_LINKS = {"left_wrist_yaw_link", "right_wrist_yaw_link"}

_prev_contact_states = {}

def print_exact_collision_paths():
    stage = omni.usd.get_context().get_stage()
    print("\n--- EXACT COLLISION MESH PATHS ---")
    
    found_any = False
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        
        # Look for anything related to your object
        if "ProxyObject" in path:
            # Check if this specific prim is a physical collider
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                print(f"[✓] Found Collider: {path}")
                found_any = True
                
    if not found_any:
        print("[!] No collision meshes found under 'ProxyObject'. (Check the name!)")
    print("----------------------------------\n")


# def _print_contact_events(scene, threshold: float = 0.01):
#     global _prev_contact_states
    
#     # Safely get the sensors
#     left_sensor = scene.sensors.get("hand_object_contact_left")
#     right_sensor = scene.sensors.get("hand_object_contact_right")
    
#     if left_sensor is None or right_sensor is None:
#         return
        
#     left_fm = left_sensor.data.force_matrix_w
#     right_fm = right_sensor.data.force_matrix_w
    
#     for env_id in range(scene.num_envs):
#         if env_id not in _prev_contact_states:
#             _prev_contact_states[env_id] = {"left": False, "right": False}
            
#         # Extract the net force magnitudes
#         # shape is (num_envs, num_sensor_bodies=1, num_filtered_bodies=1, 3)
#         l_force = torch.norm(left_fm[env_id, 0, 0, :]) if left_fm is not None else 0.0
#         r_force = torch.norm(right_fm[env_id, 0, 0, :]) if right_fm is not None else 0.0
        
#         # Boolean contact flags
#         l_contact = bool(l_force > threshold)
#         r_contact = bool(r_force > threshold)
        
#         # Print only on the ONSET of contact to keep the console readable
#         if l_contact and not _prev_contact_states[env_id]["left"]:
#             print(f"[Contact Event | Env {env_id}] ProxyObject <---> left_wrist_yaw_link (Force: {l_force:.2f} N)")
#         elif not l_contact and _prev_contact_states[env_id]["left"]:
#             print(f"[Contact Ended | Env {env_id}] ProxyObject <-X-> left_wrist_yaw_link")
            
#         if r_contact and not _prev_contact_states[env_id]["right"]:
#             print(f"[Contact Event | Env {env_id}] ProxyObject <---> right_wrist_yaw_link (Force: {r_force:.2f} N)")
#         elif not r_contact and _prev_contact_states[env_id]["right"]:
#             print(f"[Contact Ended | Env {env_id}] ProxyObject <-X-> right_wrist_yaw_link")
            
#         # Update tracking state
#         _prev_contact_states[env_id]["left"] = l_contact
#         _prev_contact_states[env_id]["right"] = r_contact

def _print_contact_events(scene, threshold: float = 0.01):
    # Get the new object-centric sensors
    l_sensor = scene.sensors.get("object_contact_left_wrist")
    r_sensor = scene.sensors.get("object_contact_right_wrist")

    for env_id in range(scene.num_envs):
        # --- LEFT WRIST FORCE ON OBJECT ---
        l_force = 0.0
        if l_sensor and l_sensor.data.force_matrix_w is not None:
            # force_matrix_w shape: (num_envs, num_sensor_bodies, num_filtered_bodies, 3)
            # We sum across ALL object meshes (dim 0) and ALL wrist matches (dim 1)
            force_vec = l_sensor.data.force_matrix_w[env_id].sum(dim=(0, 1))
            l_force = torch.norm(force_vec).item()

        # --- RIGHT WRIST FORCE ON OBJECT ---
        r_force = 0.0
        if r_sensor and r_sensor.data.force_matrix_w is not None:
            force_vec = r_sensor.data.force_matrix_w[env_id].sum(dim=(0, 1))
            r_force = torch.norm(force_vec).item()
            
        if l_force > threshold:
            print(f"[CONTACT | Env {env_id}] ProxyObject felt hit from LEFT WRIST (Force: {l_force:.2f} N)")
        if r_force > threshold:
            print(f"[CONTACT | Env {env_id}] ProxyObject felt hit from RIGHT WRIST (Force: {r_force:.2f} N)")

def _apply_proxy_hand_only_collision(scene, robot_prim_name: str = "Robot", proxy_prim_name: str = "ProxyObject"):
    from pxr import UsdPhysics, Usd

    stage = scene.stage
    env_prim_paths = scene.env_prim_paths

    for env_id in range(scene.num_envs):
        env_path = env_prim_paths[env_id]
        robot_path = f"{env_path}/{robot_prim_name}"
        proxy_path = f"{env_path}/{proxy_prim_name}"

        proxy_prim = stage.GetPrimAtPath(proxy_path)
        robot_prim = stage.GetPrimAtPath(robot_path)
        
        if not proxy_prim.IsValid() or not robot_prim.IsValid():
            continue

        # 1. Create two distinct Collision Groups
        robot_group_path = f"{env_path}/RobotGhostGroup"
        proxy_group_path = f"{env_path}/ProxyGhostGroup"

        robot_group = UsdPhysics.CollisionGroup.Define(stage, robot_group_path)
        proxy_group = UsdPhysics.CollisionGroup.Define(stage, proxy_group_path)

        # 2. Tell the groups to filter (ignore) each other
        robot_group.CreateFilteredGroupsRel().AddTarget(proxy_group_path)
        proxy_group.CreateFilteredGroupsRel().AddTarget(robot_group_path)

        # 3. Add the ENTIRE proxy object tree to the proxy group
        proxy_group.GetCollidersCollectionAPI().GetIncludesRel().AddTarget(proxy_path)

        # 4. Add the robot links to the robot group, skipping our whitelisted wrists
        for child_prim in Usd.PrimRange(robot_prim):
            if child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                link_name = child_prim.GetName()
                
                if link_name not in PROXY_CONTACT_LINKS:
                    continue
                
                # Adding a rigid body to the group automatically covers its child meshes
                robot_group.GetCollidersCollectionAPI().GetIncludesRel().AddTarget(child_prim.GetPath())

    print("[INFO] Proxy collision filter applied: Using Collision Groups for deep URDF hierarchies.")

# def _make_magic_force_object_cfg() -> RigidObjectCfg:
#     """Dynamic proxy from omomo_object thin-plate URDF; magic force drives it."""
#     bm_root = Path(__file__).resolve().parent.parent
#     urdf_path = bm_root / "omomo_object" / "omomo_box" / "test_sub16_largebox_007_thin_plate_proxy.urdf"
    
#     if not urdf_path.exists():
#         raise FileNotFoundError(f"Proxy URDF not found: {urdf_path}")

#     return RigidObjectCfg(
#         prim_path="{ENV_REGEX_NS}/ProxyObject",
#         spawn=sim_utils.UrdfFileCfg(
#             asset_path=str(urdf_path),
#             fix_base=False,
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(
#                 rigid_body_enabled=True,
#                 kinematic_enabled=False,
#                 disable_gravity=False,
#             ),
#             # Perfectly tuned mass and inertia for kp=500/200 and kd=50/20
#             mass_props=sim_utils.MassPropertiesCfg(
#                 density=400.0
#             ),
#             collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
#             joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
#                 gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
#             ),
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.95)),
#     )

def _make_magic_force_object_cfg() -> RigidObjectCfg:
    """Dynamic proxy from omomo_object thin-plate URDF; magic force drives it."""
    bm_root = Path(__file__).resolve().parent.parent
    urdf_path = bm_root / "omomo_object" / "omomo_box" / "test_sub16_largebox_007_thin_plate_proxy.urdf"
    
    if not urdf_path.exists():
        raise FileNotFoundError(f"Proxy URDF not found: {urdf_path}")

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ProxyObject",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf_path),
            fix_base=False,
            
            # ---> THIS IS THE MAGIC FIX <---
            activate_contact_sensors=True,
            
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            # Perfectly tuned mass and inertia for kp=500/200 and kd=50/20
            mass_props=sim_utils.MassPropertiesCfg(
                density=400.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.95)),
    )

@configclass
class MagicForceHandObjectSceneCfg(InteractiveSceneCfg):
    """Scene: ground, G1 robot, dynamic object with hand-only collision."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        debug_vis=False,
    )

    robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    proxy_object = _make_magic_force_object_cfg()

    # --- 1. Object Sensors ---
    hand_object_contact_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*left_wrist_yaw_link",
        # Target the specific folder containing mesh_0, mesh_1, mesh_2
        filter_prim_paths_expr=["{ENV_REGEX_NS}/ProxyObject/proxy_base/collisions"], 
        history_length=1,
    )
    
    hand_object_contact_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*right_wrist_yaw_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/ProxyObject/proxy_base/collisions"], 
        history_length=1,
    )

    # --- 2. Ground Sensors ---
    hand_ground_contact_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*left_wrist_yaw_link",
        filter_prim_paths_expr=[".*ground.*"],       # Bulletproof wildcard
        history_length=1,
    )
    hand_ground_contact_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*right_wrist_yaw_link",
        filter_prim_paths_expr=[".*ground.*"],       # Bulletproof wildcard
        history_length=1,
    )

    # --- Object-Centric Sensors ---
    # Sensor sits on the object, listens for the LEFT wrist
    object_contact_left_wrist = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ProxyObject/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*left_wrist_yaw_link"], 
        history_length=1,
    )
    
    # Sensor sits on the object, listens for the RIGHT wrist
    object_contact_right_wrist = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ProxyObject/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*right_wrist_yaw_link"], 
        history_length=1,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# Per-scene physics phase state
_physics_phase_activated: dict[int, torch.Tensor] = {}

# Keyboard-driven object target: [x, y, z] in env-0 local frame (no env_origin yet)
# Initial pose: in front of robot, reachable by arms
_OBJECT_TARGET_POS = [0.5, 0.0, 0.95]
_OBJECT_STEP = 0.03  # meters per key press

# K toggles magic force off (object becomes fully physics-driven)
_MAGIC_FORCE_DISABLED_BY_KEYBOARD = False


def _create_keyboard_object_control():
    """Subscribe to keyboard: W/S=X, A/D=Y, Q/E=Z, K=disable magic force."""
    global _MAGIC_FORCE_DISABLED_BY_KEYBOARD

    _input = carb.input.acquire_input_interface()
    _appwindow = omni.appwindow.get_default_app_window()
    _keyboard = _appwindow.get_keyboard()

    def _on_key(event, *args):
        global _MAGIC_FORCE_DISABLED_BY_KEYBOARD
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        name = event.input.name
        # W/S: X, A/D: Y, Q/E: Z
        if name == "W":
            _OBJECT_TARGET_POS[0] += _OBJECT_STEP
        elif name == "S":
            _OBJECT_TARGET_POS[0] -= _OBJECT_STEP
        elif name == "A":
            _OBJECT_TARGET_POS[1] += _OBJECT_STEP
        elif name == "D":
            _OBJECT_TARGET_POS[1] -= _OBJECT_STEP
        elif name == "Q":
            _OBJECT_TARGET_POS[2] += _OBJECT_STEP
        elif name == "E":
            _OBJECT_TARGET_POS[2] -= _OBJECT_STEP
        elif name == "K":
            _MAGIC_FORCE_DISABLED_BY_KEYBOARD = not _MAGIC_FORCE_DISABLED_BY_KEYBOARD
            print("[INFO] Magic force %s (K)" % ("disabled" if _MAGIC_FORCE_DISABLED_BY_KEYBOARD else "enabled"))
        return True

    sub = _input.subscribe_to_keyboard_events(_keyboard, _on_key)
    return _input, _keyboard, sub


def _desired_pose_from_keyboard(env_origin: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Target pose from keyboard state. Returns (7,) pos+quat."""
    x, y, z = _OBJECT_TARGET_POS
    pos = torch.tensor([x, y, z], dtype=torch.float32, device=device) + env_origin
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    return torch.cat([pos, quat])


def _apply_magic_force(
    scene,
    proxy_obj,
    desired_pose: torch.Tensor,
    kp_pos: float = 50.0,
    kd_pos: float = 10.0,
    kp_ori: float = 2.5,  # Reduced default gain for orientation to prevent instability
    kd_ori: float = 0.5,  # Reduced default damping for orientation
    contact_threshold: float = 1.0,
):
    """Apply PD magic force; disable when both hands contact object (physics phase) or K pressed."""
    global _physics_phase_activated, _MAGIC_FORCE_DISABLED_BY_KEYBOARD

    num_envs = scene.num_envs
    device = scene.device
    scene_id = id(scene)

    if scene_id not in _physics_phase_activated:
        _physics_phase_activated[scene_id] = torch.zeros(num_envs, dtype=torch.bool, device=device)
    activated = _physics_phase_activated[scene_id]

    # K key: disable magic force for all envs (toggle, does not persist to contact state)
    if _MAGIC_FORCE_DISABLED_BY_KEYBOARD:
        activated = torch.ones(num_envs, dtype=torch.bool, device=device)
    else:
        # Check both-wrists contact via force_matrix_w (hand-object only)
        both_wrists_contact = None
        try:
            left_sensor = scene.sensors["hand_object_contact_left"]
            right_sensor = scene.sensors["hand_object_contact_right"]
            left_fm = left_sensor.data.force_matrix_w
            right_fm = right_sensor.data.force_matrix_w
            if left_fm is not None and right_fm is not None:
                left_norm = torch.norm(left_fm[:, 0, :, :].sum(dim=1), dim=-1)
                right_norm = torch.norm(right_fm[:, 0, :, :].sum(dim=1), dim=-1)
                both_wrists_contact = (
                    (left_norm > contact_threshold) & (right_norm > contact_threshold)
                )
        except (KeyError, ValueError, AttributeError):
            pass

        if both_wrists_contact is not None:
            activated = activated | both_wrists_contact
            _physics_phase_activated[scene_id] = activated

    # Current state
    curr_pose = proxy_obj.data.root_link_pose_w
    curr_pos = curr_pose[:, :3]
    curr_quat = curr_pose[:, 3:7]
    curr_vel = proxy_obj.data.root_com_vel_w
    curr_lin_vel = curr_vel[:, :3]
    curr_ang_vel = curr_vel[:, 3:6]

    desired_pos = desired_pose[:, :3].to(device)
    desired_quat = desired_pose[:, 3:7].to(device)
    desired_lin_vel = torch.zeros_like(curr_lin_vel)
    desired_ang_vel = torch.zeros_like(curr_ang_vel)

    # --- Position PD and Gravity Compensation ---
    pos_error = desired_pos - curr_pos
    lin_vel_error = desired_lin_vel - curr_lin_vel
    force = kp_pos * pos_error + kd_pos * lin_vel_error

    mass = getattr(proxy_obj.data, "default_mass", None)
    if mass is not None and mass.numel() > 0:
        mass = mass.to(device)[:num_envs].view(-1, 1)
    else:
        mass = torch.ones(num_envs, 1, device=device)
        
    gravity_vec = torch.zeros(num_envs, 3, device=device)
    gravity_vec[:, 2] = -9.81
    force = force + mass * (-gravity_vec)

    # --- Orientation PD ---
    # 1. Calculate relative rotation (error quaternion)
    from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat
    quat_error = quat_mul(desired_quat, quat_conjugate(curr_quat))
    
    # 2. Enforce shortest path by checking the scalar (w) component.
    # IsaacLab typically uses (w, x, y, z) format, so w is at index 0.
    # If w < 0, negate the entire quaternion to take the shorter rotational path.
    quat_error = torch.where(quat_error[:, 0:1] < 0.0, -quat_error, quat_error)

    # 3. Convert to axis-angle representation for the torque vector
    ori_error_aa = axis_angle_from_quat(quat_error)
    ang_vel_error = desired_ang_vel - curr_ang_vel
    
    # 4. Calculate final torque
    torque = kp_ori * ori_error_aa + kd_ori * ang_vel_error

    # --- Apply Phase Logic ---
    # Zero force/torque when physics phase activated
    force = torch.where(activated.view(-1, 1), 0.0, force)
    torque = torch.where(activated.view(-1, 1), 0.0, torque)

    proxy_obj.set_external_force_and_torque(
        forces=force.unsqueeze(1),
        torques=torque.unsqueeze(1),
        body_ids=[0],
        is_global=True,
    )


def main():
    # FIX 2: Enable Continuous Collision Detection (CCD) in the PhysX solver 
    # to prevent high-velocity PD movements from tunneling through the robot's thin links.
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        physx=sim_utils.PhysxCfg(enable_ccd=True) 
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3, 3, 2], target=[0.5, 0.0, 0.5])

    # Use replicate_physics=False to match Tracking-3pt-Binded-Multi-Object (tests multi-env collision filter)
    scene = InteractiveScene(
        MagicForceHandObjectSceneCfg(
            num_envs=args_cli.num_envs,
            env_spacing=3.0,
            replicate_physics=False,
            lazy_sensor_update=False,
        )
    )

    # FIX 1: Apply the collision filter BEFORE `sim.reset()`. 
    # PhysX needs to know about these filters during its initialization compilation phase.
    _apply_proxy_hand_only_collision(scene)

    # NOW compile the scene and physics handles
    sim.reset()
    robot = scene["robot"]
    proxy_obj = scene["proxy_object"]
    sim_dt = sim.get_physics_dt()
    decimation = 4
    t = 0.0
    device = scene.device
    num_envs = args_cli.num_envs
    env_origins = scene.env_origins

    # Default joint targets; add slow periodic motion so arms can reach the object
    joint_targets = robot.data.default_joint_pos.clone()
    left_elbow_idx = robot.joint_names.index("left_elbow_joint")
    right_elbow_idx = robot.joint_names.index("right_elbow_joint")
    left_shoulder_pitch_idx = robot.joint_names.index("left_shoulder_pitch_joint")
    right_shoulder_pitch_idx = robot.joint_names.index("right_shoulder_pitch_joint")

    # Keyboard control for object target pose (W/S=X, A/D=Y, Q/E=Z)
    _input_iface, _kb, _kb_sub = _create_keyboard_object_control()

    print("[INFO] Magic force demo: object tracks target until both hands contact it.")
    print("[INFO] num_envs=%d, replicate_physics=False (matches Tracking-3pt-Binded-Multi-Object)" % num_envs)
    print("[INFO] Object target: W/S=X, A/D=Y, Q/E=Z (step=%.2fm)" % _OBJECT_STEP)
    print("[INFO] Arms move slowly toward object; when both wrists contact, magic force stops.")

    print_exact_collision_paths()
    
    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step()
            continue

        for _ in range(decimation):
            # Desired pose for object (keyboard-controlled target)
            desired = torch.stack(
                [_desired_pose_from_keyboard(env_origins[i], device) for i in range(num_envs)]
            )

            # Magic force (stops when both hands contact)
            _apply_magic_force(scene, proxy_obj, desired)

            # ---> ADD THIS LINE HERE <---
            # Print any contact events that occurred
            _print_contact_events(scene)

            # Robot: slow periodic arm motion toward object (elbow + shoulder pitch)
            targets = joint_targets.clone()
            arm_phase = 0.3 * math.sin(0.4 * t)  # slow motion
            targets[:, left_elbow_idx] = 0.5 + 0.25 * (1 + arm_phase)
            targets[:, right_elbow_idx] = 0.5 + 0.25 * (1 + arm_phase)
            targets[:, left_shoulder_pitch_idx] = 0.2 + 0.3 * (1 + arm_phase)
            targets[:, right_shoulder_pitch_idx] = 0.2 + 0.3 * (1 + arm_phase)
            robot.set_joint_position_target(targets)

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

        t += sim_dt * decimation


if __name__ == "__main__":
    main()
    simulation_app.close()