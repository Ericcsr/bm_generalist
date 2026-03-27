from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from whole_body_tracking.tasks.tracking.mdp.commands import BindedMultiMotionCommand, MotionCommand, MultiMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def rand_float(size, min_val, max_val, device):
    return (max_val - min_val) * torch.rand(size).to(device) + min_val

def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


# Contact link names for G1: foot contact -> ankle_roll, hand contact -> wrist_yaw
CONTACT_LINK_MAP = [
    ("left_ankle_roll_link", 0),   # left_foot
    ("right_ankle_roll_link", 1),  # right_foot
    ("left_wrist_yaw_link", 2),    # left_wrist
    ("right_wrist_yaw_link", 3),   # right_wrist
]

_contact_collision_instanced_warned = False
_contact_collision_applied_logged = False
_contact_label_vis_enabled_logged = False
_contact_label_vis_no_labels_warned = False
_contact_label_vis_event_called_logged = False
_debug_draw_interface = None
_contact_label_markers: VisualizationMarkers | None = None

CONTACT_LABEL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/World/Visuals/ContactLabels",
    markers={
        "on": sim_utils.SphereCfg(
            radius=0.08,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "off": sim_utils.SphereCfg(
            radius=0.08,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)


def _get_debug_draw_interface():
    """Lazy-init Isaac Sim debug draw interface for contact label visualization."""
    global _debug_draw_interface
    if _debug_draw_interface is None:
        try:
            import isaacsim.util.debug_draw._debug_draw as _dd
            _debug_draw_interface = _dd.acquire_debug_draw_interface()
        except Exception:
            _debug_draw_interface = False  # Mark as failed
    return _debug_draw_interface if _debug_draw_interface is not False else None


_contact_label_draw_warned = False


def _draw_contact_labels_debug(positions: torch.Tensor, marker_indices: torch.Tensor, device: torch.device):
    """Draw contact labels: green=contact, red=no contact. Uses debug draw or VisualizationMarkers fallback."""
    global _contact_label_draw_warned, _contact_label_markers

    def _use_debug_draw() -> bool:
        global _contact_label_draw_warned
        draw = _get_debug_draw_interface()
        if draw is None:
            return False
        pos_np = positions.detach().cpu().numpy()
        idx_np = marker_indices.detach().cpu().numpy()
        point_list = [tuple(pos_np[i].tolist()) for i in range(len(pos_np))]
        colors = [
            (0.0, 1.0, 0.0, 1.0) if idx_np[i] == 0 else (1.0, 0.0, 0.0, 1.0)
            for i in range(len(idx_np))
        ]
        sizes = [10] * len(point_list)
        try:
            if hasattr(draw, "clear_points"):
                draw.clear_points()  # type: ignore[union-attr]
            if hasattr(draw, "draw_points"):
                draw.draw_points(point_list, colors, sizes)  # type: ignore[union-attr]
            return True
        except Exception as e:
            if not _contact_label_draw_warned:
                _contact_label_draw_warned = True
                print(f"[ContactLabels] Debug draw failed: {e}")
            return False

    def _use_markers() -> bool:
        global _contact_label_markers, _contact_label_draw_warned
        if _contact_label_markers is None:
            try:
                markers = VisualizationMarkers(CONTACT_LABEL_MARKER_CFG)
                markers.set_visibility(True)
                _contact_label_markers = markers
            except Exception as e:
                if not _contact_label_draw_warned:
                    _contact_label_draw_warned = True
                    print(f"[ContactLabels] VisualizationMarkers failed: {e}")
                return False
        assert _contact_label_markers is not None
        if not _contact_label_markers.is_visible():
            _contact_label_markers.set_visibility(True)
        pos = positions.to(device)
        idx = marker_indices.to(device)
        quat = torch.tensor([[1, 0, 0, 0]], device=device, dtype=pos.dtype).expand(pos.shape[0], 4)
        try:
            _contact_label_markers.visualize(translations=pos, orientations=quat, marker_indices=idx)
            return True
        except Exception:
            return False

    # Prefer VisualizationMarkers (world-space spheres, scale with distance) over
    # debug draw (screen-space pixels, fixed size regardless of distance).
    if _use_markers():
        return
    if _use_debug_draw():
        return
    if not _contact_label_draw_warned:
        _contact_label_draw_warned = True
        print("[ContactLabels] Both debug draw and VisualizationMarkers failed. Vis disabled.")


def _collision_geometry_is_instanced(stage, link_path: str) -> bool:
    """True if collision geometry under link is instanced (modify_collision_properties would no-op).

    Checks link and all descendants: if any prim with CollisionAPI in the path is instanced,
    apply_nested will skip it. Also checks common URDF converter child names (collisions, collision).
    """
    prim = stage.GetPrimAtPath(link_path)
    if not prim.IsValid():
        return True
    to_visit = [prim]
    while to_visit:
        p = to_visit.pop()
        if p.IsInstance():
            return True
        to_visit.extend(p.GetChildren())
    return False


_contact_make_uninstanceable_done = False


def make_uninstanceable_contact_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "motion",
):
    """Run once on reset: if robot collision geometry is instanced, make it uninstanceable and re-sync PhysX.

    Required for contact-based collision to work when URDF converter creates instanced geometry
    despite make_instanceable=False. Call env.sim.reset() after make_uninstanceable to re-sync PhysX.
    """
    global _contact_make_uninstanceable_done
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, (MultiMotionCommand, BindedMultiMotionCommand)):
        return
    if getattr(command.motion, "contact_labels", None) is None:
        return
    scene = env.scene
    env_prim_paths = scene.env_prim_paths
    robot_prim_name = "Robot"
    stage = scene.stage
    sample_path = f"{env_prim_paths[0]}/{robot_prim_name}/{CONTACT_LINK_MAP[0][0]}"
    if not _collision_geometry_is_instanced(stage, sample_path):
        return
    if _contact_make_uninstanceable_done:
        return
    _contact_make_uninstanceable_done = True
    print("[ContactLabels] Collision geometry instanced; calling make_uninstanceable on robots, then sim.reset()")
    for env_id in range(env.scene.num_envs):
        robot_path = f"{env_prim_paths[env_id]}/{robot_prim_name}"
        sim_utils.make_uninstanceable(robot_path, stage=stage)
    env.sim.reset()


def set_contact_link_collision(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "motion",
    enable_contact_label_debug_vis: bool = False,
    max_contact_label_vis_envs: int = 4,
    contact_safe_frames: int = 10,
):
    """Enable/disable collision on contact links based on per-frame contact labels.

    For each env: get current time_step and motion_id, index into contact_labels,
    and set left/right ankle_roll (feet) and left/right wrist_yaw (hands) collision
    accordingly. Only runs when command has contact_labels loaded.

    Note: G1 collision geometry is instanced (link/collisions) when using replicate_physics;
    contact-based collision has no effect in that case.

    When enable_contact_label_debug_vis is True, draws green (contact) / red (no contact)
    spheres at the four contact links for the first max_contact_label_vis_envs envs.
    Requires Tracking-3pt-Binded-Multi-Terrain-Contact-Collision-G1-v0 (or similar contact-collision task) to load contact labels.

    Note: Rapid red/green blinking usually indicates frame-to-frame flicker in the
    contact label data (from velocity/acceleration thresholds). Use merged contact
    masks (contact_terrain_reconstruction with merge_contact_nodes_to_mask) to smooth.

    contact_safe_frames: First N steps of each motion keep all contact enabled to avoid
        robot sinking during initialization (default 10).

    Time scale for contact nodes: contact_labels are indexed by motion frame (time_steps).
    The event uses command.time_steps at fire time, so the correct frame's labels are read
    regardless of update interval. interval_range_s is in simulation seconds; no adjustment
    needed for sim time scale (e.g. 2x real-time).
    """
    global _contact_collision_instanced_warned, _contact_collision_applied_logged, _contact_label_vis_enabled_logged, _contact_label_vis_no_labels_warned, _contact_label_vis_event_called_logged

    if enable_contact_label_debug_vis and not _contact_label_vis_event_called_logged:
        _contact_label_vis_event_called_logged = True
        print("[ContactLabels] Event called with enable_contact_label_debug_vis=True")

    command = env.command_manager.get_term(command_name)
    if not isinstance(command, (MultiMotionCommand, BindedMultiMotionCommand)):
        return
    cl = command.get_contact_labels_masked()
    if cl is None:
        if enable_contact_label_debug_vis and not _contact_label_vis_no_labels_warned:
            _contact_label_vis_no_labels_warned = True
            print(
                "[ContactLabels] WARNING: Debug vis requested but contact labels not loaded. "
                "Ensure terrain_folder has *_contact_nodes.npy files (from contact_terrain_reconstruction)."
            )
        return

    time_steps = command.time_steps
    motion_ids = getattr(command, "motion_ids", None)
    if motion_ids is None:
        motion_ids = torch.arange(env.scene.num_envs, device=env.device) % command.motion.num_motions

    # Debug visualization: green = contact on, red = contact off
    if enable_contact_label_debug_vis:
        robot = command.robot
        if robot.is_initialized:
            body_names = command.cfg.body_names
            try:
                contact_body_indices = [body_names.index(ln) for ln, _ in CONTACT_LINK_MAP]
            except ValueError:
                pass  # Some contact links not in body_names
            else:
                N = min(env.scene.num_envs, max_contact_label_vis_envs)
                robot_body_ids = command.body_indexes[contact_body_indices]
                body_pos = robot.data.body_pos_w[:N, robot_body_ids, :]
                positions = body_pos.reshape(N * 4, 3)
                marker_indices = []
                for env_id in range(N):
                    for _, col_idx in CONTACT_LINK_MAP:
                        marker_indices.append(0 if cl[env_id, col_idx].item() else 1)
                marker_indices_t = torch.tensor(marker_indices, dtype=torch.long, device=env.device)

                # Use Isaac Sim debug draw (green=contact, red=no contact)
                _draw_contact_labels_debug(positions, marker_indices_t, env.device)
                if not _contact_label_vis_enabled_logged:
                    _contact_label_vis_enabled_logged = True
                    print("[ContactLabels] Debug vis active: green=contact, red=no contact at ankle_roll/wrist_yaw")

    scene = env.scene
    env_prim_paths = scene.env_prim_paths
    robot_prim_name = "Robot"
    stage = scene.stage

    # Skip if collision geometry is still instanced (make_uninstanceable_contact_robot should have fixed it on reset).
    sample_path = f"{env_prim_paths[0]}/{robot_prim_name}/{CONTACT_LINK_MAP[0][0]}"
    if _collision_geometry_is_instanced(stage, sample_path):
        if not _contact_collision_instanced_warned:
            _contact_collision_instanced_warned = True
            import logging
            logging.getLogger(__name__).info(
                "[ContactLabels] Collision geometry still instanced after make_uninstanceable; "
                "contact-based collision has no effect. Check make_uninstanceable_contact_robot event."
            )
        return

    if not _contact_collision_applied_logged:
        _contact_collision_applied_logged = True
        print(
            "[ContactLabels] Applying contact-based collision (replicate_physics=False, non-instanced geometry). "
            f"Sample path: {sample_path}"
        )

    for link_name, col_idx in CONTACT_LINK_MAP:
        for env_id in range(env.scene.num_envs):
            # First contact_safe_frames of each motion: keep collision enabled to avoid sink during init
            if time_steps[env_id].item() < contact_safe_frames:
                enabled = True
            else:
                enabled = cl[env_id, col_idx].item()
            base_path = f"{env_prim_paths[env_id]}/{robot_prim_name}/{link_name}"
            cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=enabled)
            sim_utils.modify_collision_properties(base_path, cfg, stage=stage)


# Links that are allowed to collide with the proxy object (hands only)
PROXY_CONTACT_LINKS = {"left_wrist_yaw_link", "right_wrist_yaw_link"}

# Hand links for compliant contact (softer contact stiffness)
HAND_LINKS = ["left_wrist_yaw_link", "right_wrist_yaw_link"]

HAND_SOFT_MATERIAL_PATH = "/World/HandSoftMaterial"

_proxy_collision_filter_applied_logged = False
_hand_compliant_contact_applied_logged = False


def apply_hand_compliant_contact(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    robot_prim_name: str = "Robot",
    compliant_contact_stiffness: float = 5e3,
    compliant_contact_damping: float = 50.0,
):
    """Apply compliant (softer) contact material to hand links only.

    Creates a physics material with compliant_contact_stiffness and compliant_contact_damping,
    then binds it to left_wrist_yaw_link and right_wrist_yaw_link. Lower stiffness = softer contact.
    """
    import isaaclab.sim as sim_utils

    scene = env.scene
    try:
        scene["robot"]
    except KeyError:
        return

    stage = scene.stage
    env_prim_paths = scene.env_prim_paths
    robot = scene["robot"]

    # Create soft hand material once (idempotent - prim may already exist)
    if not stage.GetPrimAtPath(HAND_SOFT_MATERIAL_PATH).IsValid():
        hand_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=2.5,
            dynamic_friction=2.5,
            restitution=0.0,
            compliant_contact_stiffness=compliant_contact_stiffness,
            compliant_contact_damping=compliant_contact_damping,
        )
        sim_utils.spawn_rigid_body_material(HAND_SOFT_MATERIAL_PATH, hand_material_cfg)

    for env_id in range(scene.num_envs):
        env_path = env_prim_paths[env_id]
        robot_path = f"{env_path}/{robot_prim_name}"

        for link_name in HAND_LINKS:
            if link_name not in robot.body_names:
                continue
            link_path = f"{robot_path}/{link_name}"
            link_prim = stage.GetPrimAtPath(link_path)
            if not link_prim.IsValid():
                continue
            sim_utils.bind_physics_material(
                link_path,
                HAND_SOFT_MATERIAL_PATH,
                stage=stage,
                stronger_than_descendants=True,
            )

    global _hand_compliant_contact_applied_logged
    if not _hand_compliant_contact_applied_logged:
        _hand_compliant_contact_applied_logged = True
        print(
            f"[INFO]: Hand compliant contact applied: stiffness={compliant_contact_stiffness}, "
            f"damping={compliant_contact_damping} (softer than default rigid)"
        )


def set_proxy_robot_collision_filter(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    robot_prim_name: str = "Robot",
    proxy_prim_name: str = "ProxyObject",
):
    """Restrict proxy object collision to left_wrist_yaw_link and right_wrist_yaw_link only.

    Uses UsdPhysics.CollisionGroup with FilteredGroupsRel (matching standalone_magic_force_hand_object.py).
    Creates two groups: RobotGhostGroup (all robot links except wrists) and ProxyGhostGroup (proxy object).
    Both groups filter each other, so proxy collides only with wrists. This approach works correctly
    for deep URDF hierarchies and must run before sim.reset() (use prestartup mode).

    In this config, each env loads a proxy object (scene key "proxy_object", prim name "ProxyObject").
    """
    from pxr import UsdPhysics, Usd

    scene = env.scene
    try:
        scene["proxy_object"]
    except KeyError:
        return

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

        # 1. Create two distinct Collision Groups (matching standalone)
        robot_group_path = f"{env_path}/RobotGhostGroup"
        proxy_group_path = f"{env_path}/ProxyGhostGroup"

        robot_group = UsdPhysics.CollisionGroup.Define(stage, robot_group_path)
        proxy_group = UsdPhysics.CollisionGroup.Define(stage, proxy_group_path)

        # 2. Tell the groups to filter (ignore) each other
        robot_group.CreateFilteredGroupsRel().AddTarget(proxy_group_path)
        proxy_group.CreateFilteredGroupsRel().AddTarget(robot_group_path)

        # 3. Add the ENTIRE proxy object tree to the proxy group
        proxy_group.GetCollidersCollectionAPI().GetIncludesRel().AddTarget(proxy_path)

        # 4. Add robot links to the robot group, skipping whitelisted wrists
        for child_prim in Usd.PrimRange(robot_prim):
            if child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                link_name = child_prim.GetName()
                if link_name not in PROXY_CONTACT_LINKS:
                    continue
                robot_group.GetCollidersCollectionAPI().GetIncludesRel().AddTarget(child_prim.GetPath())

    global _proxy_collision_filter_applied_logged
    if not _proxy_collision_filter_applied_logged:
        _proxy_collision_filter_applied_logged = True
        print(
            "[INFO]: Proxy collision filter applied (CollisionGroup): proxy collides only with "
            "left_wrist_yaw_link and right_wrist_yaw_link"
        )


def set_slot_based_env_origins(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    num_slots: int | None = None,
):
    """Override env_origins so envs sharing the same terrain/motion share position.

    Env i gets position of slot (i % num_slots). Uses num_slots-based grid (compact near origin)
    instead of num_envs grid, so 10 terrains stay near origin even when num_envs=4096.
    """
    import numpy as np

    scene = env.scene
    if num_slots is None:
        terrain = getattr(scene.cfg, "terrain", None)
        if terrain is not None:
            urdf_paths = getattr(getattr(terrain, "spawn", None), "urdf_paths", None)
            num_slots = len(urdf_paths) if urdf_paths else 0
        else:
            # Multi-object: no terrain; use motion_files from command manager
            cmd_mgr = getattr(env, "command_manager", None)
            if cmd_mgr is not None:
                try:
                    motion_cmd = cmd_mgr.get_term("motion")
                    motion_files = getattr(motion_cmd.cfg, "motion_files", None)
                    num_slots = len(motion_files) if motion_files else 0
                except Exception:
                    num_slots = 0
            else:
                num_slots = 0
    if num_slots <= 0:
        return
    origins = scene._default_env_origins
    if origins is None:
        return

    # Compute slot positions from num_slots grid (compact near origin), not from num_envs grid
    # Use same env_spacing as terrain spawner (terrain.spawn.env_spacing or scene.env_spacing)
    terrain = getattr(scene.cfg, "terrain", None)
    spawn = getattr(terrain, "spawn", None) if terrain else None
    env_spacing = getattr(spawn, "env_spacing", None) if spawn else None
    if env_spacing is None:
        env_spacing = getattr(scene.cfg, "env_spacing", 0.0)
    num_rows = int(np.ceil(num_slots / int(np.sqrt(num_slots))))
    num_cols = int(np.ceil(num_slots / num_rows))
    # Allocate slot_origins with num_slots rows (origins may have fewer when num_envs < num_slots)
    if origins.shape[0] >= num_slots:
        slot_origins = origins[:num_slots].clone()
    else:
        slot_origins = torch.zeros(num_slots, 3, device=origins.device, dtype=origins.dtype)
        slot_origins[: origins.shape[0]] = origins
    for slot_idx in range(num_slots):
        row = slot_idx // num_cols
        col = slot_idx % num_cols
        x = -(row - (num_rows - 1) / 2) * env_spacing
        y = (col - (num_cols - 1) / 2) * env_spacing
        slot_origins[slot_idx, 0] = x
        slot_origins[slot_idx, 1] = y
        slot_origins[slot_idx, 2] = 0.0

    # Map env i -> slot: use motion_to_slot when provided (by-exp layout), else env_id % num_slots
    motion_to_slot = getattr(spawn, "motion_to_slot", None) if spawn else None
    if motion_to_slot is not None and len(motion_to_slot) > 0:
        motion_to_slot_t = torch.tensor(motion_to_slot, device=origins.device, dtype=torch.long)
        num_motions = len(motion_to_slot_t)
        motion_ids = torch.arange(scene.num_envs, device=origins.device, dtype=torch.long) % num_motions
        indices = motion_to_slot_t[motion_ids]
    else:
        indices = torch.arange(scene.num_envs, device=origins.device, dtype=torch.long) % num_slots
    scene._default_env_origins = slot_origins[indices].clone()
    print(f"[INFO]: Env origins: {num_slots} slots near origin (spacing={env_spacing}m), envs share by slot (i % {num_slots})", flush=True)


def set_environment_collision_groups(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    terrain_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
):
    """Set collision groups for terrain and robot: terrain collides with all robots;
    robots sharing the same slot share a group (no inter-env collision within slot).

    Terrain has num_slots bodies (shared by envs); set terrain to collision_group=-1.
    When motion_to_slot is set (by-exp): env i -> slot + 1 (envs in same slot share group).
    Otherwise: env i -> i + 1 (each env unique; use when num_envs <= num_slots).
    """
    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    # Get terrain and robot assets
    terrain = env.scene[terrain_cfg.name]
    robot = env.scene[robot_cfg.name]

    # num_slots = terrain bodies (only num_slots terrain prims exist)
    spawn = getattr(getattr(env.scene.cfg, "terrain", None), "spawn", None)
    urdf_paths = getattr(spawn, "urdf_paths", None)
    num_slots = len(urdf_paths) if urdf_paths else getattr(terrain, "num_bodies", 0)
    env_ids_np = env_ids.cpu().numpy()

    # Robot collision groups: envs sharing same slot share group (no collision within slot)
    motion_to_slot = getattr(spawn, "motion_to_slot", None) if spawn else None
    if motion_to_slot is not None and len(motion_to_slot) > 0:
        motion_to_slot_t = torch.tensor(motion_to_slot, device=env.device, dtype=torch.long)
        num_motions = len(motion_to_slot_t)
        motion_ids = env_ids % num_motions
        slot_ids = motion_to_slot_t[motion_ids]
        robot_collision_groups = (slot_ids + 1).cpu().numpy().astype("int32")
    else:
        robot_collision_groups = (env_ids + 1).cpu().numpy().astype("int32")
    terrain_collide_all = -1  # Collide with all robots

    try:
        if not hasattr(robot, "root_physx_view"):
            raise AttributeError("robot root_physx_view not available")
        robot_view = robot.root_physx_view
        terrain_view = getattr(terrain, "root_physx_view", None)

        try:
            # Terrain: set each slot to collide with all (only num_slots bodies)
            # Terrain may be XFormPrim without root_physx_view; skip if so (default collision often works)
            if terrain_view is not None and hasattr(terrain_view, "get_body_handles"):
                for slot_idx in range(num_slots):
                    slot_tensor = torch.tensor([slot_idx], device=env.device, dtype=torch.long)
                    terrain_handles = terrain_view.get_body_handles(slot_tensor)
                    if hasattr(terrain_view, "set_collision_group"):
                        for h in terrain_handles:
                            terrain_view.set_collision_group(h, terrain_collide_all)
                    elif hasattr(terrain_view, "set_collision_groups"):
                        terrain_view.set_collision_groups(terrain_handles, terrain_collide_all)

            # Robot: set unique group per env
            for env_id, collision_group in zip(env_ids_np, robot_collision_groups):
                env_id_tensor = torch.tensor([env_id], device=env.device, dtype=torch.long)
                if hasattr(robot_view, "get_body_handles"):
                    robot_handles = robot_view.get_body_handles(env_id_tensor)
                    if hasattr(robot_view, "set_collision_group"):
                        for h in robot_handles:
                            robot_view.set_collision_group(h, collision_group)
                    elif hasattr(robot_view, "set_collision_groups"):
                        robot_view.set_collision_groups(robot_handles, collision_group)

            print(
                f"[INFO]: Collision groups: {num_slots} terrains collide with all; "
                f"robots share group by slot (no inter-env collision within slot).",
                flush=True,
            )
        except (AttributeError, NotImplementedError, TypeError) as api_error:
            raise AttributeError(f"Collision group API not available: {api_error}")
    
    except (AttributeError, NotImplementedError, TypeError) as e:
        # If API is not available, log warning and rely on env_spacing
        import warnings
        warnings.warn(
            f"Could not set collision groups programmatically: {e}. "
            f"Relying on env_spacing={env.scene.cfg.env_spacing} meters to prevent collisions. "
            f"Ensure terrain fits within env_spacing/2={env.scene.cfg.env_spacing/2} meters radius from environment origin."
        )


def _phase_to_weight_pyramid(phase, start=0.25, end=0.75):
    weight = torch.zeros_like(phase)
    mask1 = phase < start
    mask2 = phase > end
    weight[mask1] = 1/start * phase[mask1]
    weight[mask2] = 1/(1-end) * (1 - phase[mask2])
    weight[~(mask1 | mask2)] = 1.0
    return weight    

def force_based_push(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_duration: tuple[int, int],
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    mask_update = command.force_push_counter >= command.force_update_frequency
    if command.force_config_init is False:
        command.force_duration_per_env = torch.randint(*force_duration, (command.num_envs,), dtype=torch.int, device=command.device)
        command.force_config_init = True
    mask_finish_update = command.force_push_counter == (command.force_duration_per_env + command.force_update_frequency)
    n_finish = int(mask_finish_update.sum())

    phase = ((command.force_push_counter - command.force_update_frequency) / command.force_duration_per_env).clamp_(min=0.0, max=1.0)

    # NOTE: all body force magnitude updated at once!!
    if n_finish > 0:
        command.body_force_magnitude_buf[mask_finish_update] = torch.rand(n_finish, device=command.device, dtype=command.body_force_magnitude_buf.dtype)

    if not mask_update.any():
        command.force_push_counter.add_(1)
        return

    # Reset and re-randomize config for envs that finished their force push cycle
    if n_finish > 0:
        command.force_push_counter[mask_finish_update] = 0
        command.force_duration_per_env[mask_finish_update] = torch.randint(*force_duration, (n_finish,), dtype=torch.int, device=command.device)
        updated_dirs = torch.randn(n_finish, command.num_bodies, 3, device=command.device, dtype=command.body_force_dir_buf.dtype)
        command.body_force_dir_buf[mask_finish_update] = updated_dirs / torch.norm(updated_dirs, dim=-1, keepdim=True).clamp_(min=1e-8)

    # Compute forces only for force_push bodies (smaller allocation, no zeroing needed)
    n_force_bodies = len(command.force_push_ids)
    forces = torch.zeros(command.num_envs, n_force_bodies, 3, device=command.device, dtype=command.body_force_dir_buf.dtype)
    dir_buf = command.body_force_dir_buf[:, command.force_push_ids_rel, :]
    phase_masked = phase[mask_update]
    weight = _phase_to_weight_pyramid(phase_masked).view(-1, 1, 1)
    forces[mask_update] = (
        command.body_force_magnitude_buf[mask_update].view(-1, 1, 1)
        * weight
        * dir_buf[mask_update]
        * command.max_force
    )

    asset = env.scene[asset_cfg.name]
    positions = command.robot.data.body_pos_w[:, command.force_push_ids] + math_utils.quat_apply(
        command.robot.data.body_quat_w[:, command.force_push_ids], command.force_push_body_offsets
    )
    command.last_force_applied.copy_(forces)
    command.force_push_counter.add_(1)
    torques = torch.zeros_like(forces)
    asset.set_external_force_and_torque(forces, torques, positions=positions, body_ids=command.force_push_ids, is_global=True)


def fake_contact_force(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        command_name: str,
        distance_threshold: float = 0.05,
        contact_spring_k: float = 1000.0,
        min_friction_coeff: float = 0.5,
        squeeze_force_range: tuple[float, float] = (5.0, 20.0),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    wrist_pos_ref = command.vr_3point_body_pos_w[:,:2]
    root_pos_ref = command.anchor_pos_w
    root_orn_ref = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, 2, 1)
    robot_root_pos = command.robot_anchor_pos_w
    robot_root_orn = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, 2, 1)
    wrist_pos_ref_l = wrist_pos_ref - root_pos_ref[:,None,:]
    wrist_pos_ref_l = math_utils.quat_apply(math_utils.quat_inv(root_orn_ref), wrist_pos_ref_l)
    wrist_pos_ref_wr = math_utils.quat_apply(robot_root_orn, wrist_pos_ref_l) + robot_root_pos[:,None,:]
    l_dir = wrist_pos_ref_wr[:,0] - wrist_pos_ref_wr[:,1] # shape [envs, 3]
    l_dir = l_dir / torch.norm(l_dir, dim=-1, keepdim=True)
    r_dir = - l_dir
    
    robot_wrist_pos = command.robot_vr_3point_pos_w[:,:2]
    o2r = robot_wrist_pos - wrist_pos_ref_wr
    dist2o = torch.norm(o2r, dim=-1)
    contact_mask = (dist2o[:,0] < distance_threshold) * (dist2o[:,1] < distance_threshold) * command.wrist_grasp_label
    last_non_contact_mask = command.last_force_applied[:, :2].norm(dim=-1).sum(dim=-1) < 1e-3
    magnitude_update_mask = contact_mask & last_non_contact_mask
    command.body_force_magnitude_buf[magnitude_update_mask] = torch.rand(sum(magnitude_update_mask), device=command.device) * (squeeze_force_range[1] - squeeze_force_range[0]) + squeeze_force_range[0]
    
    l_sp_dist = (o2r[:,0]*(-l_dir)).sum(dim=-1).clamp(min=0.0)
    r_sp_dist = (o2r[:,1]*(-r_dir)).sum(dim=-1).clamp(min=0.0)
    l_normal_force = l_dir * (command.body_force_magnitude_buf + l_sp_dist * contact_spring_k)[:, None]
    r_normal_force = r_dir * (command.body_force_magnitude_buf + r_sp_dist * contact_spring_k)[:, None]
    total_force = torch.stack([l_normal_force, r_normal_force], dim=1) # shape [envs, 2, 3]
    g_force = command.down_dir * min_friction_coeff * command.body_force_magnitude_buf[:, None, None]
    total_force += g_force
    total_force[~contact_mask] = 0.0 # zero out non-contact envs
    command.last_force_applied[:,:2] = total_force.clone()
    asset = env.scene[asset_cfg.name]

    asset.set_external_force_and_torque(total_force, torch.zeros_like(total_force), positions=robot_wrist_pos, body_ids=command.force_push_ids[:2], is_global=True)

def change_compliance(env: ManagerBasedEnv,
                      env_ids, command_name: str,
                      compliance_lb: tuple[float, float],
                      compliance_ub: tuple[float, float],
                      compliance_duration: tuple[int, int],
                      start_steps: int):
    command: MotionCommand = env.command_manager.get_term(command_name)
    if env.common_step_counter < start_steps:
        return
    if command.compliance_config_init == False:
        command.compliance_duration_per_env = torch.randint(*compliance_duration, (command.num_envs,),dtype=torch.int, device=command.device)
        command.compliance_ub = torch.tensor(compliance_ub, device=command.device)
        command.compliance_lb = torch.tensor(compliance_lb, device=command.device)
        command.compliance_config_init = True
    command.compliance_counter += 1
    update_mask = command.compliance_counter == command.compliance_duration_per_env
    command.eef_stiffness_buf[update_mask] = rand_float(command.eef_stiffness_buf[update_mask].shape, command.compliance_lb, command.compliance_ub, command.device)
    command.compliance_counter[update_mask] = 0
    command.compliance_duration_per_env[update_mask] = torch.randint(*compliance_duration, (sum(update_mask),), dtype=torch.int, device=command.device)


_kinematic_object_data: dict | None = None
_kinematic_cmd_type_warned: bool = False
_kinematic_obj_debug_counter: int = 0


def _init_kinematic_object_data(env: ManagerBasedEnv, command_name: str = "motion"):
    """Lazily initialize kinematic object trajectory data. Returns the cached dict or None."""
    global _kinematic_object_data
    # Invalidate cache if env changed (e.g. new process or env recreated)
    env_id = id(env.scene)
    if _kinematic_object_data is not None:
        cached_env_id = _kinematic_object_data.get("_env_scene_id")
        if cached_env_id is not None and cached_env_id != env_id:
            _kinematic_object_data = None
        elif cached_env_id == env_id:
            return _kinematic_object_data

    # Read _object_trajectories from module at runtime (avoids import-timing: setup_motion_terrain_pairs
    # populates it after config load, so we must not capture the initial None at import time)
    # Try multi_object first, then combined terrain+object config.
    _object_trajectories = None
    for mod_name in (
        "whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_binded_multi_object",
        "whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_binded_multi_terrain_object",
    ):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            _object_trajectories = getattr(mod, "_object_trajectories", None)
            if _object_trajectories and len(_object_trajectories) > 0:
                break
        except ImportError:
            pass
    if _object_trajectories is None or len(_object_trajectories) == 0:
        print(
            "[KinObj] Object trajectories not loaded: _object_trajectories is None or empty. "
            "Ensure setup_motion_terrain_pairs() ran with --motion_dir and --object_folder, "
            "and that *_object_fitted.npz files exist in the object folder."
        )
        return None

    # Check if scene has a proxy_object rigid body
    try:
        proxy_obj = env.scene["proxy_object"]
    except KeyError:
        print("[KinObj] No 'proxy_object' in scene, skipping kinematic object init.")
        return None

    command = env.command_manager.get_term(command_name)
    from pathlib import Path as _Path

    motion_files = command.cfg.motion_files
    motion_bases = [_Path(mf).stem for mf in motion_files]
    num_pairs = len(motion_bases)

    max_frames = max(
        (t["trans"].shape[0] for t in _object_trajectories.values()),
        default=0,
    )
    if max_frames == 0:
        return None

    import numpy as np
    all_trans = np.zeros((num_pairs, max_frames, 3), dtype=np.float32)
    all_quat = np.zeros((num_pairs, max_frames, 4), dtype=np.float32)
    all_quat[:, :, 0] = 1.0
    frame_counts = np.zeros(num_pairs, dtype=np.int64)

    for pair_idx, base in enumerate(motion_bases):
        if base in _object_trajectories:
            traj = _object_trajectories[base]
            n = traj["trans"].shape[0]
            all_trans[pair_idx, :n] = traj["trans"]
            all_quat[pair_idx, :n] = traj["quat_wxyz"]
            frame_counts[pair_idx] = n

    _kinematic_object_data = {
        "trans": torch.tensor(all_trans, dtype=torch.float32, device=env.device),
        "quat_wxyz": torch.tensor(all_quat, dtype=torch.float32, device=env.device),
        "frame_counts": torch.tensor(frame_counts, dtype=torch.long, device=env.device),
        "num_pairs": num_pairs,
        "proxy_obj": proxy_obj,
        "_env_scene_id": id(env.scene),
    }
    print(f"[KinObj] Initialized {num_pairs} object trajectories, max_frames={max_frames}")
    return _kinematic_object_data


def _object_world_pos_from_hand_midpoint_motion(
    command: BindedMultiMotionCommand | MultiMotionCommand,
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """World position: midpoint of left/right **VR 3-point** reference targets.

    Matches ``command.vr_3point_body_pos_w`` (wrist link pose + ``vr_3point_body_offset``
    rotated into world frame + env origins). Uses indices 0 and 1 (first two
    ``vr_3point_body`` entries).

    When ``env_ids`` is None, uses the current motion time per env. When set (e.g.
    reset), recomputes the same formula at the first frame of each env's bound motion.
    """
    if env_ids is None:
        vr = command.vr_3point_body_pos_w
        return (vr[:, 0] + vr[:, 1]) * 0.5

    motion = command.motion
    pair_ids = command.motion_ids[env_ids]
    if hasattr(motion, "motion_start_idx"):
        ts = motion.motion_start_idx[pair_ids]
    else:
        ts = torch.zeros(env_ids.shape[0], dtype=torch.long, device=env.device)

    idx_m = command.vr_3point_body_indices_motion
    pos_b = motion.body_pos_w[ts][:, idx_m]
    quat_b = motion.body_quat_w[ts][:, idx_m]
    offsets = command.vr_3point_body_offsets[env_ids]
    shifted = pos_b + math_utils.quat_apply(quat_b, offsets) + env.scene.env_origins[env_ids, None, :]
    return (shifted[:, 0] + shifted[:, 1]) * 0.5


def update_kinematic_object_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "motion",
    object_position_from_hand_midpoint: bool = False,
):
    """Update kinematic proxy object poses from fitted object trajectories.

    Uses write_root_link_pose_to_sim (like HDMI) to drive the per-env proxy_object
    RigidObject. Each env's proxy follows its motion's fitted trajectory.

    If ``object_position_from_hand_midpoint`` is True, world position is the midpoint of
    the two hand VR 3-point reference targets (``vr_3point_body`` / offsets, same as
    policy observations); orientation still comes from the fitted object trajectory
    (``*_object_fitted.npz``).
    """
    data = _init_kinematic_object_data(env, command_name)
    if data is None or data["num_pairs"] == 0:
        return

    command = env.command_manager.get_term(command_name)
    if not isinstance(command, (BindedMultiMotionCommand, MultiMotionCommand)):
        global _kinematic_cmd_type_warned
        if not _kinematic_cmd_type_warned:
            print(
                f"[KinObj] update_kinematic_object_poses skipped: command is {type(command).__name__}, "
                "expected BindedMultiMotionCommand or MultiMotionCommand"
            )
            _kinematic_cmd_type_warned = True
        return

    num_pairs = data["num_pairs"]
    num_envs = env.scene.num_envs
    proxy_obj = data["proxy_obj"]
    device = env.device

    pair_ids = torch.arange(num_envs, device=device) % num_pairs
    time_steps = command.time_steps

    if isinstance(command, BindedMultiMotionCommand):
        motion_lengths = command.motion.motion_end_idx - command.motion.motion_start_idx
        local_steps = time_steps - command.motion.motion_start_idx[pair_ids]
        motion_len_per_env = motion_lengths[pair_ids].float()
    else:
        local_steps = time_steps
        motion_len_per_env = torch.ones(num_envs, device=device) * float(max(time_steps.max().item() + 1, 1))

    obj_frame_counts = data["frame_counts"][pair_ids]
    ratio = local_steps.float() / motion_len_per_env.clamp(min=1.0)
    obj_frames = (ratio * (obj_frame_counts.float() - 1.0)).clamp(min=0).long()
    # For non-object motions (frame_counts=0), use frame 0 (zeros/identity); else clamp to valid range
    obj_frames = torch.where(
        obj_frame_counts > 0,
        torch.minimum(obj_frames, obj_frame_counts - 1),
        torch.zeros_like(obj_frames),
    )

    obj_trans = data["trans"][pair_ids, obj_frames]
    obj_quat = data["quat_wxyz"][pair_ids, obj_frames]

    # Build pose tensor: [pos(3), quat_wxyz(4)] + env_origins for world frame
    obj_pose = torch.zeros(num_envs, 7, device=device)
    if object_position_from_hand_midpoint:
        obj_pose[:, :3] = _object_world_pos_from_hand_midpoint_motion(command, env, None)
    else:
        obj_pose[:, :3] = obj_trans + env.scene.env_origins
    obj_pose[:, 3:7] = obj_quat

    # When contact labels are switched off (dropout), move object up to avoid undesired contact
    if isinstance(command, BindedMultiMotionCommand):
        offset_z = getattr(command.cfg, "contact_label_dropout_object_offset_z", 0.0)
        if offset_z > 0:
            dropout_mask = command.is_contact_label_dropout_active()
            obj_pose[dropout_mask, 2] += offset_z

    # Zero velocity for kinematic object
    obj_vel = torch.zeros(num_envs, 6, device=device)

    # Validate pose before writing: check for NaN/inf
    if torch.isnan(obj_pose).any() or torch.isinf(obj_pose).any():
        print("[KinObj] WARNING: Invalid object pose (NaN/inf) detected, skipping write")
        return

    # Debug: print current pose (from sim), new pose, and robot-object contact before each update (all envs).
    # Only when BMG_PLAY_DEBUG=1 (set by play.py with --play_debug) to avoid spam during train.py.
    global _kinematic_obj_debug_counter
    _kinematic_obj_debug_counter += 1
    play_debug = os.environ.get("BMG_PLAY_DEBUG") == "1"
    if play_debug and (_kinematic_obj_debug_counter <= 20 or _kinematic_obj_debug_counter % 100 == 0):
        curr_poses = proxy_obj.data.root_link_pose_w.cpu()
        # Get contact status from simulation (robot links: LAnk, RAnk, LWrist, RWrist)
        contact_threshold = 1.0
        contact_mask = None
        try:
            contact_sensor = env.scene.sensors["contact_forces"]
            link_names = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
            body_ids = [contact_sensor.body_names.index(n) for n in link_names if n in contact_sensor.body_names]
            if body_ids:
                net_forces = contact_sensor.data.net_forces_w
                if net_forces is not None:
                    forces = net_forces[:, body_ids, :]
                    contact_mask = (torch.norm(forces, dim=-1) > contact_threshold).cpu()
        except (KeyError, ValueError, AttributeError):
            pass

        print(f"\n[KinObj] ========== update #{_kinematic_obj_debug_counter} ==========")
        header = f"{'env':>4} | {'ts':>5} | {'of':>4} | {'current pos (x,y,z)':>24} | {'new pos (x,y,z)':>24}"
        if contact_mask is not None:
            header += f" | {'contact (1=yes) LAnk RAnk LWr RWr':>30}"
        print(header)
        print("-" * (len(header) + 2))
        for i in range(num_envs):
            ts = time_steps[i].item()
            of = obj_frames[i].item()
            curr_p = curr_poses[i, :3].tolist()
            new_p = obj_pose[i, :3].cpu().tolist()
            curr_s = f"({curr_p[0]:.3f},{curr_p[1]:.3f},{curr_p[2]:.3f})"
            new_s = f"({new_p[0]:.3f},{new_p[1]:.3f},{new_p[2]:.3f})"
            row = f"{i:>4} | {ts:>5} | {of:>4} | {curr_s:>24} | {new_s:>24}"
            if contact_mask is not None:
                c = contact_mask[i]
                contact_s = " ".join("1" if c[j] else "0" for j in range(len(c)))
                row += f" | {contact_s:>30}"
            print(row)
        print()

    proxy_obj.write_root_link_pose_to_sim(obj_pose)
    proxy_obj.write_root_com_velocity_to_sim(obj_vel)


def reset_object_poses_direct(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "motion",
    object_position_from_hand_midpoint: bool = False,
):
    """Directly reset proxy object poses for env_ids (e.g. on robot/motion end reset).

    Uses frame 0 (motion start) for object pose since reset runs before command_manager.reset.
    Writes pose and zero velocity directly to sim - no magic force.

    If ``object_position_from_hand_midpoint`` is True, position uses the VR 3-point
    hand-target midpoint at the start of each env's bound motion; orientation from
    fitted trajectory at frame 0.
    """
    if env_ids is None or (isinstance(env_ids, torch.Tensor) and env_ids.numel() == 0):
        return
    data = _init_kinematic_object_data(env, command_name)
    if data is None or data["num_pairs"] == 0:
        return

    command = env.command_manager.get_term(command_name)
    if not isinstance(command, (BindedMultiMotionCommand, MultiMotionCommand)):
        return

    if isinstance(env_ids, (list, tuple)):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor([env_ids], device=env.device, dtype=torch.long)

    num_pairs = data["num_pairs"]
    proxy_obj = data["proxy_obj"]
    device = env.device

    pair_ids = env_ids % num_pairs
    obj_frames = torch.zeros_like(env_ids, device=device, dtype=torch.long)

    obj_trans = data["trans"][pair_ids, obj_frames]
    obj_quat = data["quat_wxyz"][pair_ids, obj_frames]

    obj_pose = torch.zeros(len(env_ids), 7, device=device)
    if object_position_from_hand_midpoint:
        obj_pose[:, :3] = _object_world_pos_from_hand_midpoint_motion(command, env, env_ids)
    else:
        obj_pose[:, :3] = obj_trans + env.scene.env_origins[env_ids]
    obj_pose[:, 3:7] = obj_quat

    obj_vel = torch.zeros(len(env_ids), 6, device=device)

    if torch.isnan(obj_pose).any() or torch.isinf(obj_pose).any():
        return

    proxy_obj.write_root_link_pose_to_sim(obj_pose, env_ids=env_ids)
    proxy_obj.write_root_com_velocity_to_sim(obj_vel, env_ids=env_ids)


def apply_object_hand_midpoint_to_events(env_cfg, use_hand_mid: bool) -> None:
    """Set ``object_position_from_hand_midpoint`` on object-related event terms.

    When True, proxy object **position** follows the reference-motion midpoint of
    ``left_wrist_yaw_link`` and ``right_wrist_yaw_link``; **orientation** still comes
    from the fitted trajectory (``*_object_fitted.npz``). When False, full pose is
    from the fitted trajectory.

    Safe to call after CLI overrides on ``env_cfg`` (e.g. from train.py / play.py).
    """
    ev = getattr(env_cfg, "events", None)
    if ev is None:
        return
    for term_name in ("init_object_poses", "reset_object_poses", "update_object_poses"):
        if not hasattr(ev, term_name):
            continue
        term = getattr(ev, term_name)
        params = dict(term.params) if term.params else {}
        params["object_position_from_hand_midpoint"] = use_hand_mid
        if hasattr(term, "replace"):
            setattr(ev, term_name, term.replace(params=params))
        else:
            object.__setattr__(term, "params", params)


# Per-env physics phase activation: keyed by env.scene id, value = (num_envs,) bool tensor
_physics_phase_activated: dict[int, torch.Tensor] = {}


def get_physics_phase_activated(env: ManagerBasedEnv) -> torch.Tensor | None:
    """Return (num_envs,) bool tensor of physics_phase_activated, or None if not initialized."""
    scene_id = id(env.scene)
    if scene_id not in _physics_phase_activated:
        return None
    return _physics_phase_activated[scene_id]


def clear_physics_phase_activated(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
):
    """Clear physics_phase_activated for reset envs. Called on env reset."""
    global _physics_phase_activated
    if env_ids is None or (isinstance(env_ids, torch.Tensor) and env_ids.numel() == 0):
        return
    scene_id = id(env.scene)
    if scene_id in _physics_phase_activated:
        _physics_phase_activated[scene_id][env_ids] = False


def update_magic_force_object_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "motion",
    kp_pos: float = 500.0,
    kd_pos: float = 50.0,
    kp_ori: float = 200.0,
    kd_ori: float = 20.0,
    gravity_compensation: bool = True,
    physics_phase: bool = False,
    physics_phase_contact_threshold: float = 1.0,
    use_pose_velocity_control: bool = False,
    object_position_from_hand_midpoint: bool = False,
):
    """Apply PD-based magic force to proxy object to track desired trajectory.

    Uses high-gain PD control to drive the object toward the desired pose from the
    fitted trajectory. Resists external perturbations (e.g. robot contact) while
    tracking. Replaces direct pose setting with force/torque application.

    If ``object_position_from_hand_midpoint`` is True, desired world position follows
    the midpoint of the two hand VR 3-point targets (``vr_3point_body`` / offsets);
    desired orientation still follows the fitted object trajectory.

    When physics_phase=True: first time both wrists make contact with object,
    magic force is disabled for that env and object is fully controlled by robot.

    When use_pose_velocity_control=True: bypass apply_forces_and_torques (avoids
    cudaErrorIllegalAddress with replicate_physics=False + MultiAssetSpawner).
    Uses direct pose/velocity setting instead; slightly different dynamics.

    Args:
        env: ManagerBasedEnv instance.
        env_ids: Unused (for EventTerm compatibility).
        command_name: Motion command term name.
        kp_pos: Position proportional gain.
        kd_pos: Position derivative gain.
        kp_ori: Orientation proportional gain.
        kd_ori: Orientation derivative gain.
        gravity_compensation: If True, add mass * (-gravity) to cancel gravity.
        physics_phase: If True, disable magic force once both wrists contact object.
        physics_phase_contact_threshold: Force magnitude threshold (N) for contact.
        use_pose_velocity_control: If True, use direct pose/velocity instead of forces
            (workaround for cudaErrorIllegalAddress with replicate_physics=False).
    """
    data = _init_kinematic_object_data(env, command_name)
    if data is None or data["num_pairs"] == 0:
        return

    command = env.command_manager.get_term(command_name)
    if not isinstance(command, (BindedMultiMotionCommand, MultiMotionCommand)):
        global _kinematic_cmd_type_warned
        if not _kinematic_cmd_type_warned:
            print(
                f"[MagicForce] update_magic_force_object_poses skipped: command is {type(command).__name__}, "
                "expected BindedMultiMotionCommand or MultiMotionCommand"
            )
            _kinematic_cmd_type_warned = True
        return

    num_pairs = data["num_pairs"]
    num_envs = env.scene.num_envs
    proxy_obj = data["proxy_obj"]
    device = env.device

    pair_ids = torch.arange(num_envs, device=device) % num_pairs
    time_steps = command.time_steps

    if isinstance(command, BindedMultiMotionCommand):
        motion_lengths = command.motion.motion_end_idx - command.motion.motion_start_idx
        local_steps = time_steps - command.motion.motion_start_idx[pair_ids]
        motion_len_per_env = motion_lengths[pair_ids].float()
    else:
        local_steps = time_steps
        motion_len_per_env = torch.ones(num_envs, device=device) * float(max(time_steps.max().item() + 1, 1))

    obj_frame_counts = data["frame_counts"][pair_ids]
    ratio = local_steps.float() / motion_len_per_env.clamp(min=1.0)
    obj_frames = (ratio * (obj_frame_counts.float() - 1.0)).clamp(min=0).long()
    # For non-object motions (frame_counts=0), use frame 0 (zeros/identity); else clamp to valid range
    obj_frames = torch.where(
        obj_frame_counts > 0,
        torch.minimum(obj_frames, obj_frame_counts - 1),
        torch.zeros_like(obj_frames),
    )

    if object_position_from_hand_midpoint:
        desired_pos = _object_world_pos_from_hand_midpoint_motion(command, env, None)
    else:
        desired_pos = data["trans"][pair_ids, obj_frames] + env.scene.env_origins
    desired_quat = data["quat_wxyz"][pair_ids, obj_frames]

    # When contact labels are switched off (dropout), move object up to avoid undesired contact
    if isinstance(command, BindedMultiMotionCommand):
        offset_z = getattr(command.cfg, "contact_label_dropout_object_offset_z", 0.0)
        if offset_z > 0:
            dropout_mask = command.is_contact_label_dropout_active()
            desired_pos = desired_pos.clone()
            desired_pos[dropout_mask, 2] += offset_z

    if torch.isnan(desired_pos).any() or torch.isinf(desired_pos).any():
        return
    if torch.isnan(desired_quat).any() or torch.isinf(desired_quat).any():
        return

    # Current state from simulation
    curr_pose = proxy_obj.data.root_link_pose_w
    curr_pos = curr_pose[:, :3]
    curr_quat = curr_pose[:, 3:7]
    curr_vel = proxy_obj.data.root_com_vel_w
    curr_lin_vel = curr_vel[:, :3]
    curr_ang_vel = curr_vel[:, 3:6]

    # Desired velocity (zero for trajectory hold; could interpolate from trajectory)
    desired_lin_vel = torch.zeros_like(curr_lin_vel)
    desired_ang_vel = torch.zeros_like(curr_ang_vel)

    # PD force: F = Kp * pos_error + Kd * vel_error
    pos_error = desired_pos - curr_pos
    lin_vel_error = desired_lin_vel - curr_lin_vel
    force = kp_pos * pos_error + kd_pos * lin_vel_error

    # Gravity compensation (Isaac Z-up: gravity = (0, 0, -9.81))
    if gravity_compensation:
        mass = getattr(proxy_obj.data, "default_mass", None)
        if mass is not None and mass.numel() > 0:
            mass = mass.to(device)[:num_envs].view(-1, 1)
        else:
            mass = torch.ones(num_envs, 1, device=device)
        gravity_vec = torch.zeros(num_envs, 3, device=device)
        gravity_vec[:, 2] = -9.81
        force = force + mass * (-gravity_vec)

    # PD torque: tau = Kp * ori_error_axis_angle + Kd * ang_vel_error
    quat_error = math_utils.quat_mul(
        desired_quat, math_utils.quat_conjugate(curr_quat)
    )
    ori_error_aa = math_utils.axis_angle_from_quat(quat_error)
    ang_vel_error = desired_ang_vel - curr_ang_vel
    torque = kp_ori * ori_error_aa + kd_ori * ang_vel_error

    # Physics phase: disable magic force once both wrists contact object
    # Read from env.cfg.physics_phase_flag so --physics_phase CLI is respected (EventTerm params are static)
    physics_phase = getattr(env.cfg, "physics_phase_flag", physics_phase)
    skip_magic_force = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if physics_phase:
        global _physics_phase_activated
        scene_id = id(env.scene)
        if scene_id not in _physics_phase_activated:
            _physics_phase_activated[scene_id] = torch.zeros(
                num_envs, dtype=torch.bool, device=device
            )
        activated = _physics_phase_activated[scene_id]
        if activated.shape[0] != num_envs:
            activated = torch.zeros(num_envs, dtype=torch.bool, device=device)
            _physics_phase_activated[scene_id] = activated

        # Use force_matrix_w (filtered: wrist-ProxyObject only) to exclude hand-hand.
        # force_matrix_w is (N, 1, M, 3): sensor body 0 vs M filter bodies (ProxyObject).
        # Gate by contact labels: only activate when reference expects wrist contact (dropout envs stay in magic force).
        # When contact_debug_zero_out: bypass gate so physics phase activates on actual contact (labels are zeroed for debug).
        contact_debug_zero_out = getattr(command.cfg, "contact_debug_zero_out", False)
        if contact_debug_zero_out:
            wrist_contact_expected = torch.ones(num_envs, dtype=torch.bool, device=device)
        else:
            cl = command.get_contact_labels_masked()
            wrist_contact_expected = (
                (cl[:, 2] | cl[:, 3]) if cl is not None else torch.ones(num_envs, dtype=torch.bool, device=device)
            )
        left_sensor = env.scene.sensors["object_contact_left_wrist"]
        right_sensor = env.scene.sensors["object_contact_right_wrist"]
        left_fm = left_sensor.data.force_matrix_w
        right_fm = right_sensor.data.force_matrix_w
        if left_fm is not None and right_fm is not None:
            left_norm = torch.norm(left_fm[:, 0, :, :].sum(dim=1), dim=-1)
            right_norm = torch.norm(right_fm[:, 0, :, :].sum(dim=1), dim=-1)
            both_wrists_contact = (
                (left_norm > physics_phase_contact_threshold)
                & (right_norm > physics_phase_contact_threshold)
            )
            activated = activated | (both_wrists_contact & wrist_contact_expected)
            # Ensure contact-label-dropout envs never enter physics mode (object stays kinematic)
            if isinstance(command, BindedMultiMotionCommand):
                dropout_mask = command.is_contact_label_dropout_active()
                activated = activated & ~dropout_mask
            _physics_phase_activated[scene_id] = activated
        skip_magic_force = activated

    force = force.unsqueeze(1)
    torque = torque.unsqueeze(1)
    if skip_magic_force.any():
        force = torch.where(skip_magic_force.view(-1, 1, 1), 0.0, force)
        torque = torch.where(skip_magic_force.view(-1, 1, 1), 0.0, torque)

    # Match proxy instance count (avoids cudaErrorIllegalAddress with MultiAssetSpawner)
    n_inst = proxy_obj.num_instances
    if n_inst != num_envs:
        force = force[:n_inst]
        torque = torque[:n_inst]
        skip_magic_force = skip_magic_force[:n_inst]

    if use_pose_velocity_control:
        # Bypass apply_forces (avoids cudaErrorIllegalAddress with replicate_physics=False).
        # Drive via direct pose + velocity; use PD output as desired velocity.
        desired_lin_vel_drive = (kp_pos * pos_error[:n_inst] + kd_pos * lin_vel_error[:n_inst]) * 0.02
        desired_ang_vel_drive = (kp_ori * ori_error_aa[:n_inst] + kd_ori * ang_vel_error[:n_inst]) * 0.02
        if gravity_compensation:
            _mass = getattr(proxy_obj.data, "default_mass", None)
            if _mass is not None and _mass.numel() > 0:
                _m = _mass.to(device)[:n_inst].view(-1, 1)
                desired_lin_vel_drive = desired_lin_vel_drive + torch.tensor([0, 0, 9.81], device=device) * 0.02
        desired_lin_vel = torch.where(skip_magic_force.view(-1, 1), curr_lin_vel[:n_inst], desired_lin_vel_drive)
        desired_ang_vel = torch.where(skip_magic_force.view(-1, 1), curr_ang_vel[:n_inst], desired_ang_vel_drive)
        root_state = torch.zeros(n_inst, 13, device=device)
        root_state[:, :3] = torch.where(skip_magic_force.view(-1, 1), curr_pos[:n_inst], desired_pos[:n_inst])
        root_state[:, 3:7] = torch.where(skip_magic_force.view(-1, 1), curr_quat[:n_inst], desired_quat[:n_inst])
        root_state[:, 7:10] = desired_lin_vel
        root_state[:, 10:13] = desired_ang_vel
        proxy_obj.write_root_state_to_sim(root_state)
    else:
        # Force-based control; ensure contiguous to avoid PhysX copy_if issues
        force = force.contiguous().clone()
        torque = torque.contiguous().clone()
        proxy_obj.set_external_force_and_torque(
            forces=force,
            torques=torque,
            body_ids=[0],
            is_global=True,
        )
