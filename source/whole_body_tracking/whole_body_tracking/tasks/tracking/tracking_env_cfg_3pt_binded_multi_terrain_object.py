"""Environment config for 3pt tracking with combined terrain + object + free-space motions.

Combines:
- Terrain motions: grouped by exp (one terrain per exp); motions in same exp share slot.
- Object motions: have {stem}_thin_plate_proxy.urdf in object_folder; all use flat slot.
- Free motions: no terrain or object match; use same flat slot as object motions.

Slot layout: [terrain slot per exp] + [flat slot for object+free].
num_slots = num_terrain_exps + 1.

Usage (train.py):
    --task Tracking-3pt-Binded-Multi-Terrain-Object-G1-v0
    # Critic conditioning (privileged [terrain_or_free, object] one-hot): append ``-CriticCat`` before ``-G1-v0``
    # e.g. Tracking-3pt-Binded-Multi-Terrain-Object-CriticCat-G1-v0 (see ``flat_env_cfg`` registrations).
    --motion_dir <dir>           # contains terrain, object, and free motions
    --terrain_folder <dir>       # {exp}_terrain.urdf for terrain motions
    --object_folder <dir>        # {stem}_thin_plate_proxy.urdf, *_object_fitted.npz, *_contact_nodes.npy

Merged assets (terrain + object + combined contact in one tree), mixed with split folders::

    --task Tracking-3pt-Binded-Multi-Terrain-Object-Merged-G1-v0
    --motion_dir ... --terrain_folder ... --object_folder ... --object_terrain_folder csv_hand_object_terrain/EXP/

Contact labels load in order: ``object_terrain_folder`` → ``object_folder`` → ``terrain_folder``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import MISSING, field
from pathlib import Path

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.terrain_mesh_spawner_cfg import (
    FLAT_PLANE_MARKER,
    MultiTerrainMeshSpawnerCfg,
)

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# Path to empty proxy URDF for motions without object interaction
_EMPTY_PROXY_URDF = Path(__file__).resolve().parent / "empty_proxy.urdf"


def _exp_prefix(motion_base: str) -> str:
    """Extract exp prefix from motion base. obstacles1_subject1_rf_motion_0 -> obstacles1_subject1_rf."""
    m = re.match(r"^(.+)_motion_\d+$", motion_base)
    return m.group(1) if m else ""


def _resolve_path(path: Path, fallback_roots: list[Path]) -> Path:
    if path.is_absolute() and path.exists():
        return path.resolve()
    for root in fallback_roots:
        candidate = root / path
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def _scan_terrain_urdf_by_stem(folder: Path | None) -> dict[str, Path]:
    """Map URDF stem (e.g. ``foo_terrain``) -> path."""
    d: dict[str, Path] = {}
    if folder is None or not folder.is_dir():
        return d
    for f in folder.glob("*.urdf"):
        d[f.stem] = f
    return d


def _scan_proxy_urdf_by_motion_stem(folder: Path | None) -> dict[str, Path]:
    """Map motion base name (e.g. ``foo_motion_0``) -> thin-plate proxy URDF path."""
    d: dict[str, Path] = {}
    if folder is None or not folder.is_dir():
        return d
    for f in folder.glob("*_thin_plate_proxy.urdf"):
        stem = f.stem.replace("_thin_plate_proxy", "")
        d[stem] = f
    return d


def _build_proxy_urdf_paths(object_folder: Path, motion_files: list[str]) -> list[str]:
    """Build URDF paths: {object_folder}/{motion_stem}_thin_plate_proxy.urdf per motion.

    For motions without a matching proxy URDF, uses empty_proxy.urdf.
    """
    result: list[str] = []
    num_empty = 0
    for mf in motion_files:
        stem = Path(mf).stem
        proxy_path = Path(object_folder) / f"{stem}_thin_plate_proxy.urdf"
        if proxy_path.exists():
            result.append(str(proxy_path.resolve()))
        else:
            result.append(str(_EMPTY_PROXY_URDF))
            num_empty += 1
    if num_empty:
        print(f"[INFO]: {num_empty} motion(s) have no matching proxy URDF (using empty_proxy)")
    return result


def _make_proxy_rigid_object_cfg_multi_urdf(
    urdf_paths: list[str], use_magic_force: bool = True
) -> RigidObjectCfg:
    """Build proxy RigidObjectCfg with MultiAssetSpawnerCfg for per-env URDFs."""
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        kinematic_enabled=not use_magic_force,
        disable_gravity=not use_magic_force,
    )
    collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    )
    assets_cfg = [
        sim_utils.UrdfFileCfg(
            asset_path=path,
            fix_base=False,
            activate_contact_sensors=True,
            rigid_props=rigid_props,
            collision_props=collision_props,
            joint_drive=joint_drive,
        )
        for path in urdf_paths
    ]
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ProxyObject",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=assets_cfg,
            random_choice=False,
            rigid_props=rigid_props,
            collision_props=collision_props,
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
        ),
    )


def _build_motion_terrain_object_pairs(
    motion_dir: str,
    terrain_folder_path: str,
    object_folder_path: str,
    object_terrain_folder_path: str = "",
    merged_object_terrain_origin_offset_xy: tuple[float, float] | None = None,
) -> tuple[list[str], list[str], list[str], list[int], list[int], list[tuple[float, float]]]:
    """Build motion-terrain-object pairs for combined config.

    Returns (motion_files, terrain_urdf_paths, proxy_urdf_paths, motion_to_slot, motion_categories,
    motion_extra_origin_xy).
    - motion_files: all motions (terrain first by exp, then object, then free)
    - terrain_urdf_paths: one per slot (terrain URDF or FLAT_PLANE_MARKER)
    - proxy_urdf_paths: one per motion (real or empty_proxy)
    - motion_to_slot: motion_idx -> slot_idx
    - motion_categories: per motion, 0=terrain, 1=object (proxy interaction), 2=free / flat only
    - motion_extra_origin_xy: always (0,0) per motion; non-zero merged canvas offsets are unsupported with
      fixed global terrain slots (see :attr:`Tracking3ptBindedMultiTerrainObjectEnvCfg.merged_object_terrain_origin_offset_xy`).

    ``object_terrain_folder_path`` (optional) is merged in: its ``*_terrain.urdf`` and proxy URDFs override
    the same keys from ``terrain_folder_path`` / ``object_folder_path`` so CSV hand-object exports can
    supply per-motion terrain + object in one directory alongside split OMOMO / robot-first trees.
    """
    fallback_roots = [
        Path.cwd(),
        Path(__file__).parent.parent.parent.parent.parent.parent,
    ]
    motion_path = _resolve_path(Path(motion_dir), fallback_roots)
    terrain_path = (
        _resolve_path(Path(terrain_folder_path), fallback_roots) if terrain_folder_path.strip() else None
    )
    object_path = (
        _resolve_path(Path(object_folder_path), fallback_roots) if object_folder_path.strip() else None
    )
    ot_path = (
        _resolve_path(Path(object_terrain_folder_path), fallback_roots)
        if object_terrain_folder_path.strip()
        else None
    )

    if not motion_path.exists():
        raise FileNotFoundError(f"Motion directory not found: {motion_path}")

    motion_by_base: dict[str, Path] = {f.stem: f for f in motion_path.glob("*.npz")}
    if not motion_by_base:
        raise FileNotFoundError(f"No .npz motion files found in {motion_path}")

    terrain_by_base: dict[str, Path] = {}
    terrain_by_base.update(_scan_terrain_urdf_by_stem(terrain_path))
    for k, v in _scan_terrain_urdf_by_stem(ot_path).items():
        terrain_by_base[k] = v

    object_proxy_by_base: dict[str, Path] = {}
    object_proxy_by_base.update(_scan_proxy_urdf_by_motion_stem(object_path))
    for k, v in _scan_proxy_urdf_by_motion_stem(ot_path).items():
        object_proxy_by_base[k] = v

    # Classify motions: terrain (has exp + exp_terrain.urdf), object (has proxy), free (neither)
    terrain_bases: list[str] = []
    object_bases: list[str] = []
    free_bases: list[str] = []

    for base in motion_by_base:
        exp = _exp_prefix(base)
        terrain_key = f"{exp}_terrain" if exp else "terrain"
        has_terrain = terrain_key in terrain_by_base
        has_object = base in object_proxy_by_base

        if has_terrain and exp:
            terrain_bases.append(base)
        elif has_object:
            object_bases.append(base)
        else:
            free_bases.append(base)

    # Sort terrain by exp then motion number
    def _motion_num(b: str) -> int:
        m = re.search(r"motion_(\d+)$", b)
        return int(m.group(1)) if m else 0

    exp_to_terrain_bases: dict[str, list[str]] = {}
    for b in terrain_bases:
        exp = _exp_prefix(b)
        if exp not in exp_to_terrain_bases:
            exp_to_terrain_bases[exp] = []
        exp_to_terrain_bases[exp].append(b)
    for exp in exp_to_terrain_bases:
        exp_to_terrain_bases[exp].sort(key=_motion_num)
    exps_ordered = sorted(exp_to_terrain_bases.keys(), key=lambda e: (e or "\x00"))

    # Build motion_files: terrain (by exp) + object + free
    motion_files: list[str] = []
    motion_to_slot: list[int] = []
    motion_categories: list[int] = []
    exp_to_slot: dict[str, int] = {exp: idx for idx, exp in enumerate(exps_ordered)}
    flat_slot = len(exps_ordered)

    for exp in exps_ordered:
        slot = exp_to_slot[exp]
        for base in exp_to_terrain_bases[exp]:
            motion_files.append(str(motion_by_base[base].resolve()))
            motion_to_slot.append(slot)
            motion_categories.append(0)

    for base in sorted(object_bases):
        motion_files.append(str(motion_by_base[base].resolve()))
        motion_to_slot.append(flat_slot)
        motion_categories.append(1)

    for base in sorted(free_bases):
        motion_files.append(str(motion_by_base[base].resolve()))
        motion_to_slot.append(flat_slot)
        motion_categories.append(2)

    # terrain_urdf_paths: one per exp + flat
    terrain_urdf_paths: list[str] = []
    for exp in exps_ordered:
        terrain_key = f"{exp}_terrain" if exp else "terrain"
        if terrain_key in terrain_by_base:
            terrain_urdf_paths.append(str(terrain_by_base[terrain_key].resolve()))
        else:
            terrain_urdf_paths.append(FLAT_PLANE_MARKER)
    terrain_urdf_paths.append(FLAT_PLANE_MARKER)

    # proxy_urdf_paths: one per motion (scan merged object_proxy_by_base so split + OT folders work)
    proxy_urdf_paths: list[str] = []
    num_empty_proxy = 0
    for mf in motion_files:
        stem = Path(mf).stem
        proxy_p = object_proxy_by_base.get(stem)
        if proxy_p is not None and proxy_p.is_file():
            proxy_urdf_paths.append(str(proxy_p.resolve()))
        else:
            proxy_urdf_paths.append(str(_EMPTY_PROXY_URDF))
            num_empty_proxy += 1
    if num_empty_proxy:
        print(f"[INFO]: {num_empty_proxy} motion(s) have no matching proxy URDF (using empty_proxy)")

    if merged_object_terrain_origin_offset_xy is not None:
        ox = float(merged_object_terrain_origin_offset_xy[0])
        oy = float(merged_object_terrain_origin_offset_xy[1])
        if abs(ox) > 1e-9 or abs(oy) > 1e-9:
            print(
                "[WARN]: merged_object_terrain_origin_offset_xy=({:.4g}, {:.4g}) is ignored. "
                "Terrain stays at /World/terrains/slot_* grid positions; shifting env origins only would "
                "misalign robot and mesh. Use env_spacing / fewer envs for viewport separation instead.",
                ox,
                oy,
                flush=True,
            )
    motion_extra_origin_xy: list[tuple[float, float]] = [(0.0, 0.0)] * len(motion_files)

    return (
        motion_files,
        terrain_urdf_paths,
        proxy_urdf_paths,
        motion_to_slot,
        motion_categories,
        motion_extra_origin_xy,
    )


##
# Scene config
##


@configclass
class CombinedTerrainObjectSceneCfg(InteractiveSceneCfg):
    """Scene with terrain (multi-slot) + proxy object (per-motion URDF).

    replicate_physics=False (required for proxy MultiAssetSpawnerCfg).
    Terrain at /World/terrains/slot_0..slot_{n-1}; height scanner over all slots.
    """

    replicate_physics: bool = False
    lazy_sensor_update: bool = False

    terrain: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/terrains/slot_0",
        collision_group=-1,
        spawn=MultiTerrainMeshSpawnerCfg(
            urdf_paths=[],
            env_spacing=6.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.4, 0.3)),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                kinematic_enabled=True,
            ),
        ),
    )

    robot: ArticulationCfg = MISSING

    proxy_object: RigidObjectCfg = field(
        default_factory=lambda: RigidObjectCfg(prim_path="{ENV_REGEX_NS}/ProxyObject"),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=True,
    ) #
    # Wrist-only contact times at 1 N so ``current_contact_time`` matches
    # ``contact_label_contact_duration_reward`` (threshold=1.0). Main ``contact_forces`` stays at 10 N
    # for feet / coarse contact; using 1 N globally would inflate foot air-time noise.
    contact_forces_wrist_airtime = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*wrist_yaw_link",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    hand_object_contact_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*left_wrist_yaw_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/ProxyObject/.*"],
        history_length=1,
    )
    hand_object_contact_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*right_wrist_yaw_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/ProxyObject/.*"],
        history_length=1,
    )
    object_contact_left_wrist = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ProxyObject/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*left_wrist_yaw_link"],
        history_length=1,
    )
    object_contact_right_wrist = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ProxyObject/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*right_wrist_yaw_link"],
        history_length=1,
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[2.0, 2.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/terrains/slot_0"],
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    motion = mdp.BindedMultiMotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        adaptive_uniform_ratio=0.25,
        debug_vis=True,
        has_wrist_grasp_label=True,
        contact_label_dropout_prob=0.5,
        contact_label_dropout_mode="episode",  # switch at reset, fixed until termination
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        command = ObsTerm(func=mdp.command_lower_body, params={"command_name": "motion"})
        vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_target, params={"command_name": "motion"})
        vr_3point_orn = ObsTerm(func=mdp.vr_3point_local_orn_target, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.1, n_max=0.1))
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-1.0, n_max=1.0))
        gravity_dir = ObsTerm(func=mdp.projected_gravity, params={"command_name": "motion"}, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)
        contact_label = ObsTerm(func=mdp.contact_label, params={"command_name": "motion"})
        # contact_label_wrists_only = ObsTerm(
        #     func=mdp.contact_label_wrists_only, params={"command_name": "motion"}
        # )
        # height_map = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        #     clip=(-1.0, 1.0),
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        contact_label = ObsTerm(func=mdp.contact_label, params={"command_name": "motion"})
        # contact_label_wrists_only = ObsTerm(
        #     func=mdp.contact_label_wrists_only, params={"command_name": "motion"}
        # )
        # height_map = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 1.0),
        # )

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    hand_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wrist_yaw.*"),
            "static_friction_range": (2.5, 2.5),
            "dynamic_friction_range": (2.5, 2.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    proxy_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("proxy_object"),
            "mass_distribution_params": (0.2, 1.2),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    set_slot_based_env_origins = EventTerm(
        func=mdp.set_slot_based_env_origins,
        mode="startup",
        params={},
    )
    set_environment_collision_groups = EventTerm(
        func=mdp.set_environment_collision_groups,
        mode="startup",
        params={
            "terrain_cfg": SceneEntityCfg("terrain"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    set_proxy_robot_collision_filter = EventTerm(
        func=mdp.set_proxy_robot_collision_filter,
        mode="prestartup",
        params={},
    )
    init_object_poses = EventTerm(
        func=mdp.update_kinematic_object_poses,
        mode="startup",
        params={"command_name": "motion"},
    )
    reset_object_poses = EventTerm(
        func=mdp.reset_object_poses_direct,
        mode="reset",
        params={"command_name": "motion"},
    )
    reset_physics_phase_flags = EventTerm(
        func=mdp.clear_physics_phase_activated,
        mode="reset",
        params={},
    )
    update_object_poses = EventTerm(
        func=mdp.update_magic_force_object_poses,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "command_name": "motion",
            "kp_pos": 20.0,
            "kd_pos": 2.0,
            "kp_ori": 20.0,
            "kd_ori": 3.0,
            "gravity_compensation": True,
            "physics_phase_contact_threshold": 0.1,
            "use_pose_velocity_control": True,
        },
    )


@configclass
class RewardsCfg:
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )
    # Contact labels: 0=L_foot, 1=R_foot, 2=L_wrist, 3=R_wrist — split foot vs wrist for tuning.
    contact_label_contact_foot = RewTerm(
        func=mdp.contact_label_contact_reward,
        weight=0.5,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "threshold": 1.0,
            "contact_indices": [0, 1],
        },
    )
    contact_label_contact_wrist = RewTerm(
        func=mdp.contact_label_contact_reward,
        weight=0.5,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
            ),
            "threshold": 1.0,
            "contact_indices": [2, 3],
            # Free-space motions have no contact_nodes labels (zeros); skip wrist matching reward.
            "disable_motion_categories": [2],
        },
    )
    contact_label_contact_duration_foot = RewTerm(
        func=mdp.contact_label_contact_duration_reward,
        weight=0.25,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "threshold": 1.0,
            "contact_indices": [0, 1],
            "max_duration_s": 0.5,
        },
    )
    contact_label_contact_duration_wrist = RewTerm(
        func=mdp.contact_label_contact_duration_reward,
        weight=0.25,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces_wrist_airtime",
                body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
            ),
            "threshold": 1.0,
            "contact_indices": [2, 3],
            "max_duration_s": 0.5,
            "disable_motion_categories": [2],
        },
    )
    # Seat / pelvis-terrain (column 4): disabled by default; enabled in 5-D seat env configs.
    contact_label_contact_seat = RewTerm(
        func=mdp.unilateral_contact_label_contact_reward_seat_support,
        weight=0.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_hip_pitch_link", "right_hip_pitch_link"],
            ),
            "threshold": 1.0,
            "contact_label_col": 4,
        },
    )
    contact_label_contact_duration_seat = RewTerm(
        func=mdp.unilateral_contact_label_contact_duration_reward_seat_support,
        weight=0.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_hip_pitch_link", "right_hip_pitch_link"],
            ),
            "threshold": 1.0,
            "contact_label_col": 4,
            "max_duration_s": 0.5,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    motion_ended = DoneTerm(func=mdp.motion_ended, params={"command_name": "motion"})
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )
    # object_dropped = DoneTerm(
    #     func=mdp.object_dropped_to_floor,
    #     params={
    #         "floor_height_threshold": 0.1,
    #         "command_name": "motion",
    #         "require_wrist_contact_label": True,
    #     },
    # )

    bad_wrist_contact_mismatch = DoneTerm(
        func=mdp.bad_wrist_contact_mismatch_duration,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
            ),
            "threshold": 1.0,
            "contact_indices": [2, 3],
            "max_mismatch_steps": 50,
            "only_object_motions": True,
        },
    )


def bad_wrist_contact_mismatch_max_steps_curriculum(
    env,
    env_ids: Sequence[int],
) -> int | None:
    """After ``cfg.wrist_mismatch_strict_after_env_steps``, set ``max_mismatch_steps`` on the wrist term.

    Uses ``termination_manager.get_term_cfg`` / ``set_term_cfg`` instead of
    :class:`isaaclab.envs.mdp.modify_term_cfg` dotted paths, which can break across Isaac Lab versions
    when termination term cfg layout differs (e.g. ``params`` vs internal accessors).

    ``env.common_step_counter`` advances once per control step; default ``15000 * 24`` matches
    15000 RSL-RL iterations when ``num_steps_per_env == 24``.
    """
    del env_ids
    term_name = "bad_wrist_contact_mismatch"
    tm = env.termination_manager
    if term_name not in tm.active_terms:
        return None
    term_cfg = tm.get_term_cfg(term_name)
    params = getattr(term_cfg, "params", None)
    if not isinstance(params, dict):
        return None
    after = getattr(env.cfg, "wrist_mismatch_strict_after_env_steps", 15000 * 24)
    target = int(getattr(env.cfg, "wrist_mismatch_strict_max_mismatch_steps", 20))
    cur = int(params.get("max_mismatch_steps", target))
    if env.common_step_counter > after and cur != target:
        params["max_mismatch_steps"] = target
        tm.set_term_cfg(term_name, term_cfg)
        return target
    return cur


@configclass
class CurriculumCfg:
    bad_wrist_contact_mismatch_max_steps = CurrTerm(
        func=bad_wrist_contact_mismatch_max_steps_curriculum,
        params={},
    )


##
# Object trajectory loading
##

_object_trajectories: dict[str, dict] | None = None


def _load_object_trajectories_merged(
    object_folder: Path | None,
    object_terrain_folder: Path | None,
    motion_files: list[str],
) -> dict[str, dict]:
    """Load ``*_object_fitted.npz`` keyed by motion base; later folders override earlier."""
    result: dict[str, dict] = {}
    folders = [p for p in (object_folder, object_terrain_folder) if p is not None and p.is_dir()]
    for mf in motion_files:
        base = Path(mf).stem
        for obj_dir in folders:
            fitted = obj_dir / f"{base}_object_fitted.npz"
            if not fitted.is_file():
                continue
            d = np.load(str(fitted), allow_pickle=True)
            result[base] = {
                "trans": np.asarray(d["object_trans"], dtype=np.float32),
                "quat_wxyz": np.asarray(d["object_quat_wxyz"], dtype=np.float32),
                "fps": float(d["fps"]) if "fps" in d else 30.0,
            }
    return result


##
# Environment configuration
##


@configclass
class Tracking3ptBindedMultiTerrainObjectEnvCfg(ManagerBasedRLEnvCfg):
    """3pt tracking with combined terrain + object + free-space motions.

    Requires --motion_dir, --terrain_folder, and --object_folder via CLI.
    Uses ManagerBasedRLEnvWithKinematicObjects for object pose updates.

    motion_category_per_motion is filled by setup_motion_terrain_pairs / __post_init__ so the env can log
    Episode_Reward/total_reward_mean_terrain|object|free (same scale as other Episode_Reward/* metrics).
    """

    motion_dir: str = ""
    terrain_folder_path: str = ""
    object_folder_path: str = ""
    #: Optional single directory with per-motion terrain URDF, proxy URDF, ``*_object_fitted.npz``,
    #: and combined ``*_contact_nodes.npy`` (merged into terrain/proxy maps; highest priority for contacts).
    object_terrain_folder_path: str = ""
    #: Reserved. Non-zero values are ignored: global terrain prims are not moved with env_origins (would
    #: misalign). Prefer adjusting ``env_spacing`` / ``num_envs`` for layout in the viewport.
    merged_object_terrain_origin_offset_xy: tuple[float, float] = (0.0, 0.0)
    #: After this many control steps (``env.common_step_counter``), curriculum sets
    #: ``bad_wrist_contact_mismatch`` termination ``max_mismatch_steps`` to
    #: :attr:`wrist_mismatch_strict_max_mismatch_steps`.
    #: Default ``15000 * 24`` matches 15000 RSL-RL iterations when ``num_steps_per_env == 24`` (see runner cfg).
    wrist_mismatch_strict_after_env_steps: int = 15000 * 24
    wrist_mismatch_strict_max_mismatch_steps: int = 20
    motion_category_per_motion: list[int] = field(default_factory=list)
    num_envs: int = 4096
    physics_phase_flag: bool = False
    env_spacing: float = 10.0

    scene: InteractiveSceneCfg = MISSING
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def _setup_common(
        self,
        motion_files: list[str],
        terrain_resolved: Path | None,
        object_resolved: Path | None,
        object_terrain_resolved: Path | None,
        num_envs: int,
    ):
        self.commands.motion.motion_files = motion_files
        self.commands.motion.terrain_folder_path = str(terrain_resolved) if terrain_resolved is not None else ""
        self.commands.motion.object_folder_path = str(object_resolved) if object_resolved is not None else ""
        self.commands.motion.object_terrain_folder_path = (
            str(object_terrain_resolved) if object_terrain_resolved is not None else ""
        )
        global _object_trajectories
        _object_trajectories = _load_object_trajectories_merged(object_resolved, object_terrain_resolved, motion_files)
        if _object_trajectories:
            print(
                f"[INFO]: Loaded {len(_object_trajectories)} object trajectories "
                f"(folders: {[p for p in (object_resolved, object_terrain_resolved) if p is not None]})"
            )

    def setup_motion_terrain_pairs(self):
        """Build motion-terrain-object pairs. Called from train.py after CLI args are set."""
        if not self.motion_dir.strip():
            print("[WARN]: motion_dir not set. Use --motion_dir.")
            return
        ot_str = (self.object_terrain_folder_path or "").strip()
        if not (self.terrain_folder_path.strip() or self.object_folder_path.strip() or ot_str):
            print(
                "[WARN]: Set at least one of terrain_folder_path, object_folder_path, "
                "object_terrain_folder_path (--terrain_folder / --object_folder / --object_terrain_folder)."
            )
            return

        fallback_roots = [Path.cwd(), Path(__file__).parent.parent.parent.parent.parent.parent]
        motion_path = _resolve_path(Path(self.motion_dir), fallback_roots)
        terrain_path = (
            _resolve_path(Path(self.terrain_folder_path), fallback_roots)
            if self.terrain_folder_path.strip()
            else None
        )
        object_path = (
            _resolve_path(Path(self.object_folder_path), fallback_roots)
            if self.object_folder_path.strip()
            else None
        )
        ot_path = _resolve_path(Path(ot_str), fallback_roots) if ot_str else None

        if not motion_path.is_dir():
            print(f"[WARN]: Motion dir not found: {motion_path}")
            return
        for label, p in (
            ("terrain", terrain_path),
            ("object", object_path),
            ("object_terrain", ot_path),
        ):
            if p is not None and not p.is_dir():
                print(f"[WARN]: {label} dir not found: {p}")
                return

        (
            motion_files,
            terrain_urdf_paths,
            proxy_urdf_paths,
            motion_to_slot,
            motion_categories,
            _,
        ) = _build_motion_terrain_object_pairs(
            self.motion_dir,
            self.terrain_folder_path or "",
            self.object_folder_path or "",
            ot_str,
            merged_object_terrain_origin_offset_xy=self.merged_object_terrain_origin_offset_xy,
        )
        self.motion_category_per_motion = motion_categories
        num_motions = len(motion_files)
        num_slots = len(terrain_urdf_paths)
        num_envs = max(self.num_envs, num_motions)

        robot_cfg = getattr(self.scene, "robot", None)
        if robot_cfg is MISSING:
            robot_cfg = None

        self.scene = CombinedTerrainObjectSceneCfg(num_envs=num_envs, env_spacing=self.env_spacing)
        self.scene.num_envs = num_envs
        self.scene.terrain.spawn.urdf_paths = terrain_urdf_paths
        self.scene.terrain.spawn.num_envs = num_envs
        self.scene.terrain.spawn.env_spacing = self.env_spacing
        self.scene.terrain.spawn.motion_to_slot = motion_to_slot
        self.scene.proxy_object = _make_proxy_rigid_object_cfg_multi_urdf(proxy_urdf_paths)
        if robot_cfg is not None:
            self.scene.robot = robot_cfg

        if self.scene.height_scanner is not None and num_slots > 0:
            self.scene.height_scanner.mesh_prim_paths = [
                f"/World/terrains/slot_{i}" for i in range(num_slots)
            ]

        self._setup_common(motion_files, terrain_path, object_path, ot_path, num_envs)

        print(f"[INFO]: Combined: {num_slots} slots ({num_slots - 1} terrain + 1 flat), {num_motions} motions")
        print(f"[INFO]: num_envs={num_envs}, replicate_physics=False")

    def __post_init__(self):
        ot_str = (self.object_terrain_folder_path or "").strip()
        has_paths = bool(self.motion_dir.strip()) and (
            bool(self.terrain_folder_path.strip())
            or bool(self.object_folder_path.strip())
            or bool(ot_str)
        )
        if not has_paths:
            self.scene = CombinedTerrainObjectSceneCfg(num_envs=1, env_spacing=10.0)
            # Use FLAT_PLANE_MARKER so terrain spawner can create scene (requires non-empty urdf_paths)
            self.scene.terrain.spawn.urdf_paths = [FLAT_PLANE_MARKER]
            self.scene.terrain.spawn.num_envs = 1
            self.scene.proxy_object = _make_proxy_rigid_object_cfg_multi_urdf([str(_EMPTY_PROXY_URDF)])
            self.commands.motion.motion_files = []
            print(
                "[INFO]: motion_dir plus at least one of terrain_folder_path, object_folder_path, "
                "object_terrain_folder_path required. Call setup_motion_terrain_pairs() after setting paths."
            )
        else:
            fallback_roots = [Path.cwd(), Path(__file__).parent.parent.parent.parent.parent.parent]
            motion_path = _resolve_path(Path(self.motion_dir), fallback_roots)
            terrain_path = (
                _resolve_path(Path(self.terrain_folder_path), fallback_roots)
                if self.terrain_folder_path.strip()
                else None
            )
            object_path = (
                _resolve_path(Path(self.object_folder_path), fallback_roots)
                if self.object_folder_path.strip()
                else None
            )
            ot_path = _resolve_path(Path(ot_str), fallback_roots) if ot_str else None

            motion_ok = motion_path.is_dir()
            terrain_ok = terrain_path is None or terrain_path.is_dir()
            object_ok = object_path is None or object_path.is_dir()
            ot_ok = ot_path is None or ot_path.is_dir()
            has_any_asset = terrain_path is not None or object_path is not None or ot_path is not None

            if motion_ok and terrain_ok and object_ok and ot_ok and has_any_asset:
                (
                    motion_files,
                    terrain_urdf_paths,
                    proxy_urdf_paths,
                    motion_to_slot,
                    motion_categories,
                    _,
                ) = _build_motion_terrain_object_pairs(
                    self.motion_dir,
                    self.terrain_folder_path or "",
                    self.object_folder_path or "",
                    ot_str,
                    merged_object_terrain_origin_offset_xy=self.merged_object_terrain_origin_offset_xy,
                )
                self.motion_category_per_motion = motion_categories
                num_motions = len(motion_files)
                num_slots = len(terrain_urdf_paths)
                num_envs = max(self.num_envs, num_motions)

                robot_cfg = getattr(self.scene, "robot", None)
                if robot_cfg is MISSING:
                    robot_cfg = None

                self.scene = CombinedTerrainObjectSceneCfg(num_envs=num_envs, env_spacing=self.env_spacing)
                self.scene.num_envs = num_envs
                self.scene.terrain.spawn.urdf_paths = terrain_urdf_paths
                self.scene.terrain.spawn.num_envs = num_envs
                self.scene.terrain.spawn.env_spacing = self.env_spacing
                self.scene.terrain.spawn.motion_to_slot = motion_to_slot
                self.scene.proxy_object = _make_proxy_rigid_object_cfg_multi_urdf(proxy_urdf_paths)
                if robot_cfg is not None:
                    self.scene.robot = robot_cfg

                if self.scene.height_scanner is not None and num_slots > 0:
                    self.scene.height_scanner.mesh_prim_paths = [
                        f"/World/terrains/slot_{i}" for i in range(num_slots)
                    ]
                self._setup_common(motion_files, terrain_path, object_path, ot_path, num_envs)
                print(f"[INFO]: Combined: {num_slots} slots, {num_motions} motions")
            else:
                self.scene = CombinedTerrainObjectSceneCfg(num_envs=1, env_spacing=10.0)
                self.scene.terrain.spawn.urdf_paths = [FLAT_PLANE_MARKER]
                self.scene.terrain.spawn.num_envs = 1
                self.scene.proxy_object = _make_proxy_rigid_object_cfg_multi_urdf([str(_EMPTY_PROXY_URDF)])
                self.commands.motion.motion_files = []
                print("[INFO]: Paths not found. Call setup_motion_terrain_pairs() after setting paths.")

        self.decimation = 4
        self.episode_length_s = 10.0
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # Multi-terrain + proxy objects + many envs: PhysX may require large aggregate broadphase
        # buffers (PxGpuDynamicsMemoryConfig foundLostAggregatePairs / totalAggregatePairs).
        self.sim.physx.gpu_found_lost_pairs_capacity = 6_000_000
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 45_000_000
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 9_000_000
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
