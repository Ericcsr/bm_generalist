"""Environment config for 3pt tracking with binded motion-object pairs.

Each environment is bound to a motion clip and optionally a kinematic proxy object.
- Object-interaction motions: paired with {motion_name}_thin_plate_proxy.urdf and
  {motion_name}_object_fitted.npz in object_folder (from export_omomo_object_to_bm_generalist.sh).
- Non-object motions: no matching proxy in object_folder; uses empty_proxy.urdf
  (invisible, no collision) so motion_dir can mix both types.

Each env loads its own URDF from object_folder: {motion_name}_thin_plate_proxy.urdf when
present, else empty_proxy.urdf. Uses Isaac Lab MultiAssetSpawnerCfg (random_choice=False).
See: https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html

The proxy is kinematic (no gravity, follows trajectory); collision with the robot is
enabled for object motions so the policy can learn to interact with it.

Usage (train.py):
    --task Tracking-3pt-Binded-Multi-Object-G1-v0
    --motion_dir omomo_box          # can contain both object and non-object motions
    --object_folder omomo_object/omomo_box
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING, field
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.terrain_mesh_spawner_cfg import FLAT_PLANE_MARKER

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


def _make_proxy_rigid_object_cfg(asset_path: str) -> RigidObjectCfg:
    """Build proxy RigidObjectCfg for a single URDF path (legacy)."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ProxyObject",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=asset_path,
            fix_base=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ),
    )


def _make_proxy_rigid_object_cfg_multi_urdf(
    urdf_paths: list[str], use_magic_force: bool = True
) -> RigidObjectCfg:
    """Build proxy RigidObjectCfg with MultiAssetSpawnerCfg for per-env URDFs.

    Each env i gets URDF at urdf_paths[i % len(urdf_paths)] (random_choice=False).
    URDF naming: {motion_name}_thin_plate_proxy.urdf in object_folder.

    When use_magic_force=True (default), proxy is dynamic and driven by PD magic force
    to track trajectory while resisting perturbations. When False, proxy is kinematic.
    """
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
            activate_contact_sensors=True,  # Required for wrist-object contact detection (standalone)
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
            activate_contact_sensors=True,  # Required for object_contact_* sensors (contact reporter API)
            mass_props=sim_utils.MassPropertiesCfg(
                density=400.0
            ),  # Match standalone: prevents object flying off when magic force turns off
        ),
    )


@configclass
class MultiObjectSceneCfgNoProxy(InteractiveSceneCfg):
    """Scene config without proxy object (used when paths not set or proxy URDF missing)."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = MISSING
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


@configclass
class MultiObjectSceneCfgWithProxy(MultiObjectSceneCfgNoProxy):
    """Scene config with kinematic proxy object. Set proxy_object to multi-URDF config before use."""

    # Aligned with standalone_magic_force_hand_object.py for contact sensor updates
    lazy_sensor_update: bool = False

    proxy_object: RigidObjectCfg = field(
        default_factory=lambda: _make_proxy_rigid_object_cfg(""),
    )
    # Hand-object contact only (excludes hand-hand); used for physics_phase activation.
    # Use ProxyObject/.* (not proxy_base/collisions): thin_plate URDF has collision on
    # left_plate, right_plate, virtual_link, not proxy_base. ProxyObject/.* matches all.
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
    # Object-centric sensors (sensor on object, listens for wrists); used for contact debug prints.
    # Matches standalone_magic_force_hand_object.py _print_contact_events.
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


# Path to empty proxy URDF for motions without object interaction (no matching thin_plate_proxy.urdf)
_EMPTY_PROXY_URDF = Path(__file__).resolve().parent / "empty_proxy.urdf"


def _build_proxy_urdf_paths(object_folder: Path, motion_files: list[str]) -> list[str]:
    """Build URDF paths: {object_folder}/{motion_stem}_thin_plate_proxy.urdf per motion.

    For motions without a matching proxy URDF in object_folder, uses empty_proxy.urdf
    (no collision, invisible) so non-object-interaction motions can be mixed in motion_dir.
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
        print(f"[INFO]: {num_empty} motion(s) have no matching proxy URDF (using empty_proxy, no object)")
    return result


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
        # Wrist contact label dropout: randomly zero out wrist contact labels for some envs.
        # When enabled, affects observation, reward, termination, and physics phase.
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
    # Hand (wrist) friction 5x higher for better grasp on proxy object
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

    # Must run before sim.reset() so PhysX compiles with correct collision filters.
    # Use prestartup (not startup) to match standalone_magic_force_hand_object.py.
    set_proxy_robot_collision_filter = EventTerm(
        func=mdp.set_proxy_robot_collision_filter,
        mode="prestartup",
        params={},
    )

    # Softer contact for hand links only (compliant contact stiffness)
    # apply_hand_compliant_contact = EventTerm(
    #     func=mdp.apply_hand_compliant_contact,
    #     mode="startup",
    #     params={
    #         "compliant_contact_stiffness": 5e3,
    #         "compliant_contact_damping": 50.0,
    #     },
    # )

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
            "physics_phase_contact_threshold": 0.1,  # More sensitive: switch off magic force at 0.1 N (was 1.0)
            "use_pose_velocity_control": True,  # Avoid cudaErrorIllegalAddress with replicate_physics=False + multi-GPU
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
    contact_label_contact = RewTerm(
        func=mdp.contact_label_contact_reward,
        weight=0.5,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            ),
            "threshold": 1.0,
            "contact_indices": [2, 3],  # hands only (0=L_foot, 1=R_foot, 2=L_wrist, 3=R_wrist)
        },
    )
    contact_label_contact_duration = RewTerm(
        func=mdp.contact_label_contact_duration_reward,
        weight=0.25,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            ),
            "threshold": 1.0,
            "contact_indices": [2, 3],
            "max_duration_s": 0.5,  # full bonus when contact held for 0.5s
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
    # Object dropped to floor during physics_phase (only when physics_phase_flag=True).
    # require_wrist_contact_label: only terminate when reference says wrist should be in contact.
    object_dropped = DoneTerm(
        func=mdp.object_dropped_to_floor,
        params={
            "floor_height_threshold": 0.1,
            "command_name": "motion",
            "require_wrist_contact_label": True,
        },
    )
    # Consecutive mismatch between reference wrist contact labels and actual wrist contact (forces).
    # Same convention as contact_label_contact_reward; both wrists must match each step or counter resets.
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
        },
    )


def bad_wrist_contact_mismatch_max_steps_curriculum(
    env,
    env_ids: Sequence[int],
) -> int | None:
    """After ``cfg.wrist_mismatch_strict_after_env_steps``, set ``max_mismatch_steps`` on the wrist term.

    Same semantics as ``tracking_env_cfg_3pt_binded_multi_terrain_object``: uses
    ``termination_manager.get_term_cfg`` / ``set_term_cfg`` for compatibility across Isaac Lab versions.

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
# Object trajectory loading and kinematic update
##

_object_trajectories: dict[str, dict] | None = None


def _load_object_trajectories(object_folder: str, motion_files: list[str]) -> dict[str, dict]:
    """Load fitted object trajectories keyed by motion base name.

    For each motion file, looks for {base_name}_object_fitted.npz in object_folder.
    Returns {base_name: {"trans": (T,3), "quat_wxyz": (T,4), "fps": float}}.
    """
    result: dict[str, dict] = {}
    obj_dir = Path(object_folder)
    if not obj_dir.exists():
        print(f"[KinObj] Object folder does not exist: {obj_dir}")
        return result
    for mf in motion_files:
        base = Path(mf).stem
        fitted = obj_dir / f"{base}_object_fitted.npz"
        if not fitted.exists():
            continue
        d = np.load(str(fitted), allow_pickle=True)
        result[base] = {
            "trans": np.asarray(d["object_trans"], dtype=np.float32),
            "quat_wxyz": np.asarray(d["object_quat_wxyz"], dtype=np.float32),
            "fps": float(d["fps"]) if "fps" in d else 30.0,
        }
    if len(result) == 0:
        print(
            f"[KinObj] No *_object_fitted.npz found in {obj_dir} for motions: "
            f"{[Path(mf).stem for mf in motion_files[:5]]}{'...' if len(motion_files) > 5 else ''}"
        )
    return result


def _resolve_path(path: Path, fallback_roots: list[Path]) -> Path:
    if path.is_absolute() and path.exists():
        return path.resolve()
    for root in fallback_roots:
        candidate = root / path
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def _build_motion_object_pairs(motion_dir: str, object_folder_path: str) -> tuple[list[str], list[str]]:
    """Build paired lists of motion and object URDF paths by matching base names.

    Motions without a matching URDF use FLAT_PLANE_MARKER (spawner creates flat plane placeholder).
    """
    fallback_roots = [
        Path.cwd(),
        Path(__file__).parent.parent.parent.parent.parent.parent,
    ]

    motion_path = _resolve_path(Path(motion_dir), fallback_roots)
    object_path = _resolve_path(Path(object_folder_path), fallback_roots)

    if not motion_path.exists():
        raise FileNotFoundError(f"Motion directory not found: {motion_path}")

    motion_by_base: dict[str, Path] = {}
    for f in motion_path.glob("*.npz"):
        motion_by_base[f.stem] = f

    if not motion_by_base:
        raise FileNotFoundError(f"No .npz motion files found in {motion_path}")

    object_by_base: dict[str, Path] = {}
    if object_folder_path.strip() and object_path.exists():
        for f in object_path.glob("*.urdf"):
            object_by_base[f.stem] = f

    all_bases = sorted(motion_by_base.keys())
    motion_files = [str(motion_by_base[b].resolve()) for b in all_bases]
    urdf_paths = [
        str(object_by_base[b].resolve()) if b in object_by_base else FLAT_PLANE_MARKER
        for b in all_bases
    ]

    num_flat = sum(1 for u in urdf_paths if u == FLAT_PLANE_MARKER)
    if num_flat:
        print(f"[INFO]: {num_flat} motion(s) have no matching object URDF (flat plane placeholder)")

    return motion_files, urdf_paths


##
# Environment configuration
##


@configclass
class Tracking3ptBindedMultiObjectEnvCfg(ManagerBasedRLEnvCfg):
    """3pt tracking with binded motion-object pairs.

    Each env gets a motion clip + kinematic proxy object. Each env loads its own
    URDF from object_folder: {motion_name}_thin_plate_proxy.urdf (MultiAssetSpawnerCfg).
    Each env's object follows its own trajectory from {base_name}_object_fitted.npz.

    Requires --motion_dir and --object_folder via CLI.
    """

    motion_dir: str = ""
    object_folder_path: str = ""
    #: After this many control steps (``env.common_step_counter``), curriculum sets
    #: ``bad_wrist_contact_mismatch`` termination ``max_mismatch_steps`` to
    #: :attr:`wrist_mismatch_strict_max_mismatch_steps`.
    #: Default ``15000 * 24`` matches 15000 RSL-RL iterations when ``num_steps_per_env == 24``.
    wrist_mismatch_strict_after_env_steps: int = 15000 * 24
    wrist_mismatch_strict_max_mismatch_steps: int = 20
    num_envs: int = 4096
    physics_phase_flag: bool = False
    magic_force_use_pose_velocity_control: bool = True  # Avoid cudaErrorIllegalAddress with replicate_physics=False

    env_spacing: float = 10.0

    scene: InteractiveSceneCfg = MISSING
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def _setup_common(self, motion_files: list[str], obj_folder_resolved: Path, num_envs: int):
        self.commands.motion.motion_files = motion_files
        self.commands.motion.terrain_folder_path = str(obj_folder_resolved)
        global _object_trajectories
        _object_trajectories = _load_object_trajectories(str(obj_folder_resolved), motion_files)
        print(f"[INFO]: Loaded {len(_object_trajectories)} object trajectories from {obj_folder_resolved}")

    def setup_motion_terrain_pairs(self):
        """Build motion-object pairs. Called from train.py after CLI args are set."""
        obj_folder = self.object_folder_path
        if not obj_folder.strip():
            print("[WARN]: object_folder_path not set. Use --object_folder to specify.")
            return

        fallback_roots = [Path.cwd(), Path(__file__).parent.parent.parent.parent.parent.parent]
        resolved_obj = _resolve_path(Path(obj_folder), fallback_roots)
        motion_path = _resolve_path(Path(self.motion_dir), fallback_roots)
        motion_files = sorted(str(f.resolve()) for f in motion_path.glob("*.npz"))
        if not motion_files:
            print(f"[WARN]: No .npz files found in {motion_path}")
            return
        # When num_envs < num motions, only load first N motions/objects
        if len(motion_files) > self.num_envs:
            motion_files = motion_files[: self.num_envs]
            print(f"[INFO]: Loaded first {len(motion_files)} motions (num_envs={self.num_envs} < total available)")
        num_pairs = len(motion_files)
        num_envs = max(self.num_envs, num_pairs)

        # Preserve robot config before replacing scene (subclasses like G1Flat3ptBindedMultiObjectEnvCfg set it)
        robot_cfg = getattr(self.scene, "robot", None)
        if robot_cfg is MISSING:
            robot_cfg = None

        urdf_paths = _build_proxy_urdf_paths(resolved_obj, motion_files)
        self.scene = MultiObjectSceneCfgWithProxy(num_envs=num_envs, env_spacing=self.env_spacing)
        self.scene.num_envs = num_envs
        self.scene.replicate_physics = False
        if robot_cfg is not None:
            self.scene.robot = robot_cfg
        self.scene.proxy_object = _make_proxy_rigid_object_cfg_multi_urdf(urdf_paths)
        self._setup_common(motion_files, resolved_obj, num_envs)

        print(f"[INFO]: Paired {num_pairs} motions with per-motion URDFs: {[Path(p).name for p in urdf_paths[:3]]}{'...' if len(urdf_paths) > 3 else ''}")
        print(f"[INFO]: Motions: {[Path(p).name for p in motion_files]}")
        print(f"[INFO]: num_envs={num_envs}, replicate_physics=False")

    def __post_init__(self):
        obj_folder = self.object_folder_path
        has_paths = bool(self.motion_dir.strip()) and bool(obj_folder.strip())
        if not has_paths:
            self.scene = MultiObjectSceneCfgNoProxy(num_envs=1, env_spacing=10.0)
            self.commands.motion.motion_files = []
            print(
                "[INFO]: motion_dir and object_folder required (--motion_dir, --object_folder). "
                "Call setup_motion_terrain_pairs() after setting paths."
            )
        else:
            fallback_roots = [Path.cwd(), Path(__file__).parent.parent.parent.parent.parent.parent]
            motion_path = _resolve_path(Path(self.motion_dir), fallback_roots)
            object_path = _resolve_path(Path(obj_folder), fallback_roots)

            if motion_path.exists() and object_path.exists():
                motion_files = sorted(str(f.resolve()) for f in motion_path.glob("*.npz"))
                if motion_files and len(motion_files) > self.num_envs:
                    motion_files = motion_files[: self.num_envs]
                    print(f"[INFO]: Loaded first {len(motion_files)} motions (num_envs={self.num_envs} < total available)")
                num_pairs = len(motion_files) if motion_files else 1
                num_envs = max(self.num_envs, num_pairs)
                urdf_paths = _build_proxy_urdf_paths(object_path, motion_files) if motion_files else []
                if not urdf_paths:
                    raise FileNotFoundError(
                        f"No motions found in {motion_path}. motion_dir must contain .npz files."
                    )
                self.scene = MultiObjectSceneCfgWithProxy(num_envs=num_envs, env_spacing=self.env_spacing)
                self.scene.replicate_physics = False
                self.scene.proxy_object = _make_proxy_rigid_object_cfg_multi_urdf(urdf_paths)
                self._setup_common(motion_files, object_path, num_envs)
                print(f"[INFO]: Paired {num_pairs} motions with per-motion URDFs: {[Path(p).name for p in urdf_paths[:3]]}{'...' if len(urdf_paths) > 3 else ''}")
            else:
                self.scene = MultiObjectSceneCfgNoProxy(num_envs=1, env_spacing=10.0)
                self.commands.motion.motion_files = []
                print(f"[INFO]: Paths not found. Call setup_motion_terrain_pairs() after setting paths.")

        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_ccd = True  # Prevent object tunneling through thin wrist links (standalone)
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_found_lost_pairs_capacity = 6_000_000  # PhysX requires >= 5.7M for multi-object; was 4M
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 6_500_000
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
