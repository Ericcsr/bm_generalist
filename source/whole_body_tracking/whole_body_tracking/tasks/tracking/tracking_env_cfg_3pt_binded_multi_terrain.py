from __future__ import annotations

import re
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from whole_body_tracking.sensors import MultiMeshRayCasterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.terrain_mesh_spawner_cfg import FLAT_PLANE_MARKER, MultiTerrainMeshSpawnerCfg

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


@configclass
class MultiTerrainSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot and multiple mesh terrains from URDF files.

    Each environment will get a different terrain selected from a list of URDF files
    based on the environment ID. Envs sharing the same terrain/motion share position
    (inter-env collision disabled). Uses replicate_physics=True with terrain at global
    paths (/World/terrains/slot_*) for FAR-tracking style performance.
    """

    replicate_physics: bool = True
    """Terrain at global paths enables replicate_physics for faster clone and enable_env_ids collision filtering."""

    # terrain at global paths /World/terrains/slot_0..slot_{n-1} (FAR-tracking style).
    # Enables replicate_physics=True; enable_env_ids handles robot-robot collisions; terrain collides with all.
    terrain: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/terrains/slot_0",
        collision_group=-1,
        spawn=MultiTerrainMeshSpawnerCfg(
            urdf_paths=[],  # Will be set in __post_init__
            env_spacing=6.0,  # Separate terrains and robots in grid (0 = clustered at origin)
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.4, 0.3)),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                kinematic_enabled=True,  # Static terrain
            ),
        ),
    )

    # robots
    robot: ArticulationCfg = MISSING
    # lights
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
    # 2m x 2m height map around each robot (21x21 grid at 0.1m resolution)
    # MultiMeshRayCasterCfg supports per-env terrain slots (env i -> mesh i % num_slots).
    height_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
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
    """Command specifications for the MDP (binded: each env bound to a motion via env_id % num_pairs)."""

    motion = mdp.BindedMultiMotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        adaptive_uniform_ratio=0.25,
        debug_vis=True,
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
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

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
        contact_labels = ObsTerm(func=mdp.contact_label, params={"command_name": "motion"})

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
        contact_labels = ObsTerm(func=mdp.contact_label, params={"command_name": "motion"})
        

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
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

    # Envs sharing same terrain/motion share position (slot-based).
    # With replicate_physics=True, enable_env_ids handles robot-robot collision filtering;
    # terrain at /World/terrains/slot_* collides with all robots.
    set_slot_based_env_origins = EventTerm(
        func=mdp.set_slot_based_env_origins,
        mode="startup",
        params={},
    )

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(1.0, 3.0),
    #     params={"velocity_range": VELOCITY_RANGE},
    # )

    # # force push robot
    # force_push_robot = EventTerm(
    #     func=mdp.force_based_push,
    #     mode="interval",
    #     interval_range_s=(0.02, 0.02),
    #     params={
    #         "force_duration": [50, 100],
    #         "command_name": "motion",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

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
    # Encourages correct links (feet, hands) to make contact per contact labels.
    # No-op (returns 0) when contact_labels not loaded; active with contact-collision task.
    contact_label_contact = RewTerm(
        func=mdp.contact_label_contact_reward,
        weight=0.5,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            ),
            "threshold": 1.0,
        },
    )

    # Duration bonus for sustained contact when reference expects it (feet + wrists).
    # Includes foot-terrain contact for terrain tracking tasks.
    contact_label_contact_duration = RewTerm(
        func=mdp.contact_label_contact_duration_reward,
        weight=0.25,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            ),
            "threshold": 1.0,
            "max_duration_s": 0.5,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

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


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


def _resolve_path(path: Path, fallback_roots: list[Path]) -> Path:
    """Resolve a path, trying fallback roots if not found."""
    if path.is_absolute() and path.exists():
        return path.resolve()
    for root in fallback_roots:
        candidate = root / path
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def _terrain_fallback_key(motion_base: str) -> str:
    """Return terrain URDF base name for motion when using one-terrain-per-exp layout.

    motion_0 -> terrain (single exp)
    obstacles1_subject1_rf_motion_0 -> obstacles1_subject1_rf_terrain (merged exp with prefix)
    """
    m = re.match(r"^(.+)_motion_\d+$", motion_base)
    return f"{m.group(1)}_terrain" if m else "terrain"


def _build_motion_terrain_pairs(motion_dir: str, terrain_folder_path: str) -> tuple[list[str], list[str]]:
    """Build paired lists of motion and terrain paths by matching base names (without extension).

    Pairs motion_dir/*.npz with terrain_folder/*.urdf where base names match.
    When no per-motion URDF exists, falls back to shared terrain.urdf per exp:
    - motion_0 -> terrain.urdf (single exp)
    - obstacles1_subject1_rf_motion_0 -> obstacles1_subject1_rf_terrain.urdf (merged)
    Motions without a matching terrain use FLAT_PLANE_MARKER (spawner creates default flat terrain).
    Returns (motion_files, urdf_paths) in matching order.
    """
    fallback_roots = [
        Path.cwd(),
        Path(__file__).parent.parent.parent.parent.parent.parent,  # bm_generalist root
    ]

    motion_path = _resolve_path(Path(motion_dir), fallback_roots)
    terrain_path = _resolve_path(Path(terrain_folder_path), fallback_roots)

    if not motion_path.exists():
        raise FileNotFoundError(f"Motion directory not found: {motion_path}")

    # Build map: base_name -> motion_file
    motion_by_base: dict[str, Path] = {}
    for f in motion_path.glob("*.npz"):
        motion_by_base[f.stem] = f

    if not motion_by_base:
        raise FileNotFoundError(f"No .npz motion files found in {motion_path}")

    # Build map: base_name -> terrain_file (empty if terrain folder missing or no matches)
    terrain_by_base: dict[str, Path] = {}
    if terrain_folder_path.strip() and terrain_path.exists():
        for f in terrain_path.glob("*.urdf"):
            terrain_by_base[f.stem] = f

    # Include ALL motions; use matching terrain, fallback to shared terrain.urdf per exp, or FLAT_PLANE_MARKER
    all_bases = sorted(motion_by_base.keys())
    motion_files = [str(motion_by_base[b].resolve()) for b in all_bases]
    urdf_paths = []
    for b in all_bases:
        if b in terrain_by_base:
            urdf_paths.append(str(terrain_by_base[b].resolve()))
        else:
            fallback = _terrain_fallback_key(b)
            if fallback in terrain_by_base:
                urdf_paths.append(str(terrain_by_base[fallback].resolve()))
            else:
                urdf_paths.append(FLAT_PLANE_MARKER)

    num_flat = sum(1 for u in urdf_paths if u == FLAT_PLANE_MARKER)
    if num_flat:
        print(f"[INFO]: {num_flat} motion(s) use default flat terrain (no matching .urdf)")

    return motion_files, urdf_paths


def _exp_prefix(motion_base: str) -> str:
    """Extract exp prefix from motion base. obstacles1_subject1_rf_motion_0 -> obstacles1_subject1_rf, motion_0 -> ''."""
    m = re.match(r"^(.+)_motion_\d+$", motion_base)
    return m.group(1) if m else ""


def _build_motion_terrain_pairs_by_exp(
    motion_dir: str, terrain_folder_path: str
) -> tuple[list[str], list[str], list[int]]:
    """Build motion-terrain pairs grouped by exp: one terrain per exp, all motions for that exp share it.

    Returns (motion_files, urdf_paths, motion_to_slot).
    - motion_files: all motions sorted by exp then motion number
    - urdf_paths: one terrain per exp (unique)
    - motion_to_slot: motion_idx -> slot_idx (0..num_slots-1)
    """
    fallback_roots = [
        Path.cwd(),
        Path(__file__).parent.parent.parent.parent.parent.parent,
    ]
    motion_path = _resolve_path(Path(motion_dir), fallback_roots)
    terrain_path = _resolve_path(Path(terrain_folder_path), fallback_roots)

    if not motion_path.exists():
        raise FileNotFoundError(f"Motion directory not found: {motion_path}")

    motion_by_base = {f.stem: f for f in motion_path.glob("*.npz")}
    if not motion_by_base:
        raise FileNotFoundError(f"No .npz motion files found in {motion_path}")

    terrain_by_base: dict[str, Path] = {}
    if terrain_folder_path.strip() and terrain_path.exists():
        for f in terrain_path.glob("*.urdf"):
            terrain_by_base[f.stem] = f

    # Group motions by exp
    exp_to_motions: dict[str, list[str]] = {}
    for base in motion_by_base:
        exp = _exp_prefix(base)
        if exp not in exp_to_motions:
            exp_to_motions[exp] = []
        exp_to_motions[exp].append(base)

    # Sort motions within each exp by motion number
    def _motion_num(b: str) -> int:
        m = re.search(r"motion_(\d+)$", b)
        return int(m.group(1)) if m else 0

    for exp in exp_to_motions:
        exp_to_motions[exp].sort(key=_motion_num)

    # Unique exps in deterministic order
    exps_ordered = sorted(exp_to_motions.keys(), key=lambda e: (e or "\x00"))

    # Build motion_files, motion_to_slot, then urdf_paths
    motion_files: list[str] = []
    motion_to_slot: list[int] = []
    exp_to_slot: dict[str, int] = {exp: idx for idx, exp in enumerate(exps_ordered)}

    for exp in exps_ordered:
        slot = exp_to_slot[exp]
        for base in exp_to_motions[exp]:
            motion_files.append(str(motion_by_base[base].resolve()))
            motion_to_slot.append(slot)

    # urdf_paths: one per exp (slot order)
    urdf_paths = []
    for exp in exps_ordered:
        terrain_key = f"{exp}_terrain" if exp else "terrain"
        if terrain_key in terrain_by_base:
            urdf_paths.append(str(terrain_by_base[terrain_key].resolve()))
        else:
            urdf_paths.append(FLAT_PLANE_MARKER)

    return motion_files, urdf_paths, motion_to_slot


##
# Environment configuration
##


@configclass
class Tracking3ptBindedMultiTerrainEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for 3pt tracking with binded motion-terrain pairs.

    Each environment uses:
    - Motion from motion_dir (e.g. run1.npz)
    - Terrain from terrain_folder_path (e.g. run1.urdf)
    Pairs are matched by base name (without extension). Env i gets pair (env_id % num_pairs).
    When num_envs > num_pairs, terrains and motions wrap around; each env has matched motion-terrain.
    """

    motion_dir: str = ""
    terrain_folder_path: str = ""
    num_envs: int = 4096
    """Number of environments. Must be >= num_motions; extra envs wrap around (env i -> pair i % num_pairs)."""

    env_spacing: float = 10.0
    """Spacing between terrain/robot slots in meters. 0 = all at origin (FAR-tracking). >0 = separated for visualization."""

    scene: InteractiveSceneCfg = MISSING
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def setup_motion_terrain_pairs(self):
        """Build motion-terrain pairs from motion_dir and terrain_folder_path.

        Call this after setting motion_dir and terrain_folder_path (e.g. from CLI).
        Sets commands.motion.motion_files, scene.terrain.spawn.urdf_paths, and scene.num_envs.
        num_envs >= num_pairs; wrap around when num_envs > num_pairs (env i -> pair i % num_pairs).
        """
        motion_files, urdf_paths = _build_motion_terrain_pairs(
            self.motion_dir,
            self.terrain_folder_path,
        )
        num_pairs = len(motion_files)
        num_envs = max(self.num_envs, num_pairs)
        self.commands.motion.motion_files = motion_files
        fallback_roots = [
            Path.cwd(),
            Path(__file__).parent.parent.parent.parent.parent.parent,
        ]
        resolved_terrain = _resolve_path(Path(self.terrain_folder_path), fallback_roots)
        self.commands.motion.terrain_folder_path = str(resolved_terrain) if resolved_terrain.exists() else self.terrain_folder_path
        self.scene.terrain.spawn.urdf_paths = urdf_paths
        self.scene.terrain.spawn.num_envs = num_envs
        self.scene.terrain.spawn.env_spacing = self.scene.env_spacing
        self.scene.num_envs = num_envs
        # Per-env height map: cast each env against its slot's terrain mesh
        if self.scene.height_scanner is not None and num_pairs > 0:
            self.scene.height_scanner.mesh_prim_paths = [
                f"/World/terrains/slot_{i}" for i in range(num_pairs)
            ]
        print(f"[INFO]: Paired {num_pairs} motion-terrain files (name-matched)")
        print(f"[INFO]: Motions: {[Path(p).name for p in motion_files]}")
        print(f"[INFO]: Terrains: {[Path(p).name for p in urdf_paths]}")
        print(f"[INFO]: num_envs={num_envs} (wrap around: env i -> pair i % {num_pairs})")

    def __post_init__(self):
        """Post initialization."""
        has_paths = bool(self.motion_dir.strip()) and bool(self.terrain_folder_path.strip())
        if not has_paths:
            # Paths must be provided via CLI; use placeholder until setup_motion_terrain_pairs() is called
            self.scene = MultiTerrainSceneCfg(num_envs=1, env_spacing=10.0)
            self.scene.terrain.spawn.urdf_paths = []
            self.scene.terrain.spawn.num_envs = 1
            self.commands.motion.motion_files = []
            print(
                "[INFO]: motion_dir and terrain_folder_path required (e.g. --motion_dir, --terrain_folder). "
                "Call setup_motion_terrain_pairs() after setting paths."
            )
        else:
            fallback_roots = [
                Path.cwd(),
                Path(__file__).parent.parent.parent.parent.parent.parent,  # bm_generalist root
            ]
            motion_path = _resolve_path(Path(self.motion_dir), fallback_roots)
            terrain_path = _resolve_path(Path(self.terrain_folder_path), fallback_roots)

            if motion_path.exists() and terrain_path.exists():
                motion_files, urdf_paths = _build_motion_terrain_pairs(
                    self.motion_dir,
                    self.terrain_folder_path,
                )
                num_pairs = len(motion_files)
                num_envs = max(self.num_envs, num_pairs)
                self.scene = MultiTerrainSceneCfg(num_envs=num_envs, env_spacing=self.env_spacing)
                self.scene.terrain.spawn.urdf_paths = urdf_paths
                self.scene.terrain.spawn.num_envs = num_envs
                self.scene.terrain.spawn.env_spacing = self.env_spacing
                self.commands.motion.motion_files = motion_files
                self.commands.motion.terrain_folder_path = str(terrain_path)
                if self.scene.height_scanner is not None and num_pairs > 0:
                    self.scene.height_scanner.mesh_prim_paths = [
                        f"/World/terrains/slot_{i}" for i in range(num_pairs)
                    ]
                print(f"[INFO]: Paired {num_pairs} motion-terrain files (name-matched)")
                print(f"[INFO]: Motions: {[Path(p).name for p in motion_files]}")
                print(f"[INFO]: Terrains: {[Path(p).name for p in urdf_paths]}")
                print(f"[INFO]: num_envs={num_envs} (wrap around: env i -> pair i % {num_pairs})")
            else:
                self.scene = MultiTerrainSceneCfg(num_envs=1, env_spacing=self.env_spacing)
                self.scene.terrain.spawn.urdf_paths = []
                self.scene.terrain.spawn.num_envs = 1
                self.commands.motion.motion_files = []
                print(
                    f"[INFO]: Motion/terrain paths not found (motion_dir={self.motion_dir}, "
                    f"terrain_folder={self.terrain_folder_path}). "
                    "Call setup_motion_terrain_pairs() after setting paths (e.g. from CLI)."
                )

        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # update height scanner at policy frequency
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # Use IsaacLab default (2**26) for gpu_collision_stack_size to reduce GPU memory
        # Increase to 2**28 only if you see collision stack overflow errors
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # Increase for multi-terrain + many envs to avoid "foundLostPairsCapacity" / "totalAggregatePairsCapacity" errors
        self.sim.physx.gpu_found_lost_pairs_capacity = 4_000_000
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 4_200_000
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
