from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg, ROUGH_TERRAINS_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt import Tracking3ptEnvCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_binded import Tracking3ptBindedEnvCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_terrain import Tracking3ptTerrainEnvCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_multi_terrain import Tracking3ptMultiTerrainEnvCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_binded_multi_terrain import Tracking3ptBindedMultiTerrainEnvCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_binded_multi_terrain_by_exp import (
    Tracking3ptBindedMultiTerrainByExpEnvCfg,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_binded_multi_terrain_contact_collision import (
    Tracking3ptBindedMultiTerrainContactCollisionEnvCfg,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_binded_multi_object import (
    Tracking3ptBindedMultiObjectEnvCfg,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_binded_multi_terrain_object import (
    Tracking3ptBindedMultiTerrainObjectEnvCfg,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_binded_multi_terrain_object_handmid import (
    Tracking3ptBindedMultiTerrainObjectHandMidEnvCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import whole_body_tracking.tasks.tracking.mdp as mdp

@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

@configclass
class G1RoughEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=ROUGH_TERRAINS_CFG,
                max_init_terrain_level=10,
                collision_group=1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                    project_uvw=True,
                ),
                debug_vis=False,
            )
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

@configclass
class G1FlatEnvGraspCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.has_wrist_grasp_label = True
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

        self.events.force_push_robot = EventTerm(
            func=mdp.fake_contact_force,
            mode="interval",
            interval_range_s = (0.02, 0.02),
            params={
                "distance_threshold": 0.05,
                "contact_spring_k": 1000.0,
                "min_friction_coeff": 0.5,
                "squeeze_force_range": (5.0, 20.0),
                "command_name": "motion",
                "asset_cfg": mdp.SceneEntityCfg("robot", joint_names=[".*"]),
            }
        )
        self.observations.policy.grasp_label = ObsTerm(func=mdp.wrist_grasp_label, params={"command_name": "motion"})
        self.observations.critic.grasp_label = ObsTerm(func=mdp.wrist_grasp_label, params={"command_name": "motion"})
        self.rewards.grasp_contact = RewTerm(func=mdp.grasp_contact_reward, weight=1.0, params={
            "command_name": "motion",
            "distance_threshold": 0.05,
        })

@configclass
class G1Flat3ptEnvCfg(Tracking3ptEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1Flat3ptBindedEnvCfg(Tracking3ptBindedEnvCfg):
    """G1 3pt tracking with binded motion (each env bound to motion via env_id % num_motions)."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1Flat3ptComplianceEnvCfg(Tracking3ptEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        self.observations.policy.compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
        self.observations.policy.vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"})
        self.observations.critic.compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
        self.observations.critic.vr_3point_pos_compliant = ObsTerm(func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"})
        self.observations.critic.vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_target, params={"command_name": "motion"})

        # For all proprioception add 4 step history
        self.observations.policy.gravity_dir = ObsTerm(func=mdp.projected_gravity, params={"command_name": "motion"}, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=4)
        self.observations.policy.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=4)
        self.observations.policy.joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=4)
        self.observations.policy.joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=4)
        self.observations.policy.actions = ObsTerm(func=mdp.last_action, history_length=4)

        self.observations.critic.base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=10)
        self.observations.critic.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=10)
        self.observations.critic.joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=10)
        self.observations.critic.joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=10)
        self.observations.critic.actions = ObsTerm(func=mdp.last_action, history_length=10)

        self.events.change_compliance = EventTerm(func=mdp.change_compliance,
                                                  mode="interval",
                                                  interval_range_s=(0.02, 0.02),
                                                  params={
                                                        "command_name": "motion",
                                                        "compliance_lb": [0.0, 0.0, 0.0],
                                                        "compliance_ub": [0.05, 0.05, 0.0],
                                                        "compliance_duration": (100, 200),
                                                        "start_steps": 0
                                                  })
        self.rewards.vr_position = RewTerm(func=mdp.vr_position_relative_error_exp, 
                                           weight=2.0,
                                           params={
                                                  "command_name": "motion",
                                                  "std": 0.1,
                                           })
        
        # self.rewards.vr_orientation = RewTerm(func=mdp.vr_orientation_relative_error_exp, 
        #                                       weight=0.4,
        #                                       params={
        #                                           "command_name": "motion",
        #                                           "std": 0.4,
        #                                    })

@configclass
class G1Flat3ptBindedComplianceEnvCfg(Tracking3ptBindedEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        self.observations.policy.compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
        self.observations.policy.vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"})
        self.observations.critic.compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
        self.observations.critic.vr_3point_pos_compliant = ObsTerm(func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"})
        self.observations.critic.vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_target, params={"command_name": "motion"})

        # For all proprioception add 4 step history
        actor_history_length = 4
        critic_history_length = 10
        self.observations.policy.gravity_dir = ObsTerm(func=mdp.projected_gravity, params={"command_name": "motion"}, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=actor_history_length)
        self.observations.policy.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=actor_history_length)
        self.observations.policy.joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=actor_history_length)
        self.observations.policy.joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=actor_history_length)
        self.observations.policy.actions = ObsTerm(func=mdp.last_action, history_length=actor_history_length)

        self.observations.critic.base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=critic_history_length)
        self.observations.critic.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=critic_history_length)
        self.observations.critic.joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=critic_history_length)
        self.observations.critic.joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=critic_history_length)
        self.observations.critic.actions = ObsTerm(func=mdp.last_action, history_length=critic_history_length)

        self.events.change_compliance = EventTerm(func=mdp.change_compliance,
                                                  mode="interval",
                                                  interval_range_s=(0.02, 0.02),
                                                  params={
                                                        "command_name": "motion",
                                                        "compliance_lb": [0.0, 0.0, 0.0],
                                                        "compliance_ub": [0.05, 0.05, 0.0],
                                                        "compliance_duration": (100, 200),
                                                        "start_steps": 0
                                                  })
        # self.rewards.vr_position = RewTerm(func=mdp.vr_position_relative_error_exp, 
        #                                    weight=2.0,
        #                                    params={
        #                                           "command_name": "motion",
        #                                           "std": 0.1,
        #                                    })
        
        # self.rewards.vr_orientation = RewTerm(func=mdp.vr_orientation_relative_error_exp, 
        #                                       weight=0.4,
        #                                       params={
        #                                           "command_name": "motion",
        #                                           "std": 0.4,
        #                                    })

@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE

@configclass
class G1Flat3ptTerrainEnvCfg(Tracking3ptTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1Flat3ptMultiTerrainEnvCfg(Tracking3ptMultiTerrainEnvCfg):
    terrain_folder_path: str = "lafan_terrain/merged"

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1Flat3ptBindedMultiTerrainEnvCfg(Tracking3ptBindedMultiTerrainEnvCfg):
    """G1 3pt tracking with binded motion-terrain pairs (each env gets motion and terrain with same base name).

    Requires --motion_dir and --terrain_folder (or --terrain_folder for terrain) when running train.py.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1Flat3ptBindedMultiTerrainByExpEnvCfg(Tracking3ptBindedMultiTerrainByExpEnvCfg):
    """G1 3pt tracking with binded motion-terrain pairs grouped by exp.

    One terrain per exp group; all motions for that exp run on the same terrain.
    Requires --motion_dir and --terrain_folder (merged robot-first format with exp prefix).
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1Flat3ptBindedMultiTerrainContactCollisionEnvCfg(Tracking3ptBindedMultiTerrainContactCollisionEnvCfg):
    """G1 3pt tracking with binded motion-terrain pairs and contact-based collision.

    Uses replicate_physics=False and per-frame collision enable/disable on
    ankle_roll (feet) and wrist_yaw (hands) based on contact labels.
    Requires --motion_dir and --terrain_folder with *_contact_nodes.npy files.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1Flat3ptBindedMultiTerrainObjectEnvCfg(Tracking3ptBindedMultiTerrainObjectEnvCfg):
    """G1 3pt tracking with combined terrain + object + free-space motions.

    Requires --motion_dir and at least one of --terrain_folder, --object_folder, --object_terrain_folder.
    Terrain motions grouped by exp; object/free motions use flat slot.
    """

    physics_phase_flag: bool = True

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


def _apply_freespace_compliance(env_cfg, free_motion_categories: list[int]) -> None:
    """Apply free-space-only compliance training to a multi-terrain-object env config.

    Adds the compliance scalar observation, switches the policy VR position target to the
    compliant target, adds observation history to proprioception, and registers force-push +
    compliance-resampling events restricted to ``free_motion_categories`` (0=terrain, 1=object,
    2=free). Envs outside those categories keep ``eef_stiffness_buf``/``last_force_applied`` at 0,
    so their compliance obs is 0 and their compliant VR target equals the original target.

    Shared by all multi-terrain-object compliance variants to keep them consistent.
    """
    free_cats = free_motion_categories

    # Compliance observation + compliant VR target.
    env_cfg.observations.policy.compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
    env_cfg.observations.policy.vr_3point_pos = ObsTerm(
        func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"}
    )
    env_cfg.observations.critic.compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
    env_cfg.observations.critic.vr_3point_pos_compliant = ObsTerm(
        func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"}
    )
    env_cfg.observations.critic.vr_3point_pos = ObsTerm(
        func=mdp.vr_3point_local_target, params={"command_name": "motion"}
    )

    # Observation history on proprioception (actor: 4 steps, critic: 10 steps).
    actor_history_length = 4
    critic_history_length = 10
    env_cfg.observations.policy.gravity_dir = ObsTerm(
        func=mdp.projected_gravity,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        history_length=actor_history_length,
    )
    env_cfg.observations.policy.base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=actor_history_length
    )
    env_cfg.observations.policy.joint_pos = ObsTerm(
        func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=actor_history_length
    )
    env_cfg.observations.policy.joint_vel = ObsTerm(
        func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=actor_history_length
    )
    env_cfg.observations.policy.actions = ObsTerm(func=mdp.last_action, history_length=actor_history_length)

    env_cfg.observations.critic.base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=critic_history_length)
    env_cfg.observations.critic.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=critic_history_length)
    env_cfg.observations.critic.joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=critic_history_length)
    env_cfg.observations.critic.joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=critic_history_length)
    env_cfg.observations.critic.actions = ObsTerm(func=mdp.last_action, history_length=critic_history_length)

    # Force push on free-space envs only (others get zero force via restrict_motion_categories).
    env_cfg.events.force_push_robot = EventTerm(
        func=mdp.force_based_push,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "force_duration": [50, 100],
            "command_name": "motion",
            "asset_cfg": mdp.SceneEntityCfg("robot", joint_names=[".*"]),
            "restrict_motion_categories": free_cats,
        },
    )
    # Compliance resampling on free-space envs only (others keep stiffness 0).
    env_cfg.events.change_compliance = EventTerm(
        func=mdp.change_compliance,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "command_name": "motion",
            "compliance_lb": [0.0, 0.0, 0.0],
            "compliance_ub": [0.05, 0.05, 0.0],
            "compliance_duration": (100, 200),
            "start_steps": 0,
            "restrict_motion_categories": free_cats,
        },
    )


@configclass
class G1Flat3ptBindedMultiTerrainObjectComplianceEnvCfg(G1Flat3ptBindedMultiTerrainObjectEnvCfg):
    """Multi-terrain-object tracking combined with free-space compliance training.

    Free-space envs (motion category 2) receive an external force push and a periodically
    resampled end-effector compliance, exactly like the BindedCompliance task. Terrain (0) and
    object (1) envs get neither: their compliance stays 0 and no external force is applied, so
    their compliant VR target observation reduces to the original target.

    Observations add the compliance scalar and switch the policy VR position target to the
    compliant target, with observation history on the proprioceptive terms.

    Gym id: ``Tracking-3pt-Binded-Multi-Terrain-Object-Compliance-G1-v0``.
    """

    # Motion categories that receive force push + compliance (0=terrain, 1=object, 2=free).
    free_motion_categories: list[int] = [2]

    def __post_init__(self):
        super().__post_init__()
        _apply_freespace_compliance(self, self.free_motion_categories)


@configclass
class EmptyCurriculumCfg:
    """Curriculum with no terms (used when wrist-mismatch curriculum does not apply)."""


@configclass
class G1Flat3ptBindedMultiTerrainObjectUnilateralContactEnvCfg(G1Flat3ptBindedMultiTerrainObjectEnvCfg):
    """Multi-terrain-object variant using unilateral foot contact rewards.

    Replaces foot contact rewards so they only reward true positives:
    desired foot contact label active AND actual foot contact active.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.contact_label_contact_foot = RewTerm(
            func=mdp.unilateral_contact_label_contact_reward,
            weight=0.5,
            params={
                "command_name": "motion",
                "sensor_cfg": mdp.SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                ),
                "threshold": 1.0,
                "contact_indices": [0, 1],
            },
        )
        self.rewards.contact_label_contact_duration_foot = RewTerm(
            func=mdp.unilateral_contact_label_contact_duration_reward,
            weight=0.25,
            params={
                "command_name": "motion",
                "sensor_cfg": mdp.SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                ),
                "threshold": 1.0,
                "contact_indices": [0, 1],
                "max_duration_s": 0.5,
            },
        )


@configclass
class G1Flat3ptBindedMultiTerrainObjectUnilateralContact5SeatEnvCfg(
    G1Flat3ptBindedMultiTerrainObjectUnilateralContactEnvCfg
):
    """5-D contact labels (feet, hands, seat), unilateral foot + seat rewards, merged-folder friendly.

    - Loads ``contact_mask`` with **5** columns; 4-column files get a fifth column of ``False``.
    - Seat (column 4): unilateral reward only when the reference wants seat contact **and** either
      ``left_hip_pitch_link`` or ``right_hip_pitch_link`` has terrain contact (force threshold).
    - **Undesired contacts** exclude hip-pitch links so seated hips are not penalized.

    Train/browse like the merged terrain+object task: set ``--motion_dir``, optional
    ``--terrain_folder`` / ``--object_folder``, and ``--object_terrain_folder`` for CSV / merged exports.

    Gym id: ``Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-G1-v0``.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.contact_label_num_dims = 5
        self.rewards.contact_label_contact_seat = RewTerm(
            func=mdp.unilateral_contact_label_contact_reward_seat_support,
            weight=0.5,
            params={
                "command_name": "motion",
                "sensor_cfg": mdp.SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_hip_pitch_link", "right_hip_pitch_link"],
                ),
                "threshold": 1.0,
                "contact_label_col": 4,
            },
        )
        self.rewards.contact_label_contact_duration_seat = RewTerm(
            func=mdp.unilateral_contact_label_contact_duration_reward_seat_support,
            weight=0.25,
            params={
                "command_name": "motion",
                "sensor_cfg": mdp.SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_hip_pitch_link", "right_hip_pitch_link"],
                ),
                "threshold": 1.0,
                "contact_label_col": 4,
                "max_duration_s": 0.5,
            },
        )
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-0.1,
            params={
                "sensor_cfg": mdp.SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)"
                        r"(?!right_wrist_yaw_link$)(?!left_hip_pitch_link$)(?!right_hip_pitch_link$).+$"
                    ],
                ),
                "threshold": 1.0,
            },
        )


@configclass
class G1Flat3ptBindedMultiTerrainObjectUnilateralContact2BitEnvCfg(
    G1Flat3ptBindedMultiTerrainObjectUnilateralContactEnvCfg
):
    """Unilateral foot contact rewards + 2-bit policy/critic contact observation (8 floats).

    Same contact rewards as :class:`G1Flat3ptBindedMultiTerrainObjectUnilateralContactEnvCfg`;
    observation matches :class:`G1Flat3ptBindedMultiTerrainObject2BitContactEnvCfg`.

    Register id: ``Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-2Bit-G1-v0``.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.contact_label = ObsTerm(
            func=mdp.contact_label_env_object_2bit, params={"command_name": "motion"}
        )
        self.observations.critic.contact_label = ObsTerm(
            func=mdp.contact_label_env_object_2bit, params={"command_name": "motion"}
        )


@configclass
class G1Flat3ptBindedMultiTerrainObjectUnilateralContact5Seat2BitEnvCfg(
    G1Flat3ptBindedMultiTerrainObjectUnilateralContact5SeatEnvCfg
):
    """5-seat unilateral rewards + 2-bit contact observation.

    Observation is **10 floats** when labels have 5 columns: eight link bits plus
    ``[seat_env, seat_obj]`` (seat_obj is 0; terrain/chair seating).

    Register id: ``Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-2Bit-G1-v0``.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.contact_label = ObsTerm(
            func=mdp.contact_label_env_object_2bit, params={"command_name": "motion"}
        )
        self.observations.critic.contact_label = ObsTerm(
            func=mdp.contact_label_env_object_2bit, params={"command_name": "motion"}
        )


@configclass
class G1Flat3ptBindedMultiTerrainObjectUnilateralContact5Seat2BitComplianceEnvCfg(
    G1Flat3ptBindedMultiTerrainObjectUnilateralContact5Seat2BitEnvCfg
):
    """5-seat 2-bit unilateral-contact task combined with free-space compliance training.

    Keeps the 5-seat unilateral contact rewards and the 10-float 2-bit contact observation, and
    adds free-space-only compliance: free-space envs (motion category 2) get an external force
    push and resampled end-effector compliance, while terrain (0) and object (1) envs get neither
    (compliance stays 0, compliant VR target equals the original target). Observation history is
    added to proprioception.

    Register id: ``Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-2Bit-Compliance-G1-v0``.
    """

    # Motion categories that receive force push + compliance (0=terrain, 1=object, 2=free).
    free_motion_categories: list[int] = [2]

    def __post_init__(self):
        super().__post_init__()
        _apply_freespace_compliance(self, self.free_motion_categories)


@configclass
class G1Flat3ptBindedMultiTerrainObjectUnilateralContact5Seat2BitNoContactObsBaselineEnvCfg(
    G1Flat3ptBindedMultiTerrainObjectUnilateralContact5Seat2BitEnvCfg
):
    """5-seat 2-bit baseline with contact labels removed from actor and critic observations.

    Contact rewards, terminations, and curriculum are unchanged from
    ``Unilateral-Contact-5Seat-2Bit``.

    Register id: ``Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-2Bit-NoContactObs-Baseline-G1-v0``.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.contact_label = None
        self.observations.critic.contact_label = None


@configclass
class G1Flat3ptBindedMultiTerrainObjectNoContactBaselineEnvCfg(
    G1Flat3ptBindedMultiTerrainObjectUnilateralContact5Seat2BitEnvCfg
):
    """5-seat 2-bit baseline with contact labels/rewards/termination disabled.

    The observation keeps the current 10-float 2-bit contact slot but fills it with zeros.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.contact_label = ObsTerm(
            func=mdp.contact_label_env_object_2bit_zeroed, params={"command_name": "motion"}
        )
        self.observations.critic.contact_label = ObsTerm(
            func=mdp.contact_label_env_object_2bit_zeroed, params={"command_name": "motion"}
        )
        self.rewards.contact_label_contact_foot.weight = 0.0
        self.rewards.contact_label_contact_wrist.weight = 0.0
        self.rewards.contact_label_contact_duration_foot.weight = 0.0
        self.rewards.contact_label_contact_duration_wrist.weight = 0.0
        self.rewards.contact_label_contact_seat.weight = 0.0
        self.rewards.contact_label_contact_duration_seat.weight = 0.0
        if hasattr(self.terminations, "bad_wrist_contact_mismatch"):
            delattr(self.terminations, "bad_wrist_contact_mismatch")
        self.curriculum = EmptyCurriculumCfg()


@configclass
class G1Flat3ptBindedMultiTerrainObjectUnilateralContactNoFootContactBaselineEnvCfg(
    G1Flat3ptBindedMultiTerrainObjectUnilateralContact5Seat2BitEnvCfg
):
    """5-seat 2-bit baseline with foot contact labels/rewards disabled.

    Wrist and seat contact labels, rewards, termination, and curriculum follow the current
    ``Unilateral-Contact-5Seat-2Bit`` config.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.contact_label_contact_foot.weight = 0.0
        self.rewards.contact_label_contact_duration_foot.weight = 0.0
        self.observations.policy.contact_label = ObsTerm(
            func=mdp.contact_label_env_object_2bit_feet_zeroed, params={"command_name": "motion"}
        )
        self.observations.critic.contact_label = ObsTerm(
            func=mdp.contact_label_env_object_2bit_feet_zeroed, params={"command_name": "motion"}
        )


@configclass
class G1Flat3ptBindedMultiTerrainObject2BitContactEnvCfg(G1Flat3ptBindedMultiTerrainObjectEnvCfg):
    """Same as multi-terrain-object, but contact label observation is 2-bit per link.

    Output layout:
    [L_foot_env, L_foot_obj, R_foot_env, R_foot_obj, L_wrist_env, L_wrist_obj, R_wrist_env, R_wrist_obj].
    With 5-column reference labels, :func:`mdp.contact_label_env_object_2bit` appends ``seat_env``, ``seat_obj``.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.contact_label = ObsTerm(
            func=mdp.contact_label_env_object_2bit, params={"command_name": "motion"}
        )
        self.observations.critic.contact_label = ObsTerm(
            func=mdp.contact_label_env_object_2bit, params={"command_name": "motion"}
        )


@configclass
class G1Flat3ptBindedMultiTerrainObjectMergedEnvCfg(G1Flat3ptBindedMultiTerrainObjectEnvCfg):
    """Same as :class:`G1Flat3ptBindedMultiTerrainObjectEnvCfg` but intended for mixed datasets with
    ``--object_terrain_folder`` (CSV hand-object / single-tree exports) alongside split terrain/object dirs.

    Register id: ``Tracking-3pt-Binded-Multi-Terrain-Object-Merged-G1-v0``.
    """


@configclass
class G1Flat3ptBindedMultiTerrainObjectHandMidEnvCfg(Tracking3ptBindedMultiTerrainObjectHandMidEnvCfg):
    """G1: object position from reference wrist midpoint; orientation from ``*_object_fitted.npz``."""

    physics_phase_flag: bool = True

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1Flat3ptBindedMultiTerrainObjectCriticCategoryEnvCfg(G1Flat3ptBindedMultiTerrainObjectEnvCfg):
    """Adds privileged critic one-hot ``[terrain_or_free, object]`` for value conditioning (Approach A)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.critic.motion_bucket_onehot = ObsTerm(
            func=mdp.motion_bucket_onehot_for_critic, params={"command_name": "motion"}
        )


@configclass
class G1Flat3ptBindedMultiTerrainObjectMergedCriticCategoryEnvCfg(G1Flat3ptBindedMultiTerrainObjectMergedEnvCfg):
    """Merged dataset variant with critic motion-bucket one-hot. See :class:`G1Flat3ptBindedMultiTerrainObjectCriticCategoryEnvCfg`."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.critic.motion_bucket_onehot = ObsTerm(
            func=mdp.motion_bucket_onehot_for_critic, params={"command_name": "motion"}
        )


@configclass
class G1Flat3ptBindedMultiTerrainObjectHandMidCriticCategoryEnvCfg(G1Flat3ptBindedMultiTerrainObjectHandMidEnvCfg):
    """Hand-mid object variant with critic motion-bucket one-hot. See :class:`G1Flat3ptBindedMultiTerrainObjectCriticCategoryEnvCfg`."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.critic.motion_bucket_onehot = ObsTerm(
            func=mdp.motion_bucket_onehot_for_critic, params={"command_name": "motion"}
        )


@configclass
class G1Flat3ptBindedMultiObjectEnvCfg(Tracking3ptBindedMultiObjectEnvCfg):
    """G1 3pt tracking with binded motion-object pairs (kinematic proxy objects).

    Requires --motion_dir and --object_folder (pointing to omomo_object/{exp_group}).
    Object collides only with wrists; magic force disabled when both wrists contact.
    """

    physics_phase_flag: bool = True  # Disable magic force when both wrists contact object

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]