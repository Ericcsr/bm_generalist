"""Same as tracking_env_cfg_3pt_binded_multi_terrain_object, but proxy object position follows
the midpoint of the two **VR 3-point** hand targets (``vr_3point_body`` / ``vr_3point_body_offset``,
same convention as ``vr_3point_local_target``); orientation still comes from ``*_object_fitted.npz``
(same time-warping as the original).

Requires ``MultiMotionCommand`` VR 3-point fields (default: first two ``vr_3point_body`` links are hands).

Usage (train.py):
    --task Tracking-3pt-Binded-Multi-Terrain-Object-HandMid-G1-v0
    --motion_dir <dir> --terrain_folder <dir> --object_folder <dir>
    # Optional: --object_terrain_folder <merged_dir> for mixed CSV hand-object + split assets
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp

from whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_binded_multi_terrain_object import (
    Tracking3ptBindedMultiTerrainObjectEnvCfg,
    EventCfg as _BaseEventCfg,
)


@configclass
class EventCfgHandMidObjectPos(_BaseEventCfg):
    """Object pose: position = reference wrist midpoint; orientation = fitted trajectory."""

    init_object_poses = EventTerm(
        func=mdp.update_kinematic_object_poses,
        mode="startup",
        params={"command_name": "motion", "object_position_from_hand_midpoint": True},
    )
    reset_object_poses = EventTerm(
        func=mdp.reset_object_poses_direct,
        mode="reset",
        params={"command_name": "motion", "object_position_from_hand_midpoint": True},
    )
    update_object_poses = EventTerm(
        func=mdp.update_magic_force_object_poses,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "command_name": "motion",
            "object_position_from_hand_midpoint": True,
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
class Tracking3ptBindedMultiTerrainObjectHandMidEnvCfg(Tracking3ptBindedMultiTerrainObjectEnvCfg):
    """Combined terrain + object + free motions; object translation from hand-midpoint reference."""

    events: EventCfgHandMidObjectPos = EventCfgHandMidObjectPos()
