"""Contact-based collision config for 3pt binded multi-terrain tracking.

All simulation tricks for contact-based collision (replicate_physics=False,
make_instanceable=False, make_uninstanceable, set_contact_link_collision) are
contained only in this config. Use Tracking-3pt-Binded-Multi-Terrain-Contact-Collision-G1-v0
when training with per-frame contact labels for feet and hands.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp

from .tracking_env_cfg_3pt_binded_multi_terrain import (
    EventCfg,
    Tracking3ptBindedMultiTerrainEnvCfg,
    _build_motion_terrain_pairs,
    _resolve_path,
)


@configclass
class EventCfgContactCollision(EventCfg):
    """Event config with contact-based collision events.

    Extends base EventCfg with make_uninstanceable_contact_robot and
    set_contact_link_collision. These require replicate_physics=False
    and make_instanceable=False (applied in setup_motion_terrain_pairs).
    """

    # contact-based collision: when geometry is instanced, make_uninstanceable + sim.reset() once on reset
    # (URDF converter may create instanced geometry despite make_instanceable=False)
    make_uninstanceable_contact_robot = EventTerm(
        func=mdp.make_uninstanceable_contact_robot,
        mode="reset",
        params={"command_name": "motion"},
    )

    # contact-based collision: enable/disable ankle_roll (feet) and wrist_yaw (hands) per frame
    # Interval 0.1s (10 Hz) reduces modify_collision_properties cost.
    set_contact_link_collision = EventTerm(
        func=mdp.set_contact_link_collision,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            "command_name": "motion",
            "enable_contact_label_debug_vis": False,
            "max_contact_label_vis_envs": 4,
        },
    )


@configclass
class Tracking3ptBindedMultiTerrainContactCollisionEnvCfg(Tracking3ptBindedMultiTerrainEnvCfg):
    """3pt binded multi-terrain with contact-based collision enabled.

    Uses replicate_physics=False and make_instanceable=False so that
    per-frame modify_collision_properties works on ankle_roll and wrist_yaw.
    Loads contact labels from terrain_folder (*_contact_nodes.npy).
    """

    events: EventCfgContactCollision = EventCfgContactCollision()

    def setup_motion_terrain_pairs(self):
        """Build motion-terrain pairs and apply contact-based collision settings.

        Always applies replicate_physics=False, make_instanceable=False,
        and enable_contact_based_collision=True for the motion command.
        """
        motion_files, urdf_paths = _build_motion_terrain_pairs(
            self.motion_dir,
            self.terrain_folder_path,
        )
        num_pairs = len(motion_files)
        num_envs = max(self.num_envs, num_pairs)
        self.commands.motion.motion_files = motion_files
        self.commands.motion.enable_contact_based_collision = True
        fallback_roots = [
            Path.cwd(),
            Path(__file__).parent.parent.parent.parent.parent.parent,
        ]
        resolved_terrain = _resolve_path(Path(self.terrain_folder_path), fallback_roots)
        self.commands.motion.terrain_folder_path = (
            str(resolved_terrain) if resolved_terrain.exists() else self.terrain_folder_path
        )

        # Contact-based collision: must disable replicate_physics and instancing
        self.scene.replicate_physics = False
        self.scene.robot = self.scene.robot.replace(
            spawn=self.scene.robot.spawn.replace(make_instanceable=False)
        )
        print(
            "[ContactLabels] Set replicate_physics=False, make_instanceable=False "
            "(required for contact-based collision)"
        )

        self.scene.terrain.spawn.urdf_paths = urdf_paths
        self.scene.terrain.spawn.num_envs = num_envs
        self.scene.terrain.spawn.env_spacing = self.scene.env_spacing
        self.scene.num_envs = num_envs
        if self.scene.height_scanner is not None and num_pairs > 0:
            self.scene.height_scanner.mesh_prim_paths = [
                f"/World/terrains/slot_{i}" for i in range(num_pairs)
            ]
        print(f"[INFO]: Paired {num_pairs} motion-terrain files (name-matched)")
        print(f"[INFO]: Motions: {[Path(p).name for p in motion_files]}")
        print(f"[INFO]: Terrains: {[Path(p).name for p in urdf_paths]}")
        print(f"[INFO]: num_envs={num_envs} (wrap around: env i -> pair i % {num_pairs})")
