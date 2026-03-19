"""3pt tracking with binded motion-terrain pairs grouped by exp.

One terrain per exp group; all motions for that exp run on the same terrain.
Env origins and collision groups are set so envs sharing a terrain share position.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from .tracking_env_cfg_3pt_binded_multi_terrain import (
    MultiTerrainSceneCfg,
    _build_motion_terrain_pairs_by_exp,
    _resolve_path,
    Tracking3ptBindedMultiTerrainEnvCfg,
)


@configclass
class Tracking3ptBindedMultiTerrainByExpEnvCfg(Tracking3ptBindedMultiTerrainEnvCfg):
    """3pt tracking with binded motion-terrain pairs grouped by exp.

    Creates num_exp_groups terrains. Each terrain has all motions for that exp.
    Env i runs motion (i % num_motions); env origin = slot for that motion's exp.
    Collision groups: robots sharing same slot don't collide; terrain collides with all.
    """

    def setup_motion_terrain_pairs(self):
        """Build motion-terrain pairs by exp group. One terrain per exp, all motions share it."""
        motion_files, urdf_paths, motion_to_slot = _build_motion_terrain_pairs_by_exp(
            self.motion_dir,
            self.terrain_folder_path,
        )
        num_motions = len(motion_files)
        num_slots = len(urdf_paths)
        num_envs = max(self.num_envs, num_motions)

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
        self.scene.terrain.spawn.motion_to_slot = motion_to_slot
        self.scene.num_envs = num_envs

        if self.scene.height_scanner is not None and num_slots > 0:
            self.scene.height_scanner.mesh_prim_paths = [
                f"/World/terrains/slot_{i}" for i in range(num_slots)
            ]

        print(f"[INFO]: By-exp: {num_slots} terrains, {num_motions} motions")
        print(f"[INFO]: Motions: {[Path(p).name for p in motion_files]}")
        print(f"[INFO]: Terrains: {[Path(p).name for p in urdf_paths]}")
        print(f"[INFO]: num_envs={num_envs}, env i -> slot motion_to_slot[i % {num_motions}]")

    def __post_init__(self):
        """Post initialization. Uses by-exp pairing (one terrain per exp, motions share)."""
        has_paths = bool(self.motion_dir.strip()) and bool(self.terrain_folder_path.strip())
        if not has_paths:
            self.scene = MultiTerrainSceneCfg(num_envs=1, env_spacing=10.0)
            self.scene.terrain.spawn.urdf_paths = []
            self.scene.terrain.spawn.num_envs = 1
            self.commands.motion.motion_files = []
            print(
                "[INFO]: motion_dir and terrain_folder_path required. "
                "Call setup_motion_terrain_pairs() after setting paths."
            )
        else:
            fallback_roots = [
                Path.cwd(),
                Path(__file__).parent.parent.parent.parent.parent.parent,
            ]
            motion_path = _resolve_path(Path(self.motion_dir), fallback_roots)
            terrain_path = _resolve_path(Path(self.terrain_folder_path), fallback_roots)

            if motion_path.exists() and terrain_path.exists():
                motion_files, urdf_paths, motion_to_slot = _build_motion_terrain_pairs_by_exp(
                    self.motion_dir,
                    self.terrain_folder_path,
                )
                num_motions = len(motion_files)
                num_slots = len(urdf_paths)
                num_envs = max(self.num_envs, num_motions)
                self.scene = MultiTerrainSceneCfg(num_envs=num_envs, env_spacing=self.env_spacing)
                self.scene.terrain.spawn.urdf_paths = urdf_paths
                self.scene.terrain.spawn.num_envs = num_envs
                self.scene.terrain.spawn.env_spacing = self.env_spacing
                self.scene.terrain.spawn.motion_to_slot = motion_to_slot
                self.commands.motion.motion_files = motion_files
                self.commands.motion.terrain_folder_path = str(terrain_path)
                if self.scene.height_scanner is not None and num_slots > 0:
                    self.scene.height_scanner.mesh_prim_paths = [
                        f"/World/terrains/slot_{i}" for i in range(num_slots)
                    ]
                print(f"[INFO]: By-exp: {num_slots} terrains, {num_motions} motions")
                print(f"[INFO]: num_envs={num_envs}, env i -> slot motion_to_slot[i % {num_motions}]")
            else:
                self.scene = MultiTerrainSceneCfg(num_envs=1, env_spacing=self.env_spacing)
                self.scene.terrain.spawn.urdf_paths = []
                self.scene.terrain.spawn.num_envs = 1
                self.commands.motion.motion_files = []
                print(
                    f"[INFO]: Motion/terrain paths not found. "
                    "Call setup_motion_terrain_pairs() after setting paths."
                )

        # General settings (same as Tracking3ptBindedMultiTerrainEnvCfg)
        self.decimation = 4
        self.episode_length_s = 10.0
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_found_lost_pairs_capacity = 4_000_000
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 4_200_000
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
