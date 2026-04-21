# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for terrain mesh spawner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.utils import configclass

from .terrain_mesh_spawner import FLAT_PLANE_MARKER, spawn_terrain_from_urdf, spawn_terrain_from_urdf_list, spawn_object_from_urdf_list


@configclass
class TerrainMeshSpawnerCfg(RigidObjectSpawnerCfg):
    """Configuration for spawning terrain from URDF file using create_prim_from_mesh.
    
    This spawner parses URDF files to extract mesh path and scale, then uses
    create_prim_from_mesh to create terrain with proper material properties.
    """
    
    func: Callable[..., Usd.Prim] = spawn_terrain_from_urdf
    """Function to use for spawning the terrain mesh."""
    
    urdf_path: str = MISSING
    """Path to the URDF file containing terrain mesh information."""
    
    visual_material: sim_utils.VisualMaterialCfg | None = None
    """Visual material properties. Defaults to PreviewSurfaceCfg with brown color."""
    
    physics_material: sim_utils.RigidBodyMaterialCfg | None = None
    """Physics material properties. Defaults to RigidBodyMaterialCfg with friction=1.0, restitution=0.0."""


@configclass
class MultiTerrainMeshSpawnerCfg(RigidObjectSpawnerCfg):
    """Configuration for spawning terrain from multiple URDF files.
    
    This spawner selects a URDF file from a list based on the environment ID,
    allowing different environments to have different terrains. The selection
    uses modulo arithmetic to cycle through available terrains.
    
    When replicate_physics=True, only env_0 exists at spawn time, so find_matching_prim_paths
    would return a single path. num_envs must be set to generate all env paths explicitly.
    """
    
    func: Callable[..., Usd.Prim] = spawn_terrain_from_urdf_list
    """Function to use for spawning the terrain mesh."""
    
    urdf_paths: list[str] = MISSING
    """List of paths to URDF files. Each environment will get a terrain based on its ID modulo len(urdf_paths)."""
    
    num_envs: int | None = None
    """Number of environments. Required when replicate_physics=True (default) since only env_0 exists at spawn time.
    When set, paths are generated as /World/envs/env_0/Terrain, env_1/Terrain, etc. instead of using find_matching_prim_paths."""

    env_spacing: float | None = None
    """Env spacing for terrain slot positions when using global paths (/World/terrains/slot_*).
    Used with GridCloner layout. env_spacing=0 puts all at origin (FAR-tracking style)."""

    motion_to_slot: list[int] | None = None
    """Optional mapping: motion_idx -> slot_idx. When set (by-exp layout), env i gets slot motion_to_slot[i % num_motions].
    Enables multiple motions per terrain (one terrain per exp group)."""

    motion_extra_origin_xy: list[tuple[float, float]] | None = None
    """Unused with global ``/World/terrains/slot_*`` layout: slot meshes are placed once on a grid, while
    per-motion env_origin offsets would shift robots without moving terrain (misalignment). Kept for API
    stability; :func:`set_slot_based_env_origins` does not consume this field."""

    visual_material: sim_utils.VisualMaterialCfg | None = None
    """Visual material properties. Defaults to PreviewSurfaceCfg with brown color."""
    
    physics_material: sim_utils.RigidBodyMaterialCfg | None = None
    """Physics material properties. Defaults to RigidBodyMaterialCfg with friction=1.0, restitution=0.0."""


@configclass
class MultiObjectSpawnerCfg(RigidObjectSpawnerCfg):
    """Configuration for spawning objects from multiple URDF files (supports box/cylinder geometry).

    Uses IsaacLab's UrdfConverter instead of the mesh-only terrain spawner,
    so it handles arbitrary URDF geometry types (box, cylinder, mesh).
    """

    func: Callable[..., Usd.Prim] = spawn_object_from_urdf_list

    urdf_paths: list[str] = MISSING
    """List of URDF paths. Each slot gets a URDF based on slot_idx modulo len(urdf_paths)."""

    num_envs: int | None = None
    """Number of environments."""

    env_spacing: float | None = None
    """Env spacing for object slot positions."""

    visual_material: sim_utils.VisualMaterialCfg | None = None

    physics_material: sim_utils.RigidBodyMaterialCfg | None = None
