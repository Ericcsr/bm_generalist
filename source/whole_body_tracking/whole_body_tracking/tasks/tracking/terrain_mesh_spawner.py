# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom terrain mesh spawner that parses URDF files and uses create_prim_from_mesh."""

from __future__ import annotations

import functools
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import trimesh
from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.sim.utils import clone, find_matching_prim_paths

if TYPE_CHECKING:
    from .terrain_mesh_spawner_cfg import TerrainMeshSpawnerCfg
else:
    TerrainMeshSpawnerCfg = None

# Marker used in urdf_paths when a motion has no matching terrain; spawner uses flat plane.
FLAT_PLANE_MARKER = "__FLAT_PLANE__"

# Module-level caches to avoid repeated work when num_envs >> num_terrains:
# - Parse URDF once per unique path instead of ~400x
# - Load mesh once per unique terrain instead of ~400x
_URDF_MESH_INFO_CACHE: dict[str, tuple[str, tuple[float, float, float]]] = {}
_MESH_CACHE: dict[str, trimesh.Trimesh] = {}


def spawn_multi_terrain_per_env(func):
    """Spawn only num_slots terrain prims total. Envs sharing terrain/motion are placed
    together; inter-env collision is disabled via collision groups.

    When prim_path starts with /World/terrains: spawn at global paths
    /World/terrains/slot_0..slot_{num_slots-1} (FAR-tracking style, enables replicate_physics=True).
    Otherwise: spawn at env_0..env_{num_slots-1} under env hierarchy (replicate_physics=False).
    """
    @functools.wraps(func)
    def wrapper(prim_path: str, cfg, *args, **kwargs):
        import carb
        from pxr import Gf, Sdf
        import isaacsim.core.utils.stage as stage_utils
        import isaacsim.core.utils.prims as prim_utils

        prim_path = str(prim_path)
        if not prim_path.startswith("/"):
            raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")
        root_path, asset_path = prim_path.rsplit("/", 1)
        num_envs = getattr(cfg, "num_envs", None)
        if num_envs is None:
            raise RuntimeError("terrain spawn requires num_envs on spawn config.")
        num_slots = len(getattr(cfg, "urdf_paths", None) or [])
        if num_slots == 0:
            raise ValueError(
                "terrain spawn requires non-empty urdf_paths. "
                "Call setup_motion_terrain_pairs() or set motion_dir/terrain_folder_path before creating the env."
            )
        env_spacing = getattr(cfg, "env_spacing", 0.0)
        use_global_paths = prim_path.startswith("/World/terrains")
        stage = stage_utils.get_current_stage()
        if use_global_paths:
            base = "/World/terrains"
            # Ensure parent scope exists (Sdf.CreatePrimInLayer may require it)
            if not stage.GetPrimAtPath(base):
                prim_utils.create_prim(base, "Scope")
            slot_dest_paths = [f"{base}/slot_{i}" for i in range(num_slots)]
        else:
            base = re.sub(r"/env_.*$", "", root_path)
            slot_dest_paths = [f"{base}/env_{i}/{asset_path}" for i in range(num_slots)]

        template_path = stage_utils.get_next_free_path("/World/Template")
        prim_utils.create_prim(template_path, "Scope")

        proto_prim_paths = []
        for slot_idx in range(num_slots):
            proto_path = f"{template_path}/Terrain_{slot_idx:04d}"
            prim = func(proto_path, cfg, slot_idx=slot_idx, *args, **kwargs)
            proto_prim_paths.append(proto_path)

        with Sdf.ChangeBlock():
            for slot_idx in range(num_slots):
                dest_path = slot_dest_paths[slot_idx]
                env_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), dest_path)
                Sdf.CopySpec(env_spec.layer, Sdf.Path(proto_prim_paths[slot_idx]), env_spec.layer, Sdf.Path(dest_path))

        prim_utils.delete_prim(template_path)

        # Position terrains when using global paths (must match set_slot_based_env_origins)
        # Use num_slots for grid so terrains stay compact near origin (not num_envs which can be 4096+)
        if use_global_paths:
            from pxr import UsdGeom
            import numpy as np
            num_rows = int(np.ceil(num_slots / int(np.sqrt(num_slots))))
            num_cols = int(np.ceil(num_slots / num_rows))
            for slot_idx in range(num_slots):
                row = slot_idx // num_cols
                col = slot_idx % num_cols
                x = -(row - (num_rows - 1) / 2) * env_spacing
                y = (col - (num_cols - 1) / 2) * env_spacing
                dest_prim = stage.GetPrimAtPath(slot_dest_paths[slot_idx])
                if dest_prim:
                    xform = UsdGeom.Xformable(dest_prim)
                    xform.ClearXformOpOrder()
                    xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))

        if not use_global_paths:
            carb.settings.get_settings().set_bool("/isaaclab/spawn/multi_assets", True)
        print(
            f"[INFO]: Terrain spawn: {num_slots} terrain prims at "
            + (f"/World/terrains/slot_0..{num_slots-1} (global, replicate_physics=True)" if use_global_paths
              else f"env_0..env_{num_slots-1} (env i -> terrain at env_(i % {num_slots}))"),
            flush=True,
        )
        return prim_utils.get_prim_at_path(slot_dest_paths[0])
    return wrapper


def parse_urdf_mesh_info(urdf_path: str) -> tuple[str, tuple[float, float, float]]:
    """Parse URDF file to extract mesh path and scale.
    
    Args:
        urdf_path: Path to the URDF file
        
    Returns:
        Tuple of (mesh_path, scale) where:
        - mesh_path: Path to the mesh file (relative to URDF directory)
        - scale: Tuple of (x, y, z) scale values
        
    Raises:
        ValueError: If mesh information cannot be found in URDF
    """
    urdf_file = Path(urdf_path)
    if not urdf_file.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    
    # Parse URDF XML
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Find mesh element in visual geometry
    mesh_path = None
    scale = (1.0, 1.0, 1.0)  # Default scale
    
    # Search in visual geometry first (as per URDF format)
    for visual in root.findall(".//visual"):
        geometry = visual.find("geometry")
        if geometry is not None:
            mesh_elem = geometry.find("mesh")
            if mesh_elem is not None:
                # Get mesh filename
                filename_attr = mesh_elem.get("filename")
                if filename_attr:
                    mesh_path = filename_attr
                    # Get scale if present
                    scale_attr = mesh_elem.get("scale")
                    if scale_attr:
                        # Parse scale string like "0.75 0.75 0.75"
                        scale_values = scale_attr.split()
                        if len(scale_values) == 3:
                            scale = tuple(float(s) for s in scale_values)
                    break
    
    if mesh_path is None:
        raise ValueError(f"Could not find mesh element in URDF file: {urdf_path}")
    
    # Resolve mesh path relative to URDF directory
    urdf_dir = urdf_file.parent
    mesh_file = urdf_dir / mesh_path
    
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
    
    return str(mesh_file.resolve()), scale


def extract_env_id_from_prim_path(prim_path: str) -> int:
    """Extract environment ID from prim path.
    
    Args:
        prim_path: Prim path like "/World/envs/env_0/Terrain" or "/World/envs/env_123/Terrain"
        
    Returns:
        Environment ID as integer
    """
    # Match patterns like env_0, env_123, etc.
    match = re.search(r"env_(\d+)", prim_path)
    if match:
        return int(match.group(1))
    else:
        # Fallback: try to extract number from path
        # This handles edge cases where naming might differ
        raise ValueError(f"Could not extract environment ID from prim_path: {prim_path}")


@clone
def spawn_terrain_from_urdf(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn terrain mesh from URDF file using create_prim_from_mesh.
    
    This function:
    1. Parses the URDF file to extract mesh path and scale
    2. Loads the mesh using trimesh
    3. Applies scale to the mesh
    4. Creates prim using create_prim_from_mesh with material properties
    
    Args:
        prim_path: Path to spawn the terrain prim at
        cfg: Configuration for terrain mesh spawner
        translation: Translation of the terrain (unused, kept for API compatibility)
        orientation: Orientation of the terrain (unused, kept for API compatibility)
        
    Returns:
        The created USD prim
    """
    # Parse URDF to get mesh path and scale
    mesh_path, urdf_scale = parse_urdf_mesh_info(cfg.urdf_path)
    
    # Load mesh using trimesh
    mesh = trimesh.load(mesh_path, process=False)
    
    # Handle Scene objects from multi-mesh files
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded object is not a valid Trimesh: {type(mesh)}")
    
    # Apply scale from URDF
    if urdf_scale != (1.0, 1.0, 1.0):
        # Scale vertices
        import numpy as np
        mesh.vertices = mesh.vertices * np.array(urdf_scale)
    
    # Configure visual material
    visual_material = cfg.visual_material if cfg.visual_material is not None else sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.5, 0.4, 0.3)
    )
    
    # Configure physics material
    physics_material = cfg.physics_material if cfg.physics_material is not None else sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    
    # Create the terrain mesh prim
    create_prim_from_mesh(
        prim_path,
        mesh,
        visual_material=visual_material,
        physics_material=physics_material,
        translation=translation,
        orientation=orientation,
    )
    
    if cfg.rigid_props is not None:
        sim_utils.define_rigid_body_properties(prim_path, cfg.rigid_props)
    
    import isaacsim.core.utils.prims as prim_utils
    return prim_utils.get_prim_at_path(prim_path)


@spawn_multi_terrain_per_env
def spawn_terrain_from_urdf_list(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    slot_idx: int | None = None,
) -> Usd.Prim:
    """Spawn terrain mesh from a list of URDF files.

    When called by spawn_multi_terrain_per_env, slot_idx is provided and we spawn
    one terrain per slot. Otherwise falls back to env_id from prim_path.
    """
    urdf_paths = getattr(cfg, "urdf_paths", None)
    if not urdf_paths:
        raise ValueError("urdf_paths list is empty in config")

    if slot_idx is not None:
        terrain_idx = slot_idx
    else:
        terrain_idx = extract_env_id_from_prim_path(prim_path) % len(urdf_paths)
    selected_urdf_path = urdf_paths[terrain_idx]

    # Use cached mesh to avoid reloading/parsing 400+ times when num_envs >> num_terrains
    cache_key = selected_urdf_path
    if cache_key in _MESH_CACHE:
        mesh = _MESH_CACHE[cache_key]
    else:
        # Default flat plane when motion has no matching terrain
        if selected_urdf_path == FLAT_PLANE_MARKER:
            # trimesh has no plane(); use thin box (20m x 20m x 0.01m) as flat terrain
            mesh = trimesh.creation.box(extents=[20.0, 20.0, 0.01])
        else:
            # Parse URDF (cached) to get mesh path and scale
            if selected_urdf_path in _URDF_MESH_INFO_CACHE:
                mesh_path, urdf_scale = _URDF_MESH_INFO_CACHE[selected_urdf_path]
            else:
                mesh_path, urdf_scale = parse_urdf_mesh_info(selected_urdf_path)
                _URDF_MESH_INFO_CACHE[selected_urdf_path] = (mesh_path, urdf_scale)

            # Load mesh using trimesh
            mesh = trimesh.load(mesh_path, process=False)

            # Handle Scene objects from multi-mesh files
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            if urdf_scale != (1.0, 1.0, 1.0):
                import numpy as np
                mesh.vertices = mesh.vertices * np.array(urdf_scale)

        _MESH_CACHE[cache_key] = mesh

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded object is not a valid Trimesh: {type(mesh)}")
    
    visual_material = cfg.visual_material if cfg.visual_material is not None else sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.5, 0.4, 0.3)
    )
    
    physics_material = cfg.physics_material if cfg.physics_material is not None else sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    
    create_prim_from_mesh(
        prim_path,
        mesh,
        visual_material=visual_material,
        physics_material=physics_material,
        translation=translation,
        orientation=orientation,
    )
    
    if cfg.rigid_props is not None:
        sim_utils.define_rigid_body_properties(prim_path, cfg.rigid_props)
    
    import isaacsim.core.utils.prims as prim_utils
    return prim_utils.get_prim_at_path(prim_path)


@spawn_multi_terrain_per_env
def spawn_object_from_urdf_list(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    slot_idx: int | None = None,
) -> Usd.Prim:
    """Spawn rigid object from a list of URDF files (supports box/cylinder/mesh geometry).

    Unlike spawn_terrain_from_urdf_list which requires mesh geometry, this function
    uses IsaacLab's UrdfConverter to handle any URDF geometry type (box, cylinder, mesh).
    Used for spawning proxy objects (thin-plate URDFs) in multi-object environments.
    """
    import isaacsim.core.utils.prims as prim_utils
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    urdf_paths = getattr(cfg, "urdf_paths", None)
    if not urdf_paths:
        raise ValueError("urdf_paths list is empty in config")

    if slot_idx is not None:
        obj_idx = slot_idx
    else:
        obj_idx = extract_env_id_from_prim_path(prim_path) % len(urdf_paths)
    selected_urdf = urdf_paths[obj_idx]

    if selected_urdf == FLAT_PLANE_MARKER:
        prim_utils.create_prim(prim_path, "Xform")
        return prim_utils.get_prim_at_path(prim_path)

    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=selected_urdf,
        fix_base=False,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    )
    urdf_converter = UrdfConverter(urdf_converter_cfg)
    usd_path = urdf_converter.usd_path

    prim_utils.create_prim(
        prim_path,
        usd_path=usd_path,
        translation=translation,
        orientation=orientation,
    )

    if cfg.rigid_props is not None:
        sim_utils.define_rigid_body_properties(prim_path, cfg.rigid_props)

    return prim_utils.get_prim_at_path(prim_path)
