# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the multi-mesh ray-cast sensor."""

from __future__ import annotations

from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.utils import configclass

from .multi_mesh_ray_caster import MultiMeshRayCaster


@configclass
class MultiMeshRayCasterCfg(RayCasterCfg):
    """Configuration for :class:`MultiMeshRayCaster`.

    Identical to :class:`~isaaclab.sensors.RayCasterCfg` except that ``mesh_prim_paths`` may
    contain more than one entry. When a single path is provided, all envs ray-cast against that
    mesh. When multiple paths are provided, env ``i`` ray-casts against
    ``mesh_prim_paths[i % num_paths]`` (for per-env terrain).
    """

    class_type: type = MultiMeshRayCaster
