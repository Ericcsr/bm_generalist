# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ray-caster sensor that supports per-environment terrain meshes.

Upstream :class:`isaaclab.sensors.RayCaster` only allows a single mesh prim and raises
``NotImplementedError`` for more. This subclass removes that restriction so each environment
can ray-cast against its own terrain slot: env ``i`` uses ``mesh_prim_paths[i % num_meshes]``.

This lives in ``whole_body_tracking`` so the Isaac Lab installation does not need to be patched.
"""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import omni.log
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.prims import XFormPrim
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster import RayCaster
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh


class MultiMeshRayCaster(RayCaster):
    """Ray-caster supporting one or more terrain meshes.

    When a single ``mesh_prim_paths`` entry is provided this behaves exactly like the base
    :class:`~isaaclab.sensors.RayCaster`. When multiple are provided, environment ``i`` ray-casts
    against ``mesh_prim_paths[i % num_meshes]``, enabling per-environment terrain height scans.
    """

    def _initialize_warp_meshes(self):
        # read prims to ray-cast (supports 1 or more meshes; when multiple, env i uses mesh i % num_meshes)
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            # check if the prim is a plane - handle PhysX plane as a special case
            # if a plane exists then we need to create an infinite mesh that is a plane
            mesh_prim = sim_utils.get_first_matching_child_prim(
                mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
            )
            # if we did not find a plane then we need to read the mesh
            if mesh_prim is None:
                # obtain the mesh prim
                mesh_prim = sim_utils.get_first_matching_child_prim(
                    mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
                )
                # check if valid
                if mesh_prim is None or not mesh_prim.IsValid():
                    raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
                # cast into UsdGeomMesh
                mesh_prim = UsdGeom.Mesh(mesh_prim)
                # read the vertices and faces
                points = np.asarray(mesh_prim.GetPointsAttr().Get())
                transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
                points = np.matmul(points, transform_matrix[:3, :3].T)
                points += transform_matrix[:3, 3]
                indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
                wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
                # print info
                omni.log.info(
                    f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(indices)} faces."
                )
            else:
                mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                # print info
                omni.log.info(f"Created infinite plane mesh prim: {mesh_prim.GetPath()}.")
            # add the warp mesh to the list
            self.meshes[mesh_prim_path] = wp_mesh

        # throw an error if no meshes are found
        if all([mesh_prim_path not in self.meshes for mesh_prim_path in self.cfg.mesh_prim_paths]):
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # obtain the poses of the sensors
        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        # note: we clone here because we are read-only operations
        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        # apply drift to ray starting position in world frame
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # check if user provided attach_yaw_only flag
        if self.cfg.attach_yaw_only is not None:
            msg = (
                "Raycaster attribute 'attach_yaw_only' property will be deprecated in a future release."
                " Please use the parameter 'ray_alignment' instead."
            )
            # set ray alignment to yaw
            if self.cfg.attach_yaw_only:
                self.cfg.ray_alignment = "yaw"
                msg += " Setting ray_alignment to 'yaw'."
            else:
                self.cfg.ray_alignment = "base"
                msg += " Setting ray_alignment to 'base'."
            # log the warning
            omni.log.warn(msg)
        # ray cast based on the sensor poses
        if self.cfg.ray_alignment == "world":
            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += self.ray_cast_drift[env_ids, 0:2]
            # no rotation is considered and directions are not rotated
            ray_starts_w = self.ray_starts[env_ids]
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "yaw":
            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += quat_apply_yaw(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            # only yaw orientation is considered and directions are not rotated
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "base":
            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += quat_apply(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            # full orientation is considered
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        else:
            raise RuntimeError(f"Unsupported ray_alignment type: {self.cfg.ray_alignment}.")

        # ray cast and store the hits
        num_meshes = len(self.cfg.mesh_prim_paths)
        if num_meshes == 1:
            self._data.ray_hits_w[env_ids] = raycast_mesh(
                ray_starts_w,
                ray_directions_w,
                max_dist=self.cfg.max_distance,
                mesh=self.meshes[self.cfg.mesh_prim_paths[0]],
            )[0]
        else:
            # Per-env mesh: env i uses mesh (i % num_meshes)
            env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor([env_ids], device=self.device)
            if env_ids_t.dim() == 0:
                env_ids_t = env_ids_t.unsqueeze(0)
            for slot_idx in range(num_meshes):
                slot_mask = (env_ids_t % num_meshes) == slot_idx
                slot_env_ids = env_ids_t[slot_mask]
                if slot_env_ids.numel() == 0:
                    continue
                mesh = self.meshes[self.cfg.mesh_prim_paths[slot_idx]]
                # Index by mask to get rays for envs in this slot
                ray_starts_slot = ray_starts_w[slot_mask]
                ray_dirs_slot = ray_directions_w[slot_mask]
                hits = raycast_mesh(
                    ray_starts_slot,
                    ray_dirs_slot,
                    max_dist=self.cfg.max_distance,
                    mesh=mesh,
                )[0]
                self._data.ray_hits_w[slot_env_ids] = hits

        # apply vertical drift to ray starting position in ray caster frame
        self._data.ray_hits_w[env_ids, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1)
