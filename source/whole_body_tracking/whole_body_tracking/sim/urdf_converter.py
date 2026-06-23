# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""URDF converter that honors ``make_instanceable`` at import time.

Upstream :class:`isaaclab.sim.converters.UrdfConverter` does not forward the ``make_instanceable``
config flag to the Isaac Sim URDF import config, so collision/visual prims are always created
instanceable. Instanced collision geometry cannot be modified per-link at runtime, which breaks
contact-based collision toggling.

This subclass forwards the flag so ``make_instanceable=False`` produces non-instanced geometry.
It lives in ``whole_body_tracking`` so the Isaac Lab installation does not need to be patched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab.sim.converters import UrdfConverter
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from pxr import Usd


class InstanceableUrdfConverter(UrdfConverter):
    """URDF converter that forwards ``make_instanceable`` to the Isaac Sim import config.

    When ``cfg.make_instanceable`` is ``False`` the imported USD keeps non-instanced collision and
    visual prims, which is required for per-link collision modification at runtime.
    """

    def _get_urdf_import_config(self):
        import_config = super()._get_urdf_import_config()
        # forward the instancing flag (older Isaac Sim versions may lack this setter)
        if hasattr(import_config, "set_make_instanceable"):
            import_config.set_make_instanceable(self.cfg.make_instanceable)
        return import_config


@clone
def spawn_from_urdf_instanceable(
    prim_path: str,
    cfg: "InstanceableUrdfFileCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> "Usd.Prim":
    """Spawn an asset from a URDF file, honoring ``make_instanceable``.

    Mirrors :func:`isaaclab.sim.spawners.from_files.spawn_from_urdf` but uses
    :class:`InstanceableUrdfConverter` so the ``make_instanceable`` flag is respected.
    """
    urdf_loader = InstanceableUrdfConverter(cfg)
    return _spawn_from_usd_file(prim_path, urdf_loader.usd_path, cfg, translation, orientation)


@configclass
class InstanceableUrdfFileCfg(UrdfFileCfg):
    """URDF file spawn config that honors ``make_instanceable``.

    Drop-in replacement for :class:`~isaaclab.sim.UrdfFileCfg` that routes spawning through
    :func:`spawn_from_urdf_instanceable`.
    """

    func: Callable = spawn_from_urdf_instanceable
