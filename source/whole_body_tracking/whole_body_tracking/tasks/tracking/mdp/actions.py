"""Action terms for whole-body tracking tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm

from whole_body_tracking.tasks.tracking.mdp.events import update_kinematic_object_poses

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class UpdateKinematicObjectPosesAction(ActionTerm):
    """Action term that updates kinematic proxy object poses before each physics step.

    Runs update_kinematic_object_poses inside apply_actions, which is called before
    write_data_to_sim and sim.step in each decimation iteration. This ensures the
    proxy object poses are correct when physics runs.

    Has action_dim=0 so it consumes no policy output.
    """

    cfg: actions_cfg.UpdateKinematicObjectPosesActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.UpdateKinematicObjectPosesActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """No-op; this term consumes no actions."""
        pass

    def apply_actions(self):
        """Update kinematic proxy object poses before physics step."""
        update_kinematic_object_poses(
            self._env,
            env_ids=None,
            command_name=self.cfg.command_name,
        )
