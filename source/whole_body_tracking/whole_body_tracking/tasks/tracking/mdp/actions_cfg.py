"""Action term configurations for whole-body tracking tasks."""

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import actions


@configclass
class UpdateKinematicObjectPosesActionCfg(ActionTermCfg):
    """Configuration for UpdateKinematicObjectPosesAction.

    Updates kinematic proxy object poses before each physics step.
    Uses asset_name='robot' only to satisfy ActionTerm base class; the term
    does not modify the robot.
    """

    class_type: type[ActionTerm] = actions.UpdateKinematicObjectPosesAction
    asset_name: str = "robot"
    command_name: str = "motion"
