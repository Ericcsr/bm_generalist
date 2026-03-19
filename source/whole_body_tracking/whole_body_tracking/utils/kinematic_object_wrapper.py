"""Wrapper to update kinematic proxy object poses before each step.

The interval event in ManagerBasedRLEnv runs after physics, so object poses
are written too late. This wrapper calls update_kinematic_object_poses before
env.step(), ensuring objects are at the correct pose when physics runs
(like the standalone_kinematic_object_robot.py example).
"""

import gymnasium as gym


def _get_manager_env(env):
    """Unwrap to ManagerBasedEnv (handles DirectMARLEnv, multi_agent wrappers)."""
    from isaaclab.envs import DirectMARLEnv

    e = env
    while e is not None:
        if hasattr(e, "scene") and hasattr(e, "command_manager"):
            return e
        if isinstance(e, DirectMARLEnv):
            e = getattr(e, "env", None)
        else:
            e = getattr(e, "env", None) if hasattr(e, "env") else None
    return None


def _needs_kinematic_object_update(env) -> bool:
    """True if env has proxy_object (multi-object task)."""
    manager_env = _get_manager_env(env)
    if manager_env is None:
        return False
    try:
        manager_env.scene["proxy_object"]
        return True
    except KeyError:
        return False


class KinematicObjectUpdateWrapper(gym.Wrapper):
    """Calls update_kinematic_object_poses before each step for multi-object tasks."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._manager_env = _get_manager_env(env)
        self._active = _needs_kinematic_object_update(env) and self._manager_env is not None

    def step(self, action):
        if self._active:
            from whole_body_tracking.tasks.tracking.mdp.events import update_kinematic_object_poses

            update_kinematic_object_poses(self._manager_env, None, command_name="motion")
        return self.env.step(action)
