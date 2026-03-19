"""ManagerBasedRLEnv subclass that updates kinematic proxy objects inside the physics loop.

Matches the update order from test_motion_object_replay.py exactly:
  1. update_kinematic_object_poses (object pose + velocity)
  2. apply_action (robot joint targets)
  3. scene.write_data_to_sim
  4. sim.step
  5. scene.update

Used for Tracking-3pt-Binded-Multi-Object-G1-v0 where objects were not moving
with the action-term or wrapper approach.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv


class ManagerBasedRLEnvWithKinematicObjects(ManagerBasedRLEnv):
    """RL env that updates kinematic proxy object poses at the start of each physics step.

    Runs update_kinematic_object_poses before apply_action in each decimation iteration,
    matching the working order in test_motion_object_replay.py.
    """

    def step(self, action):
        """Override step to inject kinematic object update at start of each decimation iteration."""
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # Update kinematic proxy objects FIRST (before apply_action), matching test_motion_object_replay.py
            # order: object pose -> robot joints -> write_data_to_sim -> sim.step -> scene.update
            try:
                from whole_body_tracking.tasks.tracking.mdp.events import update_magic_force_object_poses

                physics_phase = getattr(self.cfg, "physics_phase_flag", False)
                use_pose_velocity_control = getattr(
                    self.cfg, "magic_force_use_pose_velocity_control", True
                )
                update_magic_force_object_poses(
                    self,
                    None,
                    command_name="motion",
                    physics_phase=physics_phase,
                    use_pose_velocity_control=use_pose_velocity_control,
                )
            except (ImportError, KeyError) as e:
                # Log once to avoid spam; KeyError means no proxy_object in scene
                if not getattr(self, "_kinematic_obj_update_warned", False):
                    print(f"[KinObj] update_magic_force_object_poses skipped: {e}")
                    self._kinematic_obj_update_warned = True
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # post-step (same as parent)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
