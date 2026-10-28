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

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv


class ManagerBasedRLEnvWithKinematicObjects(ManagerBasedRLEnv):
    """RL env that updates kinematic proxy object poses at the start of each physics step.

    Runs update_kinematic_object_poses before apply_action in each decimation iteration,
    matching the working order in test_motion_object_replay.py.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        cats = getattr(cfg, "motion_category_per_motion", None)
        if cats and len(cats) > 0:
            num_motions = len(cats)
            mc = torch.tensor(cats, dtype=torch.long, device=self.device)
            self._motion_category_per_env = mc[torch.arange(self.num_envs, device=self.device) % num_motions]
            self._episode_total_reward_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

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
        # NaN guard (claude 2026-10-23): a physics blow-up's step reward is computed from the exploded state (joint_limit
        # up to -1e8, or NaN) before the env is reset; keep it finite and bounded so it cannot poison the PPO update.
        # Legit per-step rewards are |r| < ~0.3 (positive weights sum 8 x dt 0.02), so +-reward_guard_clip only bites on
        # blow-ups. The count is logged as Guard/clamped_rewards (after the resets, which rebuild extras["log"]).
        clip = float(getattr(self.cfg, "reward_guard_clip", 2.0))
        n_clamped = None
        if clip > 0:
            bad = ~torch.isfinite(self.reward_buf) | (self.reward_buf.abs() > clip)
            n_clamped = bad.sum().float()
            self.reward_buf = torch.nan_to_num(self.reward_buf, nan=0.0, posinf=clip, neginf=-clip).clamp_(-clip, clip)
        if hasattr(self, "_episode_total_reward_sum"):
            self._episode_total_reward_sum += self.reward_buf

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
        if n_clamped is not None:
            self.extras.setdefault("log", {})["Guard/clamped_rewards"] = n_clamped

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0 or not hasattr(self, "_episode_total_reward_sum"):
            return
        el_s = self.max_episode_length_s
        cats = self._motion_category_per_env[env_ids]
        totals = self._episode_total_reward_sum[env_ids]
        for name, cid in [("terrain", 0), ("object", 1), ("free", 2)]:
            mask = cats == cid
            if torch.any(mask):
                key = f"Episode_Reward/total_reward_mean_{name}"
                self.extras["log"][key] = totals[mask].mean() / el_s
        self._episode_total_reward_sum[env_ids] = 0.0
