"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

    # Usage
    python csv_to_npz.py --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
    --output_file ./motions/dance1_subject2.npz --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import os
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_folder", type=str, required=True, help="The path to the input motion npz folder.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.traj_id = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        # if self.frame_range is None:
        #     motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        # else:
        #     motion = torch.from_numpy(
        #         np.loadtxt(
        #             self.motion_file,
        #             delimiter=",",
        #             skiprows=self.frame_range[0] - 1,
        #             max_rows=self.frame_range[1] - self.frame_range[0] + 1,
        #         )
        #     )
        motion_files = os.listdir(self.motion_file)
        motions = {}
        for i in range(len(motion_files)):
            motion = np.load(f"{self.motion_file}/{motion_files[i]}")
            motions[motion_files[i].split(".")[0]] = motion
        #motions = joblib.load(self.motion_file)
        self.num_motions = len(motions)
        self.motion_base_poss_inputs = []
        self.motion_base_rots_inputs = []
        self.motion_dof_poss_inputs = []
        self.input_frames_buffer = []
        self.duration_buffer = []
        self.motion_names = []
        for key in motions.keys():
            self.motion_names.append(key)
            raw_motion = motions[key]
            motion_length = raw_motion["qpos"].shape[0]
            motion = torch.zeros(motion_length, 36)
            motion[:, :3] = torch.from_numpy(raw_motion["qpos"][:,:3])
            motion[:,3:7] = torch.from_numpy(raw_motion["qpos"][:,3:7][:, [1, 2, 3, 0]])  # convert to xyzw (?)
            motion[:,7:] = torch.from_numpy(raw_motion["qpos"][:,7:])
            motion = motion.to(torch.float32).to(self.device)
            self.motion_base_poss_inputs.append(motion[:, :3])
            self.motion_base_rots_inputs.append(motion[:, 3:7])
            self.motion_base_rots_inputs[-1] = self.motion_base_rots_inputs[-1][:, [3, 0, 1, 2]]  # convert to wxyz
            self.motion_dof_poss_inputs.append(motion[:, 7:])

            self.input_frames_buffer.append(motion.shape[0])
            self.duration_buffer.append((motion.shape[0] - 1) * self.input_dt)
            print(f"Motion loaded ({self.motion_file}), duration: {self.duration_buffer[-1]} sec, frames: {self.input_frames_buffer[-1]}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        self.motion_base_poss_buffer = []
        self.motion_base_rots_buffer = []
        self.motion_dof_poss_buffer = []
        self.output_frames_buffer = []
        for i in range(len(self.motion_base_poss_inputs)):
            times = torch.arange(0, self.duration_buffer[i], self.output_dt, device=self.device, dtype=torch.float32)
            self.output_frames_buffer.append(times.shape[0])
            index_0, index_1, blend = self._compute_frame_blend(times)
            self.motion_base_poss_buffer.append(self._lerp(
                self.motion_base_poss_inputs[i][index_0],
                self.motion_base_poss_inputs[i][index_1],
                blend.unsqueeze(1),
            ))
            self.motion_base_rots_buffer.append(self._slerp(
                self.motion_base_rots_inputs[i][index_0],
                self.motion_base_rots_inputs[i][index_1],
                blend,
            ))
            self.motion_dof_poss_buffer.append(self._lerp(
                self.motion_dof_poss_inputs[i][index_0],
                self.motion_dof_poss_inputs[i][index_1],
                blend.unsqueeze(1),
            ))
            print(
                f"Motion {i} interpolated, input frames: {self.input_frames_buffer[i]}, input fps: {self.input_fps}, output frames:"
                f" {self.output_frames_buffer[i]}, output fps: {self.output_fps}"
            )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration_buffer[self.traj_id]
        index_0 = (phase * (self.input_frames_buffer[self.traj_id] - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames_buffer[self.traj_id] - 1))
        blend = phase * (self.input_frames_buffer[self.traj_id] - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels_buffer = []
        self.motion_dof_vels_buffer = []
        self.motion_base_ang_vels_buffer = []
        for i in range(self.num_motions):
            self.motion_base_lin_vels_buffer.append(torch.gradient(self.motion_base_poss_buffer[i], spacing=self.output_dt, dim=0)[0])
            self.motion_dof_vels_buffer.append(torch.gradient(self.motion_dof_poss_buffer[i], spacing=self.output_dt, dim=0)[0])
            self.motion_base_ang_vels_buffer.append(self._so3_derivative(self.motion_base_rots_buffer[i], self.output_dt))

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss_buffer[self.traj_id][self.current_idx : self.current_idx + 1],
            self.motion_base_rots_buffer[self.traj_id][self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels_buffer[self.traj_id][self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels_buffer[self.traj_id][self.current_idx : self.current_idx + 1],
            self.motion_dof_poss_buffer[self.traj_id][self.current_idx : self.current_idx + 1],
            self.motion_dof_vels_buffer[self.traj_id][self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames_buffer[self.traj_id]:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag

    def get_next_traj(self):
        self.traj_id += 1
        reset_flag = False
        if self.traj_id >= self.num_motions:
            reset_flag = True
        return reset_flag

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_folder,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    # --------------------------------------------------------------------------
    # make output directory if not exist
    import os
    if not os.path.exists(args_cli.output_name):
        os.makedirs(args_cli.output_name)
    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        
        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag:
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            print(f"motion saved {motion.motion_names[motion.traj_id]}")
            np.savez(f"{args_cli.output_name}/{motion.motion_names[motion.traj_id]}.npz", **log)
            finished = motion.get_next_traj()
            log = {
                "fps": [args_cli.output_fps],
                "joint_pos": [],
                "joint_vel": [],
                "body_pos_w": [],
                "body_quat_w": [],
                "body_lin_vel_w": [],
                "body_ang_vel_w": [],
            }
            if finished:
                print("[INFO]: All motions saved, exiting...")
                break




def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene,
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
