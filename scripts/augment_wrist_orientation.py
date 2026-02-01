"""
This script performs data augmentation on motion sequences by adding a linear drift 
to wrist joints. 

It assumes the INPUT .npz files have the SAME SCHEMA as the output files 
(containing 'joint_pos', 'body_pos_w', etc.).

Optimization: Skips interpolation if input_fps == output_fps.

Usage:
    python augment_motions_optimized.py --input_folder ./motions_in --output_name ./motions_out \
    --input_fps 50 --output_fps 50 --offset_range 0.15
"""

import argparse
import numpy as np
import os
import torch
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Augment motion with linear offset drift using dynamic joint reading.")
parser.add_argument("--input_folder", type=str, required=True, help="The path to the input motion npz folder.")
parser.add_argument("--input_fps", type=int, default=50, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=("Frame range: START END (inclusive). If not provided, all frames loaded."),
)
parser.add_argument("--output_name", type=str, required=True, help="The output directory for npz files.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument("--offset_range", type=float, default=0.8, help="Maximum range (radians) for the random start/end offsets.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

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
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class AugmentedMotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        joint_names: list[str],
        offset_range: float
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
        
        # Augmentation params
        self.joint_names = joint_names
        self.offset_range = offset_range
        
        # ----------------------------------------------------------------------
        # DYNAMIC JOINT DISCOVERY
        # ----------------------------------------------------------------------
        # Find indices for any joint containing "wrist" in its name
        self.wrist_indices = [
            i for i, name in enumerate(self.joint_names) 
            if "wrist" in name
        ]
        
        if not self.wrist_indices:
            print("[WARNING] No joints with 'wrist' in name found in robot model. Augmentation will be skipped.")
        else:
            detected_names = [self.joint_names[i] for i in self.wrist_indices]
            print(f"[INFO] Detected wrist joints for augmentation: {detected_names}")
            print(f"[INFO] Corresponding indices: {self.wrist_indices}")

        self._load_and_augment_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _generate_linear_drift(self, length: int, num_channels: int) -> torch.Tensor:
        """
        Generates a trajectory of offsets that linearly interpolates from a random 
        start value to a random end value.
        """
        # Generate random start and end offsets within range [-offset_range, offset_range]
        start_offsets = (torch.rand(num_channels, device=self.device) * 2.0 - 1.0) * self.offset_range
        end_offsets = (torch.rand(num_channels, device=self.device) * 2.0 - 1.0) * self.offset_range
        
        # Create interpolation factor alpha from 0 to 1 (Shape: length, 1)
        alpha = torch.linspace(0, 1, length, device=self.device).unsqueeze(1)
        
        # Interpolate: (1 - alpha) * start + alpha * end
        drift = (1.0 - alpha) * start_offsets + alpha * end_offsets
        return drift

    def _load_and_augment_motion(self):
        """Loads motions, duplicates them, and applies linear drift to the duplicate."""
        motion_files = [f for f in os.listdir(self.motion_file) if f.endswith('.npz')]
        
        self.motion_base_poss_inputs = []
        self.motion_base_rots_inputs = []
        self.motion_dof_poss_inputs = []
        self.input_frames_buffer = []
        self.duration_buffer = []
        self.motion_names = []

        for f_name in motion_files:
            base_name = f_name.split(".")[0]
            file_path = os.path.join(self.motion_file, f_name)
            raw_motion = np.load(file_path)
            
            # ------------------------------------------------------------------
            # DATA EXTRACTION LOGIC (Based on Output Schema)
            # ------------------------------------------------------------------
            try:
                # 1. Joint Positions
                if "joint_pos" not in raw_motion:
                    raise KeyError(f"Missing 'joint_pos' in {f_name}")
                dof_pos_np = raw_motion["joint_pos"]
                # Ensure it's 2D: (Time, NumJoints). If 3D (Time, 1, NumJoints), squeeze.
                if dof_pos_np.ndim == 3 and dof_pos_np.shape[1] == 1:
                    dof_pos_np = dof_pos_np.squeeze(1)
                
                dof_pos = torch.from_numpy(dof_pos_np).to(torch.float32).to(self.device)

                # 2. Base Position
                if "body_pos_w" not in raw_motion:
                    raise KeyError(f"Missing 'body_pos_w' in {f_name}")
                
                # Input shape: (Time, NumBodies, 3). We want Body 0 (Root).
                base_pos_np = raw_motion["body_pos_w"]
                if base_pos_np.ndim == 3:
                    base_pos_np = base_pos_np[:, 0, :] # Extract root
                
                base_pos = torch.from_numpy(base_pos_np).to(torch.float32).to(self.device)

                # 3. Base Rotation
                if "body_quat_w" not in raw_motion:
                    raise KeyError(f"Missing 'body_quat_w' in {f_name}")
                
                # Input shape: (Time, NumBodies, 4). We want Body 0 (Root).
                base_rot_np = raw_motion["body_quat_w"]
                if base_rot_np.ndim == 3:
                    base_rot_np = base_rot_np[:, 0, :] # Extract root
                
                base_rot = torch.from_numpy(base_rot_np).to(torch.float32).to(self.device)

            except Exception as e:
                print(f"[ERROR] Failed to load {f_name}: {e}")
                print(f"        Available keys: {list(raw_motion.keys())}")
                continue

            motion_len = dof_pos.shape[0]

            # --- Processing Helper ---
            def process_single_motion(b_pos, b_rot, d_pos, suffix, apply_aug=False):
                dof_pos_curr = d_pos.clone()

                # Apply Linear Drift if requested
                if apply_aug and len(self.wrist_indices) > 0:
                    drift = self._generate_linear_drift(motion_len, len(self.wrist_indices))
                    dof_pos_curr[:, self.wrist_indices] += drift

                # Store
                self.motion_names.append(f"{base_name}_{suffix}")
                self.motion_base_poss_inputs.append(b_pos)
                self.motion_base_rots_inputs.append(b_rot)
                self.motion_dof_poss_inputs.append(dof_pos_curr)
                self.input_frames_buffer.append(motion_len)
                self.duration_buffer.append((motion_len - 1) * self.input_dt)

            # 1. Original Motion
            process_single_motion(base_pos, base_rot, dof_pos, "original", apply_aug=False)
            
            # 2. Augmented Motion
            process_single_motion(base_pos, base_rot, dof_pos, "augmented", apply_aug=True)

        self.num_motions = len(self.motion_names)
        print(f"[INFO] Loaded {len(motion_files)} files. Total trajectories (Original + Augmented): {self.num_motions}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        self.motion_base_poss_buffer = []
        self.motion_base_rots_buffer = []
        self.motion_dof_poss_buffer = []
        self.output_frames_buffer = []

        # ----------------------------------------------------------------------
        # OPTIMIZATION: SKIP INTERPOLATION IF FPS MATCH
        # ----------------------------------------------------------------------
        if self.input_fps == self.output_fps:
            print("[INFO] Input and Output FPS match. Skipping interpolation and copying buffers.")
            # Shallow copy the lists (tensors are already on device)
            self.motion_base_poss_buffer = list(self.motion_base_poss_inputs)
            self.motion_base_rots_buffer = list(self.motion_base_rots_inputs)
            self.motion_dof_poss_buffer = list(self.motion_dof_poss_inputs)
            self.output_frames_buffer = list(self.input_frames_buffer)
            return

        # ----------------------------------------------------------------------
        # NORMAL INTERPOLATION
        # ----------------------------------------------------------------------
        for i in range(len(self.motion_base_poss_inputs)):
            times = torch.arange(0, self.duration_buffer[i], self.output_dt, device=self.device, dtype=torch.float32)
            self.output_frames_buffer.append(times.shape[0])
            index_0, index_1, blend = self._compute_frame_blend(times, i)
            
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
            
    def _lerp(self, a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times, traj_id):
        phase = times / self.duration_buffer[traj_id]
        index_0 = (phase * (self.input_frames_buffer[traj_id] - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames_buffer[traj_id] - 1))
        blend = phase * (self.input_frames_buffer[traj_id] - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels_buffer = []
        self.motion_dof_vels_buffer = []
        self.motion_base_ang_vels_buffer = []
        for i in range(self.num_motions):
            print(f"[INFO] Computing velocities for trajectory {i+1}/{self.num_motions}")
            self.motion_base_lin_vels_buffer.append(torch.gradient(self.motion_base_poss_buffer[i], spacing=self.output_dt, dim=0)[0])
            self.motion_dof_vels_buffer.append(torch.gradient(self.motion_dof_poss_buffer[i], spacing=self.output_dt, dim=0)[0])
            self.motion_base_ang_vels_buffer.append(self._so3_derivative(self.motion_base_rots_buffer[i], self.output_dt))

    def _so3_derivative(self, rotations, dt):
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(self):
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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    
    # --------------------------------------------------------------------------
    # ROBOT INFO RETRIEVAL
    # --------------------------------------------------------------------------
    robot = scene["robot"]
    
    # Read the joint names directly from the loaded asset
    robot_joint_names = robot.joint_names
    print(f"[INFO] Robot loaded with {len(robot_joint_names)} joints.")
    # print(f"[INFO] Joint Order: {robot_joint_names}")
    
    # Generate indices for ALL joints to map data correctly
    robot_joint_indexes = list(range(len(robot_joint_names)))

    # --------------------------------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------------------------------
    motion = AugmentedMotionLoader(
        motion_file=args_cli.input_folder,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
        joint_names=robot_joint_names,  # Passing the dynamic list
        offset_range=args_cli.offset_range
    )

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
        
        # Apply motion data to the robot using the implicit order
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
        sim.render()
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
            for k in log.keys():
                if k != "fps":
                    log[k] = np.stack(log[k], axis=0)

            save_path = f"{args_cli.output_name}/{motion.motion_names[motion.traj_id]}.npz"
            print(f"[INFO] Saving motion: {save_path}")
            np.savez(save_path, **log)
            
            finished = motion.get_next_traj()
            robot.reset()
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
                print("[INFO]: All original and augmented motions saved. Exiting...")
                break

def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    print("[INFO]: Setup complete...")
    
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()