"""This script converts retargeted motion data (MuJoCo qpos order) to training data format (npz).

The input retargeted motion data has:
- Schema: ['qpos', 'fps']
- qpos format: MuJoCo order (root_pos[3], root_quat[4], joint_pos[29])
- qpos shape: (T, 36) where T is number of frames

The output training data has:
- Schema: ['fps', 'joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w']
- joint_pos/joint_vel: IsaacLab joint order
- All body states: world frame

.. code-block:: bash

    # Usage
    python retargeted_to_npz.py --input_file lafan_terrain/obstacles1_subject1/obstacles1_subject1_original.npz \
    --output_name obstacles1_subject1 --output_fps 50 --frame_range 0 1000
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import os
import torch
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert retargeted motion data (MuJoCo qpos) to training npz format.")
parser.add_argument("--input_file", type=str, required=True, help="Path to input retargeted motion .npz/.npy file with ['qpos', 'fps'] schema")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). Frame index starts from 0. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, required=True, help="The name of the trajectory (will create folder with this name under bm_generalist).")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument("--terrain_urdf", type=str, default=None, help="Path to terrain URDF file (optional)")
parser.add_argument("--terrain_prim_path", type=str, default="/World/Terrain", help="Prim path for terrain in IsaacLab")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
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

# MuJoCo to IsaacLab joint order mapping for G1 robot (29 DOF)
# This mapping converts from MuJoCo order to IsaacLab order
MUJOCO_TO_ISACLAB_DOF = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]


def create_scene_cfg(terrain_urdf_path: str | None = None):
    """Create scene configuration with optional terrain."""
    
    if terrain_urdf_path is not None:
        # With terrain
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
            
            # terrain as static URDF
            terrain: AssetBaseCfg = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Terrain",
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=terrain_urdf_path,
                    fix_base=True,  # Static terrain
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True,
                        retain_accelerations=False,
                    ),
                    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
                    ),
                ),
            )
    else:
        # Without terrain
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
    
    return ReplayMotionsSceneCfg


def load_retargeted_motion(motion_file: str, device: str = "cpu"):
    """Load retargeted motion data from npz/npy file with MuJoCo format.
    
    Args:
        motion_file: Path to .npz/.npy file with ['qpos', 'fps'] schema
        device: Device to load data on
        
    Returns:
        Dictionary with:
        - fps: frames per second
        - qpos: (T, N) array where N = 3 (pos) + 4 (quat) + num_joints
        - joint_pos: (T, num_joints) array in IsaacLab joint order
        - root_pos: (T, 3) array
        - root_quat: (T, 4) array (w, x, y, z)
    """
    motion_path = Path(motion_file)
    
    # Resolve path if relative
    if not motion_path.is_absolute():
        if not motion_path.exists():
            # Try relative to bm_generalist root
            bm_generalist_root = Path(__file__).parent.parent
            alt_path = bm_generalist_root / motion_file
            if alt_path.exists():
                motion_path = alt_path
            else:
                # Try in holosoma demo_data directory
                holosoma_demo = Path("/home/ericcsr/robot_imitation/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data")
                alt_path = holosoma_demo / motion_file
                if alt_path.exists():
                    motion_path = alt_path
                else:
                    # Try relative to workspace root
                    workspace_root = Path("/home/ericcsr/robot_imitation")
                    alt_path = workspace_root / motion_file
                    if alt_path.exists():
                        motion_path = alt_path
    
    motion_file = str(motion_path.resolve())
    if not Path(motion_file).exists():
        raise FileNotFoundError(
            f"Motion file not found: {motion_file}\n"
            f"Tried: {args_cli.input_file}\n"
            f"Current working directory: {Path.cwd()}\n"
            f"Script directory: {Path(__file__).parent}"
        )
    
    data = np.load(motion_file, allow_pickle=True)
    
    # Handle .npy files (which might just be the qpos array)
    if motion_path.suffix == '.npy':
        # If it's a .npy file, it might be just the qpos array
        if isinstance(data, np.ndarray):
            # Assume it's qpos array, need to get fps from elsewhere or use default
            qpos = data
            fps = 30.0  # Default fps if not available
            print(f"Warning: Loaded .npy file as qpos array. Using default fps={fps}")
        else:
            # It's a dict-like object (from .npy with allow_pickle=True)
            if 'qpos' not in data:
                raise ValueError(f"Motion file must contain 'qpos' key. Found keys: {list(data.keys()) if hasattr(data, 'keys') else 'N/A'}")
            qpos = data['qpos']
            fps = data.get('fps', 30.0)
    else:
        # Handle .npz files
        if 'qpos' not in data:
            raise ValueError(f"Motion file must contain 'qpos' key. Found keys: {list(data.keys())}")
        qpos = data['qpos']
        if 'fps' not in data:
            fps = 30.0
            print(f"Warning: 'fps' not found in motion file. Using default fps={fps}")
        else:
            fps = float(data['fps'])
    
    # Apply frame range if specified
    if args_cli.frame_range is not None:
        start_frame, end_frame = args_cli.frame_range
        qpos = qpos[start_frame:end_frame+1]
        print(f"Applied frame range: {start_frame} to {end_frame} (inclusive), resulting in {qpos.shape[0]} frames")
    
    num_frames = qpos.shape[0]
    num_dof = qpos.shape[1]
    
    # Extract root position and quaternion
    root_pos = qpos[:, 0:3] #+ np.array([0.0, 0.0, 0.15])  # (T, 3)
    root_quat = qpos[:, 3:7]  # (T, 4) - MuJoCo format: w, x, y, z
    
    # Extract joint positions (MuJoCo order)
    mujoco_joint_pos = qpos[:, 7:36]  # (T, num_joints)
    num_joints = mujoco_joint_pos.shape[1]
    
    # Convert joint order from MuJoCo to IsaacLab
    if num_joints == 29:
        # Use the mapping for 29 DOF G1 robot
        isaaclab_joint_pos = mujoco_joint_pos[:, MUJOCO_TO_ISACLAB_DOF]
    else:
        print(f"Warning: Unknown number of joints ({num_joints}). Using identity mapping.")
        isaaclab_joint_pos = mujoco_joint_pos
    
    return {
        'fps': fps,
        'qpos': qpos,
        'joint_pos': torch.tensor(isaaclab_joint_pos, dtype=torch.float32, device=device),
        'root_pos': torch.tensor(root_pos, dtype=torch.float32, device=device),
        'root_quat': torch.tensor(root_quat, dtype=torch.float32, device=device),
        'num_frames': num_frames,
    }


class RetargetedMotionLoader:
    def __init__(
        self,
        motion_file: str,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.output_fps = output_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the retargeted motion from npz/npy file."""
        motion_data = load_retargeted_motion(self.motion_file, device=self.device)
        
        self.input_fps = motion_data['fps']
        self.input_dt = 1.0 / self.input_fps
        
        # Root pose (already in correct format: w, x, y, z)
        self.motion_base_poss_input = motion_data['root_pos']  # (T, 3)
        self.motion_base_rots_input = motion_data['root_quat']  # (T, 4) - already wxyz
        
        # Joint positions (already converted to IsaacLab order)
        self.motion_dof_poss_input = motion_data['joint_pos']  # (T, 29)
        
        self.input_frames = motion_data['num_frames']
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration:.2f} sec, frames: {self.input_frames}, input fps: {self.input_fps}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
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
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

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
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Load motion
    motion = RetargetedMotionLoader(
        motion_file=args_cli.input_file,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    # Extract scene entities
    robot = scene["robot"]

    # Create output directory under bm_generalist
    bm_generalist_root = Path(__file__).parent.parent
    output_dir = bm_generalist_root / args_cli.output_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir / f"{args_cli.output_name}.npz"
    print(f"[INFO]: Output directory: {output_dir}")
    print(f"[INFO]: Output file: {output_file}")

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
    file_saved = False
    # --------------------------------------------------------------------------

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

        # set root state (matching replay_lafan_terrain.py approach)
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos + scene.env_origins
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state (joint positions are already in IsaacLab order from load_retargeted_motion)
        # Directly write all joints, matching replay_lafan_terrain.py approach
        robot.write_joint_state_to_sim(motion_dof_pos, motion_dof_vel)
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            np.savez(str(output_file), **log)
            print(f"[INFO]: Motion saved to {output_file}")
            print(f"[INFO]: Motion stats:")
            print(f"  - Frames: {log['joint_pos'].shape[0]}")
            print(f"  - FPS: {log['fps'][0]}")
            print(f"  - Joint positions shape: {log['joint_pos'].shape}")
            print(f"  - Body positions shape: {log['body_pos_w'].shape}")
            if args_cli.headless:
                print("[INFO]: Headless mode detected, exiting...")
                os._exit(0)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    
    # Resolve terrain URDF path if provided
    terrain_urdf_path = None
    if args_cli.terrain_urdf is not None:
        terrain_urdf = Path(args_cli.terrain_urdf)
        
        if not terrain_urdf.is_absolute():
            # Try relative to current working directory first
            if not terrain_urdf.exists():
                # Try relative to bm_generalist root
                bm_generalist_root = Path(__file__).parent.parent
                alt_path = bm_generalist_root / args_cli.terrain_urdf
                if alt_path.exists():
                    terrain_urdf = alt_path
                else:
                    # Try in holosoma demo_data directory
                    holosoma_demo = Path("/home/ericcsr/robot_imitation/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data")
                    alt_path = holosoma_demo / args_cli.terrain_urdf
                    if alt_path.exists():
                        terrain_urdf = alt_path
                    else:
                        # Try relative to workspace root
                        workspace_root = Path("/home/ericcsr/robot_imitation")
                        alt_path = workspace_root / args_cli.terrain_urdf
                        if alt_path.exists():
                            terrain_urdf = alt_path
        
        terrain_urdf_path = str(terrain_urdf.resolve())
        if not Path(terrain_urdf_path).exists():
            raise FileNotFoundError(
                f"Terrain URDF file not found: {terrain_urdf_path}\n"
                f"Tried: {args_cli.terrain_urdf}\n"
                f"Current working directory: {Path.cwd()}\n"
                f"Script directory: {Path(__file__).parent}"
            )
        print(f"[INFO]: Using terrain URDF: {terrain_urdf_path}")
    
    # Design scene with optional terrain
    ReplayMotionsSceneCfg = create_scene_cfg(terrain_urdf_path)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
