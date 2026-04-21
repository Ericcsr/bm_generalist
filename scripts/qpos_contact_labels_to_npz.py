"""Convert retargeted motion npz with embedded contact labels to training npz + contact_nodes.npy.

Input npz schema:
  - qpos: (T, 36) MuJoCo order — root_pos[3], root_quat[4], joint_pos[29]
  - fps: scalar or 0-d array
  - contact_labels: optional per-frame labels. If omitted, all-false labels are used (warning printed). Supported shapes:
      * (T, 2): left wrist, right wrist (feet forced to 0 / False in output mask)
      * (T, 4): full [L_foot, R_foot, L_wrist, R_wrist] (use as-is; typically feet are 0)

Output:
  - bm_generalist/{output_name}/{output_name}.npz — same as omni_to_npz
  - {prefix}_contact_nodes.npy — same dict format as omomo_object_to_npz
    (`contact_mask` bool (T_out, 4), optional `nodes` passthrough if present in source).
    Written next to the motion npz unless ``--contact_export_dir`` is set.

Motion is interpolated with lerp/slerp; contact_mask is aligned via nearest-neighbor resampling
to output frame count (same as omomo_object_to_npz._resample_contact_mask).

Usage:
    python scripts/qpos_contact_labels_to_npz.py \\
        --input_file path/to/motion.npz --output_name my_motion --output_fps 50

    # Optional: put all ``*_contact_nodes.npy`` in one directory
    #   --contact_export_dir bm_generalist/my_export_folder

    # Or pass a directory: every ``*.npz`` in the folder is converted; outputs go under
    # ``bm_generalist/{output_name}/{stem}/{stem}.npz`` and contact files use each ``stem`` as prefix.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Convert retargeted motion (qpos + contact_labels) to training npz and contact_nodes.npy."
)
parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="Input .npz file, or a directory of .npz files (non-recursive); contact_labels optional",
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Frame range START END (inclusive). Applied to qpos and contact_labels together.",
)
parser.add_argument(
    "--output_name",
    type=str,
    required=True,
    help="Trajectory folder under bm_generalist (single file: {name}/{name}.npz; folder: {name}/{stem}/{stem}.npz)",
)
parser.add_argument("--output_fps", type=int, default=50, help="Output motion FPS")
parser.add_argument(
    "--contact_export_dir",
    type=str,
    default=None,
    help="Directory for {prefix}_contact_nodes.npy (default: same folder as each motion .npz)",
)
parser.add_argument(
    "--contact_export_prefix",
    type=str,
    default=None,
    help="Filename prefix for contact export (default: output_name)",
)
parser.add_argument("--terrain_urdf", type=str, default=None, help="Optional terrain URDF path")
parser.add_argument("--terrain_prim_path", type=str, default="/World/Terrain", help="Terrain prim path")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

MUJOCO_TO_ISACLAB_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
    4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]

SCRIPT_DIR = Path(__file__).resolve().parent
BM_GENERALIST_ROOT = SCRIPT_DIR.parent


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p.resolve()
    for base in (Path.cwd(), BM_GENERALIST_ROOT, BM_GENERALIST_ROOT.parent):
        alt = base / raw
        if alt.exists():
            return alt.resolve()
    return p.resolve()


def _collect_input_npz_paths(raw: str) -> tuple[list[Path], bool]:
    """Return (paths, from_directory). Non-recursive ``*.npz`` when ``raw`` is a directory."""
    p = _resolve_path(raw)
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {p}")
    if p.is_file():
        if p.suffix.lower() != ".npz":
            raise ValueError(f"Expected .npz file, got: {p}")
        return [p], False
    if p.is_dir():
        files = sorted(p.glob("*.npz"))
        if not files:
            raise ValueError(f"No .npz files in directory: {p}")
        return files, True
    raise ValueError(f"Not a file or directory: {p}")


def _build_jobs() -> tuple[list[dict], bool]:
    """Each job: input path str, output_dir Path, npz_stem str, contact_prefix str. Second value: from_directory."""
    paths, from_directory = _collect_input_npz_paths(args_cli.input_file)
    if not from_directory:
        out_name = args_cli.output_name
        return (
            [
                {
                    "input": str(paths[0]),
                    "output_dir": BM_GENERALIST_ROOT / out_name,
                    "npz_stem": out_name,
                    "contact_prefix": args_cli.contact_export_prefix or out_name,
                }
            ],
            False,
        )
    if args_cli.contact_export_prefix is not None:
        print(
            "Warning: --contact_export_prefix ignored when --input_file is a directory; "
            "using each file's basename (without .npz) as the contact export prefix."
        )
    parent = BM_GENERALIST_ROOT / args_cli.output_name
    jobs = [
        {
            "input": str(p),
            "output_dir": parent / p.stem,
            "npz_stem": p.stem,
            "contact_prefix": p.stem,
        }
        for p in paths
    ]
    return jobs, True


def _resample_contact_mask(contact_mask: np.ndarray, num_frames_target: int) -> np.ndarray:
    """Resample contact mask (T, 4) to target frame count. Nearest-neighbor (matches omomo_object_to_npz)."""
    n_src = contact_mask.shape[0]
    if num_frames_target == n_src:
        return contact_mask.copy()
    out = np.zeros((num_frames_target, 4), dtype=bool)
    for i in range(num_frames_target):
        src_idx = int(i * (n_src - 1) / max(1, num_frames_target - 1))
        out[i] = contact_mask[src_idx]
    return out


def _contact_labels_to_mask_four(contact_labels: np.ndarray) -> np.ndarray:
    """Build (T, 4) bool mask: cols 0=L_foot, 1=R_foot, 2=L_wrist, 3=R_wrist."""
    cl = np.asarray(contact_labels)
    if cl.ndim == 1:
        raise ValueError(f"contact_labels must be 2D, got shape {cl.shape}")
    t, c = cl.shape
    mask = np.zeros((t, 4), dtype=bool)
    if c == 2:
        w = cl.astype(bool)
        mask[:, 2] = w[:, 0]
        mask[:, 3] = w[:, 1]
    elif c == 4:
        mask[:, :] = cl.astype(bool)
    else:
        raise ValueError(
            f"contact_labels must have 2 columns (L/R wrist) or 4 columns (feet+wrists), got {c}"
        )
    return mask


def load_retargeted_motion_with_contact(motion_file: str, device: torch.device):
    """Load qpos, fps, contact_labels; apply frame_range; return tensors + 4-channel mask."""
    motion_path = _resolve_path(motion_file)
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")

    data = np.load(str(motion_path), allow_pickle=True)
    if "qpos" not in data:
        raise ValueError(f"Expected 'qpos' in npz, keys: {list(data.keys())}")

    qpos = np.asarray(data["qpos"], dtype=np.float64)
    fps = float(data["fps"]) if "fps" in data else 30.0
    if "contact_labels" not in data:
        print(
            f"Warning: 'contact_labels' missing in {motion_path.name}; "
            f"using all-false labels for {qpos.shape[0]} frames."
        )
        contact_labels = np.zeros((qpos.shape[0], 4), dtype=bool)
    else:
        contact_labels = np.asarray(data["contact_labels"])
        contact_labels = (contact_labels > 0.5).astype(bool)

    if contact_labels.shape[0] != qpos.shape[0]:
        raise ValueError(
            f"contact_labels frames {contact_labels.shape[0]} != qpos frames {qpos.shape[0]}"
        )

    if args_cli.frame_range is not None:
        s, e = args_cli.frame_range
        qpos = qpos[s : e + 1]
        contact_labels = contact_labels[s : e + 1]
        print(f"Frame range [{s}, {e}] -> {qpos.shape[0]} frames")

    contact_mask_four = _contact_labels_to_mask_four(contact_labels)

    root_pos = qpos[:, 0:3]
    root_quat = qpos[:, 3:7]
    mujoco_joint_pos = qpos[:, 7:36]
    num_joints = mujoco_joint_pos.shape[1]
    if num_joints == 29:
        isaaclab_joint_pos = mujoco_joint_pos[:, MUJOCO_TO_ISACLAB_DOF]
    else:
        print(f"Warning: joint count {num_joints} != 29, using identity order")
        isaaclab_joint_pos = mujoco_joint_pos

    return {
        "fps": fps,
        "contact_mask_four": contact_mask_four,
        "joint_pos": torch.tensor(isaaclab_joint_pos, dtype=torch.float32, device=device),
        "root_pos": torch.tensor(root_pos, dtype=torch.float32, device=device),
        "root_quat": torch.tensor(root_quat, dtype=torch.float32, device=device),
        "num_frames": qpos.shape[0],
        "nodes": data["nodes"] if "nodes" in data.files else None,
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
        md = load_retargeted_motion_with_contact(self.motion_file, device=self.device)
        self.input_fps = md["fps"]
        self.input_dt = 1.0 / self.input_fps
        self.motion_base_poss_input = md["root_pos"]
        self.motion_base_rots_input = md["root_quat"]
        self.motion_dof_poss_input = md["joint_pos"]
        self.input_frames = md["num_frames"]
        self.contact_mask_input = md["contact_mask_four"]
        self._nodes_optional = md.get("nodes")
        self.duration = (self.input_frames - 1) * self.input_dt
        print(
            f"Motion loaded ({self.motion_file}), duration: {self.duration:.2f}s, "
            f"frames: {self.input_frames}, input fps: {self.input_fps}"
        )

    def _interpolate_motion(self):
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
            f"Motion interpolated: {self.input_frames} -> {self.output_frames} frames, "
            f"{self.input_fps} -> {self.output_fps} fps"
        )

    def _lerp(self, a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], blend[i])
        return out

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations, dt):
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = self.current_idx >= self.output_frames
        if reset_flag:
            self.current_idx = 0
        return state, reset_flag


def create_scene_cfg(terrain_urdf_path: str | None = None):
    if terrain_urdf_path is not None:

        @configclass
        class ReplayMotionsSceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
            sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(
                    intensity=750.0,
                    texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                ),
            )
            robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            terrain: AssetBaseCfg = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Terrain",
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=terrain_urdf_path,
                    fix_base=True,
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

        @configclass
        class ReplayMotionsSceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
            sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(
                    intensity=750.0,
                    texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                ),
            )
            robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    return ReplayMotionsSceneCfg


def _export_contact_nodes(
    contact_mask_input: np.ndarray,
    output_frames: int,
    export_dir: Path,
    prefix: str,
    nodes_optional,
) -> None:
    """Write {prefix}_contact_nodes.npy in omomo_object_to_npz format."""
    export_dir.mkdir(parents=True, exist_ok=True)
    mask_resampled = _resample_contact_mask(contact_mask_input, output_frames)
    out_path = export_dir / f"{prefix}_contact_nodes.npy"
    out_data: dict = {"contact_mask": mask_resampled}
    if nodes_optional is not None:
        out_data["nodes"] = nodes_optional
    np.save(str(out_path), out_data, allow_pickle=True)
    print(f"[INFO] Saved contact labels: {out_path} ({contact_mask_input.shape[0]} -> {output_frames} frames)")


def _process_one_job(sim: SimulationContext, scene: InteractiveScene, job: dict) -> bool:
    """Replay one motion, export contact_nodes, write training npz. Returns False if app closed early."""
    motion = RetargetedMotionLoader(
        motion_file=job["input"],
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    contact_dir = (
        Path(args_cli.contact_export_dir).resolve()
        if args_cli.contact_export_dir is not None
        else job["output_dir"]
    )
    _export_contact_nodes(
        motion.contact_mask_input,
        motion.output_frames,
        contact_dir,
        job["contact_prefix"],
        motion._nodes_optional,
    )

    robot = scene["robot"]
    job["output_dir"].mkdir(parents=True, exist_ok=True)
    output_file = job["output_dir"] / f"{job['npz_stem']}.npz"

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    for _ in range(motion.output_frames):
        if not simulation_app.is_running():
            return False
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            _,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos + scene.env_origins
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion_dof_pos, motion_dof_vel)
        scene.write_data_to_sim()
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
    print(f"[INFO] Motion saved to {output_file}")
    print(f"  Frames: {log['joint_pos'].shape[0]}, FPS: {log['fps'][0]}")
    return True


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    jobs, from_directory = _build_jobs()
    if from_directory:
        print(f"[INFO] Directory input: converting {len(jobs)} .npz file(s).")

    for ji, job in enumerate(jobs):
        print(f"[INFO] ({ji + 1}/{len(jobs)}) {job['input']}")
        if not _process_one_job(sim, scene, job):
            print("[INFO] Application closed before finishing all jobs.")
            return

    if args_cli.headless:
        print("[INFO] Headless mode, exiting...")
        os._exit(0)

    print("[INFO] All jobs finished. Close the window to exit.")
    while simulation_app.is_running():
        sim.render()
        scene.update(sim.get_physics_dt())


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    terrain_urdf_path = None
    if args_cli.terrain_urdf is not None:
        tu = Path(args_cli.terrain_urdf)
        if not tu.is_absolute() and not tu.exists():
            alt = BM_GENERALIST_ROOT / args_cli.terrain_urdf
            if alt.exists():
                tu = alt
        terrain_urdf_path = str(tu.resolve())
        if not Path(terrain_urdf_path).exists():
            raise FileNotFoundError(f"Terrain URDF not found: {terrain_urdf_path}")

    ReplayMotionsSceneCfg = create_scene_cfg(terrain_urdf_path)
    scene = InteractiveScene(ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print("[INFO] Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
