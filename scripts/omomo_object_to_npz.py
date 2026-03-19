"""Convert OMOMO object pipeline output to training format, with proxy object visualization.

Loads already-retargeted robot motion from output_object, runs headless proxy fitting
(refit + thin plate), spawns both robot and proxy URDF in IsaacLab, and saves the
converted motion npz alongside proxy assets.

Input:
  - Robot motion:  output_object/{key}/{key}.npz       (qpos, fps)
  - Object motion: output_object/{key}/{key}_object.npz (object_trans, object_rot, ...)

Output:
  - Converted motion:  bm_generalist/{output_name}/{output_name}.npz
  - Proxy URDF:        (saved to --proxy-export-dir if given)
  - Fitted trajectory: (saved to --proxy-export-dir if given)

Usage:
    python scripts/omomo_object_to_npz.py \\
        --input_file ../contact_terrain_reconstruction/output_object/test_sub16_largebox_007/test_sub16_largebox_007.npz \\
        --output_name test_sub16_largebox_007 --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert OMOMO object motion to training npz with proxy visualization.")
parser.add_argument("--input_file", type=str, required=True, help="Path to robot motion npz (qpos, fps)")
parser.add_argument(
    "--object_motion",
    type=str,
    default=None,
    help="Path to object motion npz (default: <input_stem>_object.npz)",
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Frame range: START END (both inclusive)",
)
parser.add_argument("--output_name", type=str, required=True, help="Output trajectory name")
parser.add_argument("--output_fps", type=int, default=50, help="Output FPS (default: 50)")
parser.add_argument(
    "--proxy_export_dir",
    type=str,
    default=None,
    help="Directory to save proxy URDF and fitted trajectory (default: no export)",
)
parser.add_argument(
    "--proxy_export_prefix",
    type=str,
    default=None,
    help="Prefix for exported proxy files (default: output_name)",
)
parser.add_argument(
    "--skip_proxy",
    action="store_true",
    help="Skip proxy fitting and visualization (robot-only conversion)",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
WORKSPACE_ROOT = BM_GENERALIST_ROOT.parent
CONTACT_TERRAIN_ROOT = WORKSPACE_ROOT / "contact_terrain_reconstruction"

sys.path.insert(0, str(CONTACT_TERRAIN_ROOT))


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p.resolve()
    for base in [Path.cwd(), BM_GENERALIST_ROOT, WORKSPACE_ROOT]:
        alt = base / raw
        if alt.exists():
            return alt.resolve()
    return p.resolve()


def _run_proxy_fitting(motion_npz: Path, object_npz: Path, export_dir: Path, prefix: str):
    """Run headless proxy fitting via visualize_omomo_robot_object.py functions."""
    from visualize_omomo_robot_object import (
        load_object_motion,
        infer_object_name,
        resolve_object_mesh_path,
        compute_local_center_offset_unit,
        _smooth_with_contact_control_points,
        _write_two_plate_proxy_urdf,
    )
    from refit_object import refit_object_to_palms
    from thin_plate_proxy import fit_thin_plate_proxy
    import mujoco as mj

    from general_motion_retargeting import ROBOT_XML_DICT

    robot_motion = np.load(motion_npz, allow_pickle=True)
    qpos = np.asarray(robot_motion["qpos"], dtype=np.float32)
    obj_trans, obj_quat_wxyz, mesh_scale = load_object_motion(object_npz)
    num_frames = min(len(qpos), len(obj_trans))

    robot_xml = Path(ROBOT_XML_DICT["unitree_g1"]).resolve()
    robot_model = mj.MjModel.from_xml_path(str(robot_xml))

    obj_name = infer_object_name(motion_npz.stem)
    mesh_path = resolve_object_mesh_path(motion_npz, obj_name, None)
    if not mesh_path.exists():
        mesh_path = (CONTACT_TERRAIN_ROOT / "omomo" / "captured_objects" / f"{obj_name}_cleaned_simplified.obj").resolve()

    local_offset_unit = compute_local_center_offset_unit(mesh_path) if mesh_path.exists() else np.zeros(3, dtype=np.float32)

    refit_result = None
    if mesh_path.exists():
        refit_result = refit_object_to_palms(
            robot_model=robot_model,
            qpos=qpos[:num_frames],
            obj_trans=obj_trans[:num_frames],
            obj_mesh_path=mesh_path,
            mesh_scale_base=mesh_scale,
            left_body_name="left_rubber_hand",
            right_body_name="right_rubber_hand",
            bend_to_palm_center_on_contact=True,
            palm_forward_offset=0.05,
        )
        obj_trans = refit_result.refit_trans
        local_offset_unit = refit_result.local_offset_unit
        mesh_scale = float(mesh_scale) * float(refit_result.mesh_scale_multiplier)
        if len(refit_result.contact_frames) > 0:
            obj_trans, obj_quat_wxyz = _smooth_with_contact_control_points(
                obj_trans, obj_quat_wxyz, refit_result.contact_frames
            )

    proxy = None
    proxy_urdf_path = None
    contact_frames = refit_result.contact_frames if refit_result is not None else None
    proxy = fit_thin_plate_proxy(
        robot_model=robot_model,
        qpos=qpos[:num_frames],
        obj_trans=obj_trans[:num_frames],
        obj_quat_wxyz=obj_quat_wxyz[:num_frames],
        left_body_name="left_rubber_hand",
        right_body_name="right_rubber_hand",
        contact_frames=contact_frames,
        palm_forward_offset=0.05,
    )

    export_dir.mkdir(parents=True, exist_ok=True)

    fitted_npz = export_dir / f"{prefix}_object_fitted.npz"
    np.savez(
        fitted_npz,
        object_trans=obj_trans[:num_frames].astype(np.float32),
        object_quat_wxyz=obj_quat_wxyz[:num_frames].astype(np.float32),
        fps=np.array(float(robot_motion.get("fps", 30)), dtype=np.float32),
        mesh_scale=np.array(mesh_scale, dtype=np.float32),
        local_offset_unit=np.asarray(local_offset_unit, dtype=np.float32),
    )
    print(f"[Proxy] Saved fitted trajectory: {fitted_npz}")

    if proxy is not None:
        proxy_urdf_path = export_dir / f"{prefix}_thin_plate_proxy.urdf"
        _write_two_plate_proxy_urdf(proxy_urdf_path, proxy)

        canonical_urdf = export_dir / f"{prefix}.urdf"
        _write_two_plate_proxy_urdf(canonical_urdf, proxy)

        proxy_json = export_dir / f"{prefix}_thin_plate_proxy.json"
        proxy_data = {
            "plate_size_m": [0.15, 0.15, 0.02],
            "left_pos_local": proxy.left_pos_local.tolist(),
            "right_pos_local": proxy.right_pos_local.tolist(),
            "left_quat_wxyz": proxy.left_quat_wxyz.tolist(),
            "right_quat_wxyz": proxy.right_quat_wxyz.tolist(),
            "link_fromto_local": proxy.link_fromto_local.tolist(),
            "left_body_name": proxy.left_body_name,
            "right_body_name": proxy.right_body_name,
        }
        proxy_json.write_text(json.dumps(proxy_data, indent=2))
        print(f"[Proxy] Saved URDF: {proxy_urdf_path}")
        print(f"[Proxy] Saved JSON: {proxy_json}")

    return obj_trans[:num_frames], obj_quat_wxyz[:num_frames], proxy_urdf_path


def create_scene_cfg(proxy_urdf_path: str | None = None):
    if proxy_urdf_path is not None:
        @configclass
        class SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
            sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(
                    intensity=750.0,
                    texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                ),
            )
            robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            proxy: RigidObjectCfg = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/ProxyObject",
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=proxy_urdf_path,
                    fix_base=False,
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
        class SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
            sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(
                    intensity=750.0,
                    texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                ),
            )
            robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    return SceneCfg


def load_retargeted_motion(motion_file: str, device: str = "cpu"):
    motion_path = _resolve_path(motion_file)
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")

    data = np.load(str(motion_path), allow_pickle=True)
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    fps = float(data["fps"]) if "fps" in data else 30.0

    if args_cli.frame_range is not None:
        s, e = args_cli.frame_range
        qpos = qpos[s : e + 1]

    root_pos = qpos[:, 0:3]
    root_quat = qpos[:, 3:7]
    mujoco_joint_pos = qpos[:, 7:]
    if mujoco_joint_pos.shape[1] == 29:
        isaaclab_joint_pos = mujoco_joint_pos[:, MUJOCO_TO_ISACLAB_DOF]
    else:
        isaaclab_joint_pos = mujoco_joint_pos

    return {
        "fps": fps,
        "qpos": qpos,
        "joint_pos": torch.tensor(isaaclab_joint_pos, dtype=torch.float32, device=device),
        "root_pos": torch.tensor(root_pos, dtype=torch.float32, device=device),
        "root_quat": torch.tensor(root_quat, dtype=torch.float32, device=device),
        "num_frames": qpos.shape[0],
    }


class RetargetedMotionLoader:
    def __init__(self, motion_file, output_fps, device, frame_range):
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
        md = load_retargeted_motion(self.motion_file, device=self.device)
        self.input_fps = md["fps"]
        self.input_dt = 1.0 / self.input_fps
        self.motion_base_poss_input = md["root_pos"]
        self.motion_base_rots_input = md["root_quat"]
        self.motion_dof_poss_input = md["joint_pos"]
        self.input_frames = md["num_frames"]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(
            f"Motion loaded ({self.motion_file}), duration: {self.duration:.2f}s, "
            f"frames: {self.input_frames}, input fps: {self.input_fps}"
        )

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        i0, i1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(self.motion_base_poss_input[i0], self.motion_base_poss_input[i1], blend.unsqueeze(1))
        self.motion_base_rots = self._slerp(self.motion_base_rots_input[i0], self.motion_base_rots_input[i1], blend)
        self.motion_dof_poss = self._lerp(self.motion_dof_poss_input[i0], self.motion_dof_poss_input[i1], blend.unsqueeze(1))
        print(f"Motion interpolated, {self.input_frames} -> {self.output_frames} frames, {self.input_fps} -> {self.output_fps} fps")

    def _lerp(self, a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], blend[i])
        return out

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        i0 = (phase * (self.input_frames - 1)).floor().long()
        i1 = torch.minimum(i0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - i0
        return i0, i1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rots, dt):
        q_prev, q_next = rots[:-2], rots[2:]
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
        reset = self.current_idx >= self.output_frames
        if reset:
            self.current_idx = 0
        return state, reset


def _resample_contact_mask(contact_mask: np.ndarray, num_frames_target: int) -> np.ndarray:
    """Resample contact mask (T, 4) to target frame count. Uses nearest-neighbor."""
    n_src = contact_mask.shape[0]
    if num_frames_target == n_src:
        return contact_mask.copy()
    out = np.zeros((num_frames_target, 4), dtype=bool)
    for i in range(num_frames_target):
        src_idx = int(i * (n_src - 1) / max(1, num_frames_target - 1))
        out[i] = contact_mask[src_idx]
    return out


def _export_contact_labels_if_present(
    input_path: Path,
    output_frames: int,
    export_dir: Path,
    prefix: str,
) -> None:
    """Load contact_nodes from input dir, resample to output_frames, save to export_dir."""
    contact_src = input_path.parent / f"{input_path.stem}_contact_nodes.npy"
    if not contact_src.exists():
        return
    try:
        data = np.load(str(contact_src), allow_pickle=True)
        if isinstance(data, np.ndarray) and data.ndim == 0:
            data = data.item()
        if "contact_mask" not in data:
            return
        mask = np.asarray(data["contact_mask"], dtype=bool)
        if mask.shape[1] != 4:
            return
        mask_resampled = _resample_contact_mask(mask, output_frames)
        out_path = export_dir / f"{prefix}_contact_nodes.npy"
        out_data = {"contact_mask": mask_resampled}
        if "nodes" in data:
            out_data["nodes"] = data["nodes"]  # Keep nodes for compatibility; mask takes precedence
        np.save(str(out_path), out_data, allow_pickle=True)
        print(f"[INFO] Resampled contact labels: {out_path} ({mask.shape[0]} -> {output_frames} frames)")
    except Exception as e:
        print(f"[WARN] Contact label export failed: {e}")


def _interpolate_object_traj(obj_trans, obj_quat_wxyz, input_fps, output_fps, device):
    """Interpolate object trajectory from input_fps to output_fps."""
    n_in = len(obj_trans)
    duration = (n_in - 1) / input_fps
    dt_out = 1.0 / output_fps
    times = torch.arange(0, duration, dt_out, dtype=torch.float32)
    n_out = times.shape[0]
    phase = times / duration
    i0 = (phase * (n_in - 1)).floor().long()
    i1 = torch.minimum(i0 + 1, torch.tensor(n_in - 1))
    blend = (phase * (n_in - 1) - i0).unsqueeze(1)

    trans_t = torch.tensor(obj_trans, dtype=torch.float32)
    quat_t = torch.tensor(obj_quat_wxyz, dtype=torch.float32)

    out_trans = trans_t[i0] * (1 - blend) + trans_t[i1] * blend

    out_quat = torch.zeros(n_out, 4, dtype=torch.float32)
    for i in range(n_out):
        out_quat[i] = quat_slerp(quat_t[i0[i]], quat_t[i1[i]], blend[i, 0])

    return out_trans.to(device), out_quat.to(device), n_out


def run_simulator(sim, scene, obj_trans_interp, obj_quat_interp, has_proxy):
    motion = RetargetedMotionLoader(
        motion_file=args_cli.input_file,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    robot = scene["robot"]

    bm_root = Path(__file__).parent.parent
    output_dir = bm_root / args_cli.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args_cli.output_name}.npz"

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    try:
        proxy_art = scene["proxy"] if has_proxy else None
    except KeyError:
        proxy_art = None
    n_obj = len(obj_trans_interp) if obj_trans_interp is not None else 0

    for frame_idx in range(motion.output_frames):
        (
            (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel),
            _reset,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = base_pos + scene.env_origins
        root_states[:, 3:7] = base_rot
        root_states[:, 7:10] = base_lin_vel
        root_states[:, 10:] = base_ang_vel
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(dof_pos, dof_vel)

        if proxy_art is not None and obj_trans_interp is not None and frame_idx < n_obj:
            proxy_root = proxy_art.data.default_root_state.clone()
            proxy_root[:, :3] = obj_trans_interp[frame_idx].unsqueeze(0) + scene.env_origins
            proxy_root[:, 3:7] = obj_quat_interp[frame_idx].unsqueeze(0)
            proxy_root[:, 7:] = 0.0
            proxy_art.write_root_state_to_sim(proxy_root)

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

    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)

    np.savez(str(output_file), **log)
    print(f"[INFO] Saved motion: {output_file}")
    print(f"  Frames: {log['joint_pos'].shape[0]}, FPS: {log['fps'][0]}")
    os._exit(0)


def main():
    input_path = _resolve_path(args_cli.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Robot motion not found: {input_path}")

    if args_cli.object_motion:
        object_path = _resolve_path(args_cli.object_motion)
    else:
        object_path = input_path.with_name(f"{input_path.stem}_object.npz")

    obj_trans_fitted = None
    obj_quat_fitted = None
    proxy_urdf_path = None
    has_proxy = False

    if not args_cli.skip_proxy and object_path.exists():
        export_dir = Path(args_cli.proxy_export_dir).resolve() if args_cli.proxy_export_dir else input_path.parent
        prefix = args_cli.proxy_export_prefix or args_cli.output_name
        try:
            obj_trans_fitted, obj_quat_fitted, proxy_urdf_path = _run_proxy_fitting(
                input_path, object_path, export_dir, prefix
            )
            if proxy_urdf_path is not None and proxy_urdf_path.exists():
                has_proxy = True
                print(f"[INFO] Proxy URDF: {proxy_urdf_path}")
        except Exception as e:
            print(f"[WARN] Proxy fitting failed ({e}), continuing robot-only.")
    elif not object_path.exists():
        print(f"[WARN] Object motion not found ({object_path}), running robot-only conversion.")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    SceneCfg = create_scene_cfg(str(proxy_urdf_path) if has_proxy else None)
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    obj_trans_interp = None
    obj_quat_interp = None
    if obj_trans_fitted is not None and has_proxy:
        robot_data = np.load(str(input_path), allow_pickle=True)
        input_fps = float(robot_data["fps"]) if "fps" in robot_data else 30.0
        obj_trans_interp, obj_quat_interp, _ = _interpolate_object_traj(
            obj_trans_fitted, obj_quat_fitted, input_fps, args_cli.output_fps, sim.device
        )

    # Get output frame count for contact label resampling and object trajectory export
    motion_preview = RetargetedMotionLoader(
        motion_file=args_cli.input_file,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )
    output_frames = motion_preview.output_frames

    # Resample and export contact labels if present (hand-object only, bm_generalist format)
    if args_cli.proxy_export_dir:
        export_dir = Path(args_cli.proxy_export_dir).resolve()
        prefix = args_cli.proxy_export_prefix or args_cli.output_name
        _export_contact_labels_if_present(input_path, output_frames, export_dir, prefix)

        # Overwrite object_fitted.npz with interpolated trajectory so frame count matches motion (output_fps)
        if obj_trans_interp is not None and has_proxy:
            fitted_npz = export_dir / f"{prefix}_object_fitted.npz"
            if fitted_npz.exists():
                existing = np.load(str(fitted_npz), allow_pickle=True)
                update = {
                    "object_trans": obj_trans_interp.cpu().numpy().astype(np.float32),
                    "object_quat_wxyz": obj_quat_interp.cpu().numpy().astype(np.float32),
                    "fps": np.array(float(args_cli.output_fps), dtype=np.float32),
                }
                for k in existing.keys():
                    if k not in update:
                        update[k] = np.asarray(existing[k])
                np.savez(str(fitted_npz), **update)
                print(
                    f"[Proxy] Updated object_fitted.npz with interpolated trajectory "
                    f"({output_frames} frames @ {args_cli.output_fps} fps)"
                )

    run_simulator(sim, scene, obj_trans_interp, obj_quat_interp, has_proxy)


if __name__ == "__main__":
    main()
    simulation_app.close()
