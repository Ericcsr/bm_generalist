#!/usr/bin/env python3
"""
Convert bones-style G1 CSV motion to bm_generalist training NPZ (same schema as
``scripts/omni_to_npz.py`` / ``MotionLoader``).

CSV format (``retargeted_from_gmr`` / bones_terrain):
  - Header: Frame, root_translate{X,Y,Z} (cm), root_rotate{X,Y,Z} (deg, Euler xyz),
    then 29 joint *_dof columns (deg) in **MuJoCo** order.

**Default (Isaac Lab FK)**: launches Isaac Sim via ``AppLauncher`` (same stack as
``omni_to_npz.py``), resamples with torch (lerp / ``quat_slerp``), writes robot state to
the articulation, and logs ``robot.data.*`` after FK. Pass ``--headless`` (from
``AppLauncher``) to run without a visible window; camera follow is disabled when headless.

**``--mujoco-fk``**: offline path using MuJoCo only (no Isaac); uses NumPy/SciPy resampling.

Inputs (mutually exclusive):
  - ``--csv``: single file -> ``<output-parent>/<stem>/<stem>.npz``
  - ``--csv-list``: text file with one CSV path per line ->
    ``<output-parent>/<list_basename>/<csv_stem>/<csv_stem>.npz``
  - ``--csv-dir``: directory tree; all ``*.csv`` files are found **recursively** (including
    subfolders). Outputs mirror relative paths:
    ``<output-parent>/<dir_basename>/<rel_path>.npz`` (same subfolders as under the source dir;
    ``.csv`` replaced by ``.npz``)

Usage (Isaac, headless):
  ./isaaclab.sh -p scripts/csv_bones_to_training_npz.py --headless \\
    --csv motion.csv --output-parent ./out --source-fps 120

Usage (MuJoCo offline):
  python scripts/csv_bones_to_training_npz.py --mujoco-fk \\
    --csv motion.csv --output-parent ./out
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

try:
    from isaaclab.app import AppLauncher
except ImportError:  # pragma: no cover - MuJoCo-only envs
    AppLauncher = None  # type: ignore[misc, assignment]

# Same mapping as scripts/omni_to_npz.py (MuJoCo column index -> IsaacLab joint row index)
MUJOCO_TO_ISACLAB_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19,
    26, 20, 27, 21, 28,
]

BM_GENERALIST_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROBOT_XML = (
    BM_GENERALIST_ROOT.parent / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bones CSV -> bm_generalist training npz (Isaac FK or optional MuJoCo).",
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", type=str, help="Single input CSV path")
    g.add_argument("--csv-list", type=str, help="Text file: one CSV path per line")
    g.add_argument("--csv-dir", type=str, help="Directory of CSV files")
    parser.add_argument(
        "--output-parent",
        type=str,
        required=True,
        help="Parent directory for outputs (see docstring for layout)",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=120.0,
        help="FPS of CSV rows before resampling (default 120)",
    )
    parser.add_argument("--output-fps", type=int, default=50, help="Training FPS (default 50)")
    parser.add_argument(
        "--mujoco-fk",
        action="store_true",
        help="Use MuJoCo FK only (no Isaac Sim). Ignores AppLauncher / Isaac.",
    )
    parser.add_argument(
        "--robot-xml",
        type=str,
        default=None,
        help=f"MuJoCo XML for FK when using --mujoco-fk (default: {DEFAULT_ROBOT_XML})",
    )
    if AppLauncher is not None:
        AppLauncher.add_app_launcher_args(parser)
    else:
        parser.add_argument(
            "--device",
            type=str,
            default="cuda:0",
            help="(Unused for --mujoco-fk.) Pass the same flags as omni_to_npz when using Isaac.",
        )
    return parser


def load_bones_gmr_csv_to_qpos(csv_path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        _header = next(reader, None)
        if _header is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])
    if not rows:
        raise ValueError(f"No data rows in {csv_path}")
    data = np.asarray(rows, dtype=np.float64)
    if data.shape[1] < 7 + 29:
        raise ValueError(f"{csv_path}: need at least 36 columns after frame, got width {data.shape[1]}")
    data = data[:, 1:]
    root_t = data[:, :3] / 100.0
    euler_deg = data[:, 3:6]
    joints = data[:, 6 : 6 + 29] * (np.pi / 180.0)
    while np.any(joints > 4):
        joints[joints > 4] -= np.pi * 2
    while np.any(joints < -4):
        joints[joints < -4] += np.pi * 2
    quat_xyzw = R.from_euler("xyz", euler_deg, degrees=True).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    qpos = np.concatenate([root_t, quat_wxyz, joints], axis=-1)
    return qpos.astype(np.float64)


def _ensure_at_least_two_frames(qpos: np.ndarray) -> np.ndarray:
    if qpos.shape[0] < 2:
        return np.vstack([qpos, qpos[-1:]])
    return qpos


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = np.split(q2, 4, axis=-1)
    return np.concatenate(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _axis_angle_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    w = np.clip(q[..., 0], -1.0, 1.0)
    vec = q[..., 1:]
    angle = 2.0 * np.arccos(w)
    small = np.abs(angle) < 1e-6
    axis = np.zeros_like(vec)
    axis[~small] = vec[~small] / (np.linalg.norm(vec[~small], axis=-1, keepdims=True) + 1e-12)
    return axis * angle[..., np.newaxis]


def so3_derivative_wxyz(rotations: np.ndarray, dt: float) -> np.ndarray:
    if rotations.shape[0] < 3:
        return np.zeros((rotations.shape[0], 3), dtype=np.float64)
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = _quat_mul_wxyz(q_next, _quat_conj_wxyz(q_prev))
    omega = _axis_angle_from_quat_wxyz(q_rel) / (2.0 * dt)
    return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)


def resample_qpos_mujoco(
    qpos_in: np.ndarray,
    input_fps: float,
    output_fps: float,
) -> np.ndarray:
    t_in, _ = qpos_in.shape
    if t_in < 2:
        return qpos_in.astype(np.float64)
    dt_in = 1.0 / float(input_fps)
    duration = (t_in - 1) * dt_in
    dt_out = 1.0 / float(output_fps)
    times = np.arange(0.0, duration, dt_out, dtype=np.float64)
    if times.size == 0:
        times = np.array([0.0], dtype=np.float64)

    phase = times / duration
    idx0 = np.floor(phase * (t_in - 1)).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, t_in - 1)
    blend = phase * (t_in - 1) - idx0
    blend = np.clip(blend, 0.0, 1.0)

    root_p0, root_p1 = qpos_in[idx0, :3], qpos_in[idx1, :3]
    root_pos = root_p0 * (1.0 - blend[:, None]) + root_p1 * blend[:, None]

    q0 = qpos_in[idx0, 3:7]
    q1 = qpos_in[idx1, 3:7]
    key_times = np.array([0.0, 1.0], dtype=np.float64)
    root_quat_out = np.zeros((len(blend), 4), dtype=np.float64)
    for i in range(len(blend)):
        key_rots = R.from_quat(
            np.stack([q0[i, [1, 2, 3, 0]], q1[i, [1, 2, 3, 0]]], axis=0)
        )
        slerp = Slerp(key_times, key_rots)
        q_xyzw = slerp(float(blend[i])).as_quat().reshape(4)
        root_quat_out[i] = q_xyzw[[3, 0, 1, 2]]

    j0, j1 = qpos_in[idx0, 7:36], qpos_in[idx1, 7:36]
    joints = j0 * (1.0 - blend[:, None]) + j1 * blend[:, None]

    return np.concatenate([root_pos, root_quat_out, joints], axis=-1)


def mujoco_fk_body_states(
    qpos: np.ndarray,
    robot_xml: Path,
) -> tuple[np.ndarray, np.ndarray, int]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    if qpos.shape[1] != model.nq:
        if qpos.shape[1] > model.nq:
            qpos = qpos[:, : model.nq]
        else:
            pad = np.zeros((qpos.shape[0], model.nq - qpos.shape[1]), dtype=np.float64)
            qpos = np.hstack([qpos, pad])
    nbody = model.nbody
    if nbody < 2:
        raise RuntimeError("Model has no dynamic bodies")
    b_out = nbody - 1
    t_total = qpos.shape[0]
    pos = np.zeros((t_total, b_out, 3), dtype=np.float64)
    quat = np.zeros((t_total, b_out, 4), dtype=np.float64)
    for t in range(t_total):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        pos[t] = data.xpos[1 : b_out + 1].copy()
        quat[t] = data.xquat[1 : b_out + 1].copy()
    return pos, quat, b_out


def build_training_npz_mujoco(
    csv_path: Path,
    output_npz: Path,
    source_fps: float,
    output_fps: int,
    robot_xml: Path,
) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    qpos_raw = load_bones_gmr_csv_to_qpos(csv_path)
    qpos = resample_qpos_mujoco(qpos_raw, source_fps, float(output_fps))
    t = qpos.shape[0]

    mujoco_joint = qpos[:, 7:36]
    joint_pos = mujoco_joint[:, MUJOCO_TO_ISACLAB_DOF].astype(np.float32)
    dt = 1.0 / float(output_fps)
    joint_vel = np.gradient(joint_pos, dt, axis=0, edge_order=2).astype(np.float32)

    body_pos, body_quat, _ = mujoco_fk_body_states(qpos, robot_xml)
    body_lin_vel = np.gradient(body_pos, dt, axis=0, edge_order=2).astype(np.float32)
    body_ang_vel = so3_derivative_wxyz(body_quat.astype(np.float64), dt).astype(np.float32)

    np.savez_compressed(
        str(output_npz),
        fps=np.array([output_fps], dtype=np.int32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
    )
    print(f"[OK] {csv_path.name} -> {output_npz}  frames={t} fps={output_fps} bodies={body_pos.shape[1]} (mujoco)")


def _collect_csvs_from_dir(d: Path) -> list[Path]:
    """All ``*.csv`` under ``d`` (recursive). Sorted for deterministic runs."""
    root = d.resolve()
    return sorted(p.resolve() for p in root.rglob("*.csv") if p.is_file())


def _csv_dir_output_path(batch_root: Path, root_dir: Path, csv_path: Path) -> Path:
    """NPZ path under ``batch_root`` mirroring ``csv_path`` relative to ``root_dir``."""
    rel = csv_path.resolve().relative_to(root_dir.resolve())
    return batch_root / rel.with_suffix(".npz")


def _collect_csvs_from_list(list_path: Path) -> list[Path]:
    paths: list[Path] = []
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            p = Path(line).expanduser()
            if not p.is_absolute():
                p = (list_path.parent / p).resolve()
            else:
                p = p.resolve()
            if not p.is_file():
                print(f"[WARN] skip missing file: {p}", file=sys.stderr)
                continue
            paths.append(p)
    return paths


def run_mujoco_main(args: argparse.Namespace) -> None:
    robot_xml = Path(args.robot_xml).resolve() if args.robot_xml else DEFAULT_ROBOT_XML.resolve()
    if not robot_xml.is_file():
        print(f"Error: robot XML not found: {robot_xml}", file=sys.stderr)
        sys.exit(1)

    out_parent = Path(args.output_parent).resolve()
    out_parent.mkdir(parents=True, exist_ok=True)

    if args.csv:
        csv_path = Path(args.csv).resolve()
        if not csv_path.is_file():
            print(f"Error: not a file: {csv_path}", file=sys.stderr)
            sys.exit(1)
        stem = csv_path.stem
        batch_dir = out_parent / stem
        build_training_npz_mujoco(
            csv_path, batch_dir / f"{stem}.npz", args.source_fps, args.output_fps, robot_xml
        )
        return

    if args.csv_list:
        list_path = Path(args.csv_list).resolve()
        if not list_path.is_file():
            print(f"Error: not a file: {list_path}", file=sys.stderr)
            sys.exit(1)
        batch_root = out_parent / list_path.stem
        csvs = _collect_csvs_from_list(list_path)
        if not csvs:
            print("No valid CSV paths in list.", file=sys.stderr)
            sys.exit(1)
        for csv_path in csvs:
            stem = csv_path.stem
            build_training_npz_mujoco(
                csv_path,
                batch_root / stem / f"{stem}.npz",
                args.source_fps,
                args.output_fps,
                robot_xml,
            )
        return

    if args.csv_dir:
        dir_path = Path(args.csv_dir).resolve()
        if not dir_path.is_dir():
            print(f"Error: not a directory: {dir_path}", file=sys.stderr)
            sys.exit(1)
        batch_root = out_parent / dir_path.name
        csvs = _collect_csvs_from_dir(dir_path)
        if not csvs:
            print(f"No .csv files under {dir_path} (recursive)", file=sys.stderr)
            sys.exit(1)
        for csv_path in csvs:
            out_npz = _csv_dir_output_path(batch_root, dir_path, csv_path)
            build_training_npz_mujoco(
                csv_path,
                out_npz,
                args.source_fps,
                args.output_fps,
                robot_xml,
            )


# ----- Isaac path (imports after AppLauncher in __main__) -----


def _run_isaac_main_after_launch(args: argparse.Namespace) -> None:
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

    from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

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

    class QposMotionLoader:
        """Same resampling / velocities as omni_to_npz.RetargetedMotionLoader, from CSV qpos."""

        def __init__(
            self,
            qpos: np.ndarray,
            input_fps: float,
            output_fps: int,
            device: torch.device,
            label: str,
        ):
            self.output_fps = output_fps
            self.output_dt = 1.0 / self.output_fps
            self.device = device
            self.input_fps = input_fps
            self.input_dt = 1.0 / self.input_fps
            self.label = label

            qpos = _ensure_at_least_two_frames(qpos)
            root_pos = torch.tensor(qpos[:, 0:3], dtype=torch.float32, device=device)
            root_quat = torch.tensor(qpos[:, 3:7], dtype=torch.float32, device=device)
            mujoco_j = qpos[:, 7:36]
            isaac_j = mujoco_j[:, MUJOCO_TO_ISACLAB_DOF]
            joint_pos = torch.tensor(isaac_j, dtype=torch.float32, device=device)

            self.motion_base_poss_input = root_pos
            self.motion_base_rots_input = root_quat
            self.motion_dof_poss_input = joint_pos
            self.input_frames = qpos.shape[0]
            self.duration = (self.input_frames - 1) * self.input_dt
            print(
                f"[INFO] {label}: duration {self.duration:.3f}s, input frames {self.input_frames},"
                f" input fps {self.input_fps}"
            )
            self._interpolate_motion()
            self._compute_velocities()

        def _interpolate_motion(self) -> None:
            times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
            if times.numel() == 0:
                times = torch.zeros(1, device=self.device, dtype=torch.float32)
            self.output_frames = int(times.shape[0])
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
                f"[INFO] {self.label}: output frames {self.output_frames}, output fps {self.output_fps}"
            )

        def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
            return a * (1 - blend) + b * blend

        def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
            slerped_quats = torch.zeros_like(a)
            for i in range(a.shape[0]):
                slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
            return slerped_quats

        def _compute_frame_blend(self, times: torch.Tensor):
            phase = times / self.duration
            index_0 = (phase * (self.input_frames - 1)).floor().long()
            bound = torch.tensor(self.input_frames - 1, device=self.device, dtype=torch.long)
            index_1 = torch.minimum(index_0 + 1, bound)
            blend = phase * (self.input_frames - 1) - index_0.float()
            return index_0, index_1, blend

        def _compute_velocities(self) -> None:
            self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
            self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
            self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

        def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
            q_prev, q_next = rotations[:-2], rotations[2:]
            q_rel = quat_mul(q_next, quat_conjugate(q_prev))
            omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
            return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def export_one_csv(
        sim: SimulationContext,
        scene: InteractiveScene,
        csv_path: Path,
        output_npz: Path,
        headless: bool,
        motion_label: str | None = None,
    ) -> None:
        qpos_np = load_bones_gmr_csv_to_qpos(csv_path)
        motion = QposMotionLoader(
            qpos_np,
            args.source_fps,
            args.output_fps,
            sim.device,
            label=motion_label if motion_label is not None else csv_path.name,
        )
        robot = scene["robot"]
        output_npz.parent.mkdir(parents=True, exist_ok=True)

        log: dict = {
            "fps": [args.output_fps],
            "joint_pos": [],
            "joint_vel": [],
            "body_pos_w": [],
            "body_quat_w": [],
            "body_lin_vel_w": [],
            "body_ang_vel_w": [],
        }

        for i in range(motion.output_frames):
            motion_base_pos = motion.motion_base_poss[i : i + 1]
            motion_base_rot = motion.motion_base_rots[i : i + 1]
            motion_base_lin_vel = motion.motion_base_lin_vels[i : i + 1]
            motion_base_ang_vel = motion.motion_base_ang_vels[i : i + 1]
            motion_dof_pos = motion.motion_dof_poss[i : i + 1]
            motion_dof_vel = motion.motion_dof_vels[i : i + 1]

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

            if not headless:
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

        np.savez(str(output_npz), **log)
        print(f"[OK] {csv_path.name} -> {output_npz}  frames={log['joint_pos'].shape[0]} fps={args.output_fps} (isaac)")

    sim_cfg = sim_utils.SimulationCfg(device=args.device)
    sim_cfg.dt = 1.0 / args.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    out_parent = Path(args.output_parent).resolve()
    out_parent.mkdir(parents=True, exist_ok=True)
    headless = bool(getattr(args, "headless", False))

    if args.csv:
        csv_path = Path(args.csv).resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        stem = csv_path.stem
        export_one_csv(sim, scene, csv_path, out_parent / stem / f"{stem}.npz", headless=headless)
        return

    if args.csv_list:
        list_path = Path(args.csv_list).resolve()
        if not list_path.is_file():
            raise FileNotFoundError(list_path)
        batch_root = out_parent / list_path.stem
        csvs = _collect_csvs_from_list(list_path)
        if not csvs:
            print("No valid CSV paths in list.", file=sys.stderr)
            sys.exit(1)
        for csv_path in csvs:
            stem = csv_path.stem
            export_one_csv(sim, scene, csv_path, batch_root / stem / f"{stem}.npz", headless=headless)
        return

    if args.csv_dir:
        dir_path = Path(args.csv_dir).resolve()
        if not dir_path.is_dir():
            raise NotADirectoryError(dir_path)
        batch_root = out_parent / dir_path.name
        csvs = _collect_csvs_from_dir(dir_path)
        if not csvs:
            print(f"No .csv files under {dir_path} (recursive)", file=sys.stderr)
            sys.exit(1)
        for csv_path in csvs:
            out_npz = _csv_dir_output_path(batch_root, dir_path, csv_path)
            rel_label = csv_path.resolve().relative_to(dir_path.resolve()).as_posix()
            export_one_csv(sim, scene, csv_path, out_npz, headless=headless, motion_label=rel_label)


if __name__ == "__main__":
    parser = build_parser()
    args_cli = parser.parse_args()

    if args_cli.mujoco_fk:
        run_mujoco_main(args_cli)
    else:
        if AppLauncher is None:
            print(
                "Isaac Lab (isaaclab.app.AppLauncher) is not importable. "
                "Install Isaac Lab / run from the Isaac conda env, or use --mujoco-fk.",
                file=sys.stderr,
            )
            sys.exit(1)
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
        try:
            _run_isaac_main_after_launch(args_cli)
        finally:
            simulation_app.close()
