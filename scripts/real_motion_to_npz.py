"""Convert already-processed real motion npz (dataset_real_filtered format) to training npz + contact_nodes.npy.

This mimics ``qpos_contact_labels_to_npz.py`` but reads a different input schema. The input here is
already sampled at a fixed fps with base/joint states AND velocities precomputed, so NO interpolation is
performed — we replay the frames as-is through Isaac Sim purely to obtain forward-kinematics body states.

Input npz schema (dataset_real_filtered/*.npz):
  - motion_base_pos:     (T, 3)   root world position
  - motion_base_rot:     (T, 4)   root world quaternion (w, x, y, z)
  - motion_base_lin_vel: (T, 3)   root world linear velocity
  - motion_base_ang_vel: (T, 3)   root world angular velocity
  - motion_dof_pos:      (T, 29)  joint positions, order given by ``dof_names``
  - motion_dof_vel:      (T, 29)  joint velocities, same order as ``dof_names``
  - timestamp:           (T,)     seconds; fps is derived from this (overridable via --output_fps)
  - dof_names:           (29,)    joint names (MuJoCo/native order: legs, waist, arms)
  - motion_contact_mask: (T, 4)   [L_foot, R_foot, L_wrist, R_wrist], 0/1

Joint order:
  ``dof_names`` is the native (MuJoCo) order. Isaac Lab uses a different (BFS) order. Instead of a
  hardcoded index table we build the remap dynamically from ``dof_names`` -> ``robot.data.joint_names``,
  so the input joints are placed into the exact order the articulation expects. The logged
  ``joint_pos``/``joint_vel`` are therefore in Isaac Lab order (same as the reference script's output).

Output (identical key names / layout to qpos_contact_labels_to_npz.py):
  - {output_name}/{stem}.npz with keys:
        fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
  - {prefix}_contact_nodes.npy with {"contact_mask": bool (T, 4)}

Usage:
    python scripts/real_motion_to_npz.py \\
        --input_file dataset_real_filtered/up/h0.190_w0.250_s0.4_t0.npz --output_name my_motion

    # Directory: every *.npz (non-recursive) is converted to {output_name}/{stem}/{stem}.npz
    python scripts/real_motion_to_npz.py --input_file dataset_real_filtered/up --output_name up_export
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Convert processed real motion (dataset_real_filtered format) to training npz and contact_nodes.npy."
)
parser.add_argument(
    "--input_file",
    type=str,
    default=None,
    help="Input .npz file, or a directory of .npz files (non-recursive). Omit when using --manifest.",
)
parser.add_argument(
    "--manifest",
    type=str,
    default=None,
    help="Batch mode: CSV with columns input_npz,output_name (from terrain_xml_to_data.py). "
    "Motion npz -> <data_root>/motion_dir/<output_name>.npz, contact -> <data_root>/terrain_folder/. "
    "Overrides --input_file/--output_name/--contact_export_dir.",
)
parser.add_argument(
    "--data_root",
    type=str,
    default=None,
    help="With --manifest: data root holding motion_dir/ and terrain_folder/ (default: manifest's parent).",
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Frame range START END (inclusive). Applied to all per-frame arrays together.",
)
parser.add_argument(
    "--output_name",
    type=str,
    default=None,
    help="Trajectory folder under repo root (single file: {name}/{name}.npz; folder: {name}/{stem}/{stem}.npz). "
    "Required unless --manifest is given.",
)
parser.add_argument(
    "--output_fps",
    type=int,
    default=None,
    help="Override output FPS written to the npz (default: derived from timestamps, rounded)",
)
parser.add_argument(
    "--contact_export_dir",
    type=str,
    default=None,
    help="Directory for {prefix}_contact_nodes.npy (default: same folder as each output .npz)",
)
parser.add_argument(
    "--contact_export_prefix",
    type=str,
    default=None,
    help="Filename prefix for contact export (default: output_name; ignored for directory input)",
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

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

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


def _build_manifest_jobs() -> list[dict]:
    """Batch jobs from a CSV (input_npz, output_name). Motion -> motion_dir, contact -> terrain_folder."""
    import csv

    manifest_path = _resolve_path(args_cli.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    data_root = (
        Path(args_cli.data_root).resolve()
        if args_cli.data_root is not None
        else manifest_path.parent
    )
    motion_dir = data_root / "motion_dir"
    terrain_dir = data_root / "terrain_folder"
    motion_dir.mkdir(parents=True, exist_ok=True)
    terrain_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row["output_name"].strip()
            jobs.append(
                {
                    "input": str(_resolve_path(row["input_npz"].strip())),
                    "output_dir": motion_dir,  # motion npz -> motion_dir/<name>.npz (flat, no nesting)
                    "npz_stem": name,
                    "contact_prefix": name,
                    "contact_dir": terrain_dir,  # contact -> terrain_folder/<name>_contact_nodes.npy
                }
            )
    if not jobs:
        raise ValueError(f"Manifest has no rows: {manifest_path}")
    print(f"[INFO] Manifest mode: {len(jobs)} motion(s) -> {motion_dir} + {terrain_dir}")
    return jobs


def _build_jobs() -> tuple[list[dict], bool]:
    """Each job: input path str, output_dir Path, npz_stem str, contact_prefix str. Second value: from_directory."""
    if args_cli.manifest is not None:
        return _build_manifest_jobs(), True
    if args_cli.input_file is None or args_cli.output_name is None:
        raise ValueError("Provide --input_file and --output_name, or use --manifest.")
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


def _fps_from_timestamps(timestamp: np.ndarray) -> float:
    """Derive fps from the timestamp vector (median dt for robustness)."""
    if timestamp.shape[0] < 2:
        return 50.0
    dt = np.median(np.diff(timestamp.astype(np.float64)))
    if dt <= 0:
        return 50.0
    return float(1.0 / dt)


def load_processed_motion(motion_file: str):
    """Load base/dof states, velocities, contact mask and dof_names; apply --frame_range."""
    motion_path = _resolve_path(motion_file)
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")

    data = np.load(str(motion_path), allow_pickle=True)
    required = [
        "motion_base_pos",
        "motion_base_rot",
        "motion_base_lin_vel",
        "motion_base_ang_vel",
        "motion_dof_pos",
        "motion_dof_vel",
        "dof_names",
    ]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"Missing keys {missing} in {motion_path.name}; keys: {list(data.files)}")

    base_pos = np.asarray(data["motion_base_pos"], dtype=np.float64)
    base_rot = np.asarray(data["motion_base_rot"], dtype=np.float64)
    base_lin_vel = np.asarray(data["motion_base_lin_vel"], dtype=np.float64)
    base_ang_vel = np.asarray(data["motion_base_ang_vel"], dtype=np.float64)
    dof_pos = np.asarray(data["motion_dof_pos"], dtype=np.float64)
    dof_vel = np.asarray(data["motion_dof_vel"], dtype=np.float64)
    dof_names = [str(n) for n in data["dof_names"]]

    num_frames = base_pos.shape[0]

    if "motion_contact_mask" in data.files:
        contact_mask = (np.asarray(data["motion_contact_mask"]) > 0.5).astype(bool)
    else:
        print(
            f"Warning: 'motion_contact_mask' missing in {motion_path.name}; "
            f"using all-false labels for {num_frames} frames."
        )
        contact_mask = np.zeros((num_frames, 4), dtype=bool)

    if contact_mask.shape[0] != num_frames:
        raise ValueError(f"contact_mask frames {contact_mask.shape[0]} != motion frames {num_frames}")

    if "timestamp" in data.files:
        fps = _fps_from_timestamps(np.asarray(data["timestamp"]))
    else:
        fps = 50.0

    if args_cli.frame_range is not None:
        s, e = args_cli.frame_range
        sl = slice(s, e + 1)
        base_pos, base_rot = base_pos[sl], base_rot[sl]
        base_lin_vel, base_ang_vel = base_lin_vel[sl], base_ang_vel[sl]
        dof_pos, dof_vel = dof_pos[sl], dof_vel[sl]
        contact_mask = contact_mask[sl]
        num_frames = base_pos.shape[0]
        print(f"Frame range [{s}, {e}] -> {num_frames} frames")

    out_fps = int(round(fps)) if args_cli.output_fps is None else int(args_cli.output_fps)

    return {
        "base_pos": base_pos,
        "base_rot": base_rot,
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "dof_names": dof_names,
        "contact_mask": contact_mask,
        "num_frames": num_frames,
        "fps": fps,
        "out_fps": out_fps,
    }


def _build_dof_reorder(src_names: list[str], dst_names: list[str]) -> list[int]:
    """Index list mapping source (dof_names) columns into destination (robot) joint order."""
    name_to_src = {n: i for i, n in enumerate(src_names)}
    missing = [n for n in dst_names if n not in name_to_src]
    if missing:
        raise ValueError(
            f"Robot joints not present in input dof_names: {missing}\n"
            f"input dof_names: {src_names}"
        )
    extra = [n for n in src_names if n not in set(dst_names)]
    if extra:
        print(f"Warning: input dof_names not used by robot (dropped): {extra}")
    return [name_to_src[n] for n in dst_names]


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


def _export_contact_nodes(contact_mask: np.ndarray, export_dir: Path, prefix: str) -> None:
    """Write {prefix}_contact_nodes.npy in omomo_object_to_npz format (contact_mask already at output frames)."""
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / f"{prefix}_contact_nodes.npy"
    np.save(str(out_path), {"contact_mask": contact_mask.astype(bool)}, allow_pickle=True)
    print(f"[INFO] Saved contact labels: {out_path} ({contact_mask.shape[0]} frames)")


def _process_one_job(sim: SimulationContext, scene: InteractiveScene, job: dict) -> bool:
    """Replay one motion frame-by-frame, export contact_nodes, write training npz. False if app closed early."""
    md = load_processed_motion(job["input"])
    num_frames = md["num_frames"]
    print(
        f"Motion loaded ({job['input']}), frames: {num_frames}, "
        f"input fps: {md['fps']:.2f}, output fps: {md['out_fps']}"
    )

    robot = scene["robot"]
    reorder = _build_dof_reorder(md["dof_names"], list(robot.data.joint_names))

    device = sim.device
    base_pos = torch.tensor(md["base_pos"], dtype=torch.float32, device=device)
    base_rot = torch.tensor(md["base_rot"], dtype=torch.float32, device=device)
    base_lin_vel = torch.tensor(md["base_lin_vel"], dtype=torch.float32, device=device)
    base_ang_vel = torch.tensor(md["base_ang_vel"], dtype=torch.float32, device=device)
    dof_pos = torch.tensor(md["dof_pos"][:, reorder], dtype=torch.float32, device=device)
    dof_vel = torch.tensor(md["dof_vel"][:, reorder], dtype=torch.float32, device=device)

    # contact export (already at output frame count, no resampling needed)
    if job.get("contact_dir") is not None:
        contact_dir = job["contact_dir"]  # manifest mode: explicit terrain_folder
    elif args_cli.contact_export_dir is not None:
        contact_dir = Path(args_cli.contact_export_dir).resolve()
    else:
        contact_dir = job["output_dir"]
    _export_contact_nodes(md["contact_mask"], contact_dir, job["contact_prefix"])

    job["output_dir"].mkdir(parents=True, exist_ok=True)
    output_file = job["output_dir"] / f"{job['npz_stem']}.npz"

    log = {
        "fps": [md["out_fps"]],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    for i in range(num_frames):
        if not simulation_app.is_running():
            return False

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = base_pos[i : i + 1] + scene.env_origins
        root_states[:, 3:7] = base_rot[i : i + 1]
        root_states[:, 7:10] = base_lin_vel[i : i + 1]
        root_states[:, 10:] = base_ang_vel[i : i + 1]
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(dof_pos[i : i + 1], dof_vel[i : i + 1])
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
