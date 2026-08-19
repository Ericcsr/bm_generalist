#!/usr/bin/env python3
"""Replay a combined scene (motion + object + terrain) in Isaac Sim, with video export.

Given one motion .npz from ``data/motion_dir``, this script auto-discovers the
matching object and terrain assets and replays them together in one environment
(kinematic replay, no physics on the robot):

  - robot   : G1 humanoid driven by the motion (root from body_pos_w, joint_pos)
  - object  : {stem}_thin_plate_proxy.urdf + {stem}_object_fitted.npz  (object_folder)
  - terrain : {base}_terrain.urdf  where base = stem with trailing _motion_<N> stripped

Whichever of object / terrain is missing for the given motion is simply skipped
(the two dataset groups are disjoint: OMOMO motions have an object on flat
ground; terrain motions have terrain but no object).

Three modes:

  1. Interactive (default): open the GUI and replay one motion.

        python scripts/replay_scene.py --motion_file data/motion_dir/train_sub7_largebox_029.npz

  2. Record one video (headless): a follow-camera tracks the robot, RGB frames
     are written to <out_dir>/<stem>.mp4, then the app exits.

        python scripts/replay_scene.py --motion_file data/motion_dir/train_sub7_largebox_029.npz \
            --record --headless --enable_cameras --out_dir videos

  3. Batch (headless): render one video per motion file. Isaac Sim is launched
     ONCE; each motion is rendered by building a fresh USD stage (create_new_stage)
     with a new SimulationContext + scene, so different object/terrain URDFs never
     clash. This avoids the ~20s app-startup cost paid per file by a subprocess.
     --headless/--enable_cameras are enabled automatically. --batch takes a
     directory or a glob.

        python scripts/replay_scene.py --batch data/motion_dir --out_dir videos
        python scripts/replay_scene.py --batch "data/motion_dir/train_sub7_*.npz" --out_dir videos --limit 20
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import glob as _glob
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay motion + object + terrain scene, with video export.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to a single motion .npz file.")
parser.add_argument("--batch", type=str, default=None, help="Directory or glob of motion .npz files -> one video each (single Isaac process, fresh stage per file).")
parser.add_argument("--object_folder", type=str, default=None, help="Object folder (default: <data_root>/object_folder).")
parser.add_argument("--terrain_folder", type=str, default=None, help="Terrain folder (default: <data_root>/terrain_folder).")
parser.add_argument("--no_object", action="store_true", help="Do not load the object even if found.")
parser.add_argument("--no_terrain", action="store_true", help="Do not load the terrain even if found.")
parser.add_argument("--loop", action="store_true", help="Loop the motion (interactive mode only).")
# Video / recording options
parser.add_argument("--record", action="store_true", help="Render a follow-camera video for one motion, then exit.")
parser.add_argument("--out_dir", type=str, default="videos", help="Output directory for videos.")
parser.add_argument("--width", type=int, default=1280, help="Video width in pixels.")
parser.add_argument("--height", type=int, default=720, help="Video height in pixels.")
parser.add_argument("--cam_offset", type=float, nargs=3, default=[2.5, 2.5, 1.5], help="Follow-camera offset from robot root (x y z).")
parser.add_argument("--overwrite", action="store_true", help="Re-render even if the output video already exists.")
parser.add_argument("--limit", type=int, default=None, help="Batch mode: process at most this many files.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.batch and not args_cli.motion_file:
    parser.error("either --motion_file (single) or --batch (many) is required")

# Batch and single-record modes both need the renderer + a camera sensor. Enable them
# up-front so the user does not have to pass --headless/--enable_cameras themselves.
if args_cli.batch or args_cli.record:
    args_cli.enable_cameras = True
    if args_cli.batch:
        args_cli.headless = True

# Pre-import the torch -> _dynamo -> triton chain BEFORE the renderer starts.
# With --enable_cameras, AppLauncher loads the rendering kit (omni.replicator);
# if triton is then imported lazily during scene setup (via torchvision -> torch._dynamo)
# it segfaults under Vulkan ("Fatal Python error: Segmentation fault", triton/knobs.py).
# Importing the chain up-front in a clean process makes the later import a no-op.
import torch  # noqa: E402, F401
try:
    import torch._dynamo  # noqa: E402, F401  (pulls in triton)
    import torchvision  # noqa: E402, F401  (the actual lazy importer during scene setup)
except Exception:
    pass

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import gc
import re

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from isaacsim.core.utils.stage import create_new_stage

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg_3pt_binded_multi_object import (
    _make_proxy_rigid_object_cfg_multi_urdf,
)

# MuJoCo -> IsaacLab joint order (only used if a motion stores raw qpos; the
# data/motion_dir files already store joint_pos in IsaacLab order).
MUJOCO_TO_ISACLAB_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
    4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]


def _resolve(path: Path) -> Path:
    """Resolve a path relative to cwd or the repo root."""
    if path.is_absolute():
        return path
    for root in (Path.cwd(), Path(__file__).resolve().parent.parent):
        cand = root / path
        if cand.exists():
            return cand.resolve()
    return path.resolve()


def _enumerate_motions(batch_arg: str) -> list[Path]:
    """Resolve --batch (a directory or a glob) to a sorted list of .npz files."""
    p = Path(batch_arg)
    if p.is_dir():
        files = sorted(p.glob("*.npz"))
    else:
        files = sorted(Path(f) for f in _glob.glob(batch_arg))
    return files


def _load_motion(npz_path: Path) -> dict:
    """Load one motion npz. Supports body_pos_w (data/motion_dir) and qpos formats."""
    d = dict(np.load(npz_path, allow_pickle=True))
    fps = float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d else 50.0

    if "qpos" in d:
        qpos = np.asarray(d["qpos"], dtype=np.float32)
        root_pos, root_quat = qpos[:, 0:3], qpos[:, 3:7]
        joints = qpos[:, 7:]
        joint_pos = joints[:, MUJOCO_TO_ISACLAB_DOF] if joints.shape[1] == 29 else joints
    else:
        joint_pos = np.asarray(d["joint_pos"], dtype=np.float32)
        body_pos = np.asarray(d["body_pos_w"], dtype=np.float32)
        body_quat = np.asarray(d["body_quat_w"], dtype=np.float32)
        root_pos = body_pos[:, 0, :]   # body 0 = root (pelvis)
        root_quat = body_quat[:, 0, :]

    joint_vel = np.asarray(d["joint_vel"], dtype=np.float32) if "joint_vel" in d else np.zeros_like(joint_pos)
    if joint_vel.shape != joint_pos.shape:
        joint_vel = np.zeros_like(joint_pos)

    return {
        "joint_pos": joint_pos, "joint_vel": joint_vel,
        "root_pos": root_pos, "root_quat": root_quat,
        "fps": fps, "num_frames": joint_pos.shape[0],
    }


def _find_object(object_folder: Path, stem: str):
    """Return (proxy_urdf_path, fitted_npz_path) if both exist, else None."""
    urdf = object_folder / f"{stem}_thin_plate_proxy.urdf"
    fitted = object_folder / f"{stem}_object_fitted.npz"
    if urdf.exists() and fitted.exists():
        return str(urdf.resolve()), fitted
    return None


def _find_terrain(terrain_folder: Path, stem: str):
    """Terrain base = motion stem with trailing _motion_<N> stripped."""
    base = re.sub(r"_motion_\d+$", "", stem)
    for cand_base in (stem, base):
        urdf = terrain_folder / f"{cand_base}_terrain.urdf"
        if urdf.exists():
            return str(urdf.resolve())
    return None


def _load_object_traj(fitted_npz: Path) -> dict:
    d = dict(np.load(fitted_npz, allow_pickle=True))
    trans = np.asarray(d["object_trans"], dtype=np.float32)          # (T, 3)
    quat = np.asarray(d["object_quat_wxyz"], dtype=np.float32)       # (T, 4) wxyz
    return {"trans": trans, "quat": quat, "num_frames": trans.shape[0]}


def _resolve_folders(motion_path: Path):
    """Object / terrain folders: CLI overrides, else <data_root>/{object,terrain}_folder."""
    data_root = motion_path.resolve().parent.parent  # data/motion_dir/x.npz -> data/
    object_folder = _resolve(Path(args_cli.object_folder)) if args_cli.object_folder else data_root / "object_folder"
    terrain_folder = _resolve(Path(args_cli.terrain_folder)) if args_cli.terrain_folder else data_root / "terrain_folder"
    return object_folder, terrain_folder


def _discover_assets(motion_path: Path):
    """Return (proxy_urdf, fitted_npz, terrain_urdf) for a motion, honoring --no_object/--no_terrain."""
    object_folder, terrain_folder = _resolve_folders(motion_path)
    stem = motion_path.stem

    proxy_urdf, fitted_npz = None, None
    if not args_cli.no_object and object_folder.exists():
        found = _find_object(object_folder, stem)
        if found:
            proxy_urdf, fitted_npz = found

    terrain_urdf = None
    if not args_cli.no_terrain and terrain_folder.exists():
        terrain_urdf = _find_terrain(terrain_folder, stem)

    return proxy_urdf, fitted_npz, terrain_urdf


def _build_scene_cfg(proxy_urdf: str | None, terrain_urdf: str | None, record: bool):
    """Build a single-env scene cfg with robot (+ optional object proxy + terrain + camera)."""

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        # NOTE: no GroundPlaneCfg / textured DomeLight here -- both reference remote
        # Omniverse-nucleus assets that fail to load on offline machines. A local
        # cuboid ground is added below (only when there is no terrain mesh).
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
        )
        robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cfg = SceneCfg(num_envs=1, env_spacing=5.0)

    if terrain_urdf is None:
        # Local visual ground (top surface at z=0); skipped when a terrain mesh is present.
        cfg.ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.CuboidCfg(
                size=(100.0, 100.0, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.32, 0.35)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        )

    if record:
        # Follow-camera: repositioned every frame via set_world_poses_from_view.
        cfg.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/FollowCamera",
            update_period=0.0,
            height=args_cli.height,
            width=args_cli.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 1e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=tuple(args_cli.cam_offset), convention="world"),
        )

    if proxy_urdf is not None:
        # Kinematic proxy (use_magic_force=False): pose written directly each step.
        cfg.proxy_object = _make_proxy_rigid_object_cfg_multi_urdf([proxy_urdf], use_magic_force=False)

    if terrain_urdf is not None:
        cfg.terrain = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Terrain",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=terrain_urdf,
                fix_base=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, retain_accelerations=False),
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
                ),
            ),
        )
    return cfg


def _build_sim_scene(motion: dict, proxy_urdf: str | None, terrain_urdf: str | None, record: bool):
    """Tear down any previous sim, open a FRESH stage, then build sim + scene for one motion.

    create_new_stage() wipes all prims from the previous motion, so object/terrain URDFs
    never clash across files. A new SimulationContext is created per motion because dt is
    tied to the motion fps.
    """
    # Drop the previous SimulationContext singleton and clear the stage before rebuilding.
    SimulationContext.clear_instance()
    create_new_stage()

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / motion["fps"], device=args_cli.device))
    scene_cfg = _build_scene_cfg(proxy_urdf, terrain_urdf, record=record)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    return sim, scene


def _make_apply_frame(sim, scene, motion: dict, obj_traj: dict | None, record: bool):
    """Build the per-frame pose-writer closure plus the tensors it needs."""
    device = scene.device
    env_origin = scene.env_origins[0]  # (3,)
    cam_offset = torch.tensor(args_cli.cam_offset, dtype=torch.float32, device=device)

    proxy = scene["proxy_object"] if "proxy_object" in scene.keys() else None
    robot = scene["robot"]
    camera = scene["camera"] if record else None

    joint_pos = torch.tensor(motion["joint_pos"], dtype=torch.float32, device=device)
    joint_vel = torch.tensor(motion["joint_vel"], dtype=torch.float32, device=device)
    root_pos = torch.tensor(motion["root_pos"], dtype=torch.float32, device=device)
    root_quat = torch.tensor(motion["root_quat"], dtype=torch.float32, device=device)
    num_frames = motion["num_frames"]

    if obj_traj is not None:
        obj_trans = torch.tensor(obj_traj["trans"], dtype=torch.float32, device=device)
        obj_quat = torch.tensor(obj_traj["quat"], dtype=torch.float32, device=device)
        obj_frames = obj_traj["num_frames"]
        zero_vel = torch.zeros(1, 6, device=device)

    def _apply_frame(f: int):
        """Write robot + object poses for motion frame f, and aim the follow-camera at the robot."""
        if proxy is not None and obj_traj is not None:
            ratio = f / max(num_frames - 1, 1)
            of = min(int(round(ratio * (obj_frames - 1))), obj_frames - 1)
            pose = torch.zeros(1, 7, device=device)
            pose[0, :3] = obj_trans[of] + env_origin
            pose[0, 3:7] = obj_quat[of]
            proxy.write_root_link_pose_to_sim(pose)
            proxy.write_root_com_velocity_to_sim(zero_vel)

        root_states = robot.data.default_root_state.clone()
        root_states[0, :3] = root_pos[f] + env_origin
        root_states[0, 3:7] = root_quat[f]
        root_states[0, 7:] = 0.0
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos[f].unsqueeze(0), joint_vel[f].unsqueeze(0))

        tgt = root_states[0, :3]
        if camera is not None:
            camera.set_world_poses_from_view((tgt + cam_offset).unsqueeze(0), tgt.unsqueeze(0))
        else:
            t = tgt.cpu().numpy()
            sim.set_camera_view(eye=(t + args_cli.cam_offset).tolist(), target=t.tolist())
        return tgt

    return _apply_frame, camera, num_frames


def _record_one(motion_path: Path, out_dir: Path) -> Path | None:
    """Render one motion to <out_dir>/<stem>.mp4 in the current Isaac process. Returns the path."""
    import cv2

    proxy_urdf, fitted_npz, terrain_urdf = _discover_assets(motion_path)
    motion = _load_motion(motion_path)
    obj_traj = _load_object_traj(fitted_npz) if fitted_npz is not None else None

    print(f"[INFO] motion : {motion_path.name}  ({motion['num_frames']} frames @ {motion['fps']} fps)")
    print(f"[INFO] object : {'-' if proxy_urdf is None else Path(proxy_urdf).name}")
    print(f"[INFO] terrain: {'-' if terrain_urdf is None else Path(terrain_urdf).name}")

    sim, scene = _build_sim_scene(motion, proxy_urdf, terrain_urdf, record=True)
    apply_frame, camera, num_frames = _make_apply_frame(sim, scene, motion, obj_traj, record=True)
    sim_dt = sim.get_physics_dt()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{motion_path.stem}.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
        float(motion["fps"]), (args_cli.width, args_cli.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {out_path}")

    try:
        # Warm up: place at frame 0 and render a few steps so the camera buffer is valid.
        apply_frame(0)
        scene.write_data_to_sim()
        for _ in range(3):
            sim.step()
            scene.update(sim_dt)

        for f in range(num_frames):
            apply_frame(f)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            rgb = camera.data.output["rgb"][0, ..., :3]        # (H, W, 3) uint8
            frame_bgr = cv2.cvtColor(rgb.cpu().numpy(), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()

    print(f"[INFO] wrote {num_frames} frames -> {out_path.resolve()}")
    return out_path


def _run_interactive(motion_path: Path):
    """Open the GUI and loop one motion (no video)."""
    proxy_urdf, fitted_npz, terrain_urdf = _discover_assets(motion_path)
    motion = _load_motion(motion_path)
    obj_traj = _load_object_traj(fitted_npz) if fitted_npz is not None else None

    print(f"[INFO] motion : {motion_path.name}  ({motion['num_frames']} frames @ {motion['fps']} fps)")
    print(f"[INFO] object : {'-' if proxy_urdf is None else Path(proxy_urdf).name}")
    print(f"[INFO] terrain: {'-' if terrain_urdf is None else Path(terrain_urdf).name}")

    sim, scene = _build_sim_scene(motion, proxy_urdf, terrain_urdf, record=False)
    apply_frame, _camera, num_frames = _make_apply_frame(sim, scene, motion, obj_traj, record=False)
    sim_dt = sim.get_physics_dt()

    apply_frame(0)
    frame = 0
    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step()
            continue

        apply_frame(min(frame, num_frames - 1))
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        frame += 1
        if frame >= num_frames:
            frame = 0 if args_cli.loop else num_frames - 1


def _run_batch():
    """Render one video per motion file, reusing a single Isaac Sim process."""
    files = _enumerate_motions(args_cli.batch)
    if not files:
        raise FileNotFoundError(f"No .npz motion files matched --batch {args_cli.batch!r}")
    if args_cli.limit is not None:
        files = files[: args_cli.limit]

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BATCH] {len(files)} motion(s) -> {out_dir.resolve()}")

    n_ok, n_skip, n_fail = 0, 0, 0
    failures: list[str] = []
    for i, mf in enumerate(files, 1):
        out_mp4 = out_dir / f"{mf.stem}.mp4"
        if out_mp4.exists() and not args_cli.overwrite:
            print(f"[BATCH] ({i}/{len(files)}) skip (exists): {out_mp4.name}")
            n_skip += 1
            continue

        print(f"[BATCH] ({i}/{len(files)}) rendering: {mf.name}")
        try:
            _record_one(_resolve(mf), out_dir)
            if out_mp4.exists():
                n_ok += 1
            else:
                n_fail += 1
                failures.append(mf.name)
                print(f"[BATCH] FAILED (no output): {mf.name}")
        except Exception as e:  # keep going: one bad motion shouldn't kill the batch
            n_fail += 1
            failures.append(mf.name)
            print(f"[BATCH] FAILED ({type(e).__name__}: {e}): {mf.name}")
        finally:
            gc.collect()

    print(f"\n[BATCH] done: {n_ok} ok, {n_skip} skipped, {n_fail} failed")
    if failures:
        print("[BATCH] failures:\n  " + "\n  ".join(failures))
    return n_fail


def main():
    if args_cli.batch:
        n_fail = _run_batch()
        return 0 if n_fail == 0 else 1

    motion_path = _resolve(Path(args_cli.motion_file))
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")

    if args_cli.record:
        if not getattr(args_cli, "enable_cameras", False):
            raise RuntimeError("--record requires --enable_cameras (and usually --headless).")
        _record_one(motion_path, Path(args_cli.out_dir))
    else:
        _run_interactive(motion_path)
    return 0


if __name__ == "__main__":
    rc = main()
    if args_cli.batch:
        # Every video is flushed to disk per-file (cv2 writer.release in _record_one).
        # simulation_app.close() reliably hangs on this Isaac Sim build, which would
        # block automation after the whole batch is already done, so hard-exit instead.
        import os
        os._exit(rc)
    simulation_app.close()
    if rc:
        import sys
        sys.exit(rc)
