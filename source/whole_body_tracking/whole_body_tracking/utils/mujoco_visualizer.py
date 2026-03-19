"""MuJoCo passive viewer for real-time visualization of Isaac Sim robot and proxy poses.

Runs in the same process as the main loop using mujoco.viewer.launch_passive().
Isaac Sim and MuJoCo both use Z-up world frame; no coordinate transform needed.
Uses name-based joint mapping when robot_joint_names is provided; otherwise falls back
to index-based ISAAC_TO_MUJOCO (IsaacLab/motion order).

When proxy_urdf_path is provided, parses the URDF and combines it with the robot MJCF
so both robot and object are visualized.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Isaac joint index -> MuJoCo hinge index (for G1 29-DOF, motion/IsaacLab order)
# Used only when robot_joint_names is not provided
ISAAC_TO_MUJOCO = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28,
])


def _urdf_rigid_body_to_mjcf(urdf_path: str | Path) -> tuple[str, str]:
    """Convert a rigid-body URDF to MJCF asset + body XML.

    Supports box, cylinder, and mesh geometry. Returns (asset_xml, body_xml).
    For box/cylinder: asset_xml is empty. For mesh: asset_xml adds the mesh.
    """
    urdf_file = Path(urdf_path)
    urdf_dir = urdf_file.parent
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    links = {e.get("name"): e for e in root.findall("link")}
    joints = [e for e in root.findall("joint") if e.get("type") == "fixed"]

    # Find root link
    child_links = {j.find("child").get("link") for j in joints}
    parent_links = {j.find("parent").get("link") for j in joints}
    root_name = next((p for p in parent_links if p not in child_links), list(links)[0])
    root_link = links.get(root_name)

    # Collect geometry from: (1) fixed joint children, (2) root link itself (single-link URDF)
    geoms: list[tuple[str, str, list[float], list[float], list[float] | None]] = []
    # (type, size_or_mesh, pos, quat_wxyz, scale or None for mesh)

    def process_geom(geom_elem, origin_xyz, origin_rpy):
        if geom_elem is None:
            return
        rot = Rotation.from_euler("xyz", origin_rpy)
        quat = rot.as_quat()
        quat_mj = [quat[3], quat[0], quat[1], quat[2]]  # wxyz
        xyz = list(origin_xyz)
        box = geom_elem.find("box")
        cyl = geom_elem.find("cylinder")
        mesh_elem = geom_elem.find("mesh")
        if box is not None and box.get("size"):
            parts = [float(x) for x in box.get("size", "").split()]
            if len(parts) >= 3:
                half = [p / 2 for p in parts[:3]]
                geoms.append(("box", " ".join(f"{h:.6g}" for h in half), xyz, quat_mj, None))
        elif cyl is not None:
            r = float(cyl.get("radius", 0.01))
            l = float(cyl.get("length", 0.1))
            geoms.append(("cylinder", f"{r:.6g} {l/2:.6g}", xyz, quat_mj, None))
        elif mesh_elem is not None:
            filename = mesh_elem.get("filename")
            if filename:
                scale = [1.0, 1.0, 1.0]
                if mesh_elem.get("scale"):
                    parts = mesh_elem.get("scale", "").split()
                    if len(parts) >= 3:
                        scale = [float(x) for x in parts[:3]]
                mesh_path = (urdf_dir / filename).resolve()
                if mesh_path.exists():
                    geoms.append(("mesh", str(mesh_path), xyz, quat_mj, scale))

    # From fixed joint children
    for j in joints:
        if j.find("parent").get("link") != root_name:
            continue
        child_name = j.find("child").get("link")
        child_link = links.get(child_name)
        if not child_link:
            continue
        origin = j.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            if origin.get("xyz"):
                xyz = [float(x) for x in origin.get("xyz", "").split()]
            if origin.get("rpy"):
                rpy = [float(x) for x in origin.get("rpy", "").split()]
        geom_elem = child_link.find("collision/geometry") or child_link.find("visual/geometry")
        process_geom(geom_elem, xyz, rpy)

    # From root link (single-link URDF like terrain.urdf)
    if root_link and not geoms:
        for geom_parent in ("visual", "collision"):
            vis = root_link.find(geom_parent)
            if vis is not None:
                origin = vis.find("origin")
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]
                if origin is not None:
                    if origin.get("xyz"):
                        xyz = [float(x) for x in origin.get("xyz", "").split()]
                    if origin.get("rpy"):
                        rpy = [float(x) for x in origin.get("rpy", "").split()]
                geom_elem = vis.find("geometry")
                process_geom(geom_elem, xyz, rpy)
                if geoms:
                    break  # use first successful

    if not geoms:
        return ("", "")

    asset_lines = []
    body_lines = [
        '    <body name="proxy_root" pos="0 0 0">',
        '      <freejoint name="proxy_free"/>',
    ]
    for i, item in enumerate(geoms):
        gtype, size_or_mesh, pos, quat, scale = item
        pos_str = " ".join(f"{x:.6g}" for x in pos)
        quat_str = " ".join(f"{q:.6g}" for q in quat)
        rgba = "0.5 0.5 0.5 1" if gtype == "mesh" else "0.2 0.6 0.9 0.9"
        if gtype == "mesh":
            mesh_name = f"proxy_mesh_{i}"
            scale_str = ""
            if scale and (scale[0] != 1 or scale[1] != 1 or scale[2] != 1):
                scale_str = f' scale="{scale[0]:.6g} {scale[1]:.6g} {scale[2]:.6g}"'
            asset_lines.append(
                f'    <mesh name="{mesh_name}" file="{size_or_mesh}"{scale_str}/>'
            )
            body_lines.append(
                f'      <geom name="proxy_geom_{i}" type="mesh" mesh="{mesh_name}" '
                f'pos="{pos_str}" quat="{quat_str}" rgba="{rgba}"/>'
            )
        else:
            body_lines.append(
                f'      <geom name="proxy_geom_{i}" type="{gtype}" size="{size_or_mesh}" '
                f'pos="{pos_str}" quat="{quat_str}" rgba="{rgba}"/>'
            )
    body_lines.append("    </body>")
    asset_xml = "\n".join(asset_lines) if asset_lines else ""
    body_xml = "\n".join(body_lines)
    return (asset_xml, body_xml)


def _inject_proxy_into_robot_mjcf(
    robot_xml: str, proxy_asset_xml: str, proxy_body_xml: str
) -> str:
    """Inject proxy asset and body into robot MJCF."""
    if not proxy_body_xml.strip():
        return robot_xml
    # Inject asset (mesh) before </asset> if present
    if proxy_asset_xml.strip():
        if "</asset>" in robot_xml:
            idx = robot_xml.index("</asset>")
            robot_xml = robot_xml[:idx] + "\n" + proxy_asset_xml + "\n  " + robot_xml[idx:]
        elif "<worldbody>" in robot_xml:
            idx = robot_xml.index("<worldbody>")
            robot_xml = (
                robot_xml[:idx] + "<asset>\n" + proxy_asset_xml + "\n  </asset>\n  " + robot_xml[idx:]
            )
    # Inject body before </worldbody>
    if "</worldbody>" not in robot_xml:
        return robot_xml
    idx = robot_xml.index("</worldbody>")
    return robot_xml[:idx] + "\n" + proxy_body_xml + "\n  " + robot_xml[idx:]


def _find_g1_mjcf_path() -> Path | None:
    """Resolve path to G1 MuJoCo XML (from GMR or HDMI, sibling of bm_generalist)."""
    # __file__: bm_generalist/.../utils/mujoco_visualizer.py -> parents[5] = robot_imitation
    root = Path(__file__).resolve().parents[5]
    candidates = [
        root / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
        root / "HDMI" / "active_adaptation" / "assets_mjcf" / "g1_29dof_nohand" / "g1_29dof_nohand.xml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _make_minimal_mjcf(include_proxy: bool = True) -> str:
    """Minimal MJCF with floor, simple robot skeleton (capsules), and optional proxy box."""
    proxy_part = ""
    if include_proxy:
        proxy_part = """
    <body name="proxy_root" pos="0 0 0.5">
      <freejoint name="proxy_free"/>
      <geom name="proxy_box" type="box" size="0.1 0.1 0.1" rgba="0.2 0.6 0.9 0.8"/>
    </body>"""
    return f"""
<mujoco model="isaac_mirror">
  <compiler angle="radian"/>
  <option timestep="0.005"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
  </visual>
  <worldbody>
    <light pos="0 0 5"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="robot_root" pos="0 0 0.9">
      <freejoint name="robot_free"/>
      <geom name="robot_pelvis" type="capsule" fromto="0 0 0 0 0 -0.1" size="0.08" rgba="0.3 0.3 0.3 1"/>
      <geom name="robot_torso" type="capsule" fromto="0 0 0 0 0 0.3" size="0.06" rgba="0.5 0.5 0.5 1"/>
    </body>{proxy_part}
  </worldbody>
</mujoco>
"""


class MujocoVisualizer:
    """Passive MuJoCo viewer that mirrors Isaac Sim robot and proxy poses in real time."""

    def __init__(
        self,
        env_id: int = 0,
        g1_mjcf_path: str | Path | None = None,
        proxy_urdf_path: str | Path | None = None,
    ):
        """Initialize MuJoCo model and launch passive viewer.

        Args:
            env_id: Environment index to visualize.
            g1_mjcf_path: Path to G1 MuJoCo XML. If None, tries GMR/HDMI paths or uses minimal scene.
            proxy_urdf_path: Path to proxy object URDF. When provided, parses URDF and combines
                with robot MJCF so both robot and object are visualized.
        """
        import mujoco
        import mujoco.viewer

        self.env_id = env_id
        self._mujoco = mujoco
        self._viewer_mod = mujoco.viewer

        path = Path(g1_mjcf_path) if g1_mjcf_path else _find_g1_mjcf_path()
        proxy_path = Path(proxy_urdf_path) if proxy_urdf_path else None

        if path and path.exists():
            robot_xml = path.read_text()
            if proxy_path and proxy_path.exists():
                try:
                    proxy_asset, proxy_body = _urdf_rigid_body_to_mjcf(proxy_path)
                    if proxy_body:
                        robot_xml = _inject_proxy_into_robot_mjcf(
                            robot_xml, proxy_asset, proxy_body
                        )
                        # Write to temp file in same dir as robot XML so meshdir resolves
                        with tempfile.NamedTemporaryFile(
                            mode="w",
                            suffix=".xml",
                            dir=str(path.parent),
                            delete=False,
                        ) as f:
                            f.write(robot_xml)
                            tmp_path = f.name
                        try:
                            self.model = mujoco.MjModel.from_xml_path(tmp_path)
                        finally:
                            Path(tmp_path).unlink(missing_ok=True)
                    else:
                        self.model = mujoco.MjModel.from_xml_path(str(path))
                except Exception as e:
                    print(f"[MujocoVisualizer] Failed to add proxy from URDF: {e}")
                    self.model = mujoco.MjModel.from_xml_path(str(path))
            else:
                self.model = mujoco.MjModel.from_xml_path(str(path))
            self._use_full_model = True
        else:
            has_proxy_urdf = proxy_path and proxy_path.exists()
            combined = _make_minimal_mjcf(include_proxy=not has_proxy_urdf)
            if has_proxy_urdf:
                try:
                    proxy_asset, proxy_body = _urdf_rigid_body_to_mjcf(proxy_path)
                    if proxy_body:
                        combined = _inject_proxy_into_robot_mjcf(
                            combined, proxy_asset, proxy_body
                        )
                except Exception as e:
                    print(f"[MujocoVisualizer] Failed to add proxy from URDF: {e}")
                    combined = _make_minimal_mjcf(include_proxy=True)
            self.model = mujoco.MjModel.from_xml_string(combined)
            self._use_full_model = False

        self.data = mujoco.MjData(self.model)
        self._viewer = None
        self._proxy_body_id = None
        self._robot_free_id = None
        self._proxy_free_id = None
        self._qpos_layout = None

        self._init_qpos_layout()
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _init_qpos_layout(self) -> None:
        """Determine qpos layout for robot freejoint, hinges, and proxy freejoint."""
        m = self.model
        self._robot_free_id = self._mujoco.mj_name2id(m, self._mujoco.mjtObj.mjOBJ_JOINT, "robot_free")
        if self._robot_free_id < 0:
            self._robot_free_id = self._mujoco.mj_name2id(m, self._mujoco.mjtObj.mjOBJ_JOINT, "pelvis")
        self._proxy_free_id = self._mujoco.mj_name2id(m, self._mujoco.mjtObj.mjOBJ_JOINT, "proxy_free")
        if self._proxy_free_id < 0:
            self._proxy_free_id = -1

        self._qpos_layout = {}
        self._hinge_qpos_addrs = []  # qpos addr for each hinge (MuJoCo joint index order)
        self._joint_name_to_qpos_addr = {}  # joint name -> qpos addr for name-based mapping
        for i in range(m.njnt):
            jnt_type = m.jnt_type[i]
            jnt_addr = m.jnt_qposadr[i]
            jnt_name = self._mujoco.mj_id2name(m, self._mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_type == self._mujoco.mjtJoint.mjJNT_FREE:
                self._qpos_layout[i] = ("free", jnt_addr, 7)
            elif jnt_type == self._mujoco.mjtJoint.mjJNT_BALL:
                self._qpos_layout[i] = ("ball", jnt_addr, 4)
            else:
                self._qpos_layout[i] = ("hinge", jnt_addr, 1)
                self._hinge_qpos_addrs.append(jnt_addr)
                if jnt_name:
                    self._joint_name_to_qpos_addr[jnt_name] = jnt_addr

    def update(
        self,
        robot_root_pos: np.ndarray,
        robot_root_quat: np.ndarray,
        robot_joint_pos: np.ndarray,
        robot_joint_names: list[str] | None = None,
        proxy_pos: np.ndarray | None = None,
        proxy_quat: np.ndarray | None = None,
        contact_positions: list[np.ndarray] | None = None,
        contact_forces: list[np.ndarray] | None = None,
    ) -> None:
        """Update MuJoCo state from Isaac poses and sync viewer.

        Args:
            robot_root_pos: (3,) world position.
            robot_root_quat: (4,) quaternion (w,x,y,z).
            robot_joint_pos: (29,) joint positions in Isaac/URDF order.
            robot_joint_names: Joint names from robot.joint_names (same order as joint_pos).
                When provided, uses name-based mapping; otherwise falls back to ISAAC_TO_MUJOCO.
            proxy_pos: (3,) proxy world position, or None to skip.
            proxy_quat: (4,) proxy quat (w,x,y,z), or None.
            contact_positions: List of (3,) hand contact positions for visualization.
            contact_forces: List of (3,) contact force vectors (N) for arrow visualization.
        """
        m, d = self.model, self.data
        qpos = d.qpos

        # Robot root (freejoint)
        if self._robot_free_id >= 0 and self._robot_free_id < len(self._qpos_layout):
            _, addr, _ = self._qpos_layout[self._robot_free_id]
            qpos[addr : addr + 3] = robot_root_pos
            qpos[addr + 3 : addr + 7] = robot_root_quat  # w,x,y,z

        # Robot joints: name-based mapping when joint_names provided, else index-based
        if robot_joint_pos is not None and self._hinge_qpos_addrs:
            if robot_joint_names is not None and len(robot_joint_names) == len(robot_joint_pos):
                for isaac_idx, name in enumerate(robot_joint_names):
                    if name in self._joint_name_to_qpos_addr:
                        qpos_addr = self._joint_name_to_qpos_addr[name]
                        qpos[qpos_addr] = float(robot_joint_pos[isaac_idx])
            else:
                for isaac_idx in range(min(len(robot_joint_pos), len(ISAAC_TO_MUJOCO))):
                    mj_idx = int(ISAAC_TO_MUJOCO[isaac_idx])
                    if mj_idx < len(self._hinge_qpos_addrs):
                        qpos_addr = self._hinge_qpos_addrs[mj_idx]
                        qpos[qpos_addr] = float(robot_joint_pos[isaac_idx])

        # Proxy (freejoint)
        if self._proxy_free_id >= 0 and proxy_pos is not None and proxy_quat is not None:
            _, addr, _ = self._qpos_layout[self._proxy_free_id]
            qpos[addr : addr + 3] = proxy_pos
            qpos[addr + 3 : addr + 7] = proxy_quat

        self._mujoco.mj_forward(m, d)

        # Camera track robot
        with self._viewer.lock():
            self._viewer.cam.lookat[:] = robot_root_pos
            self._viewer.cam.distance = 3.0
            # Hand-object contact visualization: spheres at contact points + force arrows
            if contact_positions is not None and contact_forces is not None:
                self._draw_hand_contact_vis(contact_positions, contact_forces)

        self._viewer.sync()

    def _draw_hand_contact_vis(
        self,
        contact_positions: list[np.ndarray],
        contact_forces: list[np.ndarray],
        force_threshold: float = 1.0,
        sphere_radius: float = 0.02,
        arrow_scale: float = 0.015,
    ) -> None:
        """Draw hand contact points (spheres) and force arrows in MuJoCo user scene.

        Args:
            contact_positions: List of (3,) world positions (e.g. wrist link positions).
            contact_forces: List of (3,) force vectors in world frame (N).
            force_threshold: Only draw when force magnitude exceeds this (N).
            sphere_radius: Radius of contact point spheres.
            arrow_scale: Arrow length = arrow_scale * force_magnitude (m per N).
        """
        mj = self._mujoco
        viewer = self._viewer
        if not hasattr(viewer, "user_scn") or viewer.user_scn is None:
            return
        scn = viewer.user_scn
        scn.ngeom = 0
        mat_identity = np.eye(3).flatten().astype(np.float64)
        rgba_sphere = np.array([1.0, 0.6, 0.0, 0.95], dtype=np.float32)  # orange
        rgba_arrow = np.array([1.0, 0.2, 0.2, 0.9], dtype=np.float32)  # red

        for pos, force in zip(contact_positions, contact_forces):
            pos = np.asarray(pos, dtype=np.float64)
            force = np.asarray(force, dtype=np.float64)
            fnorm = float(np.linalg.norm(force))
            if fnorm < force_threshold:
                continue
            if scn.ngeom >= scn.maxgeom - 2:
                break
            # Sphere at contact position
            mj.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=np.array([sphere_radius, 0, 0], dtype=np.float64),
                pos=pos,
                mat=mat_identity,
                rgba=rgba_sphere,
            )
            scn.ngeom += 1
            # Arrow: capsule from pos to pos + scale*force
            half_len = 0.5 * min(0.4, arrow_scale * fnorm)
            tip = pos + (force / fnorm) * (2 * half_len)
            center = (pos + tip) / 2.0
            d = tip - pos
            dist = np.linalg.norm(d)
            if dist < 1e-6:
                continue
            d = d / dist
            z = np.array([0.0, 0.0, 1.0])
            ax = np.cross(z, d)
            ax_norm = np.linalg.norm(ax)
            if ax_norm < 1e-6:
                rot = np.eye(3)
            else:
                ang = np.arccos(np.clip(np.dot(z, d), -1.0, 1.0))
                from scipy.spatial.transform import Rotation as R

                rot = R.from_rotvec((ax / ax_norm) * ang).as_matrix()
            mj.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mj.mjtGeom.mjGEOM_CAPSULE,
                size=np.array([0.008, half_len, 0], dtype=np.float64),
                pos=center.astype(np.float64),
                mat=rot.flatten().astype(np.float64),
                rgba=rgba_arrow,
            )
            scn.ngeom += 1

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


def create_mujoco_visualizer(
    env_id: int = 0,
    g1_mjcf_path: str | Path | None = None,
    proxy_urdf_path: str | Path | None = None,
) -> MujocoVisualizer | None:
    """Create MuJoCo visualizer if mujoco is available."""
    try:
        import mujoco  # noqa: F401
        import mujoco.viewer  # noqa: F401
    except ImportError:
        print("[MujocoVisualizer] mujoco not installed; --mujoco_vis disabled.")
        return None
    return MujocoVisualizer(
        env_id=env_id,
        g1_mjcf_path=g1_mjcf_path,
        proxy_urdf_path=proxy_urdf_path,
    )
