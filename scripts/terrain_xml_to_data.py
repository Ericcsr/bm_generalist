#!/usr/bin/env python3
"""Convert MuJoCo terrain XMLs (dataset_real_filtered/*.xml) into the ``data/`` terrain category.

Companion to ``real_motion_to_npz.py``. Given a directory of ``<stem>.npz`` (motion) + ``<stem>.xml``
(terrain) pairs, this script generates the terrain assets that ``motion_stats.py`` classifies as
``motion+terrain``, following the exact layout of ``data/terrain_folder``:

  <output_root>/terrain_folder/<terrain_base>.obj            (heightfield mesh, ONE per unique terrain)
  <output_root>/terrain_folder/<terrain_base>_terrain.urdf   (URDF referencing the .obj, ONE per terrain)

and emits a manifest mapping each input motion to its output name, which feeds
``real_motion_to_npz.py --manifest`` to produce the matching motion npz + ``_contact_nodes.npy``.

Naming (mirrors data/terrain_folder, e.g. ``come_down_..._motion_0``):
  * terrain_base   = ``<dir_tag>_h<H>_w<W>`` (e.g. ``up_h0.190_w0.250``), one per unique terrain.
  * motion stem    = ``<terrain_base>_motion_<k>``; the s/t variants sharing a terrain get k=0,1,2,...
    ``motion_stats._terrain_base`` strips the trailing ``_motion_<k>`` so every variant matches the
    single ``<terrain_base>_terrain.urdf``.

De-duplication: XMLs with the same ``h,w`` (different ``s,t``) are byte-identical terrains, so the .obj
+ .urdf are written ONCE per unique terrain. Terrain identity is verified by a hash of the box geoms;
a mismatch between the ``<dir_tag>_h_w`` name and the geom hash aborts (name collision guard).

The mesh is a regular XY heightfield (matching the existing dataset: x,y in [-10, 9.95], step 0.03,
666x666 grid). Height at each grid point = max top (pos_z + size_z) among axis-aligned ``<geom type=box>``
whose XY footprint contains the point, else 0 (floor plane).

This is a pure-geometry script: NO Isaac Sim launch.

Usage:
    # 1) Generate terrain assets + manifest for a folder of pairs
    python scripts/terrain_xml_to_data.py --input_dir dataset_real_filtered/up --output_root data_real

    # 2) Generate the matching motions + contact nodes (single Isaac session) from the manifest
    python scripts/real_motion_to_npz.py --manifest data_real/motion_manifest.csv --data_root data_real --headless

    # 3) Sanity-check the resulting data/ layout
    python scripts/motion_stats.py --data_root data_real
"""

import argparse
import csv
import hashlib
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Match the existing data/terrain_folder heightfield grid exactly.
DEFAULT_GRID_MIN = -10.0
DEFAULT_GRID_RES = 0.03
DEFAULT_GRID_N = 666

# Filename pattern: h<H>_w<W>_s<S>_t<T>.xml  ->  captures the (h,w) terrain key.
_HW_RE = re.compile(r"(h[0-9.]+_w[0-9.]+)")


def _parse_boxes(xml_path: Path) -> list[dict]:
    """Return axis-aligned box geoms from <worldbody>: dict(pos=(3,), size=(3,))."""
    root = ET.parse(str(xml_path)).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return []
    boxes = []
    for geom in worldbody.iter("geom"):
        if geom.get("type") != "box":
            continue
        pos = np.array([float(v) for v in geom.get("pos", "0 0 0").split()], dtype=np.float64)
        size = np.array([float(v) for v in geom.get("size", "0 0 0").split()], dtype=np.float64)
        quat = geom.get("quat", "1 0 0 0").split()
        quat = [float(v) for v in quat]
        # Only axis-aligned boxes are representable as an XY heightfield. Warn (don't silently skip)
        # on any rotation so a rotated terrain isn't misconverted into a wrong flat step.
        if not (abs(quat[0] - 1.0) < 1e-6 and all(abs(q) < 1e-6 for q in quat[1:])):
            print(f"  WARNING: non-identity quat {quat} in {xml_path.name}; treating box as axis-aligned AABB.")
        boxes.append({"pos": pos, "size": size})
    return boxes


def _geom_hash(boxes: list[dict]) -> str:
    """Stable hash of the box set (order-independent) — the terrain identity."""
    sig = sorted(
        (round(b["pos"][0], 6), round(b["pos"][1], 6), round(b["pos"][2], 6),
         round(b["size"][0], 6), round(b["size"][1], 6), round(b["size"][2], 6))
        for b in boxes
    )
    return hashlib.md5(repr(sig).encode()).hexdigest()[:12]


def _heightfield(boxes: list[dict], grid_min: float, grid_res: float, grid_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (coords (grid_n,), Z (grid_n, grid_n)) with Z[iy, ix] = terrain height at (X[ix], Y[iy])."""
    coords = grid_min + grid_res * np.arange(grid_n, dtype=np.float64)
    Z = np.zeros((grid_n, grid_n), dtype=np.float64)
    for b in boxes:
        px, py, pz = b["pos"]
        sx, sy, sz = b["size"]
        top = pz + sz
        xmask = (coords >= px - sx) & (coords <= px + sx)
        ymask = (coords >= py - sy) & (coords <= py + sy)
        if not xmask.any() or not ymask.any():
            continue
        sel = np.ix_(ymask, xmask)
        Z[sel] = np.maximum(Z[sel], top)
    return coords, Z


def _build_faces(grid_n: int) -> np.ndarray:
    """1-indexed triangle faces for a grid_n x grid_n grid (x-major vertices). Two tris per quad.

    Matches the existing dataset winding: quad (v00,v10,v01,v11) -> (v00,v10,v01) + (v10,v11,v01).
    """
    n = grid_n
    iy, ix = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    v00 = (iy * n + ix).ravel() + 1  # 1-indexed
    v10 = v00 + 1
    v01 = v00 + n
    v11 = v01 + 1
    tri1 = np.stack([v00, v10, v01], axis=1)
    tri2 = np.stack([v10, v11, v01], axis=1)
    faces = np.empty((tri1.shape[0] * 2, 3), dtype=np.int64)
    faces[0::2] = tri1
    faces[1::2] = tri2
    return faces


def _write_obj(path: Path, coords: np.ndarray, Z: np.ndarray, faces: np.ndarray) -> None:
    """Write heightfield OBJ (# Terrain Mesh header, x-major verts, then faces)."""
    n = coords.shape[0]
    xv = np.tile(coords, n)          # ix fastest
    yv = np.repeat(coords, n)        # iy slowest
    zv = Z.reshape(-1)               # Z[iy, ix] C-order -> iy outer, ix inner (matches xv/yv)
    verts = np.column_stack([xv, yv, zv])
    with open(path, "w") as f:
        f.write("# Terrain Mesh\n")
        np.savetxt(f, verts, fmt="v %.4f %.4f %.4f")
        np.savetxt(f, faces, fmt="f %d %d %d")


_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="{name}">
    <!-- Terrain mesh model for viser and pybullet -->

    <link name="base_link">
        <!-- Visual geometry -->
        <visual name="terrain_visual">
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{obj}" scale="1 1 1"/>
            </geometry>
            <material name="terrain_material">
                <color rgba="0.5 0.5 0.5 1.0"/>
            </material>
        </visual>

        <!-- Collision geometry (same as visual for terrain) -->
        <collision name="terrain_collision">
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{obj}" scale="1 1 1"/>
            </geometry>
        </collision>

        <!-- Inertial properties (static terrain) -->
        <inertial>
            <mass value="0"/>
            <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
        </inertial>
    </link>
</robot>
"""


def _write_urdf(path: Path, name: str, obj_filename: str) -> None:
    path.write_text(_URDF_TEMPLATE.format(name=name, obj=obj_filename))


def main():
    parser = argparse.ArgumentParser(description="Generate terrain .obj + .urdf (data/ terrain category) from MuJoCo XMLs.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of paired <stem>.npz + <stem>.xml (non-recursive).")
    parser.add_argument("--output_root", type=str, default="data_real", help="Output data root; terrain assets go under <root>/terrain_folder.")
    parser.add_argument("--dir_tag", type=str, default=None, help="Terrain-name prefix (default: input_dir basename, e.g. 'up').")
    parser.add_argument("--grid_min", type=float, default=DEFAULT_GRID_MIN, help="Grid start coordinate (x and y).")
    parser.add_argument("--grid_res", type=float, default=DEFAULT_GRID_RES, help="Grid resolution (step).")
    parser.add_argument("--grid_n", type=int, default=DEFAULT_GRID_N, help="Grid point count per axis.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite .obj/.urdf even if they already exist.")
    parser.add_argument("--manifest_name", type=str, default="motion_manifest.csv", help="Manifest filename written under output_root.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        alt = REPO_ROOT / args.input_dir
        input_dir = alt if alt.is_dir() else input_dir
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {args.input_dir}")

    dir_tag = args.dir_tag or input_dir.name
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    terrain_dir = output_root / "terrain_folder"
    terrain_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "motion_dir").mkdir(parents=True, exist_ok=True)

    xml_paths = sorted(input_dir.glob("*.xml"))
    if not xml_paths:
        raise ValueError(f"No .xml files in {input_dir}")

    # ---- Group motions by terrain identity; verify (dir_tag,h,w) name <-> geom hash bijection. ----
    # groups[terrain_base] = {"hash", "boxes", "members": [ (stem, npz_path) ... ]}
    groups: dict[str, dict] = {}
    base_of_hash: dict[str, str] = {}
    skipped_no_npz = []
    for xml_path in xml_paths:
        stem = xml_path.stem
        npz_path = xml_path.with_suffix(".npz")
        if not npz_path.exists():
            skipped_no_npz.append(stem)
            continue
        m = _HW_RE.search(stem)
        if not m:
            print(f"  WARNING: cannot parse h/w from '{stem}'; skipping.")
            continue
        terrain_base = f"{dir_tag}_{m.group(1)}"
        boxes = _parse_boxes(xml_path)
        h = _geom_hash(boxes)

        if terrain_base in groups:
            if groups[terrain_base]["hash"] != h:
                raise RuntimeError(
                    f"Name collision: '{terrain_base}' maps to two different terrains "
                    f"({groups[terrain_base]['hash']} vs {h}). Same (dir,h,w) must be one terrain."
                )
        else:
            if h in base_of_hash and base_of_hash[h] != terrain_base:
                print(f"  NOTE: identical geometry under two names: {base_of_hash[h]} and {terrain_base}.")
            base_of_hash.setdefault(h, terrain_base)
            groups[terrain_base] = {"hash": h, "boxes": boxes, "members": []}
        groups[terrain_base]["members"].append((stem, npz_path))

    if skipped_no_npz:
        print(f"[INFO] {len(skipped_no_npz)} xml(s) had no matching .npz (skipped), e.g. {skipped_no_npz[:3]}")

    # ---- Precompute faces once (grid-fixed, shared by all terrains). ----
    faces = _build_faces(args.grid_n)

    # ---- Emit terrain assets (once per terrain) + manifest rows (per motion). ----
    manifest_rows = []
    n_written = n_skipped = 0
    for terrain_base in sorted(groups):
        g = groups[terrain_base]
        obj_path = terrain_dir / f"{terrain_base}.obj"
        urdf_path = terrain_dir / f"{terrain_base}_terrain.urdf"

        if args.overwrite or not obj_path.exists():
            coords, Z = _heightfield(g["boxes"], args.grid_min, args.grid_res, args.grid_n)
            _write_obj(obj_path, coords, Z, faces)
            print(f"[OBJ ] {obj_path.name}  (z max={Z.max():.4f}, {len(g['boxes'])} boxes)")
            n_written += 1
        else:
            n_skipped += 1
        if args.overwrite or not urdf_path.exists():
            _write_urdf(urdf_path, terrain_base, f"{terrain_base}.obj")

        for k, (stem, npz_path) in enumerate(sorted(g["members"])):
            motion_name = f"{terrain_base}_motion_{k}"
            manifest_rows.append(
                {
                    "input_npz": str(npz_path.resolve()),
                    "output_name": motion_name,
                    "terrain_base": terrain_base,
                    "motion_index": k,
                    "source_stem": stem,
                }
            )

    manifest_path = output_root / args.manifest_name
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["input_npz", "output_name", "terrain_base", "motion_index", "source_stem"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(
        f"\n[DONE] terrains: {len(groups)} unique (obj written {n_written}, skipped existing {n_skipped}); "
        f"motions: {len(manifest_rows)}"
    )
    print(f"[DONE] terrain assets -> {terrain_dir}")
    print(f"[DONE] manifest       -> {manifest_path}")
    print("\nNext: generate matching motions + contact nodes in one Isaac session:")
    print(f"  python scripts/real_motion_to_npz.py --manifest {manifest_path} --data_root {output_root} --headless")


if __name__ == "__main__":
    main()
