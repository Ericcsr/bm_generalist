#!/usr/bin/env python3
"""Merge motion and terrain assets from multiple robot-first experiment folders.

Each exp folder has: motion_dir/{exp}/motion_N.npz, terrain_dir/{exp}/terrain.obj,
terrain.urdf (one per exp), motion_N_contact_nodes.npy. This script merges them
with exp name as prefix to avoid collisions. Output is compatible with
_build_motion_terrain_pairs (motion_dir/*.npz and terrain_folder/*.urdf matched by base name).

Usage:
    python scripts/merge_robot_first_exps.py --folders obstacles1_subject1_rf obstacles2_subject1_rf \\
        --output merged_rf

    # Or from a file list
    python scripts/merge_robot_first_exps.py --folders-file folders.txt --output merged_rf

Output:
    bm_generalist/{output}/           - merged motions (e.g. obstacles1_subject1_rf_motion_0.npz)
    bm_generalist/lafan_terrain/{output}/ - merged terrains (one mesh + one terrain.urdf per exp, contact_nodes with prefix)
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

# Resolve bm_generalist root
SCRIPT_DIR = Path(__file__).resolve().parent
BM_GENERALIST_ROOT = SCRIPT_DIR.parent

# Common mesh filenames in robot-first terrain export
TERRAIN_MESH_CANDIDATES = ("terrain.obj", "terrain_mesh.obj", "terrain_convex_combined.obj")


def _find_mesh_ref_in_urdf(urdf_content: str) -> str | None:
    """Extract mesh filename from URDF content (filename=\"*.obj\")."""
    match = re.search(r'filename=["\']?([^"\']+\.obj)["\']?', urdf_content)
    return match.group(1).strip() if match else None


def _find_terrain_mesh(terrain_dir: Path, urdf_path: Path | None) -> tuple[Path | None, str | None]:
    """Find terrain mesh file and its basename. Returns (src_path, basename) or (None, None)."""
    # Try parsing URDF first
    if urdf_path and urdf_path.exists():
        mesh_name = _find_mesh_ref_in_urdf(urdf_path.read_text())
        if mesh_name:
            src = terrain_dir / mesh_name
            if src.exists():
                return src, mesh_name

    # Fallback: check common candidates
    for name in TERRAIN_MESH_CANDIDATES:
        src = terrain_dir / name
        if src.exists():
            return src, name
    return None, None


def _update_urdf_mesh_ref(urdf_content: str, old_mesh: str, new_mesh: str) -> str:
    """Replace mesh filename references in URDF content."""
    return re.sub(
        rf'filename=["\']?' + re.escape(old_mesh) + r'["\']?',
        f'filename="{new_mesh}"',
        urdf_content,
    )


def merge_folders(
    folders: list[str],
    output_name: str,
    motion_base: Path | None = None,
    terrain_base: Path | None = None,
) -> None:
    """Merge robot-first motions and terrains from folders into output, with folder name as prefix."""
    motion_base = motion_base or BM_GENERALIST_ROOT
    terrain_base = terrain_base or (BM_GENERALIST_ROOT / "lafan_terrain")

    out_motion_dir = motion_base / output_name
    out_terrain_dir = terrain_base / output_name
    out_motion_dir.mkdir(parents=True, exist_ok=True)
    out_terrain_dir.mkdir(parents=True, exist_ok=True)

    merged_motions: list[str] = []
    merged_terrains: list[str] = []

    for folder in folders:
        src_motion = motion_base / folder
        src_terrain = terrain_base / folder
        prefix = folder

        if not src_motion.exists():
            print(f"[WARN] Motion dir not found: {src_motion}")
            continue
        if not src_terrain.exists():
            print(f"[WARN] Terrain dir not found: {src_terrain}")
            continue

        motion_npzs = {f.stem: f for f in src_motion.glob("*.npz")}
        contact_files = {
            f.stem.replace("_contact_nodes", ""): f
            for f in src_terrain.glob("motion_*_contact_nodes.npy")
        }

        # Robot-first: one shared terrain mesh per exp
        terrain_urdf = src_terrain / "terrain.urdf"
        mesh_src, mesh_basename = _find_terrain_mesh(src_terrain, terrain_urdf)
        if not mesh_src or not mesh_basename:
            print(f"[WARN] No terrain mesh found in {src_terrain}, copying motions only")
            for base in motion_npzs:
                new_base = f"{prefix}_{base}"
                shutil.copy2(motion_npzs[base], out_motion_dir / f"{new_base}.npz")
                merged_motions.append(str(out_motion_dir / f"{new_base}.npz"))
                print(f"  [motion]  {folder}/{base}.npz -> {output_name}/{new_base}.npz")
            continue

        new_mesh_name = f"{prefix}_{mesh_basename}"
        dst_mesh = out_terrain_dir / new_mesh_name
        shutil.copy2(mesh_src, dst_mesh)
        print(f"  [terrain] {folder}/{mesh_basename} -> {output_name}/{new_mesh_name}")

        # One terrain.urdf per exp (no per-motion URDF copies)
        template_urdf = terrain_urdf if terrain_urdf.exists() else None
        if not template_urdf:
            print(f"[WARN] No terrain URDF found in {src_terrain}")
        else:
            urdf_content = template_urdf.read_text()
            if mesh_basename:
                urdf_content = _update_urdf_mesh_ref(
                    urdf_content, mesh_basename, f"{prefix}_{mesh_basename}"
                )
            dst_terrain_urdf = out_terrain_dir / f"{prefix}_terrain.urdf"
            dst_terrain_urdf.write_text(urdf_content)
            merged_terrains.append(str(dst_terrain_urdf))
            print(f"  [terrain] {folder}/terrain.urdf -> {output_name}/{prefix}_terrain.urdf")

        # Motions and contact nodes (one terrain.urdf per exp, no per-motion URDF)
        all_bases = sorted(set(motion_npzs.keys()) | set(contact_files.keys()))

        for base in all_bases:
            new_base = f"{prefix}_{base}"

            # Copy motion
            if base in motion_npzs:
                shutil.copy2(motion_npzs[base], out_motion_dir / f"{new_base}.npz")
                merged_motions.append(str(out_motion_dir / f"{new_base}.npz"))
                print(f"  [motion]  {folder}/{base}.npz -> {output_name}/{new_base}.npz")

            # Copy contact nodes
            if base in contact_files:
                shutil.copy2(
                    contact_files[base],
                    out_terrain_dir / f"{new_base}_contact_nodes.npy",
                )
                print(f"  [contact] {folder}/{base}_contact_nodes.npy -> {output_name}/{new_base}_contact_nodes.npy")

    print(f"\n=== Merged {len(merged_motions)} motions, {len(set(merged_terrains))} terrain URDFs ===")
    print(f"  Motion dir:  {out_motion_dir}")
    print(f"  Terrain dir: {out_terrain_dir}")
    print(f"\nTrain with:")
    print(f"  --motion_dir {output_name} --terrain_folder lafan_terrain/{output_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge robot-first motion and terrain from multiple experiment folders"
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=[],
        help="Folder names to merge (e.g. obstacles1_subject1_rf obstacles2_subject1_rf)",
    )
    parser.add_argument(
        "--folders-file",
        type=str,
        default=None,
        help="Path to file with one folder name per line (alternative to --folders)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="merged_rf",
        help="Output folder name (default: merged_rf)",
    )
    parser.add_argument(
        "--motion-base",
        type=str,
        default=None,
        help="Base path for motion dirs (default: bm_generalist root)",
    )
    parser.add_argument(
        "--terrain-base",
        type=str,
        default=None,
        help="Base path for terrain dirs (default: bm_generalist/lafan_terrain)",
    )
    args = parser.parse_args()

    folders = list(args.folders)
    if args.folders_file:
        path = Path(args.folders_file)
        if not path.is_absolute():
            path = BM_GENERALIST_ROOT / path
        if path.exists():
            folders.extend(
                line.strip()
                for line in path.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
        else:
            print(f"Error: Folders file not found: {args.folders_file}", file=sys.stderr)
            sys.exit(1)

    if not folders:
        print("Error: No folders specified. Use --folders or --folders-file.", file=sys.stderr)
        sys.exit(1)

    motion_base = Path(args.motion_base) if args.motion_base else None
    terrain_base = Path(args.terrain_base) if args.terrain_base else None

    print(f"Merging {len(folders)} robot-first folders: {folders}")
    print(f"Output: {args.output}\n")
    merge_folders(folders, args.output, motion_base, terrain_base)


if __name__ == "__main__":
    main()
