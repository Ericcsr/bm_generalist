#!/usr/bin/env python3
"""Merge motion and terrain assets from multiple lafan_terrain folders into a new folder.

Copies motions and terrains from a given list of folders, prefixing each asset with
the source folder name to avoid collisions. Output is compatible with
_build_motion_terrain_pairs (motion_dir/*.npz and terrain_folder/*.urdf matched by base name).

Usage:
    python scripts/merge_lafan_terrain.py --folders obstacles1_subject1 obstacles2_subject1 \\
        --output merged

    # Or from a file list
    python scripts/merge_lafan_terrain.py --folders-file folders.txt --output merged

Output:
    bm_generalist/{output}/           - merged motions (e.g. obstacles1_subject1_motion_0.npz)
    bm_generalist/lafan_terrain/{output}/ - merged terrains (URDF + OBJ with prefix)
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

# Resolve bm_generalist root
SCRIPT_DIR = Path(__file__).resolve().parent
BM_GENERALIST_ROOT = SCRIPT_DIR.parent


def _update_urdf_mesh_ref(urdf_content: str, old_mesh: str, new_mesh: str) -> str:
    """Replace mesh filename references in URDF content."""
    # Replace filename="old.obj" with filename="new.obj"
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
    """Merge motions and terrains from folders into output, with folder name as prefix."""
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
        if not src_terrain.exists():
            print(f"[WARN] Terrain dir not found: {src_terrain}")

        # Collect motion-terrain pairs by base name (motion_N)
        motion_npzs = {f.stem: f for f in src_motion.glob("*.npz")} if src_motion.exists() else {}
        terrain_urdfs = {f.stem: f for f in src_terrain.glob("*.urdf")} if src_terrain.exists() else {}

        # Use union of motion and terrain base names
        all_bases = sorted(set(motion_npzs.keys()) | set(terrain_urdfs.keys()))

        for base in all_bases:
            new_base = f"{prefix}_{base}"

            # Copy motion
            if base in motion_npzs:
                src_npz = motion_npzs[base]
                dst_npz = out_motion_dir / f"{new_base}.npz"
                shutil.copy2(src_npz, dst_npz)
                merged_motions.append(str(dst_npz))
                print(f"  [motion]  {folder}/{base}.npz -> {output_name}/{new_base}.npz")

            # Copy terrain (URDF + mesh)
            if base in terrain_urdfs:
                src_urdf = terrain_urdfs[base]
                dst_urdf = out_terrain_dir / f"{new_base}.urdf"

                # Mesh: same base name, e.g. motion_0.obj
                mesh_name = f"{base}.obj"
                src_mesh = src_terrain / mesh_name
                new_mesh_name = f"{new_base}.obj"
                dst_mesh = out_terrain_dir / new_mesh_name

                if src_mesh.exists():
                    shutil.copy2(src_mesh, dst_mesh)
                else:
                    print(f"  [WARN] Mesh not found: {src_mesh}")

                # Copy URDF and update mesh reference
                urdf_content = src_urdf.read_text()
                urdf_content = _update_urdf_mesh_ref(urdf_content, mesh_name, new_mesh_name)
                dst_urdf.write_text(urdf_content)
                merged_terrains.append(str(dst_urdf))
                print(f"  [terrain] {folder}/{base}.urdf -> {output_name}/{new_base}.urdf")

    print(f"\n=== Merged {len(merged_motions)} motions, {len(merged_terrains)} terrains ===")
    print(f"  Motion dir:  {out_motion_dir}")
    print(f"  Terrain dir: {out_terrain_dir}")
    print(f"\nTrain with:")
    print(f"  --motion_dir {output_name} --terrain_folder lafan_terrain/{output_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge motion and terrain from multiple lafan_terrain folders"
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=[],
        help="Folder names to merge (e.g. obstacles1_subject1 obstacles2_subject1)",
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
        default="merged",
        help="Output folder name (default: merged)",
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

    print(f"Merging {len(folders)} folders: {folders}")
    print(f"Output: {args.output}\n")
    merge_folders(folders, args.output, motion_base, terrain_base)


if __name__ == "__main__":
    main()
