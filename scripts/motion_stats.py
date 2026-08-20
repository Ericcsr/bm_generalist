#!/usr/bin/env python3
"""Classify each motion in --motion_dir by which paired assets exist, and print a summary.

Categories (mutually exclusive):
  - motion only            : no object proxy and no terrain
  - motion+object          : has {stem}_thin_plate_proxy.urdf + {stem}_object_fitted.npz
  - motion+terrain         : has {base}_terrain.urdf  (base = stem minus trailing _motion_<N>)
  - motion+object+terrain  : has both

Asset lookup mirrors the training config / replay_scene.py naming, and additionally
scans --object_terrain_folder RECURSIVELY (its combined object+terrain assets live in
subdirectories such as bones_box/). Pure Isaac-free filesystem scan -- no sim launch.

Also reports:
  - asset fan-out: how many unique object / terrain assets are shared by how many motions
  - a typical file layout per category (the files a sample motion has across the 4 folders)

Usage:
    python scripts/motion_stats.py
    python scripts/motion_stats.py --data_root data
    python scripts/motion_stats.py --print_names all            # print every motion, grouped
    python scripts/motion_stats.py --print_names terrain        # only, object, terrain, object_terrain, all
    python scripts/motion_stats.py --list_csv /tmp/motion_categories.csv
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = ["motion only", "motion+object", "motion+terrain", "motion+object+terrain"]

# --print_names short keys -> category label ("all" prints every category)
NAME_KEYS = {
    "only": "motion only",
    "object": "motion+object",
    "terrain": "motion+terrain",
    "object_terrain": "motion+object+terrain",
    "all": "all",
}

ALL_FOLDERS = ["motion_dir", "object_folder", "terrain_folder", "object_terrain_folder"]


def _terrain_base(stem: str) -> str:
    """Terrain base = motion stem with a trailing _motion_<N> stripped (matches _find_terrain)."""
    return re.sub(r"_motion_\d+$", "", stem)


def _scan_suffix(folders_rec: list[tuple[Path, bool]], suffix: str) -> dict[str, Path]:
    """Map {name-without-suffix -> path} for every *<suffix> file under the folders.

    Later folders override earlier ones (object_terrain_folder wins), matching the
    training config's merge order.
    """
    out: dict[str, Path] = {}
    for folder, rec in folders_rec:
        if not folder.is_dir():
            continue
        it = folder.rglob(f"*{suffix}") if rec else folder.glob(f"*{suffix}")
        for f in it:
            out[f.name[: -len(suffix)]] = f
    return out


def classify(motion_dir: Path, object_folder: Path, terrain_folder: Path, object_terrain_folder: Path):
    """Return records: {stem -> {"cat", "object_key", "terrain_key"}}.

    object_key / terrain_key are the matched asset identifiers (None if unmatched); used
    for fan-out stats. object_terrain_folder is scanned recursively (assets in subfolders).
    """
    # object_terrain_folder scanned recursively; other folders top-level only (matches training).
    obj_dirs = [(object_folder, False), (object_terrain_folder, True)]
    ter_dirs = [(terrain_folder, False), (object_terrain_folder, True)]

    proxy = _scan_suffix(obj_dirs, "_thin_plate_proxy.urdf")
    fitted = _scan_suffix(obj_dirs, "_object_fitted.npz")
    terrain = _scan_suffix(ter_dirs, "_terrain.urdf")

    records: dict[str, dict] = {}
    for m in sorted(motion_dir.glob("*.npz")):
        stem = m.stem
        object_key = stem if (stem in proxy and stem in fitted) else None

        terrain_key = None
        if stem in terrain:
            terrain_key = stem
        elif _terrain_base(stem) in terrain:
            terrain_key = _terrain_base(stem)

        has_o, has_t = object_key is not None, terrain_key is not None
        if has_o and has_t:
            cat = "motion+object+terrain"
        elif has_o:
            cat = "motion+object"
        elif has_t:
            cat = "motion+terrain"
        else:
            cat = "motion only"
        records[stem] = {"cat": cat, "object_key": object_key, "terrain_key": terrain_key}

    return records


def _fanout_report(records: dict, key_name: str) -> str:
    """Summarize how many unique assets (by key_name) are shared by how many motions."""
    per_asset = Counter(r[key_name] for r in records.values() if r[key_name] is not None)
    if not per_asset:
        return "  (none)"
    n_assets = len(per_asset)
    n_motions = sum(per_asset.values())
    avg = n_motions / n_assets
    dist = Counter(per_asset.values())  # motions-per-asset -> how many such assets
    lines = [f"  {n_assets} unique assets used by {n_motions} motions (avg {avg:.2f} motions/asset, max {max(per_asset.values())})"]
    lines.append("  motions-per-asset distribution:")
    for k in sorted(dist):
        lines.append(f"      shared by {k:>2} motion(s): {dist[k]:>4} asset(s)")
    return "\n".join(lines)


def _related_files(stem: str, folders: dict[str, Path]) -> dict[str, list[str]]:
    """Files across the 4 folders whose name starts with the motion stem or its terrain base."""
    base = _terrain_base(stem)
    prefixes = (stem, base)
    out: dict[str, list[str]] = {}
    for label, folder in folders.items():
        if not folder.is_dir():
            continue
        hits = []
        for f in sorted(folder.rglob("*")):
            if f.is_file() and f.name.startswith(prefixes):
                hits.append(f.name)
        if hits:
            out[label] = hits
    return out


def main():
    parser = argparse.ArgumentParser(description="Categorize motions by paired assets.")
    parser.add_argument("--data_root", type=str, default="data", help="Root holding the 4 folders.")
    parser.add_argument("--motion_dir", type=str, default=None, help="Override (default: <data_root>/motion_dir).")
    parser.add_argument("--object_folder", type=str, default=None, help="Override (default: <data_root>/object_folder).")
    parser.add_argument("--terrain_folder", type=str, default=None, help="Override (default: <data_root>/terrain_folder).")
    parser.add_argument("--object_terrain_folder", type=str, default=None, help="Override (default: <data_root>/object_terrain_folder).")
    parser.add_argument("--print_names", type=str, default=None, choices=sorted(NAME_KEYS),
                        help="Print all motion names for a category (only|object|terrain|object_terrain|all).")
    parser.add_argument("--no_layout", action="store_true", help="Skip the per-category typical-file-layout section.")
    parser.add_argument("--list_csv", type=str, default=None, help="Optional: write per-motion stem,category,object_key,terrain_key CSV.")
    args = parser.parse_args()

    root = Path(args.data_root)
    folders = {
        "motion_dir": Path(args.motion_dir) if args.motion_dir else root / "motion_dir",
        "object_folder": Path(args.object_folder) if args.object_folder else root / "object_folder",
        "terrain_folder": Path(args.terrain_folder) if args.terrain_folder else root / "terrain_folder",
        "object_terrain_folder": Path(args.object_terrain_folder) if args.object_terrain_folder else root / "object_terrain_folder",
    }
    motion_dir = folders["motion_dir"]
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"motion_dir not found: {motion_dir}")

    records = classify(motion_dir, folders["object_folder"], folders["terrain_folder"], folders["object_terrain_folder"])
    total = len(records)
    counter = Counter(r["cat"] for r in records.values())

    # ---- paths + category counts ----
    print("Folders:")
    for label in ALL_FOLDERS:
        f = folders[label]
        tag = "" if f.is_dir() else "  (MISSING)"
        if label == "object_terrain_folder" and f.is_dir():
            tag = "  (scanned recursively)"
        print(f"  {label:<22}: {f}{tag}")
    print(f"\nTotal motions: {total}\n")
    print(f"{'category':<26}{'count':>7}{'pct':>9}")
    print("-" * 42)
    for c in CATEGORIES:
        n = counter.get(c, 0)
        pct = (n / total * 100) if total else 0.0
        print(f"{c:<26}{n:>7}{pct:>8.1f}%")
    print("-" * 42)
    print(f"{'total':<26}{sum(counter.values()):>7}")

    # ---- asset fan-out ----
    print("\nObject asset fan-out (proxy shared across motions):")
    print(_fanout_report(records, "object_key"))
    print("\nTerrain asset fan-out (terrain shared across motions):")
    print(_fanout_report(records, "terrain_key"))

    # ---- typical file layout per category ----
    if not args.no_layout:
        print("\nTypical file layout per category (files a sample motion has across the 4 folders):")
        seen_sample: dict[str, str] = {}
        for stem, r in records.items():
            seen_sample.setdefault(r["cat"], stem)
        for c in CATEGORIES:
            stem = seen_sample.get(c)
            if not stem:
                continue
            print(f"\n  [{c}]  sample: {stem}")
            rel = _related_files(stem, folders)
            for label in ALL_FOLDERS:
                if label in rel:
                    print(f"    {label}:")
                    for name in rel[label]:
                        print(f"        {name}")

    # ---- optional: print all names ----
    if args.print_names:
        want = NAME_KEYS[args.print_names]
        cats = CATEGORIES if want == "all" else [want]
        print("\n" + "=" * 42)
        for c in cats:
            names = sorted(s for s, r in records.items() if r["cat"] == c)
            print(f"\n### {c}  ({len(names)})")
            for name in names:
                print(name)

    # ---- optional: CSV ----
    if args.list_csv:
        out = Path(args.list_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            f.write("stem,category,object_key,terrain_key\n")
            for stem in sorted(records):
                r = records[stem]
                f.write(f"{stem},{r['cat']},{r['object_key'] or ''},{r['terrain_key'] or ''}\n")
        print(f"\nwrote per-motion categories -> {out.resolve()}")


if __name__ == "__main__":
    main()
