#!/usr/bin/env python3
"""
Recursively find CSV files under a root whose filenames contain a keyword,
then copy them into a destination directory.

Handles deep / repeated basenames via --preserve-structure or flatten with
path-based disambiguation on name collision.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recursively copy CSV files whose names contain a keyword."
    )
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Root directory to search (walked recursively).",
    )
    p.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Output directory (created if missing).",
    )
    p.add_argument(
        "--keyword",
        type=str,
        required=True,
        help="Substring that must appear in the file name (e.g. 'contact'). "
        "Path directories are not matched, only the final file name.",
    )
    p.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Match keyword with case sensitivity (default: insensitive).",
    )
    p.add_argument(
        "--preserve-structure",
        action="store_true",
        help="Mirror relative paths from --source under --dest (avoids basename collisions).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    return p.parse_args()


def name_matches(path: Path, needle: str, case_sensitive: bool) -> bool:
    name = path.name if case_sensitive else path.name.lower()
    key = needle if case_sensitive else needle.lower()
    return key in name


def flatten_dest_path(dest_root: Path, source_root: Path, src_file: Path) -> Path:
    """Single-folder layout: use basename; disambiguate duplicates with encoded relative path."""
    rel = src_file.relative_to(source_root)
    candidate = dest_root / src_file.name
    if not candidate.exists():
        return candidate
    # Same basename from another subfolder: encode full relative path (safe for fs)
    flat_name = "__".join(rel.parts)
    return dest_root / flat_name


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    dest = args.dest.expanduser().resolve()

    if not source.is_dir():
        print(f"Error: source is not a directory: {source}", file=sys.stderr)
        return 1

    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    needle = args.keyword
    matched: list[Path] = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".csv":
            continue
        if not name_matches(path, needle, args.case_sensitive):
            continue
        matched.append(path)

    matched.sort(key=lambda p: str(p.relative_to(source)))

    print(f"Found {len(matched)} CSV file(s) matching keyword {needle!r} under {source}")

    for src in matched:
        rel = src.relative_to(source)
        if args.preserve_structure:
            out = dest / rel
            if not args.dry_run:
                out.parent.mkdir(parents=True, exist_ok=True)
        else:
            out = flatten_dest_path(dest, source, src)

        if args.dry_run:
            print(f"  [dry-run] {src} -> {out}")
        else:
            if out.resolve() == src.resolve():
                print(f"  skip (same path): {src}")
                continue
            shutil.copy2(src, out)
            print(f"  copied: {rel} -> {out.relative_to(dest)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
