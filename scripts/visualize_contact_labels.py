#!/usr/bin/env python3
"""Visualize contact labels for a motion using matplotlib.

Useful for debugging blinking: plot contact state over time for each EEF
(L/R foot, L/R wrist). Run from bm_generalist root:

  python scripts/visualize_contact_labels.py --motion_dir obstacles1_subject1 \\
      --terrain_folder lafan_terrain/obstacles1_subject1 --motion_idx 0

Saves contact_labels_motion{N}.png and prints stats (transitions, min duration).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Load contact_labels module directly (avoid mdp __init__ which pulls isaaclab)
BM_ROOT = Path(__file__).resolve().parent.parent
CONTACT_LABELS_PATH = (
    BM_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking"
    / "tasks" / "tracking" / "mdp" / "contact_labels.py"
)
import importlib.util
_spec = importlib.util.spec_from_file_location("contact_labels", CONTACT_LABELS_PATH)
_contact_labels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_contact_labels)
load_contact_labels_for_motions = _contact_labels.load_contact_labels_for_motions
_is_left_foot = _contact_labels._is_left_foot
_is_right_foot = _contact_labels._is_right_foot
_is_left_hand = _contact_labels._is_left_hand
_is_right_hand = _contact_labels._is_right_hand


EEF_NAMES = ["L_foot", "R_foot", "L_wrist", "R_wrist"]


def _resolve_path(p: Path, roots: list[Path]) -> Path:
    """Resolve path; try roots if not absolute."""
    if p.is_absolute() and p.exists():
        return p
    for r in roots:
        cand = (r / p).resolve()
        if cand.exists():
            return cand
    return p.resolve()


def _build_motion_terrain_pairs(motion_dir: str, terrain_folder: str) -> tuple[list[str], list[str]]:
    """Same logic as tracking_env_cfg_3pt_binded_multi_terrain."""
    roots = [Path.cwd(), BM_ROOT]
    motion_path = _resolve_path(Path(motion_dir), roots)
    terrain_path = _resolve_path(Path(terrain_folder), roots)

    motion_by_base = {f.stem: f for f in motion_path.glob("*.npz")}
    if not motion_by_base:
        raise FileNotFoundError(f"No .npz in {motion_path}")
    all_bases = sorted(motion_by_base.keys())
    motion_files = [str(motion_by_base[b].resolve()) for b in all_bases]
    return motion_files, all_bases


def analyze_contact_mask(mask: np.ndarray, eef_names: list[str]) -> dict:
    """Compute stats: transitions, min/max run length per EEF."""
    stats = {}
    for col, name in enumerate(eef_names):
        vec = mask[:, col].astype(float)
        # Transitions: diff != 0
        transitions = np.sum(np.abs(np.diff(vec)) > 0.5)
        # Run lengths
        runs = []
        in_run = False
        run_len = 0
        for v in vec:
            if v > 0.5:
                run_len += 1
                in_run = True
            else:
                if in_run:
                    runs.append(run_len)
                    run_len = 0
                in_run = False
        if in_run:
            runs.append(run_len)
        stats[name] = {
            "transitions": int(transitions),
            "runs": runs,
            "min_run": min(runs) if runs else 0,
            "max_run": max(runs) if runs else 0,
            "total_contact_frames": int(np.sum(vec > 0.5)),
        }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Visualize contact labels for debugging")
    parser.add_argument("--motion_dir", type=str, default="obstacles1_subject1")
    parser.add_argument("--terrain_folder", type=str, default="lafan_terrain/obstacles1_subject1")
    parser.add_argument("--motion_idx", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    parser.add_argument("--compare-raw", action="store_true", help="Also plot raw (from nodes) vs merged for comparison")
    args = parser.parse_args()

    motion_files, all_bases = _build_motion_terrain_pairs(args.motion_dir, args.terrain_folder)
    roots = [Path.cwd(), BM_ROOT]
    terrain_folder = _resolve_path(Path(args.terrain_folder), roots)

    # Load motion frame counts
    num_frames_per_motion = []
    for mf in motion_files:
        data = np.load(mf)
        num_frames_per_motion.append(data["joint_pos"].shape[0])

    # Load contact labels (same as MultiMotionLoader)
    cl = load_contact_labels_for_motions(
        str(terrain_folder),
        motion_files,
        num_frames_per_motion,
    )
    if cl is None:
        print("[ERROR] No contact labels loaded. Check terrain_folder has *_contact_nodes.npy")
        return 1

    # Pick motion
    idx = args.motion_idx
    if idx >= len(motion_files):
        print(f"[ERROR] motion_idx {idx} >= num_motions {len(motion_files)}")
        return 1

    start = sum(num_frames_per_motion[:idx])
    end = start + num_frames_per_motion[idx]
    mask = cl[start:end].copy()  # (T, 4)

    motion_name = Path(motion_files[idx]).stem
    n_frames_motion = num_frames_per_motion[idx]
    print(f"[INFO] Motion {idx}: {motion_name}, {mask.shape[0]} frames")
    print(f"[INFO] Contact file: {terrain_folder / f'{motion_name}_contact_nodes.npy'}")

    # Check source; optionally load raw for --compare-raw
    contact_path = terrain_folder / f"{motion_name}_contact_nodes.npy"
    if not contact_path.exists():
        contact_path = terrain_folder / "scene_contact_nodes.npy"
    mask_raw = None
    if contact_path.exists():
        data = np.load(str(contact_path), allow_pickle=True)
        if isinstance(data, np.ndarray) and data.ndim == 0:
            data = data.item()
        if "contact_mask" in data:
            cm = np.asarray(data["contact_mask"], dtype=bool)
            if cm.shape[0] != n_frames_motion:
                print(f"[WARN] Frame mismatch: motion has {n_frames_motion} frames, contact_mask has {cm.shape[0]}. "
                      f"Contact may be from different source (e.g. original pkl vs retargeted npz).")
            print("[INFO] Using merged contact_mask from file")
        else:
            print("[INFO] Using merged mask (built from raw nodes at load time)")
        if args.compare_raw and "nodes" in data and len(data["nodes"]) > 0:
            # Build raw mask from nodes (no merge)
            mask_raw = np.zeros((n_frames_motion, 4), dtype=bool)
            for node in data["nodes"]:
                ts = node.get("timestamp", 0)
                label = node.get("label", "floor")
                entity = node.get("entity_info", {})
                jn = entity.get("joint_name", "")
                if ts < 0 or ts >= n_frames_motion:
                    continue
                if label == "floor":
                    if _is_left_foot(jn): mask_raw[ts, 0] = True
                    elif _is_right_foot(jn): mask_raw[ts, 1] = True
                elif label == "hand_terrain":
                    if _is_left_hand(jn): mask_raw[ts, 2] = True
                    elif _is_right_hand(jn): mask_raw[ts, 3] = True
            print("[INFO] Loaded raw mask from nodes for comparison")

    # Stats
    stats = analyze_contact_mask(mask, EEF_NAMES)
    print("\n[STATS] Transitions and run lengths per EEF:")
    for name, s in stats.items():
        print(f"  {name}: {s['transitions']} transitions, runs={s['runs'][:10]}{'...' if len(s['runs'])>10 else ''} "
              f"(min={s['min_run']}, max={s['max_run']} frames)")

    # Plot
    n_rows = 4
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 6), sharex=True)
    if n_rows == 1:
        axes = [axes]
    t = np.arange(mask.shape[0])
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]  # green, red, blue, purple

    for col, (ax, name) in enumerate(zip(axes, EEF_NAMES)):
        ax.fill_between(t, 0, mask[:, col].astype(float), color=colors[col], alpha=0.7, step="post", label="merged")
        if mask_raw is not None:
            ax.step(t, mask_raw[:, col].astype(float) * 0.5 + 0.5, color="black", alpha=0.5, where="post", label="raw")
        ax.set_ylabel(name, fontsize=10)
        ax.set_ylim(-0.1, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["off", "on"])
        ax.grid(True, alpha=0.3)
        if mask_raw is not None:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Frame")
    title = f"Contact labels: {motion_name} (env/motion {idx})"
    if mask_raw is not None:
        title += " [black=raw, color=merged]"
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    out_path = args.out or f"contact_labels_{motion_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Saved {out_path}")
    plt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
