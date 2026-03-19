"""Compute average motion length (frames and seconds) for all NPZ files in a motion directory.

.. code-block:: bash

    python scripts/compute_avg_motion_length.py --motion_dir merged
    python scripts/compute_avg_motion_length.py --motion_dir lafan --unit frames
"""

import argparse
import os

import numpy as np


def get_motion_length(npz_path: str) -> tuple[int, float] | None:
    """Load NPZ and return (num_frames, duration_sec). Returns None if invalid."""
    try:
        data = np.load(npz_path)
        # Prefer joint_pos, fallback to body_pos_w (merge_contact_dataset format)
        if "joint_pos" in data:
            num_frames = data["joint_pos"].shape[0]
        elif "body_pos_w" in data:
            num_frames = data["body_pos_w"].shape[0]
        else:
            return None
        fps = float(data["fps"]) if "fps" in data else 50.0
        duration_sec = num_frames / fps
        return num_frames, duration_sec
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute average motion length (frames and seconds) for NPZ files in a directory."
    )
    parser.add_argument(
        "--motion_dir",
        type=str,
        required=True,
        help="Path to the motion directory containing .npz files.",
    )
    parser.add_argument(
        "--unit",
        type=str,
        choices=["frames", "seconds", "both"],
        default="both",
        help="Unit to print for averages: frames, seconds, or both (default).",
    )
    args = parser.parse_args()

    motion_dir = args.motion_dir
    if not os.path.isdir(motion_dir):
        print(f"[ERROR]: motion_dir not found: {motion_dir}")
        return 1

    npz_files = sorted(f for f in os.listdir(motion_dir) if f.endswith(".npz"))
    if not npz_files:
        print(f"[WARN]: No .npz files found in {motion_dir}")
        return 0

    lengths_frames = []
    lengths_sec = []
    failed = []

    for f in npz_files:
        path = os.path.join(motion_dir, f)
        result = get_motion_length(path)
        if result is not None:
            lengths_frames.append(result[0])
            lengths_sec.append(result[1])
        else:
            failed.append(f)

    if not lengths_frames:
        print("[ERROR]: No valid NPZ motion files could be loaded.")
        return 1

    n = len(lengths_frames)
    avg_frames = sum(lengths_frames) / n
    avg_sec = sum(lengths_sec) / n
    min_frames = min(lengths_frames)
    max_frames = max(lengths_frames)
    min_sec = min(lengths_sec)
    max_sec = max(lengths_sec)
    total_frames = sum(lengths_frames)
    total_sec = sum(lengths_sec)

    print(f"Motion directory: {motion_dir}")
    print(f"Total motions: {n}")
    if failed:
        print(f"Skipped (invalid): {len(failed)}")
    print()
    if args.unit in ("frames", "both"):
        print(f"Frames  - avg: {avg_frames:.1f}, min: {min_frames}, max: {max_frames}, total: {total_frames}")
    if args.unit in ("seconds", "both"):
        print(f"Seconds - avg: {avg_sec:.2f}, min: {min_sec:.2f}, max: {max_sec:.2f}, total: {total_sec:.2f}")

    return 0


if __name__ == "__main__":
    exit(main())
