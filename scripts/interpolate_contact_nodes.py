#!/usr/bin/env python3
"""Interpolate contact node timestamps and contact_mask to a new FPS.

Contact nodes from contact_terrain_reconstruction have timestamps at source_fps
(e.g. 30 from GMR). When motion is resampled to output_fps (e.g. 50), contact
labels must be interpolated so timestamps align with the new frame indices.

- Nodes: nearest-neighbor timestamp scaling: new_timestamp = round(old_timestamp * output_fps / source_fps)
- contact_mask: resampled to output frame count via nearest-neighbor (T_src -> T_out).
  Supports (T, C) for any C (e.g. C=4 feet+hands, C=5 with seat / hips column).
"""

import argparse
import numpy as np
from pathlib import Path


def _resample_contact_mask(contact_mask: np.ndarray, num_frames_target: int) -> np.ndarray:
    """Resample contact mask (T, C) to target frame count. Uses nearest-neighbor; preserves C."""
    n_src = contact_mask.shape[0]
    n_cols = contact_mask.shape[1]
    if num_frames_target == n_src:
        return contact_mask.copy()
    out = np.zeros((num_frames_target, n_cols), dtype=bool)
    for i in range(num_frames_target):
        src_idx = int(i * (n_src - 1) / max(1, num_frames_target - 1))
        out[i] = contact_mask[src_idx]
    return out


def interpolate_contact_nodes(
    contact_path: str | Path,
    output_path: str | Path,
    source_fps: float,
    output_fps: float,
) -> bool:
    """Load contact nodes, interpolate timestamps, save to output_path.

    Args:
        contact_path: Path to *_contact_nodes.npy
        output_path: Path to save interpolated contact nodes
        source_fps: FPS of original motion (contact timestamps are frame indices at this fps)
        output_fps: Target FPS (matches omni_to_npz output)

    Returns:
        True if successful, False if contact file not found or empty.
    """
    contact_path = Path(contact_path)
    output_path = Path(output_path)
    if not contact_path.exists():
        return False

    data = np.load(str(contact_path), allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()
    data = dict(data) if hasattr(data, "keys") else {}
    scale = output_fps / source_fps

    def _interpolate_node_list(nodes):
        if nodes is None or len(nodes) == 0:
            return nodes if nodes is not None else []
        nodes = list(nodes)  # handle numpy array
        out = []
        for node in nodes:
            new_node = dict(node)
            old_ts = node.get("timestamp", 0)
            new_ts = int(round(old_ts * scale))
            new_ts = max(0, new_ts)
            new_node["timestamp"] = new_ts
            out.append(new_node)
        return out

    nodes = data.get("nodes", [])
    nodes_all = data.get("nodes_all", [])

    # Resample contact_mask to output FPS (this is what training uses)
    out_data = dict(data)
    if "contact_mask" in data:
        mask = np.asarray(data["contact_mask"], dtype=bool)
        if mask.ndim == 2 and mask.shape[0] > 0 and mask.shape[1] > 0:
            n_src = mask.shape[0]
            n_out = int(round(n_src * output_fps / source_fps))
            n_out = max(1, n_out)
            out_data["contact_mask"] = _resample_contact_mask(mask, n_out)
            if n_out != n_src:
                print(
                    f"[interpolate_contact_nodes] Resampled contact_mask: "
                    f"{n_src} -> {n_out} frames, {mask.shape[1]} label dim(s)"
                )
        else:
            out_data["contact_mask"] = mask

    if not nodes and not nodes_all:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), out_data, allow_pickle=True)
        return True

    out_data["positions"] = data.get("positions", np.empty((0, 3)))
    out_data["labels"] = data.get("labels", np.empty(0, dtype=object))
    out_data["nodes"] = _interpolate_node_list(nodes)
    if nodes_all:
        out_data["nodes_all"] = _interpolate_node_list(nodes_all)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), out_data, allow_pickle=True)
    return True


def main():
    parser = argparse.ArgumentParser(description="Interpolate contact node timestamps to new FPS")
    parser.add_argument("--input", "-i", required=True, help="Path to *_contact_nodes.npy")
    parser.add_argument("--output", "-o", required=True, help="Output path for interpolated contact nodes")
    parser.add_argument("--source-fps", type=float, required=True, help="FPS of original motion (GMR output)")
    parser.add_argument("--output-fps", type=float, required=True, help="Target FPS (matches omni_to_npz)")
    args = parser.parse_args()

    ok = interpolate_contact_nodes(
        args.input,
        args.output,
        args.source_fps,
        args.output_fps,
    )
    if ok:
        print(f"[interpolate_contact_nodes] Saved to {args.output} (source_fps={args.source_fps} -> output_fps={args.output_fps})")
    else:
        print(f"[interpolate_contact_nodes] Input not found: {args.input}")
        exit(1)


if __name__ == "__main__":
    main()
