#!/usr/bin/env python3
"""Interpolate contact node timestamps to a new FPS.

Contact nodes from contact_terrain_reconstruction have timestamps at source_fps
(e.g. 30 from GMR). When motion is resampled to output_fps (e.g. 50), contact
labels must be interpolated so timestamps align with the new frame indices.

Uses nearest-neighbor: new_timestamp = round(old_timestamp * output_fps / source_fps)
"""

import argparse
import numpy as np
from pathlib import Path


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
    if not nodes and not nodes_all:
        np.save(str(output_path), data, allow_pickle=True)
        return True

    out_data = {
        "positions": data.get("positions", np.empty((0, 3))),
        "labels": data.get("labels", np.empty(0, dtype=object)),
        "nodes": _interpolate_node_list(nodes),
    }
    if nodes_all:
        out_data["nodes_all"] = _interpolate_node_list(nodes_all)
    if "contact_mask" in data:
        out_data["contact_mask"] = data["contact_mask"]

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
