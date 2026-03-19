# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Load contact labels from terrain folder and convert to per-frame masks for training.

Contact nodes from contact_terrain_reconstruction have nodes with timestamp, label, entity_info.
Maps to robot links: floor -> ankle_roll (feet), hand_terrain -> wrist_yaw (hands).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Sequence

# Human skeleton joint names and robot link names that map to robot links
LEFT_FOOT_NAMES = (
    "L_Foot", "L_Toe", "LeftFoot", "LeftToe", "LeftToe_EndSite", "LeftToeBase",
    "left_ankle_roll_link", "left_ankle_pitch_link",
)
RIGHT_FOOT_NAMES = (
    "R_Foot", "R_Toe", "RightFoot", "RightToe", "RightToe_EndSite", "RightToeBase",
    "right_ankle_roll_link", "right_ankle_pitch_link",
)
LEFT_HAND_NAMES = ("L_Hand", "L_Wrist", "LeftHand", "LeftForeArm")
RIGHT_HAND_NAMES = ("R_Hand", "R_Wrist", "RightHand", "RightForeArm")


def _is_left_foot(joint_name: str) -> bool:
    return any(n in joint_name for n in LEFT_FOOT_NAMES) or (
        "Left" in joint_name and ("Foot" in joint_name or "Toe" in joint_name)
    )


def _is_right_foot(joint_name: str) -> bool:
    return any(n in joint_name for n in RIGHT_FOOT_NAMES) or (
        "Right" in joint_name and ("Foot" in joint_name or "Toe" in joint_name)
    )


def _is_left_hand(joint_name: str) -> bool:
    return any(n in joint_name for n in LEFT_HAND_NAMES) or (
        "Left" in joint_name and ("Hand" in joint_name or "Wrist" in joint_name)
    )


def _is_right_hand(joint_name: str) -> bool:
    return any(n in joint_name for n in RIGHT_HAND_NAMES) or (
        "Right" in joint_name and ("Hand" in joint_name or "Wrist" in joint_name)
    )


def _entity_to_column(label: str, joint_name: str) -> int | None:
    """Map (label, joint_name) to mask column: 0=L_foot, 1=R_foot, 2=L_wrist, 3=R_wrist."""
    if label == "floor":
        if _is_left_foot(joint_name):
            return 0
        if _is_right_foot(joint_name):
            return 1
    elif label == "hand_terrain":
        if _is_left_hand(joint_name):
            return 2
        if _is_right_hand(joint_name):
            return 3
    return None


def _merge_nodes_to_mask(
    nodes: list,
    num_frames: int,
    max_time_gap: int = 10,
    max_spatial_dist: float = 0.15,
    gap_fill_frames: int = 5,
    min_contact_duration_frames: int = 6,
) -> np.ndarray:
    """Merge temporally/spatially adjacent contact nodes to reduce blinking (same logic as contact_label_merge)."""
    mask = np.zeros((num_frames, 4), dtype=bool)
    groups: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
    for node in nodes:
        ts = node.get("timestamp", 0)
        label = node.get("label", "floor")
        entity = node.get("entity_info", {})
        joint_name = entity.get("joint_name", "")
        pos = np.asarray(node.get("position", [0, 0, 0]), dtype=float)
        col = _entity_to_column(label, joint_name)
        if col is None or ts < 0 or ts >= num_frames:
            continue
        key = (label, joint_name)
        if key not in groups:
            groups[key] = []
        groups[key].append((ts, pos))

    for key, items in groups.items():
        col = _entity_to_column(key[0], key[1])
        if col is None:
            continue
        items.sort(key=lambda x: x[0])
        intervals: list[tuple[int, int]] = []
        start, end = items[0][0], items[0][0]
        last_pos = items[0][1]
        for i in range(1, len(items)):
            ts, pos = items[i]
            dt = ts - end
            dist = float(np.linalg.norm(pos - last_pos))
            if dt <= max_time_gap and dist <= max_spatial_dist:
                end = ts
                last_pos = pos
            else:
                intervals.append((start, end))
                start, end = ts, ts
                last_pos = pos
        intervals.append((start, end))

        if gap_fill_frames > 0 and len(intervals) > 1:
            merged: list[tuple[int, int]] = [intervals[0]]
            for s, e in intervals[1:]:
                last_s, last_e = merged[-1]
                gap = s - last_e - 1
                if 0 <= gap <= gap_fill_frames:
                    merged[-1] = (last_s, e)
                else:
                    merged.append((s, e))
            intervals = merged

        if min_contact_duration_frames > 0:
            extended: list[tuple[int, int]] = []
            for s, e in intervals:
                duration = e - s + 1
                if duration < min_contact_duration_frames:
                    deficit = min_contact_duration_frames - duration
                    extend_before = deficit // 2
                    extend_after = deficit - extend_before
                    new_s = max(0, s - extend_before)
                    new_e = min(num_frames - 1, e + extend_after)
                    extended.append((new_s, new_e))
                else:
                    extended.append((s, e))
            intervals = extended
            if len(intervals) > 1:
                intervals.sort(key=lambda x: x[0])
                merged = [intervals[0]]
                for s, e in intervals[1:]:
                    last_s, last_e = merged[-1]
                    if s <= last_e + 1:
                        merged[-1] = (last_s, max(last_e, e))
                    else:
                        merged.append((s, e))
                intervals = merged

        for s, e in intervals:
            mask[s : e + 1, col] = True

    return mask


def load_contact_labels_for_motion(
    terrain_folder: str | Path,
    motion_stem: str,
    num_frames: int,
) -> np.ndarray | None:
    """Load contact nodes and build per-frame mask (T, 4): [left_foot, right_foot, left_wrist, right_wrist].

    Args:
        terrain_folder: Path to terrain folder (contains {motion_stem}_contact_nodes.npy).
        motion_stem: Base name of motion (e.g. motion_0).
        num_frames: Number of frames in motion (for output shape).

    Returns:
        (T, 4) bool array, or None if contact file not found or empty.
    """
    terrain_folder = Path(terrain_folder)
    contact_path = terrain_folder / f"{motion_stem}_contact_nodes.npy"
    if not contact_path.exists():
        contact_path = terrain_folder / "scene_contact_nodes.npy"
    if not contact_path.exists():
        return None

    data = np.load(str(contact_path), allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()

    # Prefer merged contact_mask if present (reduces frame-to-frame flicker)
    if "contact_mask" in data:
        mask = np.asarray(data["contact_mask"], dtype=bool)
        if mask.shape != (num_frames, 4):
            # Truncate or pad to match motion length (motion may have different frame count)
            out = np.zeros((num_frames, 4), dtype=bool)
            n = min(mask.shape[0], num_frames)
            out[:n] = mask[:n]
            return out
        return mask

    if "nodes" not in data or len(data["nodes"]) == 0:
        return None

    nodes = data["nodes"]

    # Apply merge when contact_mask not present (reduces blinking from raw noisy nodes)
    mask = _merge_nodes_to_mask(
        nodes,
        num_frames,
        max_time_gap=10,
        max_spatial_dist=0.15,
        gap_fill_frames=5,
        min_contact_duration_frames=6,  # 0.2s at 30fps
    )
    return mask


def load_contact_labels_for_motions(
    terrain_folder: str | Path,
    motion_files: Sequence[str],
    num_frames_per_motion: Sequence[int],
) -> np.ndarray | None:
    """Load contact labels for multiple motions and concatenate into single (T_total, 4) array.

    Args:
        terrain_folder: Path to terrain folder.
        motion_files: List of motion file paths (e.g. [/path/motion_0.npz, ...]).
        num_frames_per_motion: Number of frames for each motion.

    Returns:
        (T_total, 4) bool array, or None if no contact files found.
    """
    terrain_folder = Path(terrain_folder)
    masks = []
    for motion_path, nf in zip(motion_files, num_frames_per_motion):
        stem = Path(motion_path).stem
        m = load_contact_labels_for_motion(terrain_folder, stem, nf)
        if m is not None:
            masks.append(m)
        else:
            masks.append(np.zeros((nf, 4), dtype=bool))

    if not masks:
        return None
    return np.concatenate(masks, axis=0)


def load_contact_labels_for_motions_dual_folder(
    terrain_folder: str | Path,
    object_folder: str | Path,
    motion_files: Sequence[str],
    num_frames_per_motion: Sequence[int],
) -> np.ndarray | None:
    """Load contact labels for multiple motions, trying object_folder first then terrain_folder.

    For each motion: try object_folder/{stem}_contact_nodes.npy first (object motions),
    then terrain_folder/{stem}_contact_nodes.npy or scene_contact_nodes.npy (terrain motions).
    If neither found, use zeros (free-space motions).

    Args:
        terrain_folder: Path to terrain folder (contact nodes from terrain pipeline).
        object_folder: Path to object folder (contact nodes from OMOMO object pipeline).
        motion_files: List of motion file paths.
        num_frames_per_motion: Number of frames for each motion.

    Returns:
        (T_total, 4) bool array, or None if both folders empty/invalid.
    """
    terrain_folder = Path(terrain_folder)
    object_folder = Path(object_folder)
    masks = []
    for motion_path, nf in zip(motion_files, num_frames_per_motion):
        stem = Path(motion_path).stem
        m = load_contact_labels_for_motion(object_folder, stem, nf)
        if m is None:
            m = load_contact_labels_for_motion(terrain_folder, stem, nf)
        if m is not None:
            masks.append(m)
        else:
            masks.append(np.zeros((nf, 4), dtype=bool))

    if not masks:
        return None
    return np.concatenate(masks, axis=0)
