"""Scan every training motion's contact label (training priority) for fps mismatch and dropped hand nodes.

Run on foxtrot from ~/sky_workdir/bm_generalist (CPU only)."""
import collections
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp")
import contact_labels as CL  # noqa: E402

MD = "all_object_all_terrain_merged_bones_freespace_seat_stairs_pickplace500"
FOLDERS = ("csv_hand_object_terrain", "omomo_object/omomo_all_foot", "lafan_terrain/all_lafan_terrain_rf_bones_seat_stair")
TERR = ("csv_hand_object_terrain", "lafan_terrain/all_lafan_terrain_rf_bones_seat_stair")


def group(stem):
    for pat, g in ((r"^h\d", "param_stair"), (r"^obstacles", "lafan_obstacles"), (r"^(pushAndStumble|walk|run|jump|"
                   r"dance|fallAndGetUp|sprint|ground|aiming|fight|push)", "lafan_other"), (r"bones", "bones"),
                   (r"KIT|kit", "kit"), (r"^sub\d+|omomo", "omomo")):
        if re.search(pat, stem):
            return g
    return "other"


rows = []
for f in sorted(os.listdir(MD)):
    if not f.endswith(".npz"):
        continue
    stem = f[:-4]
    lab = next((d for d in FOLDERS if os.path.exists(f"{d}/{stem}_contact_nodes.npy")), None)
    if lab is None:
        continue
    z = np.load(os.path.join(MD, f))
    T, fps = len(z["joint_pos"]), float(np.asarray(z["fps"]).ravel()[0]) if "fps" in z.files else np.nan
    d = np.load(f"{lab}/{stem}_contact_nodes.npy", allow_pickle=True)
    d = d.item() if d.ndim == 0 else d
    m = re.match(r"^(.+)_motion_\d+$", stem)
    terr = bool(m and any(os.path.exists(f"{t}/{m.group(1)}_terrain.urdf") for t in TERR))
    nodes = list(d.get("nodes") or [])
    hn = [n for n in nodes if n.get("label") == "hand_terrain"]
    hnames = sorted({n.get("entity_info", {}).get("joint_name", "") for n in hn})
    hand_mapped = [nm for nm in hnames if CL._entity_to_column("hand_terrain", nm) is not None]
    if "contact_mask" in d:
        cm = np.asarray(d["contact_mask"], bool)
        mT, mcols, mhand = len(cm), cm.shape[1], int(cm[:, 2:4].sum()) if cm.shape[1] >= 4 else 0
    else:
        mT, mcols, mhand = -1, -1, -1
    ts_max = max((n["timestamp"] for n in nodes), default=-1)
    rows.append(dict(stem=stem, folder=lab, group=group(stem), terrain=terr, T=T, fps=fps, mask_T=mT, mask_cols=mcols,
                     ratio=round(T / mT, 3) if mT > 0 else None, node_ts_max=int(ts_max), n_hand_nodes=len(hn),
                     hand_names=hnames, hand_names_mapped=hand_mapped, mask_hand_frames=mhand,
                     labels=dict(collections.Counter(n.get("label") for n in nodes))))
json.dump(rows, open(sys.argv[1], "w"))
print(len(rows), "labelled motions")
