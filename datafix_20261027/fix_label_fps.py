"""Fix 2026-10-27: contact_mask stored at the source 30 fps on 50 fps motions (LaFAN obstacles* clips).

For every training motion whose resolved label (training priority) has a contact_mask whose length differs from the
motion by > 2 %, write a corrected copy to datafix_20261027/label_fps_fixed/: contact_mask resampled (nearest
neighbour, endpoints aligned) to the motion's frame count. Everything else in the file is kept; the wrist columns are
left as stored (all zero on these clips: no hand labels on non-object motions). Run from ~/sky_workdir/bm_generalist.
"""
import json
import os

import numpy as np

MD = "all_object_all_terrain_merged_bones_freespace_seat_stairs_pickplace500"
FOLDERS = ("csv_hand_object_terrain", "omomo_object/omomo_all_foot", "lafan_terrain/all_lafan_terrain_rf_bones_seat_stair")
OUT = "datafix_20261027/label_fps_fixed"
os.makedirs(OUT, exist_ok=True)

manifest = []
for f in sorted(os.listdir(MD)):
    if not f.endswith(".npz"):
        continue
    stem = f[:-4]
    lab = next((d for d in FOLDERS if os.path.exists(f"{d}/{stem}_contact_nodes.npy")), None)
    if lab is None:
        continue
    src = f"{lab}/{stem}_contact_nodes.npy"
    d = np.load(src, allow_pickle=True).item()
    if "contact_mask" not in d:
        continue
    T = len(np.load(os.path.join(MD, f))["joint_pos"])
    cm = np.asarray(d["contact_mask"], bool)
    if abs(len(cm) / T - 1.0) <= 0.02:
        continue
    assert lab.startswith("lafan_terrain"), (stem, lab)                    # only the LaFAN obstacles labels expected
    assert not cm[:, 2:4].any(), stem                                       # non-object motion: wrist columns stay zero
    idx = np.rint(np.arange(T) * (len(cm) - 1) / max(T - 1, 1)).astype(int)
    d["contact_mask"] = cm[idx]
    d["contact_mask_fix_20261027"] = f"resampled {len(cm)} -> {T} frames (30 -> 50 fps)"
    np.save(os.path.join(OUT, f"{stem}_contact_nodes.npy"), d, allow_pickle=True)
    manifest.append(dict(stem=stem, label=src, mask_T=len(cm), T=T))
json.dump(manifest, open("datafix_20261027/label_fps_fixed.json", "w"), indent=1)
print(f"wrote {len(manifest)} fixed label files to {OUT}", sorted({m['stem'].split('_rf_')[0] for m in manifest}))
