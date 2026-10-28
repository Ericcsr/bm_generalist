"""Fix the 50 parametric stair clips (`h<rise>_w<tread>_s<speed>_motion_0.npz`): their body arrays (body_pos_w,
body_quat_w, body_lin_vel_w, body_ang_vel_w) are stored in MuJoCo DFS body order, while bm_generalist's
MotionLoader indexes them with the Isaac (BFS) robot body order -- so every tracked body except the pelvis and the
right wrist gets another link's trajectory (e.g. left_ankle_roll <- left_shoulder_yaw), and ee_body_pos
terminates these envs on every step.

Body k (k >= 1) is the child link of joint k-1 in both conventions, so
  isaac bodies  = [pelvis] + [link of j for j in Isaac joint order]
  mujoco bodies = [pelvis] + [link of j for j in MuJoCo joint order]
and the fix is a pure permutation (joint_pos / joint_vel are already in Isaac order and are left untouched).

  python tools/fix_param_body_order.py SRC_DIR DST_DIR [--glob 'h0.*_motion_0.npz']
Writes fixed copies to DST_DIR (never edits SRC_DIR) and validates every output: FK of (root, joint_pos) on the
training robot must reproduce body_pos_w in Isaac order (< 1 mm). Point the training motion dir (a symlink farm)
at DST_DIR for these 50 stems.
"""
import argparse
import glob
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import robot as RB  # noqa: E402

BODY_KEYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def orders():
    m = RB.robot_spec(visual=False).compile()
    link = lambda j: m.body(m.jnt_bodyid[m.joint(j).id]).name
    isaac_joints = [RB.MUJOCO_JOINT_NAMES[i] for i in RB.MUJOCO_TO_ISAAC]     # isaac joint i
    isaac = ["pelvis"] + [link(j) for j in isaac_joints]
    mj = ["pelvis"] + [link(j) for j in RB.MUJOCO_JOINT_NAMES]
    return m, isaac, mj


def fk_isaac(m, d, isaac, root_pos, root_quat, joint_pos_isaac):
    d.qpos[:3], d.qpos[3:7] = root_pos, root_quat
    d.qpos[7:36] = joint_pos_isaac[RB.ISAAC_TO_MUJOCO]
    mujoco.mj_kinematics(m, d)
    return np.array([d.xpos[m.body(n).id] for n in isaac])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--glob", default="h0.*_motion_0.npz")
    a = ap.parse_args()
    m, isaac, mj = orders()
    d = mujoco.MjData(m)
    perm = np.array([mj.index(n) for n in isaac])          # isaac slot i <- mujoco-stored slot perm[i]
    os.makedirs(a.dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.src, a.glob)))
    assert files, f"no files match {a.glob} in {a.src}"
    for f in files:
        z = dict(np.load(f))
        t = len(z["joint_pos"]) // 2
        before = np.abs(fk_isaac(m, d, isaac, z["body_pos_w"][t, 0], z["body_quat_w"][t, 0], z["joint_pos"][t])
                        - z["body_pos_w"][t]).max()
        for k in BODY_KEYS:
            z[k] = z[k][:, perm]
        err = max(np.abs(fk_isaac(m, d, isaac, z["body_pos_w"][s, 0], z["body_quat_w"][s, 0], z["joint_pos"][s])
                         - z["body_pos_w"][s]).max() for s in range(0, len(z["joint_pos"]), 25))
        assert err < 1e-3, (f, err)
        np.savez(os.path.join(a.dst, os.path.basename(f)), **z)
        print(f"{os.path.basename(f)}: FK mismatch before {before * 100:.1f} cm -> after {err * 1000:.2f} mm")
    print(f"fixed {len(files)} files -> {a.dst}")


if __name__ == "__main__":
    main()
