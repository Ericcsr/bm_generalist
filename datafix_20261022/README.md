Fix 1 of the 2026-10-22 training-data audit (report: ~/Projects/wbc_bench/docs/training_data_audit_2026-10-22.md on the
local machine). The 50 parametric stair clips h<rise>_w<tread>_s<speed>_motion_0.npz stored body_pos_w / body_quat_w /
body_lin_vel_w / body_ang_vel_w in MuJoCo DFS body order; MotionLoader indexes them in Isaac body order.
param_body_order_fixed/: permuted to Isaac order (wbc_bench/tools/fix_param_body_order.py; FK-validated 0.00 mm;
joint_pos/joint_vel/fps untouched). apply.sh swaps the base-dir files for symlinks (originals -> param_body_order_orig/),
rollback.sh undoes it. Takes effect at the NEXT training launch (running jobs keep their loaded data).
