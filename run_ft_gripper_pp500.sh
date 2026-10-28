#!/bin/bash
# Tracker training: rccu_camera + bimanual gripper MASS (flat-hand collision), +500 pick/place free-space clips
# (350 pick / 150 place), extra pelvis-frame wrist tracking reward on free-space motions (task ...-FreeWrist-...).
# Speed fixes (2026-10-21): no height scanner, spread flat-slot envs, vectorized adaptive sampling,
# per-process USD conversion dirs  ->  ~6.5 s/iter (was ~31 s).
#
#   MAX_ITERS=60000 RUN_NAME=gripper_pp500 LOAD_RUN=<run dir name> CKPT=model_XXXX.pt ./run_ft_gripper_pp500.sh
#   LOAD_RUN empty  -> train from scratch.   SAVE_INTERVAL=1000 -> checkpoint every 1000 iterations (default 500).   With LOAD_RUN: resumes and trains MAX_ITERS *more* iterations.
cd ~/sky_workdir/bm_generalist
MAX_ITERS=${MAX_ITERS:-60000}
RUN_NAME=${RUN_NAME:-new_robot_gripper_pp500}
LOAD_RUN=${LOAD_RUN-2026-07-11_23-51-10_new_robot}
CKPT=${CKPT:-model_59999.pt}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}   # checkpoint every N iterations (hydra agent.save_interval)
NUM_ENVS=${NUM_ENVS:-}              # envs per GPU (empty = task default 4096)
ENVS=()
[ -n "$NUM_ENVS" ] && ENVS=(--num_envs "$NUM_ENVS")
RESUME=()
[ -n "$LOAD_RUN" ] && RESUME=(--resume True --load_run "$LOAD_RUN" --checkpoint "$CKPT")
# remove per-process USD conversion dirs left by processes that are no longer alive
for d in /tmp/IsaacLab/usd_pid*_*; do [ -d "$d" ] || continue; p=${d#/tmp/IsaacLab/usd_pid}; p=${p%%_*}; kill -0 "$p" 2>/dev/null || rm -rf "$d"; done
mkdir -p ft_logs
OMNI_KIT_ACCEPT_EULA=YES setsid nohup ~/miniconda3/envs/bm/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  scripts/rsl_rl/train.py \
  --task=Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-2Bit-FreeWrist-G1-v0 --headless \
  --logger wandb --log_project_name tracking --run_name "$RUN_NAME" \
  --robot rccu_camera_gripper \
  --motion_dir all_object_all_terrain_merged_bones_freespace_seat_stairs_pickplace500 \
  --object_folder omomo_object/omomo_all_foot \
  --terrain_folder lafan_terrain/all_lafan_terrain_rf_bones_seat_stair \
  --object_terrain_folder csv_hand_object_terrain \
  --physics_phase --distributed "${RESUME[@]}" "${ENVS[@]}" \
  --max_iterations "$MAX_ITERS" agent.save_interval="$SAVE_INTERVAL" > "ft_logs/${RUN_NAME}.log" 2>&1 < /dev/null &
echo "launched pid $! -> ft_logs/${RUN_NAME}.log (max_iters=$MAX_ITERS, save_interval=$SAVE_INTERVAL, num_envs=${NUM_ENVS:-default}, resume=${LOAD_RUN:-none}/${CKPT})"
