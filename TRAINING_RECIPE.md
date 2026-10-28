# Whole-body tracker (WBC) finetune — training recipe

This is the recipe for the current G1 whole-body-tracker finetune:
- **Robot:** the gripper + backpack robot.
- **Data:** the full terrain / object / free-space mix, plus 500 pick/place clips.
- **Source:** reconstructed from the code and runs on foxtrot, as of 2026-10-24.
- **Uncommitted local edits:** everything this recipe depends on is listed in §8, with its backup file.

## 1. Lineage

| run (`logs/rsl_rl/g1_flat/...`) | from | iterations | status |
|---|---|---|---|
| `2026-07-11_23-51-10_new_robot` | scratch, `--robot rccu_camera`, 5Seat-2Bit task | 0 → 59999 | done; **`model_20000.pt` is the finetune base** |
| `2026-10-21_02-58-06_new_robot_gripper_pp500_from20k` | base `model_20000` | 20000 → 40624 | **stopped** (trained on the broken parametric-stair data, §4); keeps `model_26500`, `model_40500` |
| `2026-10-22_21-33-11_..._datafix1022` | base `model_20000`, data fixes 1+2 | 20000 → 22480 | **crashed** (NaN policy std after physics blow-ups, §6) |
| `2026-10-23_17-07-48_..._datafix1022_r2` | `..._datafix1022/model_22000.pt`, + NaN guard | 22000 → 32885 | stopped 2026-10-24 (best evaluated: `model_32500`) |
| `2026-10-24_16-12-1x_new_robot_gripper_pp500_scratch_datafix1022` | **scratch**, all fixes + NaN guard, `SAVE_INTERVAL=1000` | 0 → 27000 | stopped 2026-10-27 01:3x at `model_27000` (checkpoints in the `16-12-16` dir) |
| `2026-10-27_..._new_robot_gripper_pp500_scratch_datafix1022_labfix1027` | scratch run's `model_27000.pt`, + label fps fix 3 and wrist-reward gate (§4, §8) | 27000 → 35202 | **stopped 2026-10-27 19:0x (plateau)**; `model_27000`..`model_35000` in the `01-34-30` dir; best `sl_35000` 86.8% success (below `sc_27000` 89.8%) |
| `2026-10-27_19-37-2x_new_robot_gripper_pp500_scratch_labfix1027_8k` | **scratch**, all fixes (1, 2, 3, wrist-reward gate, NaN guard), **8192 envs/GPU**, `SAVE_INTERVAL=1000` | 0 → 60000 | **current** (launched 2026-10-27 19:37; ~11.7 s/iter, ~31/46 GB per GPU) |

Only one of the three rank-stamped run dirs of a distributed launch holds `model_*.pt`.

**Storage (2026-10-27):** `logs/rsl_rl/g1_flat` is a symlink to `/data/bm_generalist_logs/rsl_rl/g1_flat` on the 6.9 TB
`/data` volume (root is 247 GB). New runs write there automatically. `datafix_20261027/move_ckpts.sh` did the move.

## 2. Robot — `--robot rccu_camera_gripper`

- **URDF:** `assets/unitree_description/urdf/g1/g1_rccu_with_camera_pla_grippers.urdf`. The robot is a G1 29-DoF with:
  - an RCCU + ZED X + PLA backpack, with its inertia merged into `torso_link` and a box collider;
  - an inertia-only YAM gripper payload (0.801 kg) on each `*_wrist_yaw_link`;
  - flat box-hand colliders.
  - 37.03 kg in total; the wrist_roll subtree weighs 1.625 kg.
- **Actuators (`robots/g1.py`):**
  - implicit PD with stiffness = armature·(2π·10 Hz)² and damping ratio 2;
  - per-joint effort limits: hip roll / knee 139, hip pitch / yaw 88, ankles / waist roll+pitch 50, arms 25,
    wrist pitch / yaw 5 N·m;
  - self-collision on, solver 8/4, fixed joints merged.

## 3. Task and environment

- **Task:** `Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-2Bit-FreeWrist-G1-v0`.
  - The env class is `envs/manager_based_rl_env_kinematic_objects.py`.
  - The cfg is `G1Flat3ptBindedMultiTerrainObjectUnilateralContact5Seat2BitFreeWristEnvCfg`.
- **Timing:** sim dt 0.005 s (200 Hz), decimation 4 (50 Hz policy), episodes 10 s.
- **Envs:** 4096 envs per GPU × 8 GPUs (`--distributed`). Env i is bound to motion i % 2643, and the start phase
  comes from adaptive bin sampling.
- **Observations (160-d actor):**
  - reference lower-body joint pos/vel;
  - VR 3-point targets (wrists `[0.18, ∓0.025, 0]`, torso `[0, 0, 0.35]` in the body frame; pelvis anchor);
  - anchor pose in the body frame;
  - proprioception and the last action;
  - a 10-d 2-bit contact label (env/object bit per foot, wrist and seat; wrist bits become object bits on object
    motions).
- **Rewards:**
  - tracking: anchor and body pos/ori/vel;
  - regularizers: action rate −0.1, joint limit −10, undesired contacts −0.1;
  - contact-label agreement and duration for feet / wrists / seat;
  - FreeWrist: on free-space motions (category 2) only, pelvis-frame VR hand-target position (w 1.0, std 0.1 m) and
    orientation (w 0.5, std 0.4 rad).
- **Terminations:**
  - anchor |dz| > 0.25 m;
  - anchor projected-gravity |Δg_z| > 0.8;
  - ankle or wrist |dz| > 0.25 m (`ee_body_pos`);
  - wrist contact mismatch over 50 steps (20 after `wrist_mismatch_strict_after_env_steps`);
  - `physics_blowup` (§6);
  - motion end / time-out.
- **Objects (OMOMO):**
  - each env gets a thin-plate proxy URDF (density 400, mass randomized U(0.2, 1.2) kg);
  - it is pose/velocity "magic force" driven along `*_object_fitted.npz` (kp 20 / kd 2 / ori 20 / 3);
  - with `--physics_phase`, the magic force switches off once both wrists touch it (> 0.1 N) while the labels expect
    wrist contact;
  - wrist contact labels drop out for 50% of episodes.
- **Domain randomization:** friction 0.3–1.6, hands 2.5, torso COM ±2.5/5/5 cm, default joint pos ±0.01 rad.
- **Speed flags (FreeWrist cfg):**
  - `disable_height_scanner=True` (unused, ~400 ms/step);
  - `spread_flat_envs=True` (flat-slot envs get their own 4 m grid + ground box at x ≥ 100 m);
  - vectorized adaptive sampler.
- **Added 2026-10-22/23 (§8):**
  - `terrain_ground_plane=True`: a z = 0 plane under the terrain-slot grid;
  - `physics_blowup` termination;
  - `reward_guard_clip=2.0`.

## 4. Data

Launcher flags:
```
--motion_dir all_object_all_terrain_merged_bones_freespace_seat_stairs_pickplace500    # symlink farm, 2643 motions
--object_folder omomo_object/omomo_all_foot
--terrain_folder lafan_terrain/all_lafan_terrain_rf_bones_seat_stair
--object_terrain_folder csv_hand_object_terrain
```
- **Mix:** 2143 base motions plus 500 pick/place clips (350 pick + 150 place).
  - 386 terrain motions (category 0): LaFAN / bones scans, 16 LaFAN stairs, 20 KIT stairs and 50 parametric stairs.
  - 974 OMOMO object motions (category 1).
  - 783 free-space motions (category 2): AMASS, LaFAN, …
  - The pick/place clips are free-space with all-zero contacts, converted FK-exact from `pickplace500_qpos/`.
- **Pairing (non-recursive scans, by file stem):**
  - terrain: motion `<exp>_motion_N` paired with `<terrain_folder>/<exp>_terrain.urdf`, one slot per exp, on a 10 m
    slot grid;
  - object: `<object_folder>/<stem>_thin_plate_proxy.urdf`;
  - otherwise free.
  - Contacts: `<stem>_contact_nodes.npy`, priority object_terrain → object → terrain, else zeros.
- **Changing the mix:** the motion dir is a symlink farm. Build a new farm; never edit the sources.

### Data fixes (all reversible)

- **2026-10-20 (`datafix_20261020/`, `rollback.sh`):**
  - 70 top-level symlinks expose `csv_hand_object_terrain/bones_box/`;
  - the 50 parametric stairs `h<rise>_w<tread>_s<speed>` were renamed to `<name>_motion_0` so they pair with their
    terrain;
  - two bad lift_crate A160 clips were excluded.
- **2026-10-22, fix 1 (`datafix_20261022/`, `apply.sh` / `rollback.sh`):**
  - The parametric stair clips stored `body_pos_w / body_quat_w / body_lin_vel_w / body_ang_vel_w` in MuJoCo DFS
    body order, while `MotionLoader` indexes them in Isaac order. As a result, left_ankle got the left_shoulder_yaw
    target, torso got right_hip_yaw, and so on.
  - Effect: `ee_body_pos` terminated those 100 envs per GPU on every step (~100 of the logged ~105 per step).
  - Fix: the base-dir files are symlinks to `param_body_order_fixed/` (a pure permutation, FK-validated to 0 mm).
- **2026-10-22, fix 2 (code, §8):**
  - The terrain-object scene had no ground plane, and the stair meshes are staircase-only.
  - Unsupported phases: the parametric approach (first 58–200 frames), LaFAN pre/post-stair and KIT post-stair
    phases, 6–25% of those clips' frames. Envs starting there fell through.
  - Confirmed in Isaac with PhysX raycasts (`scripts/rsl_rl/probe_ground_claude.py`); fixed by `terrain_ground_plane`.
- **2026-10-27, fix 3 (`datafix_20261027/`, `apply.sh` / `rollback.sh`; audit in
  `~/Projects/wbc_bench/docs/contact_label_audit_2026-10-27.md`):**
  - The 214 LaFAN `obstacles{1..6}_subject*` terrain clips (18 experiments, ~79% of terrain-motion frames) stored
    `contact_mask` at the source 30 fps on 50 fps motions. `_normalize_contact_mask` index-copies, so the labels ran
    1.67x fast and the last ~40% of each clip read "no foot contact".
  - Fix: the label files are symlinks to `label_fps_fixed/` copies with the mask resampled (nearest, endpoints
    aligned) to the motion frame count (`fix_label_fps.py`). Foot F1 vs kinematic contact on 13 sampled clips
    0.36 -> 0.74; a ~6-frame (0.12 s) label lead remains (same as the parametric stairs; not corrected).
  - Hand contacts: those clips' `hand_terrain` nodes (`left/right_rubber_hand`) never reach the mask (the name
    matcher drops them); by decision the wrist columns stay **zero on all non-object motions**. Wrist labels exist
    only on the 974 OMOMO and 8 `csv_hand_object_terrain` lift_crate clips.
- **2026-10-27, wrist-reward gate (code, §8):** `contact_label_contact_wrist` (bilateral match) penalized any
  hand–terrain contact on terrain motions, whose wrist labels are zero. Both wrist contact-label rewards now have
  `object_motions_only=True` (FreeWrist cfg): they apply only on motions with an object proxy
  (`motion_has_object_per_motion`, logged at startup as `982 motion(s) with an object proxy`). `undesired_contacts`
  already excludes `*_wrist_yaw_link` (the hand box collider); `bad_wrist_contact_mismatch` is object-only.
- **Validation:** before a launch, run the checks in §7.

## 5. Optimization (`config/g1/agents/rsl_rl_ppo_cfg.py`)

PPO (rsl_rl 2.3.3):
- 24 steps per env per iteration; save every 500 iterations;
- empirical observation normalization;
- actor and critic MLPs 2048-2048-1024-1024-512-512, ELU;
- noise std type `log`, init 1.0 (it sits at ~0.39 around iteration 20k+);
- 5 epochs × 4 mini-batches, lr 1e-3 with an adaptive schedule (desired KL 0.01);
- clip 0.2, clipped value loss, value coefficient 1.0, entropy 0.005;
- γ 0.99, λ 0.95, max grad norm 1.0.

Throughput: ~7.5 s per iteration on 8 × L40S (collection ~5.6 s, learning ~1.8 s), so 60k iterations take about 5 days.

## 6. NaN guard (2026-10-23)

- **Problem:** isolated physics blow-ups produced a huge `joint_limit` penalty and value losses up to 1e18.
  - Rate: about 8 per 1000 iterations after the restart, mostly in OMOMO object envs.
  - On 2026-10-23 02:55 one blow-up turned the policy log-std NaN, and the run crashed with
    `normal expects all elements of std >= 0.0`.
  - Root cause: not pinned down. The only env change vs the old run was the ground plane.
- **Guard:**
  - `mdp.physics_blowup` terminates on a non-finite robot state, any joint > 100 rad/s or root speed > 20 m/s.
  - `ManagerBasedRLEnvWithKinematicObjects.step` maps non-finite per-step rewards to 0 and clamps to ±2. Legit
    per-step rewards are < ~0.3.
  - Logged as `Episode_Termination/physics_blowup` and `Guard/clamped_rewards`.
  - In practice the termination fires ~0.04 per step (~0.08 at 8192 envs). The clamp also fires, on ~1.8% of
    iterations at 4096 envs and ~5.8% at 8192 (one env, coinciding with a blow-up's `joint_limit` spike). That is
    expected; value loss stays < 0.35. Only a sustained clamp, or value-loss spikes > 100, are alarming.

## 7. Launch / resume / stop

```bash
cd ~/sky_workdir/bm_generalist
# resume LOAD_RUN/CKPT and train MAX_ITERS *more* iterations (new run dir + new wandb run each time)
MAX_ITERS=58000 RUN_NAME=new_robot_gripper_pp500_from20k_datafix1022_r2 \
  LOAD_RUN=2026-10-22_21-33-11_new_robot_gripper_pp500_from20k_datafix1022 CKPT=model_22000.pt \
  ./run_ft_gripper_pp500.sh          # setsid nohup torchrun 8 GPUs -> ft_logs/<RUN_NAME>.log
# from the base:   LOAD_RUN=2026-07-11_23-51-10_new_robot CKPT=model_20000.pt MAX_ITERS=60000
# from scratch (current run), checkpoint every 1000 iterations:
#   MAX_ITERS=60000 RUN_NAME=new_robot_gripper_pp500_scratch_datafix1022 LOAD_RUN= SAVE_INTERVAL=1000 ./run_ft_gripper_pp500.sh
# current run (2026-10-27), resumed from the scratch run's model_27000 after fix 3 + the wrist-reward gate:
# 8192-env from-scratch run (2026-10-27 19:37):
#   MAX_ITERS=60000 RUN_NAME=new_robot_gripper_pp500_scratch_labfix1027_8k LOAD_RUN= SAVE_INTERVAL=1000 NUM_ENVS=8192 \
#     ./run_ft_gripper_pp500.sh
#   MAX_ITERS=33000 RUN_NAME=new_robot_gripper_pp500_scratch_datafix1022_labfix1027 \
#     LOAD_RUN=2026-10-24_16-12-16_new_robot_gripper_pp500_scratch_datafix1022 CKPT=model_27000.pt SAVE_INTERVAL=1000 \
#     ./run_ft_gripper_pp500.sh
# stop right after a checkpoint is written: datafix_20261027/stop_after_ckpt.sh <iter> <run dir>
# monitor of a resumed log: seed a new state file with the previous run's "base" (its 20300-20800 window is not in it)
```
- **Envs per GPU:** `NUM_ENVS=8192` -> `--num_envs` (added 2026-10-27, backup `run_ft_gripper_pp500.sh.bak_claude_numenvs`;
  empty = task default 4096). The task cfg prints `num_envs=4096` while building the scene; train.py overrides
  `scene.num_envs` / `terrain.spawn.num_envs` afterwards — check `Number of environments: 8192` in the log.
  At 8192: ~31/46 GB per L40S, ~11.7 s/iter (1.55x the 4096 time for 2x the samples).
- **Launcher variables:** `SAVE_INTERVAL` (default 500) is passed as the Hydra override `agent.save_interval`. It was
  added 2026-10-24; the backup is `run_ft_gripper_pp500.sh.bak_claude_saveint`. An empty `LOAD_RUN` means from scratch.
- **What the launcher sets:** the FreeWrist task, `--robot rccu_camera_gripper`, the data flags of §4,
  `--physics_phase --distributed`, wandb project `tracking`. It also clears stale `/tmp/IsaacLab/usd_pid*` dirs.
  Startup takes ~10 min.
- **Stopping:** put the kill logic in a script file. `pkill -f` from an ssh command line kills its own shell. Send
  SIGTERM to the torchrun parent (`ps -eo pid,cmd | grep "torch.distributed.run.*train.py"`); it stops the 8 ranks.
- **Pre-flight checklist:**
  1. No training processes are running and all GPUs are idle.
  2. Disk space: ≥ 25 GB free for a 60k run (checkpoints ~200 MB × 120 + the log ~4 GB). Prune old runs'
     checkpoints, keeping the bases / finals.
  3. Data: all motions in Isaac body order. The heuristic is that body 3 (`waist_yaw_link`) sits within 5 cm of the
     pelvis. After any data change, re-run the FK check
     (`~/Projects/wbc_bench/tools/fix_param_body_order.py` validates; `tools/audit_training_data.py` audits).
  4. Ground / support: `scripts/rsl_rl/probe_ground_claude.py` (400 envs, PhysX raycasts; needs
     `enable_scene_query_support`, which the probe sets) should report "none" = 0%.
  5. Smoke check after launch: the checkpoint loaded (`Loading model checkpoint`), `[ContactLabels] Loaded`,
     `Combined: 191 slots`, and the first iterations look sane (`ee_body_pos` drops to ~5–10 per step within
     ~50 iterations of a resume).

## 8. Uncommitted local edits this recipe depends on

| file | change | backup |
|---|---|---|
| robots/g1.py | `G1_ROBOT_CFGS` registry + `rccu_camera_gripper` | — |
| scripts/rsl_rl/train.py | `--robot`; per-pid USD conversion dirs | `train.py.bak_claude_speed` |
| config/g1/flat_env_cfg.py, `__init__.py` | FreeWrist task cfg + registration | — |
| mdp/rewards.py | `vr_hand_position/orientation_anchor_frame_error_exp`; `object_motions_only` on the contact-label match / duration rewards | `rewards.py.bak_claude_20261020`, `rewards.py.bak_claude_labfix1027` |
| tracking_env_cfg_3pt_binded_multi_terrain_object.py | speed flags; `terrain_ground_plane`; `physics_blowup` term; `reward_guard_clip`; `motion_has_object_per_motion` | `*.bak_claude_speed`, `*.bak_claude_ground`, `*.bak_claude_nanguard`, `*.bak_claude_labfix1027` |
| config/g1/flat_env_cfg.py (FreeWrist cfg) | wrist contact-label rewards `object_motions_only=True` | `flat_env_cfg.py.bak_claude_labfix1027` |
| mdp/commands.py | vectorized `_adaptive_sampling` | `commands.py.bak_claude_speed` |
| mdp/events.py | `spread_flat_layout` | `events.py.bak_claude_speed` |
| mdp/terminations.py | `physics_blowup` | `terminations.py.bak_claude_nanguard` |
| envs/manager_based_rl_env_kinematic_objects.py | reward NaN guard / clamp | `*.bak_claude_nanguard` |
| utils/my_on_policy_runner.py | no checkpoint / ONNX upload to wandb (quota); `WANDB_UPLOAD_CKPT=1` re-enables | `*.bak_wandb_upload` |

## 9. Monitoring

```bash
~/miniconda3/envs/bm/bin/python ft_logs/monitor_ft.py ft_logs/<run>.log ft_logs/monitor_state_<x>.json <run>
```
- **Output:** one JSON line, `ok | alarm | done`. A fresh state file takes its baseline from iterations
  20300–20800 of that log.
- **Healthy ranges (r2, iteration ~22–27k):**

| signal | healthy |
|---|---|
| mean reward | ~7.5 |
| episode length | ~95 |
| `ee_body_pos` per step | ~4–8 |
| `anchor_pos` per step | ~0.7 |
| `physics_blowup` per step | ≤ ~0.05 |
| `Guard/clamped_rewards` | 0 |
| value loss | < 0.1 |
| iteration time | ~7.5 s |

- **Spike check:** any iteration with `Mean value_function loss` > 100. `Episode_Reward/total_reward_mean_{terrain,
  object,free}` shows which env type blew up.
- **Bias warning:** `Metrics/motion/*` are the errors at episode end, averaged over the envs resetting that step.
  They are not comparable across runs with different termination mixes. Judge quality with the benchmark (§10).

## 10. Export and evaluation

- **Export:** `~/miniconda3/envs/bm/bin/python scripts/rsl_rl/export_run_pt_to_onnx.py logs/rsl_rl/g1_flat/<run>
  --pattern model_N.pt --output-dir exported_onnx_claude/<tag>`. It needs no Isaac, and its output is bit-identical
  to the runner's own export.
- **Benchmark (local machine, `~/Projects/wbc_bench`, skill `wbc-eval`):** MuJoCo, the same training robot, 80 clips ×
  5 seeds:
  - held-out pick/place with table;
  - held-out GMA stair rollouts on the apartment stairs;
  - 20 training stair clips on their training terrain;
  - 20 OMOMO object clips.

  ```
  python -m bench.ckpt fetch --iters latest --run <run dir> --prefix fx
  python -m bench.run_eval --ckpts fx_N --proxy wrist   # and --proxy off
  python -m bench.report
  ```
- **Results so far (`--proxy wrist`):**

| | legacy model_44000 | base newrobot_20000 | old run 26500 | fx_22000 | fx_25000 | fx_26500 |
|---|---|---|---|---|---|---|
| held-out stair ascent success / traversed % | 62 / 70 | 34 / 58 | 2 / 12 | 74 / 92 | 70 / 72 | 28 / 88 |
| training-stair parametric ascent % | 0 | 0 | 0 | 17.5 | 37.5 | 42.5 |
| pick/place R-hand err / TCP at grasp cm | 11.9 / 26.3 | 10.9 / 20.2 | 1.6 / 9.4 | 2.2 / 9.5 | 1.9 / 10.0 | 1.6 / 8.9 |
| overall success / fall % | 79 / 16 | 73 / 19 | 73 / 23.5 | 83 / 12 | 84.5 / 13 | 78 / 10 |

  `fx_26500`'s low ascent *success* comes from timing lag past the strict 25 cm rule. It still reaches the top with
  few falls. The old run genuinely collapsed (88% of its ascents fell).

## 11. Known issues / open items

- **Blow-ups:** contained by the guard, but the cause is unknown. Suspects:
  - a global PhysX effect of the large ground plane, which overlaps all 772 terrain-slot robots per GPU and doubles
    their floor contacts;
  - contact-rich OMOMO hand / proxy interaction.
  - A per-env blow-up logger (motion, phase, contacts) would identify them at full scale.
- **Proxy collision filter:** `set_proxy_robot_collision_filter` adds the wrist links (not the others) to the
  filtered group, which is the opposite of its docstring. As coded, the wrists may never touch the proxy, and the
  physics phase may never activate.
- **Foot-contact reward on all-zero-label clips:** `contact_label_contact_foot` has no `disable_motion_categories`.
  On the ~1283 clips without contact labels (free + pick/place) it rewards feet *off* the ground.
- **Wrist-label dropout reaches rewards / termination:** `get_contact_labels_masked()` (50% of episodes, wrist bits
  zeroed) feeds the wrist rewards and `bad_wrist_contact_mismatch`, not only the observation. On OMOMO in dropout
  episodes a correct grasp loses the wrist match reward and counts as a mismatch. Not changed.
- **Knee / elbow contacts on crawl and vault clips:** penalized by `undesired_contacts` (−0.1 per body), and there is
  no label column for them.
- **Parametric stair label quality:** their contact labels lead the motion by ~5 frames (71% agreement with
  kinematic contact; KIT 91%). The clips also start with a ~22 rad/s arm snap in the first ~10 frames.
- **Slot overlap:** 68 terrain meshes reach > 5 m from their slot origin on the 10 m grid. This gives occasional
  phantom neighbour geometry (13 non-stair clips).
