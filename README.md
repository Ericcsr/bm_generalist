# SceneBot Training

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/24.04/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository trains a single **whole-body motion tracker** for the Unitree G1 humanoid. One policy tracks a large,
mixed set of reference motions:
- **terrain motions** (stairs, boxes, obstacles, seats) on their own scanned or parametric terrain meshes;
- **object-interaction motions** (OMOMO carry / lift / push) with a physical object proxy;
- **free-space motions** (locomotion, dance, pick-and-place arm motions).

The tracker is driven by a VR-style 3-point command (both wrists + torso, relative to the pelvis) plus the reference
lower-body motion and a per-limb **contact label** (foot / wrist / seat, environment vs. object).

**Robot.** The main robot is the G1 with an RCCU + ZED X camera + PLA backpack and a 0.801 kg gripper payload on
each wrist (`--robot rccu_camera_gripper`, 37 kg). Plain G1 and backpack-only variants are also registered.

**Training.** PPO (rsl_rl), 4096 environments per GPU on 8 GPUs, in Isaac Sim 4.5 / Isaac Lab 2.2.

The detailed, up-to-date training recipe is in [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md). It covers:
- the run lineage and data;
- rewards and terminations;
- launch, monitoring, export and evaluation.

## Hardware

- One node with **8× NVIDIA L40S** (46 GB). The default 4096 envs/GPU take ~7.5 s per PPO iteration (60k iterations
  ≈ 5 days). 8192 envs/GPU fit (~31 GB per GPU) but learned no faster per wall-clock hour.
- ~25 GB of free disk per 60k-iteration run (checkpoints ~200 MB each, saved every 1000 iterations, plus the log).

## Setup

Tested environment: Python 3.10.18, Isaac Sim 4.5.0, Isaac Lab v2.2.0, rsl-rl-lib 2.3.3, PyTorch 2.7 (CUDA 12.8),
Ubuntu 24.04.

1. **Clone** this repository next to (not inside) the Isaac Lab checkout:

   ```bash
   git clone https://github.com/Ericcsr/bm_generalist.git
   cd bm_generalist
   ```

2. **Install** with `setup.sh`. It does the following:
   - creates the conda env `bm` (Python 3.10);
   - installs PyTorch and `isaacsim[all,extscache]==4.5.0`;
   - clones Isaac Lab next to this repo at tag `v2.2.0` and runs `./isaaclab.sh -i`;
   - downloads the Unitree robot descriptions into `source/whole_body_tracking/whole_body_tracking/assets/`;
   - installs this package in editable mode (`pip install -e source/whole_body_tracking`).

   ```bash
   bash setup.sh
   conda activate bm
   ```

   The gripper / backpack G1 URDFs (`assets/unitree_description/urdf/g1/g1_rccu_with_camera_pla*.urdf`) are part of the
   robot description assets. `scripts/standalone_visualize_g1.py --robot rccu_camera_gripper` shows a variant in
   Isaac Sim.

3. **Weights & Biases** (metrics logging only; checkpoints are not uploaded unless `WANDB_UPLOAD_CKPT=1`):

   ```bash
   wandb login
   export WANDB_ENTITY=<your_wandb_entity>
   ```

   Do not commit API keys: set them in your shell, not in `setup.sh`.

## Data

Training reads four folders (relative to the repo root by default). Download and unpack:

| Folder (launcher default) | Contents | Download |
|---|---|---|
| motion dir, e.g. `all_object_all_terrain_merged_bones_freespace_seat_stairs_pickplace500/` | retargeted robot motions `*.npz` (50 fps, body states in Isaac Lab body order) | [Google Drive](https://drive.google.com/drive/folders/15KlkF6yV75TLoqoDOQFNbSmKQT3e45yf?usp=drive_link) |
| `lafan_terrain/all_lafan_terrain_rf_bones_seat_stair/` | per-experiment terrain `<exp>_terrain.urdf` + mesh, foot/seat contact labels `*_contact_nodes.npy` | [Google Drive](https://drive.google.com/drive/folders/12vDs6aZ6H4LTewBGAfM_X_tLNHDmNsqX?usp=drive_link) |
| `omomo_object/omomo_all_foot/` | object proxies `*_thin_plate_proxy.urdf`, fitted object trajectories `*_object_fitted.npz`, contact labels | [Google Drive](https://drive.google.com/drive/folders/1VPa3-J0IQlYMmwqprHmHdakIHt5uwg2k?usp=drive_link) |
| `csv_hand_object_terrain/` | merged terrain + object exports (e.g. crate lifting) with hand + foot contact labels | [Google Drive](https://drive.google.com/drive/folders/1MUGz6WzrigWD1rOLYRlP3tEr5C0XR2dk?usp=drive_link) |

How the folders are paired (by file stem):
- **Terrain motions:** `<exp>_motion_N.npz` pairs with `<exp>_terrain.urdf`.
- **Object motions:** pair with `<stem>_thin_plate_proxy.urdf`.
- **Everything else** is free-space.
- **Contact labels:** `<stem>_contact_nodes.npy` is taken from the first folder that has it, in the order
  object-terrain → object → terrain. Motions without labels get all-zero labels.
- **Wrist labels** are used only on object-interaction motions.

**Data fixes.** The training data needs three fixes. They are already applied on the training node. If your copy
predates them, apply them from the repo root; all three are reversible. The scripts assume the folder names above
and that the motion dir is a symlink farm over `all_object_all_terrain_merged_bones_freespace_seat_stairs/`.
1. **Pairing** (`datafix_20261020/`):
   - renames the parametric-stair clips to `<name>_motion_0` so they pair with their terrain;
   - exposes `csv_hand_object_terrain/bones_box/` to the non-recursive scanners;
   - excludes two clips with a bad object fit.

   Run `bash datafix_20261020/datafix.sh` (dry run), then `--apply`; it writes its own `rollback.sh`.
2. **Parametric-stair body order** (`datafix_20261022/`): the 50 `h<rise>_w<tread>_s<speed>` clips stored body
   arrays in MuJoCo body order. `fix_param_body_order.py` writes permuted, FK-validated copies; it needs the training
   robot's MuJoCo model from the separate `wbc_bench` tools, so the simplest route is to copy the fixed
   `param_body_order_fixed/` folder from the training node. Then run `bash datafix_20261022/apply.sh`
   (undo: `rollback.sh`).
3. **Contact-label frame rate** (`datafix_20261027/`): the 214 LaFAN `obstacles*` clips stored their contact masks at
   30 fps. Run `python datafix_20261027/fix_label_fps.py && bash datafix_20261027/apply.sh` (undo: `rollback.sh`).

Then check the labels with `python scripts/scan_labels_claude.py /tmp/labels.json`. Every mask should match its
motion length, and wrist labels should appear only on object motions.

## Training

The launcher wraps `torchrun` on 8 GPUs with the current task, robot and data flags:

```bash
# from scratch, checkpoint every 1000 iterations
MAX_ITERS=60000 RUN_NAME=my_run LOAD_RUN= SAVE_INTERVAL=1000 ./run_ft_gripper_pp500.sh
# resume <run dir>/model_N.pt for MAX_ITERS more iterations
MAX_ITERS=30000 RUN_NAME=my_run_r2 LOAD_RUN=<run dir> CKPT=model_N.pt ./run_ft_gripper_pp500.sh
```

| Variable | Meaning | Default |
|---|---|---|
| `MAX_ITERS` | iterations to train (more iterations when resuming) | 60000 |
| `RUN_NAME` | log / wandb run name; the log goes to `ft_logs/<RUN_NAME>.log` | — |
| `LOAD_RUN`, `CKPT` | run dir under `logs/rsl_rl/g1_flat/` and checkpoint to resume; empty `LOAD_RUN` = from scratch | — |
| `SAVE_INTERVAL` | checkpoint interval | 500 |
| `NUM_ENVS` | envs per GPU | 4096 |

- **What the launcher passes:**
  - task `Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-2Bit-FreeWrist-G1-v0`;
  - `--robot rccu_camera_gripper`;
  - the four data folders;
  - `--physics_phase --distributed`.
- **Startup:** ~10 minutes.
- **Stopping:** send SIGTERM to the `torch.distributed.run` parent (`datafix_20261027/stop_after_ckpt.sh <iter> <run dir>`
  waits for a checkpoint first).
- **Run dirs:** a distributed launch creates one run dir per rank timestamp; only one of them holds the
  `model_*.pt`.

Equivalent direct call:
```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 scripts/rsl_rl/train.py \
  --task=Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-2Bit-FreeWrist-G1-v0 --headless \
  --robot rccu_camera_gripper --logger wandb --log_project_name tracking --run_name my_run \
  --motion_dir <motion dir> --terrain_folder <terrain folder> --object_folder <object folder> \
  --object_terrain_folder <object-terrain folder> --physics_phase --distributed --max_iterations 60000 \
  agent.save_interval=1000
```

**Health signals.** `TRAINING_RECIPE.md` §9 lists the ranges. In a healthy run:
- mean reward ~7.5–7.9;
- episode length ~95;
- `Episode_Termination/physics_blowup` ≲ 0.05 per step;
- value loss < 0.35.

`Guard/clamped_rewards` may be non-zero on a few % of iterations (isolated physics blow-ups); value-loss spikes above
100 are not normal.

## Export and playback

```bash
# ONNX export of any checkpoint (no Isaac Sim needed)
python scripts/rsl_rl/export_run_pt_to_onnx.py logs/rsl_rl/g1_flat/<run dir> --pattern model_N.pt --output-dir <out dir>

# play a local checkpoint in Isaac Sim (same data flags as training)
python scripts/rsl_rl/play.py --task=Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-2Bit-FreeWrist-G1-v0 \
  --num_envs=2 --load_run <run dir> --checkpoint model_N.pt \
  --motion_dir <motion dir> --terrain_folder <terrain folder> --object_folder <object folder> \
  --object_terrain_folder <object-terrain folder>
```
`play.py` has no `--robot` option yet and spawns the task's default G1, not the gripper robot.

## Code structure

- `source/whole_body_tracking/whole_body_tracking/robots/g1.py`: G1 actuators and the `G1_ROBOT_CFGS` registry used by
  `--robot`.
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/`:
  - `tracking_env_cfg_3pt_binded_multi_terrain_object.py`: the multi-terrain / object scene, motion–terrain–object
    pairing, rewards, terminations, and flags (`terrain_ground_plane`, `reward_guard_clip`, speed flags).
  - `config/g1/flat_env_cfg.py`, `config/g1/__init__.py`: G1 task variants and their gym ids (5Seat-2Bit, FreeWrist, …).
  - `config/g1/agents/rsl_rl_ppo_cfg.py`: PPO hyperparameters.
  - `mdp/`: commands (motion loading, adaptive sampling, contact labels), observations, rewards, events,
    terminations; `mdp/contact_labels.py` loads the `*_contact_nodes.npy` labels.
- `source/whole_body_tracking/whole_body_tracking/envs/manager_based_rl_env_kinematic_objects.py`: env with object proxy
  driving and the reward guard.
- `scripts/rsl_rl/`: `train.py`, `play.py`, `export_run_pt_to_onnx.py`, and checks:
  - `probe_ground_claude.py`: PhysX ground-support check;
  - `probe_blowup_claude.py`: blow-up check.
- `scripts/`: motion conversion and export tools, `scan_labels_claude.py`, `standalone_visualize_g1.py`.
- `datafix_*/`: data-fix scripts (apply / rollback).

## Changes and fixes (2026-10)

**Robot and task**
- `--robot` registry with the gripper + backpack G1.
- FreeWrist task: pelvis-frame VR hand-target rewards on free-space motions.
- Speed changes (~31 s → ~7.5 s per iteration):
  - no height scanner;
  - flat envs spread on their own grid;
  - a vectorized adaptive sampler;
  - per-process USD conversion.

**Bug fixes**
- **Parametric-stair body order:** these clips stored body arrays in MuJoCo order, so they terminated every step
  (`datafix_20261022`).
- **Ground under stair terrains:** `terrain_ground_plane=True`.
- **NaN guard:** `physics_blowup` termination plus a per-step reward clamp, after a physics blow-up crashed training
  with NaN policy std.
- **LaFAN obstacles contact masks:** they were stored at 30 fps on 50 fps motions and are now resampled
  (`datafix_20261027`).
- **Wrist contact-label rewards:** now limited to object-interaction motions (`object_motions_only`). Before, they
  penalized hand–terrain contact on terrain motions, which have no hand labels.

## Acknowledgments

This code base is built on the motion tracking code of **BeyondMimic**
([website](https://beyondmimic.github.io/), [paper](https://arxiv.org/abs/2508.08241),
[code](https://github.com/HybridRobotics/whole_body_tracking)) and on [Isaac Lab](https://isaac-sim.github.io/IsaacLab).
If you use it, please cite BeyondMimic:

```bibtex
@article{liao2025beyondmimic,
  title   = {BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion},
  author  = {Liao, Qiayuan and Truong, Takara E. and Huang, Xiaoyu and Tevet, Guy and Sreenath, Koushil and Liu, C. Karen},
  journal = {arXiv preprint arXiv:2508.08241},
  year    = {2025}
}
```
