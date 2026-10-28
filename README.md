# SceneBot Training

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

SceneBot Training extends BeyondMimic-style whole-body motion tracking to multi-terrain, object-interaction, and contact-aware training on the Unitree G1. Policies are trained in Isaac Sim / Isaac Lab with distributed RL and logged to Weights & Biases.

## Hardware Requirements

Training is intended for a single node with **8× NVIDIA L40 GPUs** (40 GB memory each).

## Installation

Clone this repository, then run the setup script from the repo root:

```bash
bash setup.sh
```

`setup.sh` creates a conda environment (`bm`), installs Isaac Sim 4.5.0 and Isaac Lab v2.2.0, pulls robot assets, installs this package in editable mode, and configures W&B login.

Before training, set your W&B credentials in `setup.sh` (or export them in your shell):

```bash
export WANDB_API_KEY="<your_api_key>"
export WANDB_ENTITY="<your_organization>"   # use your org name, not your personal username
```

Log in to [Weights & Biases](https://wandb.ai/) and confirm `WANDB_ENTITY` matches your organization slug.

## Data Preparation

Download and unpack the four dataset folders below. Each folder should contain the motion and asset files expected by the training task (`.npz` motions, terrain URDFs, object proxy URDFs, contact labels, etc.).

| Folder | Description | Download |
|--------|-------------|----------|
| `motion_dir` | Retargeted robot motions (`.npz`) | [Google Drive](https://drive.google.com/drive/folders/15KlkF6yV75TLoqoDOQFNbSmKQT3e45yf?usp=drive_link) |
| `terrain_folder` | Per-motion terrain URDFs | [Google Drive](https://drive.google.com/drive/folders/12vDs6aZ6H4LTewBGAfM_X_tLNHDmNsqX?usp=drive_link) |
| `object_folder` | Object proxy URDFs and fitted trajectories | [Google Drive](https://drive.google.com/drive/folders/1VPa3-J0IQlYMmwqprHmHdakIHt5uwg2k?usp=drive_link) |
| `object_terrain_folder` | Merged terrain + object assets and contact labels | [Google Drive](https://drive.google.com/drive/folders/1MUGz6WzrigWD1rOLYRlP3tEr5C0XR2dk?usp=drive_link) |

Example layout after download:

```
/path/to/data/
├── motion_dir/              # *.npz
├── terrain_folder/          # *_terrain.urdf, *_contact_nodes.npy
├── object_folder/           # *_thin_plate_proxy.urdf, *_object_fitted.npz
└── object_terrain_folder/   # merged per-motion exports
```

## Training

Activate the environment and launch distributed training on 8 GPUs:

```bash
conda activate bm
cd /path/to/bm_generalist

python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=8 \
  scripts/rsl_rl/train.py \
  --task=Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-2Bit-G1-v0 \
  --headless \
  --logger wandb \
  --log_project_name tracking \
  --run_name <your_run_name> \
  --motion_dir <your_motion_dir> \
  --object_folder <your_object_folder> \
  --terrain_folder <your_terrain_folder> \
  --object_terrain_folder <your_object_terrain_folder> \
  --physics_phase \
  --distributed \
  --max_iterations 60000
```

Replace the four data paths and `<your_run_name>` with your local paths and desired run name.

### Policy Evaluation

Play a trained checkpoint (use the same data paths as training):

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-2Bit-G1-v0 \
  --num_envs=2 \
  --wandb_path=<wandb-run-path> \
  --motion_dir <your_motion_dir> \
  --object_folder <your_object_folder> \
  --terrain_folder <your_terrain_folder> \
  --object_terrain_folder <your_object_terrain_folder>
```

The W&B run path is shown on the run overview page and follows `{organization}/{project}/{run_id}`.

## Gripper-robot tracker: changes and fixes (2026-10)

The full recipe (lineage, data, rewards, launch, monitoring, evaluation) is in
[`TRAINING_RECIPE.md`](TRAINING_RECIPE.md). Summary of what changed:

**Robot and task**
- `--robot` selects a G1 variant from `G1_ROBOT_CFGS` (`robots/g1.py`). `rccu_camera_gripper` is the
  RCCU + ZED X + PLA backpack robot plus an inertia-only 0.801 kg gripper payload per wrist (37.03 kg in total).
  `scripts/standalone_visualize_g1.py --robot <key>` shows a variant in Isaac Sim.
- New task `Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-2Bit-FreeWrist-G1-v0`: the 5Seat-2Bit
  task plus pelvis-frame VR hand-target position / orientation rewards on free-space motions. Observations are
  unchanged.
- Speed (~31 s -> ~7.5 s per iteration on 8 GPUs):
  - `disable_height_scanner` (the scanner is unused);
  - `spread_flat_envs` (flat-ground envs on their own grid);
  - a vectorized adaptive sampler;
  - per-process USD conversion dirs in `train.py`.

**Bug fixes**
- **Parametric stair data** (`datafix_20261022/`): the 50 `h<rise>_w<tread>_s<speed>` clips stored body arrays in
  MuJoCo body order. `ee_body_pos` therefore terminated them on every step. The fix is a pure permutation,
  FK-validated.
- **No ground under stair terrains:** env cfg flag `terrain_ground_plane=True` adds a z = 0 plane under the
  terrain-slot grid. Approach and post-stair phases previously had no ground under them.
- **NaN guard:** isolated physics blow-ups made the policy std NaN and crashed training. Fix:
  - `mdp.physics_blowup` termination (non-finite state, joint > 100 rad/s, root > 20 m/s);
  - a per-step reward clamp, `reward_guard_clip=2.0` (logged as `Guard/clamped_rewards`).
- **Contact labels at 30 fps** (`datafix_20261027/`): the 214 LaFAN `obstacles*` terrain clips stored
  `contact_mask` at 30 fps on 50 fps motions. The loader index-copies, so their labels ran 1.67x fast and the last
  ~40% of each clip read "no contact". The masks are resampled to the motion length; foot F1 vs kinematic contact
  went 0.36 -> 0.74.
- **Hand contact labels:** wrist labels exist only on object-interaction motions (OMOMO + `csv_hand_object_terrain`).
  The bilateral wrist contact reward used to penalize any hand–terrain contact on terrain motions. Both wrist
  contact-label rewards now take `object_motions_only=True` (FreeWrist cfg; per-motion flag
  `motion_has_object_per_motion`).

**Data fixes are not in git.** `datafix_*/apply.sh` writes the corrected copies into the (untracked) data folders as
symlinks; `rollback.sh` restores the originals. `scripts/scan_labels_claude.py <out.json>` checks label lengths and
hand-label coverage. `scripts/rsl_rl/probe_ground_claude.py` checks ground support with PhysX raycasts.

**Launcher** `run_ft_gripper_pp500.sh`
```bash
# from scratch, checkpoint every 1000 iterations, 4096 envs per GPU (default)
MAX_ITERS=60000 RUN_NAME=my_run LOAD_RUN= SAVE_INTERVAL=1000 ./run_ft_gripper_pp500.sh
# resume LOAD_RUN/CKPT for MAX_ITERS more iterations; NUM_ENVS sets envs per GPU
MAX_ITERS=30000 RUN_NAME=my_run_r2 LOAD_RUN=<run dir> CKPT=model_27000.pt NUM_ENVS=8192 ./run_ft_gripper_pp500.sh
```
- `NUM_ENVS=8192` (~31 / 46 GB per L40S, ~11.5 s per iteration) learned at the same wall-clock rate as 4096. It
  gave no benchmark gain by ~7k iterations.
- Best checkpoint so far: the from-scratch 4096-env run's `model_27000`. On the local MuJoCo tracking benchmark
  (80 clips x 5 seeds) it reaches 89.8% success / 8.5% falls.
- Checkpoints are not uploaded to wandb (quota). Set `WANDB_UPLOAD_CKPT=1` to re-enable.
- `scripts/rsl_rl/export_run_pt_to_onnx.py <run dir> --pattern model_N.pt --output-dir <dir>` exports without Isaac.

## Code Structure

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/`** — MDP terms (commands, rewards, observations, events, terminations)
- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/`** — G1 task and environment configs
- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`** — PPO hyperparameters
- **`scripts/rsl_rl/`** — Training (`train.py`), evaluation (`play.py`), and export utilities

## Acknowledgments

Built on [BeyondMimic](https://beyondmimic.github.io/) / [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) and [Isaac Lab](https://isaac-sim.github.io/IsaacLab).
