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

## Code Structure

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/`** — MDP terms (commands, rewards, observations, events, terminations)
- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/`** — G1 task and environment configs
- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`** — PPO hyperparameters
- **`scripts/rsl_rl/`** — Training (`train.py`), evaluation (`play.py`), and export utilities

## Acknowledgments

Built on [BeyondMimic](https://beyondmimic.github.io/) / [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) and [Isaac Lab](https://isaac-sim.github.io/IsaacLab).
