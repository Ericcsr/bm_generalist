#!/bin/bash
#
# SBATCH --job-name="arcap_test"
# SBATCH --output=arcap.out
# SBATCH --cpus-per-task=16
# SBATCH --mem-per-cpu=32G
# SBATCH --gres=gpu:1 
# SBATCH --partition=move-interactive

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

srun bash -c '/sailhome/ericcsr/miniconda3/envs/bm/bin/python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 --headless --logger wandb --log_project_name tracking --run_name amass_twist --motion_dir amass_twist'

echo "Done"
