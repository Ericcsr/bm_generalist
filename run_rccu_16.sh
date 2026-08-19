set -ex
cd ~/lab42/src/FAR-skypilot-wrapper
far-set-env SKY

export SKIP_UNTRACKED=1

cluster_name_prefix=$USER

./bin/sky-ws42-launch.sh EKS:scenebot \
    --cluster benchmark-${cluster_name_prefix}-rccu-16-high-priority \
    --num-nodes 2 --gpus L40S:8 --cpus 192+ --memory 256+ \
    --infra k8s/sky-us-east-2 \
    --yes --idle-minutes-to-autostop 3600 --down \
    --detach-run \
    --async \
    --retry-until-up \
    --env ENTRYPOINT=" \
            ulimit -n 65535; WANDB_USERNAME=far-wandb NCCL_P2P_DISABLE=1 HYDRA_FULL_ERROR=1 python -m torch.distributed.run \
                --nnodes=2 \
                --nproc_per_node=8 \
                --node_rank=\$SKYPILOT_NODE_RANK \
                --master_addr=\$SKYPILOT_MASTER_ADDR \
                --master_port=29500 \
                --redirects=3 --tee=3 \
                --log_dir=/root/far_scenebot_training/torchrun_logs \
                scripts/rsl_rl/train.py \
                --task=Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-2Bit-G1-v0 \
                --headless \
                --logger wandb \
                --log_project_name sihengz-scenebot \
                --run_name sihengz_rccu_16 \
                --motion_dir data/motion_dir \
                --object_folder data/object_folder \
                --terrain_folder data/terrain_folder \
                --object_terrain_folder data/object_terrain_folder \
                --physics_phase \
                --distributed \
                --num_envs 2048 \
                --max_iterations 60000
        "
