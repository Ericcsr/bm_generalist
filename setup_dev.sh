#!/bin/bash
set -ex

conda create -y -n bm python=3.10.18
conda activate bm
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
cd ..
pip install pip==23
pip install setuptools==65
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.2.0
apt-get install -y cmake build-essential
./isaaclab.sh --install
cd ../far_scenebot_training
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/
rm unitree_description.tar.gz
pip install -e source/whole_body_tracking
mkdir -p data
cd data
aws s3 cp s3://far-research-internal/sihengz/datasets/motion_dir.zip .
aws s3 cp s3://far-research-internal/sihengz/datasets/object_folder.zip .
aws s3 cp s3://far-research-internal/sihengz/datasets/terrain_folder.zip .
aws s3 cp s3://far-research-internal/sihengz/datasets/object_terrain_folder.zip .
unzip motion_dir.zip
unzip object_folder.zip
unzip terrain_folder.zip
unzip object_terrain_folder.zip
cd ..
