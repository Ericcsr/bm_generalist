conda create -n bm python=3.10.18
conda activate bm
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
cd ..
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.1.0
sudo apt install cmake build-essential
./isaaclab.sh -i
cd ../bm_generalist
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/
rm unitree_description.tar.gz
pip install -e source/whole_body_tracking
wandb login --cloud --relogin
export WANDB_API_KEY="9b35dd09702e956a7752a96872e08787b6f7401e"
export WANDB_ENTITY="ericcsr-bm"
