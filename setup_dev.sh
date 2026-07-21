# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Create overall workspace
WORKSPACE_DIR=$SCRIPT_DIR/thirdparty
CONDA_ROOT=$WORKSPACE_DIR/miniconda3
ENV_ROOT=$CONDA_ROOT/envs/bm
SENTINEL_FILE=.env_setup_finished

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n bm python=3.10.18
  fi

  source $CONDA_ROOT/bin/activate bm

  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

  # Also make LD_PRELOAD permanent for every future `conda activate bm`
  mkdir -p $ENV_ROOT/etc/conda/activate.d
  echo 'export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"' \
    > $ENV_ROOT/etc/conda/activate.d/ld_preload.sh

  # Install PyTorch (CUDA 12.1)
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

  # Below follows https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
  # Install IsaacSim
  pip install --upgrade pip
  pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com

  # Pin older build tools for IsaacLab's installer
  pip install pip==23
  pip install setuptools==65

  if [[ ! -d $WORKSPACE_DIR/IsaacLab ]]; then
    git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.2.0 $WORKSPACE_DIR/IsaacLab
  fi

  sudo apt install -y cmake build-essential
  cd $WORKSPACE_DIR/IsaacLab
  ./isaaclab.sh --install

  # Download the Unitree robot descriptions into the whole_body_tracking assets
  cd $SCRIPT_DIR
  curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
  tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/
  rm unitree_description.tar.gz

  # Install the project
  python -m pip install -e source/whole_body_tracking

  sudo apt-get install -y libglu1-mesa libxi6 libxrender1 libxrandr2 libxxf86vm1 libxcursor1 libsm6 libice6

  touch $SENTINEL_FILE
fi
