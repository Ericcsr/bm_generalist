# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Use sudo only when not already root (the EKS container runs as root with no sudo binary)
SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
  SUDO="sudo"
fi

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

  $SUDO apt install -y cmake build-essential
  cd $WORKSPACE_DIR/IsaacLab
  ./isaaclab.sh --install

  # Download the Unitree robot descriptions into the whole_body_tracking assets
  cd $SCRIPT_DIR
  curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
  tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/
  rm unitree_description.tar.gz

  # Install the project
  python -m pip install -e source/whole_body_tracking

  $SUDO apt-get install -y libglu1-mesa libxi6 libxrender1 libxrandr2 libxxf86vm1 libxcursor1 libsm6 libice6 unzip

  # Download and unpack datasets from S3 into data/
  # Uploaded via, e.g.: aws s3 cp motion_dir.zip s3://far-research-internal/sihengz/datasets
  # Credentials: on EKS the pod's service-account role (IRSA) is picked up by the
  # default credential chain, so NO named profile is needed. Only set S3_PROFILE
  # when running somewhere that needs an explicit profile (e.g. a laptop where
  # 'far-compute' has read access; the default 'far-asimov' is denied on this bucket).
  S3_DATASET_URI="s3://far-research-internal/sihengz/datasets"
  S3_PROFILE="${S3_PROFILE:-}"
  DATA_DIR=$SCRIPT_DIR/data
  mkdir -p $DATA_DIR

  # Build the optional --profile flag. If S3_PROFILE is set but the profile
  # doesn't exist, fail loudly instead of after a slow full reinstall.
  AWS_PROFILE_ARGS=()
  if [[ -n "$S3_PROFILE" ]]; then
    if ! aws configure list-profiles 2>/dev/null | grep -qx "$S3_PROFILE"; then
      echo "[setup] S3_PROFILE='$S3_PROFILE' not found in AWS config; " \
           "unset it to use the default credential chain (IRSA on EKS)." >&2
      exit 1
    fi
    AWS_PROFILE_ARGS=(--profile "$S3_PROFILE")
  fi

  # Fetch + unpack one dataset zip. Args: <zip-name>
  fetch_dataset() {
    local name="$1"
    aws "${AWS_PROFILE_ARGS[@]}" s3 cp "$S3_DATASET_URI/$name" "$DATA_DIR/$name"
    unzip -o -q "$DATA_DIR/$name" -d "$DATA_DIR"
    rm "$DATA_DIR/$name"
  }

  cd $SCRIPT_DIR
  fetch_dataset motion_dir.zip
  fetch_dataset object_folder.zip
  fetch_dataset object_terrain_folder.zip
  fetch_dataset terrain_folder.zip

  touch $SENTINEL_FILE
fi
