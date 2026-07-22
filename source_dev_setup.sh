# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# source ${SCRIPT_DIR}/thirdparty/miniconda3/bin/activate bm
conda activate bm
export OMNI_KIT_ACCEPT_EULA=1
wandb login --host=https://far.wandb.io
