#!/bin/bash
# Wait until model_$1.pt is fully written in run dir $2, then SIGTERM the torchrun parent of the training.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."   # repo root
CK=logs/rsl_rl/g1_flat/$2/model_$1.pt
until [ -f "$CK" ]; do sleep 15; done
s0=-1; s1=$(stat -c %s "$CK")
while [ "$s0" != "$s1" ]; do sleep 20; s0=$s1; s1=$(stat -c %s "$CK"); done
python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).testzip()" "$CK" && echo "checkpoint ok: $CK ($s1 bytes)"
P=$(ps -eo pid,cmd | grep "torch.distributed.run.*train.py" | grep -v grep | awk '{print $1}')
echo "torchrun pid(s): $P"
[ -n "$P" ] && kill -TERM $P
for i in $(seq 1 60); do
  n=$(ps -eo cmd | grep "scripts/rsl_rl/train.py" | grep -v grep | wc -l)
  [ "$n" -eq 0 ] && { echo "all training processes gone after ${i}x5s"; break; }
  sleep 5
done
ps -eo pid,cmd | grep "scripts/rsl_rl/train.py" | grep -v grep || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
