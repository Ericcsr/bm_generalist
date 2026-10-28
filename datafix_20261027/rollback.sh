#!/bin/bash
# Undo apply.sh: put the original 30 fps label files back.
set -euo pipefail
cd ~/sky_workdir/bm_generalist
L=lafan_terrain/all_lafan_terrain_rf_bones_seat_stair
D=datafix_20261027
n=0
for f in $D/label_fps_orig/*_contact_nodes.npy; do
  b=$(basename $f)
  [ -L $L/$b ] || { echo "not a symlink, skipping: $b"; continue; }
  rm $L/$b
  cp -p $f $L/$b
  n=$((n+1))
done
echo "rolled back: $n files"
