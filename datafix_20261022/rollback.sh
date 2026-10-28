#!/bin/bash
# Undo apply.sh: restore the original parametric stair files into the base motion dir.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."   # repo root
B=all_object_all_terrain_merged_bones_freespace_seat_stairs
D=datafix_20261022
for f in $D/param_body_order_orig/h0.*_motion_0.npz; do
  b=$(basename $f)
  [ -L $B/$b ] && rm $B/$b && cp -p $f $B/$b
done
echo "rolled back"
