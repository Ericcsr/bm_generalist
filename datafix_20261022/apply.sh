#!/bin/bash
# Fix 1 (2026-10-22): parametric stair clips stored body arrays in MuJoCo body order; replace the 50 base-dir files with
# symlinks to body-order-fixed copies. Originals are copied to param_body_order_orig/ first (sha-verified).
set -euo pipefail
cd ~/sky_workdir/bm_generalist
B=all_object_all_terrain_merged_bones_freespace_seat_stairs
D=datafix_20261022
mkdir -p $D/param_body_order_orig
n=0
for f in $D/param_body_order_fixed/h0.*_motion_0.npz; do
  b=$(basename $f)
  if [ -L $B/$b ]; then echo "already applied: $b"; continue; fi
  cp -p $B/$b $D/param_body_order_orig/$b
  [ "$(sha256sum < $B/$b)" == "$(sha256sum < $D/param_body_order_orig/$b)" ] || { echo "backup mismatch $b"; exit 1; }
  rm $B/$b
  ln -s ../$D/param_body_order_fixed/$b $B/$b
  n=$((n+1))
done
echo "applied: $n files"
