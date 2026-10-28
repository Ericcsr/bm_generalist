#!/bin/bash
# Fix 2026-10-27: replace the 30 fps contact labels of the LaFAN obstacles clips with symlinks to 50 fps copies
# (label_fps_fixed/, built by fix_label_fps.py). Originals are copied to label_fps_orig/ first (sha-verified).
set -euo pipefail
cd ~/sky_workdir/bm_generalist
L=lafan_terrain/all_lafan_terrain_rf_bones_seat_stair
D=datafix_20261027
mkdir -p $D/label_fps_orig
n=0
for f in $D/label_fps_fixed/*_contact_nodes.npy; do
  b=$(basename $f)
  if [ -L $L/$b ]; then echo "already applied: $b"; continue; fi
  cp -p $L/$b $D/label_fps_orig/$b
  [ "$(sha256sum < $L/$b)" == "$(sha256sum < $D/label_fps_orig/$b)" ] || { echo "backup mismatch $b"; exit 1; }
  rm $L/$b
  ln -s ../../$D/label_fps_fixed/$b $L/$b
  n=$((n+1))
done
echo "applied: $n files"
