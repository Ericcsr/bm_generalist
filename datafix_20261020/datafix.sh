#!/bin/bash
# Fix (1) csv_hand_object_terrain/bones_box invisible to the non-recursive scanners, and
# (2) parametric stairs (hHHH_wWWW_sS) not paired with their terrain. Usage: datafix.sh [--apply]
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."   # repo root
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
M=all_object_all_terrain_merged_bones_freespace_seat_stairs
T=lafan_terrain/all_lafan_terrain_rf_bones_seat_stair
OT=csv_hand_object_terrain
EXCL=${M}__excluded_bad_object_fit
LOG=$PWD/datafix_20261020
RB=$LOG/rollback.sh
run() { echo "+ $*"; [ $APPLY = 1 ] && "$@" || true; }
if [ $APPLY = 1 ]; then
  [ -e $RB ] && { echo "already applied (see $RB)"; exit 1; }
  mkdir -p $LOG; echo '#!/bin/bash' > $RB; echo "set -e; cd $PWD" >> $RB
fi
rb() { [ $APPLY = 1 ] && echo "$*" >> $RB || true; }
n1=0; n2=0; n3=0; n4=0
# (1) top-level relative symlinks to every bones_box file
for f in $OT/bones_box/*; do
  b=$(basename "$f"); [ -e "$OT/$b" ] && { echo "SKIP exists $OT/$b"; continue; }
  run ln -s "bones_box/$b" "$OT/$b"; rb "rm -f '$OT/$b'"; n1=$((n1+1))
done
# (1b) exclude the two lift_crate A160 clips whose fitted crate trajectory is static (bad fit)
run mkdir -p "$EXCL"
for b in lift_crate_come_up_50cm_box_R_001__A160_M_motion_0 lift_crate_come_up_50cm_box_R_001__A160_motion_0; do
  run mv "$M/$b.npz" "$EXCL/$b.npz"; rb "mv '$EXCL/$b.npz' '$M/$b.npz'"; n2=$((n2+1))
done
# (2) param stairs: motion -> <name>_motion_0.npz; terrain/contacts get matching symlink names (originals kept)
for f in $M/h0*.npz; do
  b=$(basename "$f" .npz); [[ $b == *_motion_* ]] && continue
  [ -e "$T/$b.urdf" ] && [ -e "$T/${b}_contact_nodes.npy" ] || { echo "MISSING terrain/contacts for $b"; exit 1; }
  run mv "$M/$b.npz" "$M/${b}_motion_0.npz"; rb "mv '$M/${b}_motion_0.npz' '$M/$b.npz'"
  run ln -s "$b.urdf" "$T/${b}_terrain.urdf"; rb "rm -f '$T/${b}_terrain.urdf'"
  run ln -s "${b}_contact_nodes.npy" "$T/${b}_motion_0_contact_nodes.npy"; rb "rm -f '$T/${b}_motion_0_contact_nodes.npy'"
  n3=$((n3+1))
done
[ $APPLY = 1 ] && { echo "rmdir '$EXCL' 2>/dev/null || true" >> $RB; chmod +x $RB; cp "$0" $LOG/datafix.sh; }
echo "== bones_box symlinks: $n1   A160 clips excluded: $n2   param stairs fixed: $n3   (apply=$APPLY)"
