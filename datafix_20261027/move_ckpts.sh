#!/bin/bash
# Move bm_generalist's checkpoint tree (logs/rsl_rl/g1_flat) to the /data volume and leave a symlink in its place.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."   # repo root
SRC=logs/rsl_rl/g1_flat
DST=/data/bm_generalist_logs/rsl_rl/g1_flat
[ -L "$SRC" ] && { echo "already a symlink: $(readlink $SRC)"; exit 0; }
if ps -eo cmd | grep -v grep | grep -q "scripts/rsl_rl/train.py"; then echo "training is running, abort"; exit 1; fi
mkdir -p "$(dirname $DST)"
rsync -a "$SRC/" "$DST/"
n1=$(find "$SRC" -type f | wc -l); n2=$(find "$DST" -type f | wc -l)
b1=$(du -sb --apparent-size "$SRC" | cut -f1); b2=$(du -sb --apparent-size "$DST" | cut -f1)
echo "files $n1 -> $n2, bytes $b1 -> $b2"
[ "$n1" == "$n2" ] && [ "$b1" == "$b2" ] || { echo "copy mismatch, original kept"; exit 1; }
# spot-check: checksum of the newest checkpoints
for f in $(ls -t $SRC/*/model_*.pt | head -3); do
  r=${f#$SRC/}
  [ "$(sha256sum < $f)" == "$(sha256sum < $DST/$r)" ] || { echo "checksum mismatch $r, original kept"; exit 1; }
done
echo "checksums ok"
mv "$SRC" "${SRC}.moved_claude"
ln -s "$DST" "$SRC"
rm -rf "${SRC}.moved_claude"
ls -la logs/rsl_rl/
ls "$SRC" | wc -l
df -h / /data | tail -2
