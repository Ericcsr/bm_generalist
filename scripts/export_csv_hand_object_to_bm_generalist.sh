#!/usr/bin/env bash
# Export run_csv_robot_object_terrain_pipeline output to bm_generalist.
#
# Expects the same per-motion layout as run_csv_robot_terrain_pipeline / --bones export:
#   SOURCE/<stem>/<stem>.npz
#   SOURCE/<stem>/<stem>_object.npz
#   SOURCE/<stem>/<stem>_contact_nodes.npy
#   SOURCE/<stem>/*_scaled_*.urdf (or terrain.urdf, etc.)
#
# Optional nested layout (group folders):
#   SOURCE/<group>/<stem>/...
#
# This script:
# 1) Puts converted motions in bm_generalist/{exp_group}/{stem}_motion_0.npz
#    (via omomo_object_to_npz: IsaacLab conversion + thin-plate proxy + resampled contact).
# 2) Puts terrain URDF/meshes, proxy URDF, *_object_fitted.npz, *_contact_nodes.npy, proxy JSON
#    in bm_generalist/csv_hand_object_terrain/{exp_group}/ (one directory for terrain+object+labels).
#
# Train (same path for terrain and object folders):
#   python scripts/rsl_rl/train.py --task Tracking-3pt-Binded-Multi-Terrain-Object-G1-v0 \
#     --motion_dir <exp_group> --terrain_folder csv_hand_object_terrain/<exp_group> \
#     --object_folder csv_hand_object_terrain/<exp_group>
#
# Usage:
#   ./scripts/export_csv_hand_object_to_bm_generalist.sh --exp-group my_exp --source-dir /path/to/pipeline/out/group
#   ./scripts/export_csv_hand_object_to_bm_generalist.sh --exp-group bones_box \\
#       --source-dir ../contact_terrain_reconstruction/output_omni/csv_bones_csv_hand_object_terrain/bones_box \\
#       --headless
#
# Optional: --motion-keys k1 k2  or  --motion-keys-file keys.txt
#           --skip-conversion   (only copy terrain bundles; no Isaac / omomo_object_to_npz)
#           --output-fps N (default 50)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BM_GENERALIST_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$BM_GENERALIST_ROOT/.." && pwd)"

CONTACT_TERRAIN_ROOT="${CONTACT_TERRAIN_ROOT:-$WORKSPACE_ROOT/contact_terrain_reconstruction}"
OUTPUT_OMNI="${OUTPUT_OMNI:-$CONTACT_TERRAIN_ROOT/output_omni}"

# Second output tree: terrain + object + contact (single folder for both CLI paths).
ASSETS_SUBDIR="${ASSETS_SUBDIR:-csv_hand_object_terrain}"

EXP_GROUP=""
SOURCE_DIR=""
SKIP_CONVERSION=false
OUTPUT_FPS=50
VISUALIZE=false
MOTION_KEYS=()
MOTION_KEYS_FILE=""

copy_terrain_bundle_from_motion_dir() {
    local motion_src="$1"
    local dest_stem="$2"
    local dest_dir="$3"
    local src_urdf="$4"
    _CT_MOTION_SRC=$motion_src _CT_DEST_STEM=$dest_stem _CT_DEST_DIR=$dest_dir _CT_SRC_URDF=$src_urdf python3 <<'PY'
import pathlib
import os
import re
import shutil
import sys

motion_src = pathlib.Path(os.environ["_CT_MOTION_SRC"]).resolve()
dest_stem = os.environ["_CT_DEST_STEM"]
dest_dir = pathlib.Path(os.environ["_CT_DEST_DIR"]).resolve()
src_urdf = pathlib.Path(os.environ["_CT_SRC_URDF"]).resolve()
dest_urdf = dest_dir / f"{dest_stem}_terrain.urdf"

dest_dir.mkdir(parents=True, exist_ok=True)
text = src_urdf.read_text(encoding="utf-8")
dest_urdf.write_text(text, encoding="utf-8")
for m in sorted(set(re.findall(r'filename="([^"]+)"', text))):
    src_m = motion_src / m
    if not src_m.is_file():
        print(f"  [terrain][warn] mesh missing, skip: {src_m}", file=sys.stderr)
        continue
    shutil.copy2(src_m, dest_dir / m)
    print(f"  [terrain] copied mesh {m}")
PY
}

find_motion_folder_urdf() {
    local mdir="$1"
    local stem="$2"
    local u=""
    u=$(ls -1 "$mdir"/*_scaled_*.urdf 2>/dev/null | head -1 || true)
    if [[ -n "$u" && -f "$u" ]]; then
        echo "$u"
        return
    fi
    for candidate in "$mdir/$stem.urdf" "$mdir/terrain.urdf" "$mdir/terrain_convex.urdf"; do
        if [[ -f "$candidate" ]]; then
            echo "$candidate"
            return
        fi
    done
    echo ""
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp-group)
            EXP_GROUP="$2"
            shift 2
            ;;
        --source-dir)
            SOURCE_DIR="$2"
            shift 2
            ;;
        --skip-conversion)
            SKIP_CONVERSION=true
            shift
            ;;
        --output-fps)
            OUTPUT_FPS="$2"
            shift 2
            ;;
        --visualize)
            VISUALIZE=true
            shift
            ;;
        --motion-keys)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                MOTION_KEYS+=("$1")
                shift
            done
            ;;
        --motion-keys-file)
            MOTION_KEYS_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --exp-group NAME --source-dir PATH [options]"
            echo ""
            echo "Export CSV hand-object terrain pipeline output to bm_generalist."
            echo ""
            echo "Required:"
            echo "  --exp-group NAME     Experiment group (bm_generalist/<name>/ for motions)"
            echo "  --source-dir PATH    Directory containing motion subfolders (or group/stem nested)"
            echo ""
            echo "Optional:"
            echo "  --motion-keys K ...   Process only these stem names (folder basenames)"
            echo "  --motion-keys-file F  One stem per line"
            echo "  --skip-conversion     Only copy terrain URDF/meshes (no omomo_object_to_npz)"
            echo "  --output-fps N        Target FPS (default: 50)"
            echo "  --visualize           Preview in Isaac (default: --headless conversion)"
            echo ""
            echo "Output:"
            echo "  Motions:  $BM_GENERALIST_ROOT/{exp_group}/{stem}_motion_0.npz"
            echo "  Assets:   $BM_GENERALIST_ROOT/$ASSETS_SUBDIR/{exp_group}/"
            echo "            {stem}_terrain.urdf + meshes, {stem}_motion_0_thin_plate_proxy.*,"
            echo "            {stem}_motion_0_object_fitted.npz, {stem}_motion_0_contact_nodes.npy"
            echo ""
            echo "Paths (env): CONTACT_TERRAIN_ROOT, OUTPUT_OMNI, ASSETS_SUBDIR"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$EXP_GROUP" || -z "$SOURCE_DIR" ]]; then
    echo "Error: --exp-group and --source-dir are required"
    exit 1
fi

RF_SOURCE="$(cd "$SOURCE_DIR" && pwd)"
if [[ ! -d "$RF_SOURCE" ]]; then
    echo "Error: Source directory not found: $SOURCE_DIR"
    exit 1
fi

if [[ -n "$MOTION_KEYS_FILE" ]]; then
    if [[ ! -f "$MOTION_KEYS_FILE" ]]; then
        echo "Error: Motion keys file not found: $MOTION_KEYS_FILE"
        exit 1
    fi
    while IFS= read -r line; do
        line="$(echo "$line" | xargs)"
        [[ -z "$line" || "$line" == \#* ]] && continue
        MOTION_KEYS+=("$line")
    done < "$MOTION_KEYS_FILE"
fi

MOTION_DIRS=()
for d in "$RF_SOURCE"/*/; do
    [[ -d "$d" ]] || continue
    name="$(basename "${d%/}")"
    [[ "$name" == "." || "$name" == ".." ]] && continue
    if [[ -f "$d/${name}.npz" && -f "$d/${name}_object.npz" ]]; then
        MOTION_DIRS+=("$d")
    fi
done

if [[ ${#MOTION_DIRS[@]} -eq 0 ]]; then
    for g in "$RF_SOURCE"/*/; do
        [[ -d "$g" ]] || continue
        for d in "$g"/*/; do
            [[ -d "$d" ]] || continue
            name="$(basename "${d%/}")"
            if [[ -f "$d/${name}.npz" && -f "$d/${name}_object.npz" ]]; then
                MOTION_DIRS+=("$d")
            fi
        done
    done
fi

if [[ ${#MOTION_KEYS[@]} -gt 0 ]]; then
    FILTERED=()
    declare -A ALLOW
    for k in "${MOTION_KEYS[@]}"; do ALLOW["$k"]=1; done
    for d in "${MOTION_DIRS[@]}"; do
        name="$(basename "${d%/}")"
        if [[ -n "${ALLOW[$name]+x}" ]]; then
            FILTERED+=("$d")
        fi
    done
    MOTION_DIRS=("${FILTERED[@]}")
fi

if [[ ${#MOTION_DIRS[@]} -eq 0 ]]; then
    echo "Error: No motion folders with {stem}.npz and {stem}_object.npz under $RF_SOURCE"
    exit 1
fi

MOTION_DIR="$BM_GENERALIST_ROOT/$EXP_GROUP"
ASSETS_DIR="$BM_GENERALIST_ROOT/$ASSETS_SUBDIR/$EXP_GROUP"

echo "=== Export CSV hand-object terrain -> bm_generalist ==="
echo "  exp_group:   $EXP_GROUP"
echo "  Source:      $RF_SOURCE"
echo "  Motions ->   $MOTION_DIR"
echo "  Assets ->    $ASSETS_DIR (terrain + object + contact)"
echo "  Count:       ${#MOTION_DIRS[@]}"
echo ""

mkdir -p "$MOTION_DIR"
mkdir -p "$ASSETS_DIR"

SUCCESS=0
FAIL=0

for motion_dir in "${MOTION_DIRS[@]}"; do
    stem="$(basename "${motion_dir%/}")"
    export_base="${stem}_motion_0"

    echo ""
    echo "--- $stem -> $export_base ---"

    input_npz="$motion_dir/${stem}.npz"
    object_npz="$motion_dir/${stem}_object.npz"

    if [[ ! -f "$input_npz" ]]; then
        echo "  [skip] Missing $input_npz"
        FAIL=$((FAIL + 1))
        continue
    fi
    if [[ ! -f "$object_npz" ]]; then
        echo "  [skip] Missing $object_npz"
        FAIL=$((FAIL + 1))
        continue
    fi

    urdf_src=$(find_motion_folder_urdf "$motion_dir" "$stem")
    if [[ -z "$urdf_src" || ! -f "$urdf_src" ]]; then
        echo "  [terrain][warn] No URDF in $motion_dir; skipping terrain copy"
    else
        copy_terrain_bundle_from_motion_dir "$motion_dir" "$stem" "$ASSETS_DIR" "$urdf_src"
        echo "  [terrain] ${stem}_terrain.urdf"
    fi

    if [[ "$SKIP_CONVERSION" == true ]]; then
        echo "  [skip-conversion] Not running omomo_object_to_npz"
        SUCCESS=$((SUCCESS + 1))
        continue
    fi

    PROXY_ARGS=()
    if [[ "$VISUALIZE" != true ]]; then
        PROXY_ARGS+=(--headless)
    fi

    echo "  [convert] omomo_object_to_npz ($export_base) @ ${OUTPUT_FPS} fps ..."
    if (
        cd "$BM_GENERALIST_ROOT"
        python scripts/omomo_object_to_npz.py \
            --input_file "$input_npz" \
            --object_motion "$object_npz" \
            --output_name "$export_base" \
            --output_fps "$OUTPUT_FPS" \
            --proxy_export_dir "$ASSETS_DIR" \
            --proxy_export_prefix "$export_base" \
            "${PROXY_ARGS[@]}"
    ); then
        :
    else
        echo "  [convert] Failed for $stem"
        FAIL=$((FAIL + 1))
        continue
    fi

    src_converted="$BM_GENERALIST_ROOT/$export_base/${export_base}.npz"
    if [[ -f "$src_converted" ]]; then
        mv "$src_converted" "$MOTION_DIR/${export_base}.npz"
        rmdir "$BM_GENERALIST_ROOT/$export_base" 2>/dev/null || true
        echo "  [convert] Saved $MOTION_DIR/${export_base}.npz"
    else
        echo "  [convert][warn] Expected npz not found: $src_converted"
        FAIL=$((FAIL + 1))
        continue
    fi

    SUCCESS=$((SUCCESS + 1))
done

echo ""
echo "=== Done ==="
echo "  Success: $SUCCESS / $((${SUCCESS} + ${FAIL}))"
[[ $FAIL -gt 0 ]] && echo "  Failed:  $FAIL"
echo ""
echo "  Motions: $MOTION_DIR"
echo "  Assets:  $ASSETS_DIR"
echo ""
echo "Example train (mixed dataset: point --object_terrain_folder at this export; split dirs optional):"
echo "  --task Tracking-3pt-Binded-Multi-Terrain-Object-Merged-G1-v0 \\"
echo "    --motion_dir $EXP_GROUP --terrain_folder lafan_terrain/OTHER --object_folder omomo_object/OTHER \\"
echo "    --object_terrain_folder $ASSETS_SUBDIR/$EXP_GROUP"
