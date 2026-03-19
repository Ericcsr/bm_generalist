#!/usr/bin/env bash
# Post-retargeting export script: After robot_retarget.py finishes for an exp_group,
# this script:
# 1. Creates bm_generalist/{exp_group}/ for motions and bm_generalist/lafan_terrain/{exp_group}/ for terrain
# 2. Runs omni_to_npz conversion for each *_original.npz in demo_results
# 3. Copies scaled URDF and mesh to terrain folder, renaming URDF to match motion clip names
#
# Usage:
#   ./scripts/post_retarget_export.sh --exp-group lafan [options]
#
# Required: --exp-group (e.g. lafan)
# Optional: --skip-conversion (skip omni_to_npz, only copy terrain)
#           --headless (run omni_to_npz in headless mode)
#           --output-fps 50

set -e

# Resolve paths relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BM_GENERALIST_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$BM_GENERALIST_ROOT/.." && pwd)"

# Default paths (can be overridden via env)
HOLOSOMA_ROOT="${HOLOSOMA_ROOT:-$WORKSPACE_ROOT/holosoma}"
CONTACT_TERRAIN_ROOT="${CONTACT_TERRAIN_ROOT:-$WORKSPACE_ROOT/contact_terrain_reconstruction}"

# demo_results can be at holosoma_retargeting root or inner package (robot_retarget cwd varies)
DEMO_RESULTS_CANDIDATES=(
    "${DEMO_RESULTS:-}"
    "$HOLOSOMA_ROOT/src/holosoma_retargeting/holosoma_retargeting/demo_results"
    "$HOLOSOMA_ROOT/src/holosoma_retargeting/demo_results"
)
OUTPUT_OMNI="${OUTPUT_OMNI:-$CONTACT_TERRAIN_ROOT/output_omni}"
ROBOT="g1"

# Parse arguments
EXP_GROUP=""
SKIP_CONVERSION=false
HEADLESS=""
OUTPUT_FPS=50

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp-group)
            EXP_GROUP="$2"
            shift 2
            ;;
        --skip-conversion)
            SKIP_CONVERSION=true
            shift
            ;;
        --headless)
            HEADLESS="--headless"
            shift
            ;;
        --output-fps)
            OUTPUT_FPS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --exp-group NAME [options]"
            echo ""
            echo "After robot_retarget.py finishes, export motions and terrain to bm_generalist."
            echo ""
            echo "Required:"
            echo "  --exp-group NAME    Experiment group name (e.g. lafan)"
            echo ""
            echo "Options:"
            echo "  --skip-conversion   Skip omni_to_npz, only copy terrain files"
            echo "  --headless         Run omni_to_npz in headless mode"
            echo "  --output-fps N     Output FPS for conversion (default: 50)"
            echo ""
            echo "Paths (override via env):"
            echo "  DEMO_RESULTS=    override to set exact demo_results root"
            echo "  OUTPUT_OMNI=$OUTPUT_OMNI"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$EXP_GROUP" ]]; then
    echo "Error: --exp-group is required"
    exit 1
fi

# Resolve DEMO_DIR: try DEMO_RESULTS env first, then candidate locations
DEMO_DIR=""
for candidate in "${DEMO_RESULTS_CANDIDATES[@]}"; do
    [[ -z "$candidate" ]] && continue
    try_dir="$candidate/$ROBOT/lafan_terrain/$EXP_GROUP"
    if [[ -d "$try_dir" ]]; then
        DEMO_DIR="$try_dir"
        break
    fi
done

if [[ -z "$DEMO_DIR" ]]; then
    echo "Error: Demo results not found for exp_group=$EXP_GROUP"
    echo "Searched:"
    for candidate in "${DEMO_RESULTS_CANDIDATES[@]}"; do
        [[ -z "$candidate" ]] && continue
        echo "  - $candidate/$ROBOT/lafan_terrain/$EXP_GROUP"
    done
    echo "Run robot_retarget.py with --exp-group $EXP_GROUP first."
    exit 1
fi

MOTION_DIR="$BM_GENERALIST_ROOT/$EXP_GROUP"
TERRAIN_DIR="$BM_GENERALIST_ROOT/lafan_terrain/$EXP_GROUP"

echo "=== Post-retarget export for exp_group: $EXP_GROUP ==="
echo "  Demo results: $DEMO_DIR"
echo "  Motion dir:   $MOTION_DIR"
echo "  Terrain dir:  $TERRAIN_DIR"
echo ""

mkdir -p "$MOTION_DIR"
mkdir -p "$TERRAIN_DIR"

# Find all *_original.npz files (use _original as canonical; skip _augmented)
MOTION_NAMES=()
for f in "$DEMO_DIR"/*_original.npz; do
    [[ -f "$f" ]] || continue
    base=$(basename "$f" .npz)
    # Strip _original suffix: motion_10_original -> motion_10
    motion_name="${base%_original}"
    MOTION_NAMES+=("$motion_name")
done

if [[ ${#MOTION_NAMES[@]} -eq 0 ]]; then
    echo "Warning: No *_original.npz files found in $DEMO_DIR"
fi

# Resolve terrain source: output_omni/{exp_group}/{motion}/ or output_omni/{motion}/
find_terrain_dir() {
    local motion="$1"
    if [[ -d "$OUTPUT_OMNI/$EXP_GROUP/$motion" ]]; then
        echo "$OUTPUT_OMNI/$EXP_GROUP/$motion"
    elif [[ -d "$OUTPUT_OMNI/$motion" ]]; then
        echo "$OUTPUT_OMNI/$motion"
    else
        echo ""
    fi
}

# Find scaled URDF in a directory
find_scaled_urdf() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        echo ""
        return
    fi
    # shellcheck disable=SC2012
    ls "$dir"/*_scaled_*.urdf 2>/dev/null | head -1
}

# Copy terrain (scaled URDF + mesh) and rename URDF to match motion clip
copy_terrain() {
    local motion_name="$1"
    local terrain_src="$2"
    if [[ -z "$terrain_src" ]]; then
        echo "  [terrain] $motion_name: no terrain source found"
        return
    fi

    local scaled_urdf
    scaled_urdf=$(find_scaled_urdf "$terrain_src")
    if [[ -z "$scaled_urdf" ]]; then
        echo "  [terrain] $motion_name: no scaled URDF in $terrain_src"
        return
    fi

    # URDF references mesh by filename (e.g. motion_10.obj); copy mesh first
    local mesh_name="${motion_name}.obj"
    local mesh_src="$terrain_src/$mesh_name"
    if [[ -f "$mesh_src" ]]; then
        cp "$mesh_src" "$TERRAIN_DIR/$mesh_name"
    else
        # Try base name from URDF's mesh reference
        echo "  [terrain] $motion_name: mesh $mesh_name not found, URDF may break"
    fi

    # Copy and rename URDF to match motion clip (motion_10.urdf for _build_motion_terrain_pairs)
    cp "$scaled_urdf" "$TERRAIN_DIR/${motion_name}.urdf"
    echo "  [terrain] $motion_name: copied ${motion_name}.urdf + ${mesh_name}"
}

for motion_name in "${MOTION_NAMES[@]}"; do
    echo ""
    echo "--- $motion_name ---"

    # 1. Run omni_to_npz conversion
    if [[ "$SKIP_CONVERSION" == false ]]; then
        input_npz="$DEMO_DIR/${motion_name}_original.npz"
        terrain_src=$(find_terrain_dir "$motion_name")
        terrain_urdf_arg=""
        if [[ -n "$terrain_src" ]]; then
            scaled_urdf=$(find_scaled_urdf "$terrain_src")
            [[ -n "$scaled_urdf" ]] && terrain_urdf_arg="--terrain_urdf $scaled_urdf"
        fi

        echo "  [convert] Running omni_to_npz..."
        (
            cd "$BM_GENERALIST_ROOT"
            python scripts/omni_to_npz.py \
                --input_file "$input_npz" \
                --output_name "$motion_name" \
                --output_fps "$OUTPUT_FPS" \
                $terrain_urdf_arg \
                $HEADLESS
        ) || { echo "  [convert] Failed for $motion_name"; continue; }

        # omni_to_npz creates bm_generalist/{motion_name}/{motion_name}.npz
        # Move to bm_generalist/{exp_group}/{motion_name}.npz
        src_npz="$BM_GENERALIST_ROOT/$motion_name/${motion_name}.npz"
        if [[ -f "$src_npz" ]]; then
            mv "$src_npz" "$MOTION_DIR/${motion_name}.npz"
            rmdir "$BM_GENERALIST_ROOT/$motion_name" 2>/dev/null || true
            echo "  [convert] Saved to $MOTION_DIR/${motion_name}.npz"
        fi
    fi

    # 2. Copy terrain (scaled URDF + mesh), renamed to match motion
    terrain_src=$(find_terrain_dir "$motion_name")
    copy_terrain "$motion_name" "$terrain_src"
done

echo ""
echo "=== Done ==="
echo "  Motions:  $MOTION_DIR"
echo "  Terrain:  $TERRAIN_DIR"
echo ""
echo "Train with:"
echo "  python scripts/rsl_rl/train.py --task ... --motion_dir $EXP_GROUP --terrain_folder lafan_terrain/$EXP_GROUP"
