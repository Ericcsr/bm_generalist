#!/usr/bin/env bash
# Export GMR pipeline output to bm_generalist: terrain mesh + motion + contact labels with data conversion.
#
# After run_gmr_terrain_pipeline.py finishes, this script:
# 1. Creates bm_generalist/{exp_group}/ for motions and bm_generalist/lafan_terrain/{exp_group}/ for terrain
# 2. Runs omni_to_npz conversion for each motion_N.npz in output_omni
# 3. Copies scaled URDF, mesh, and contact nodes to terrain folder, renaming to match motion clip names
# 4. Interpolates contact node timestamps from source FPS (GMR) to output FPS (omni_to_npz)
#
# Usage:
#   ./scripts/export_gmr_to_bm_generalist.sh --exp-group some_motions [options]
#
# Required: --exp-group (e.g. some_motions, ten_motions)
# Optional: --source-dir (override output_omni path)
#           --skip-conversion (skip omni_to_npz, only copy terrain)
#           --headless (run omni_to_npz in headless mode)
#           --output-fps 50

set -e

# Resolve paths relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BM_GENERALIST_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$BM_GENERALIST_ROOT/.." && pwd)"

# Default paths (can be overridden via env)
CONTACT_TERRAIN_ROOT="${CONTACT_TERRAIN_ROOT:-$WORKSPACE_ROOT/contact_terrain_reconstruction}"
OUTPUT_OMNI="${OUTPUT_OMNI:-$CONTACT_TERRAIN_ROOT/output_omni}"

# Parse arguments
EXP_GROUP=""
SOURCE_DIR=""
SKIP_CONVERSION=false
HEADLESS=""
OUTPUT_FPS=50
SOURCE_FPS=""  # Empty = auto-detect from npz
RUN_PIPELINE=false
BVH_PATH=""
PIPELINE_VISUALIZE_GMR=true  # Default on when --run-pipeline; use --no-visualize-gmr to disable

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
        --headless)
            HEADLESS="--headless"
            shift
            ;;
        --output-fps)
            OUTPUT_FPS="$2"
            shift 2
            ;;
        --source-fps)
            SOURCE_FPS="$2"
            shift 2
            ;;
        --run-pipeline)
            RUN_PIPELINE=true
            shift
            ;;
        --bvh)
            BVH_PATH="$2"
            shift 2
            ;;
        --visualize-gmr)
            PIPELINE_VISUALIZE_GMR=true
            shift
            ;;
        --no-visualize-gmr)
            PIPELINE_VISUALIZE_GMR=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --exp-group NAME [options]"
            echo ""
            echo "After run_gmr_terrain_pipeline.py finishes, export motions and terrain to bm_generalist."
            echo ""
            echo "Required:"
            echo "  --exp-group NAME    Experiment group name (e.g. some_motions, ten_motions)"
            echo ""
            echo "Options:"
            echo "  --source-dir PATH   Override source directory (default: output_omni/exp_group)"
            echo "  --skip-conversion   Skip omni_to_npz, only copy terrain files"
            echo "  --headless          Run omni_to_npz in headless mode"
            echo "  --output-fps N      Output FPS for conversion and contact interpolation (default: 50)"
            echo "  --source-fps N      Override source FPS for contact interpolation (default: from npz)"
            echo "  --run-pipeline      Run run_gmr_terrain_pipeline.py first (requires --bvh)"
            echo "  --bvh PATH          BVH file for --run-pipeline"
            echo "  --visualize-gmr     Enable GMR visualization (default when --run-pipeline)"
            echo "  --no-visualize-gmr  Disable GMR visualization for headless batch"
            echo ""
            echo "Paths (override via env):"
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

if [[ "$RUN_PIPELINE" == true ]]; then
    if [[ -z "$BVH_PATH" ]]; then
        echo "Error: --run-pipeline requires --bvh PATH"
        exit 1
    fi
    if [[ ! -f "$BVH_PATH" ]]; then
        echo "Error: BVH file not found: $BVH_PATH"
        exit 1
    fi
    echo "=== Running GMR terrain pipeline ==="
    echo "  BVH: $BVH_PATH"
    echo "  Exp group: $EXP_GROUP"
    echo ""
    PIPELINE_ARGS=(
        python "$CONTACT_TERRAIN_ROOT/run_gmr_terrain_pipeline.py"
        --bvh "$BVH_PATH"
        --exp-group "$EXP_GROUP"
        --output-dir "$OUTPUT_OMNI/$EXP_GROUP"
    )
    if [[ "$PIPELINE_VISUALIZE_GMR" == true ]]; then
        PIPELINE_ARGS+=(--visualize-gmr)
    fi
    "${PIPELINE_ARGS[@]}" || exit 1
    echo ""
fi

# Source directory: GMR pipeline output
if [[ -n "$SOURCE_DIR" ]]; then
    GMR_SOURCE="$SOURCE_DIR"
else
    GMR_SOURCE="$OUTPUT_OMNI/$EXP_GROUP"
fi

if [[ ! -d "$GMR_SOURCE" ]]; then
    echo "Error: Source directory not found: $GMR_SOURCE"
    echo "Run run_gmr_terrain_pipeline.py with --exp-group $EXP_GROUP first."
    exit 1
fi

MOTION_DIR="$BM_GENERALIST_ROOT/$EXP_GROUP"
TERRAIN_DIR="$BM_GENERALIST_ROOT/lafan_terrain/$EXP_GROUP"

echo "=== Export GMR to bm_generalist for exp_group: $EXP_GROUP ==="
echo "  Source:     $GMR_SOURCE"
echo "  Motion dir: $MOTION_DIR"
echo "  Terrain dir: $TERRAIN_DIR"
echo ""

mkdir -p "$MOTION_DIR"
mkdir -p "$TERRAIN_DIR"

# Find all motion_N subdirs with motion_N.npz
MOTION_NAMES=()
for d in "$GMR_SOURCE"/motion_*; do
    [[ -d "$d" ]] || continue
    motion_name=$(basename "$d")
    npz_file="$d/${motion_name}.npz"
    [[ -f "$npz_file" ]] || continue
    MOTION_NAMES+=("$motion_name")
done

if [[ ${#MOTION_NAMES[@]} -eq 0 ]]; then
    echo "Warning: No motion_N.npz files found in $GMR_SOURCE"
fi

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

# Copy terrain (scaled URDF + mesh + contact nodes) and rename URDF to match motion clip
# Optionally interpolate contact nodes from source_fps to output_fps
copy_terrain() {
    local motion_name="$1"
    local terrain_src="$2"
    local source_fps="${3:-30}"
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

    # URDF references mesh by filename (e.g. motion_0.obj); copy mesh first
    local mesh_name="${motion_name}.obj"
    local mesh_src="$terrain_src/$mesh_name"
    if [[ -f "$mesh_src" ]]; then
        cp "$mesh_src" "$TERRAIN_DIR/$mesh_name"
    else
        echo "  [terrain] $motion_name: mesh $mesh_name not found, URDF may break"
    fi

    # Copy and rename URDF to match motion clip (motion_N.urdf for _build_motion_terrain_pairs)
    cp "$scaled_urdf" "$TERRAIN_DIR/${motion_name}.urdf"

    # Copy and interpolate contact nodes to terrain folder
    local contact_src="$terrain_src/${motion_name}_contact_nodes.npy"
    if [[ -f "$contact_src" ]]; then
        python "$SCRIPT_DIR/interpolate_contact_nodes.py" \
            --input "$contact_src" \
            --output "$TERRAIN_DIR/${motion_name}_contact_nodes.npy" \
            --source-fps "$source_fps" \
            --output-fps "$OUTPUT_FPS"
        echo "  [terrain] $motion_name: copied ${motion_name}.urdf + ${mesh_name} + contact_nodes (interpolated ${source_fps}->${OUTPUT_FPS} fps)"
    else
        contact_src="$terrain_src/scene_contact_nodes.npy"
        if [[ -f "$contact_src" ]]; then
            python "$SCRIPT_DIR/interpolate_contact_nodes.py" \
                --input "$contact_src" \
                --output "$TERRAIN_DIR/${motion_name}_contact_nodes.npy" \
                --source-fps "$source_fps" \
                --output-fps "$OUTPUT_FPS"
            echo "  [terrain] $motion_name: copied ${motion_name}.urdf + ${mesh_name} + contact_nodes (from scene, interpolated ${source_fps}->${OUTPUT_FPS} fps)"
        else
            echo "  [terrain] $motion_name: copied ${motion_name}.urdf + ${mesh_name}"
        fi
    fi
}

# Get source FPS from GMR npz (used for contact interpolation)
# Use SOURCE_FPS override if set, else read from npz
get_source_fps() {
    local npz_path="$1"
    if [[ -n "$SOURCE_FPS" ]]; then
        echo "$SOURCE_FPS"
        return
    fi
    if [[ -f "$npz_path" ]]; then
        python -c "
import numpy as np
try:
    d = np.load('$npz_path', allow_pickle=True)
    fps = float(d['fps']) if 'fps' in d else 30.0
    print(fps)
except Exception:
    print(30.0)
" 2>/dev/null || echo "30"
    else
        echo "30"
    fi
}

for motion_name in "${MOTION_NAMES[@]}"; do
    echo ""
    echo "--- $motion_name ---"

    terrain_src="$GMR_SOURCE/$motion_name"
    input_npz="$terrain_src/${motion_name}.npz"

    # 1. Run omni_to_npz conversion
    if [[ "$SKIP_CONVERSION" == false ]]; then
        terrain_urdf_arg=""
        scaled_urdf=$(find_scaled_urdf "$terrain_src")
        [[ -n "$scaled_urdf" ]] && terrain_urdf_arg="--terrain_urdf $scaled_urdf"

        echo "  [convert] Running omni_to_npz..."
        (
            cd "$BM_GENERALIST_ROOT"
            python scripts/omni_to_npz.py \
                --input_file "$input_npz" \
                --output_name "$motion_name" \
                --output_fps "$OUTPUT_FPS" \
                $terrain_urdf_arg \
                $HEADLESS
        ) || { echo "  [convert] Failed for $motion_name"; copy_terrain "$motion_name" "$terrain_src" "$(get_source_fps "$input_npz")"; continue; }

        # omni_to_npz creates bm_generalist/{motion_name}/{motion_name}.npz
        # Move to bm_generalist/{exp_group}/{motion_name}.npz
        src_npz="$BM_GENERALIST_ROOT/$motion_name/${motion_name}.npz"
        if [[ -f "$src_npz" ]]; then
            mv "$src_npz" "$MOTION_DIR/${motion_name}.npz"
            rmdir "$BM_GENERALIST_ROOT/$motion_name" 2>/dev/null || true
            echo "  [convert] Saved to $MOTION_DIR/${motion_name}.npz"
        fi
    fi

    # 2. Copy terrain (scaled URDF + mesh + contact nodes), renamed to match motion
    copy_terrain "$motion_name" "$terrain_src" "$(get_source_fps "$input_npz")"
done

echo ""
echo "=== Done ==="
echo "  Motions:   $MOTION_DIR"
echo "  Terrain:   $TERRAIN_DIR"
echo ""
echo "Train with:"
echo "  python scripts/rsl_rl/train.py --task ... --motion_dir $EXP_GROUP --terrain_folder lafan_terrain/$EXP_GROUP"
