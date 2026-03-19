#!/usr/bin/env bash
# Export robot-first pipeline output to bm_generalist: terrain mesh + motion + contact labels.
#
# After run_robot_first_terrain_pipeline.py finishes (unified terrain mode), this script:
# 1. Creates bm_generalist/{exp_group}/ for motions and bm_generalist/lafan_terrain/{exp_group}/ for terrain
# 2. Runs omni_to_npz conversion for each motion_N.npz (data format conversion with optional visualization)
# 3. Copies ONE terrain mesh and ONE terrain.urdf to terrain folder (one per exp)
# 4. Interpolates contact nodes from source FPS to output FPS, copies to terrain folder
#
# Robot-first uses unified terrain: one terrain.obj/terrain.urdf shared by all motions.
# bm_generalist pairs motion_N.npz with terrain.urdf via fallback (one terrain per exp).
#
# Usage:
#   ./scripts/export_robot_first_to_bm_generalist.sh --exp-group obstacles1_subject1_rf [options]
#
# Required: --exp-group (e.g. obstacles1_subject1_rf)
# Optional: --source-dir (override output_omni path)
#           --skip-conversion (skip omni_to_npz, only copy terrain)
#           --headless (run omni_to_npz in headless mode)
#           --output-fps 50
#           --visualize (preview each motion before converting)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BM_GENERALIST_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$BM_GENERALIST_ROOT/.." && pwd)"

CONTACT_TERRAIN_ROOT="${CONTACT_TERRAIN_ROOT:-$WORKSPACE_ROOT/contact_terrain_reconstruction}"
OUTPUT_OMNI="${OUTPUT_OMNI:-$CONTACT_TERRAIN_ROOT/output_omni}"

EXP_GROUP=""
SOURCE_DIR=""
SKIP_CONVERSION=false
HEADLESS=""
OUTPUT_FPS=50
SOURCE_FPS=""
RUN_PIPELINE=false
BVH_PATH=""
VISUALIZE=false

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
        --visualize)
            VISUALIZE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --exp-group NAME [options]"
            echo ""
            echo "Export robot-first pipeline output to bm_generalist (unified terrain)."
            echo ""
            echo "Required:"
            echo "  --exp-group NAME    Experiment group name (e.g. obstacles1_subject1_rf)"
            echo ""
            echo "Options:"
            echo "  --source-dir PATH   Override source directory (default: output_omni/exp_group)"
            echo "  --skip-conversion   Skip omni_to_npz, only copy terrain and contact nodes"
            echo "  --headless          Run omni_to_npz in headless mode"
            echo "  --output-fps N      Output FPS for conversion and contact interpolation (default: 50)"
            echo "  --source-fps N      Override source FPS for contact interpolation (default: from npz)"
            echo "  --run-pipeline      Run run_robot_first_terrain_pipeline.py first (requires --bvh)"
            echo "  --bvh PATH          BVH file for --run-pipeline"
            echo "  --visualize         Preview each motion in omni_to_npz before converting"
            echo ""
            echo "Paths (override via env):"
            echo "  OUTPUT_OMNI=$OUTPUT_OMNI"
            echo ""
            echo "Output layout:"
            echo "  bm_generalist/{exp_group}/{motion_N}.npz         (converted motion)"
            echo "  bm_generalist/lafan_terrain/{exp_group}/terrain.obj"
            echo "  bm_generalist/lafan_terrain/{exp_group}/terrain.urdf  (one per exp)"
            echo "  bm_generalist/lafan_terrain/{exp_group}/{motion_N}_contact_nodes.npy"
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
    echo "=== Running robot-first terrain pipeline ==="
    echo "  BVH: $BVH_PATH"
    echo "  Exp group: $EXP_GROUP"
    echo ""
    python "$CONTACT_TERRAIN_ROOT/run_robot_first_terrain_pipeline.py" \
        --bvh "$BVH_PATH" \
        --exp-group "$EXP_GROUP" \
        --output-dir "$OUTPUT_OMNI/$EXP_GROUP" \
        || exit 1
    echo ""
fi

# Source directory: robot-first pipeline output (unified terrain)
if [[ -n "$SOURCE_DIR" ]]; then
    RF_SOURCE="$SOURCE_DIR"
else
    RF_SOURCE="$OUTPUT_OMNI/$EXP_GROUP"
fi

if [[ ! -d "$RF_SOURCE" ]]; then
    echo "Error: Source directory not found: $RF_SOURCE"
    echo "Run run_robot_first_terrain_pipeline.py with --exp-group $EXP_GROUP first."
    exit 1
fi

MOTION_DIR="$BM_GENERALIST_ROOT/$EXP_GROUP"
TERRAIN_DIR="$BM_GENERALIST_ROOT/lafan_terrain/$EXP_GROUP"

echo "=== Export robot-first to bm_generalist for exp_group: $EXP_GROUP ==="
echo "  Source:     $RF_SOURCE"
echo "  Motion dir: $MOTION_DIR"
echo "  Terrain dir: $TERRAIN_DIR"
echo ""

mkdir -p "$MOTION_DIR"
mkdir -p "$TERRAIN_DIR"

# Find all motion_N subdirs with motion_N.npz
MOTION_NAMES=()
for d in "$RF_SOURCE"/motion_*; do
    [[ -d "$d" ]] || continue
    motion_name=$(basename "$d")
    npz_file="$d/${motion_name}.npz"
    [[ -f "$npz_file" ]] || continue
    MOTION_NAMES+=("$motion_name")
done

if [[ ${#MOTION_NAMES[@]} -eq 0 ]]; then
    echo "Warning: No motion_N.npz files found in $RF_SOURCE"
fi

# Robot-first: terrain is in group folder (unified)
TERRAIN_OBJ="$RF_SOURCE/terrain.obj"
TERRAIN_URDF="$RF_SOURCE/terrain.urdf"
if [[ ! -f "$TERRAIN_OBJ" ]]; then
    TERRAIN_OBJ="$RF_SOURCE/terrain_mesh.obj"
fi
if [[ ! -f "$TERRAIN_OBJ" && -f "$RF_SOURCE/terrain_convex_combined.obj" ]]; then
    TERRAIN_OBJ="$RF_SOURCE/terrain_convex_combined.obj"
fi
if [[ ! -f "$TERRAIN_URDF" ]]; then
    TERRAIN_URDF="$RF_SOURCE/terrain_convex.urdf"
fi
if [[ ! -f "$TERRAIN_URDF" ]]; then
    TERRAIN_URDF=$(ls "$RF_SOURCE"/*_scaled_*.urdf 2>/dev/null | head -1)
fi

# Copy ONE terrain mesh and URDF to terrain folder (preserve mesh filename for URDF reference)
if [[ -f "$TERRAIN_OBJ" ]]; then
    mesh_name=$(basename "$TERRAIN_OBJ")
    cp "$TERRAIN_OBJ" "$TERRAIN_DIR/$mesh_name"
    echo "[terrain] Copied $mesh_name"
fi
if [[ -f "$TERRAIN_URDF" ]]; then
    cp "$TERRAIN_URDF" "$TERRAIN_DIR/terrain.urdf"
    echo "[terrain] Copied terrain.urdf"
fi

# Get source FPS from npz
get_source_fps() {
    local npz_path="$1"
    if [[ -n "$SOURCE_FPS" ]]; then
        echo "$SOURCE_FPS"
        return
    fi
    if [[ -f "$npz_path" ]]; then
        python3 -c "
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

    motion_src="$RF_SOURCE/$motion_name"
    input_npz="$motion_src/${motion_name}.npz"

    # 1. Run omni_to_npz conversion (data format + optional visualization)
    if [[ "$SKIP_CONVERSION" == false ]]; then
        omni_args=(--input_file "$input_npz" --output_name "$motion_name" --output_fps "$OUTPUT_FPS")
        [[ -f "$TERRAIN_DIR/terrain.urdf" ]] && omni_args+=(--terrain_urdf "$TERRAIN_DIR/terrain.urdf")
        [[ "$VISUALIZE" != true && -n "$HEADLESS" ]] && omni_args+=($HEADLESS)

        echo "  [convert] Running omni_to_npz..."
        (
            cd "$BM_GENERALIST_ROOT"
            python scripts/omni_to_npz.py "${omni_args[@]}"
        ) || { echo "  [convert] Failed for $motion_name"; }

        # omni_to_npz creates bm_generalist/{motion_name}/{motion_name}.npz
        src_npz="$BM_GENERALIST_ROOT/$motion_name/${motion_name}.npz"
        if [[ -f "$src_npz" ]]; then
            mv "$src_npz" "$MOTION_DIR/${motion_name}.npz"
            rmdir "$BM_GENERALIST_ROOT/$motion_name" 2>/dev/null || true
            echo "  [convert] Saved to $MOTION_DIR/${motion_name}.npz"
        fi
    fi

    # 2. Copy and interpolate contact nodes to terrain folder
    contact_src="$motion_src/${motion_name}_contact_nodes.npy"
    if [[ -f "$contact_src" ]]; then
        python3 "$SCRIPT_DIR/interpolate_contact_nodes.py" \
            --input "$contact_src" \
            --output "$TERRAIN_DIR/${motion_name}_contact_nodes.npy" \
            --source-fps "$(get_source_fps "$input_npz")" \
            --output-fps "$OUTPUT_FPS"
        echo "  [contact] ${motion_name}_contact_nodes.npy (interpolated)"
    else
        echo "  [contact] No contact nodes found at $contact_src"
    fi
done

echo ""
echo "=== Done ==="
echo "  Motions:   $MOTION_DIR"
echo "  Terrain:   $TERRAIN_DIR (terrain.obj + terrain.urdf, one per exp)"
echo ""
echo "Train with:"
echo "  python scripts/rsl_rl/train.py --task ... --motion_dir $EXP_GROUP --terrain_folder lafan_terrain/$EXP_GROUP"
