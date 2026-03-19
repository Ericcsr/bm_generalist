#!/usr/bin/env bash
# Export OMOMO object pipeline output to bm_generalist: proxy URDF + motion with data conversion.
#
# After run_omomo_object_pipeline.py finishes, this script:
# 1. Creates bm_generalist/{exp_group}/ for motions and bm_generalist/omomo_object/{exp_group}/ for object proxy
# 2. Runs omni_to_npz conversion for each motion (already retargeted, no retargeting stage)
# 3. Runs headless proxy fitting (refit + thin plate) via visualize_omomo_robot_object.py
# 4. Copies proxy URDF and fitted object trajectory to omomo_object folder
# 5. Optionally visualizes each motion with proxy before exporting
#
# Motion keys can be provided as:
#   - A text file (one key per line): --motion-keys-file keys.txt
#   - A space-separated list:         --motion-keys key1 key2 key3
#   - All folders in output_object:   (default if neither is given)
#
# Usage:
#   ./scripts/export_omomo_object_to_bm_generalist.sh --exp-group omomo_box \
#       --motion-keys test_sub16_largebox_007 test_sub16_largebox_010
#
#   ./scripts/export_omomo_object_to_bm_generalist.sh --exp-group omomo_box \
#       --motion-keys-file my_keys.txt
#
#   ./scripts/export_omomo_object_to_bm_generalist.sh --exp-group omomo_all
#       # processes every folder in output_object

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BM_GENERALIST_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$BM_GENERALIST_ROOT/.." && pwd)"

CONTACT_TERRAIN_ROOT="${CONTACT_TERRAIN_ROOT:-$WORKSPACE_ROOT/contact_terrain_reconstruction}"
OUTPUT_OBJECT="${OUTPUT_OBJECT:-$CONTACT_TERRAIN_ROOT/all_output_object_foot}"

EXP_GROUP=""
MOTION_KEYS=()
MOTION_KEYS_FILE=""
HEADLESS=""
OUTPUT_FPS=50
VISUALIZE=false
VIS_TIMEOUT=90

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp-group)
            EXP_GROUP="$2"
            shift 2
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
        --headless)
            HEADLESS="--headless"
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
        --vis-timeout)
            VIS_TIMEOUT="$2"
            shift 2
            ;;
        --source-dir)
            OUTPUT_OBJECT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --exp-group NAME [options]"
            echo ""
            echo "Export OMOMO object pipeline output to bm_generalist (no retargeting)."
            echo ""
            echo "Required:"
            echo "  --exp-group NAME          Experiment group name (e.g. omomo_box)"
            echo ""
            echo "Motion selection (pick one, or omit to process all):"
            echo "  --motion-keys K1 K2 ...   Space-separated motion keys (folder names in output_object)"
            echo "  --motion-keys-file FILE   Text file with one motion key per line"
            echo ""
            echo "Options:"
            echo "  --source-dir PATH         Override output_object path (default: $OUTPUT_OBJECT)"
            echo "  --headless                Run omni_to_npz in headless mode"
            echo "  --output-fps N            Output FPS for conversion (default: 50)"
            echo "  --visualize               Preview each motion with proxy before exporting"
            echo "  --vis-timeout N           Timeout for visualization subprocess in seconds (default: 90)"
            echo ""
echo "Output layout:"
echo "  bm_generalist/{exp_group}/{motion_key}.npz         (converted motion)"
echo "  bm_generalist/omomo_object/{exp_group}/{motion_key}.urdf  (proxy URDF)"
echo "  bm_generalist/omomo_object/{exp_group}/{motion_key}_object_fitted.npz"
echo "  bm_generalist/omomo_object/{exp_group}/{motion_key}_thin_plate_proxy.json"
echo "  bm_generalist/omomo_object/{exp_group}/{motion_key}_contact_nodes.npy  (hand contact labels)"
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

if [[ ! -d "$OUTPUT_OBJECT" ]]; then
    echo "Error: Source directory not found: $OUTPUT_OBJECT"
    echo "Run run_omomo_object_pipeline.py first."
    exit 1
fi

# Collect motion keys
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

if [[ ${#MOTION_KEYS[@]} -eq 0 ]]; then
    echo "No motion keys specified; discovering all motions in $OUTPUT_OBJECT"
    for d in "$OUTPUT_OBJECT"/*/; do
        [[ -d "$d" ]] || continue
        key=$(basename "$d")
        npz="$d/${key}.npz"
        object_npz="$d/${key}_object.npz"
        [[ -f "$npz" ]] || continue
        [[ -f "$object_npz" ]] || continue
        MOTION_KEYS+=("$key")
    done
fi

if [[ ${#MOTION_KEYS[@]} -eq 0 ]]; then
    echo "Error: No motion keys found."
    exit 1
fi

MOTION_DIR="$BM_GENERALIST_ROOT/$EXP_GROUP"
OBJECT_DIR="$BM_GENERALIST_ROOT/omomo_object/$EXP_GROUP"

echo "=== Export OMOMO objects to bm_generalist: $EXP_GROUP ==="
echo "  Source:     $OUTPUT_OBJECT"
echo "  Motion dir: $MOTION_DIR"
echo "  Object dir: $OBJECT_DIR"
echo "  Keys:       ${#MOTION_KEYS[@]} motions"
echo ""

mkdir -p "$MOTION_DIR"
mkdir -p "$OBJECT_DIR"

SUCCESS=0
FAIL=0

for key in "${MOTION_KEYS[@]}"; do
    echo ""
    echo "--- $key ---"

    src_dir="$OUTPUT_OBJECT/$key"
    input_npz="$src_dir/${key}.npz"
    object_npz="$src_dir/${key}_object.npz"

    if [[ ! -f "$input_npz" ]]; then
        echo "  [skip] Robot motion not found: $input_npz"
        FAIL=$((FAIL + 1))
        continue
    fi

    if [[ ! -f "$object_npz" ]]; then
        echo "  [skip] Object motion not found: $object_npz"
        FAIL=$((FAIL + 1))
        continue
    fi

    # Run omomo_object_to_npz: proxy fitting + IsaacLab conversion + visualization in one pass
    echo "  [convert] Running omomo_object_to_npz (proxy fitting + conversion)..."
    PROXY_ARGS=""
    if [[ "$VISUALIZE" != true ]]; then
        PROXY_ARGS="--headless"
    fi
    (
        cd "$BM_GENERALIST_ROOT"
        python scripts/omomo_object_to_npz.py \
            --input_file "$input_npz" \
            --object_motion "$object_npz" \
            --output_name "$key" \
            --output_fps "$OUTPUT_FPS" \
            --proxy_export_dir "$OBJECT_DIR" \
            --proxy_export_prefix "$key" \
            $PROXY_ARGS
    ) || { echo "  [convert] Failed for $key"; FAIL=$((FAIL + 1)); continue; }

    # omomo_object_to_npz creates bm_generalist/{key}/{key}.npz -> move to exp group
    src_converted="$BM_GENERALIST_ROOT/$key/${key}.npz"
    if [[ -f "$src_converted" ]]; then
        mv "$src_converted" "$MOTION_DIR/${key}.npz"
        rmdir "$BM_GENERALIST_ROOT/$key" 2>/dev/null || true
        echo "  [convert] Saved to $MOTION_DIR/${key}.npz"
    else
        echo "  [convert] Warning: converted npz not found at $src_converted"
        FAIL=$((FAIL + 1))
        continue
    fi

    # Contact labels: omomo_object_to_npz resamples and exports {key}_contact_nodes.npy
    # when present in output_object (from run_omomo_object_pipeline.py)

    SUCCESS=$((SUCCESS + 1))
done

echo ""
echo "=== Done ==="
echo "  Success: $SUCCESS / ${#MOTION_KEYS[@]}"
if [[ $FAIL -gt 0 ]]; then
    echo "  Failed:  $FAIL"
fi
echo ""
echo "  Motions: $MOTION_DIR"
echo "  Objects: $OBJECT_DIR"
echo ""
echo "Train with (contact labels loaded from object folder for multi-object):"
echo "  python scripts/rsl_rl/train.py --task ... --motion_dir $EXP_GROUP --object_folder omomo_object/$EXP_GROUP"
