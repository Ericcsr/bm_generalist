#!/usr/bin/env bash
# Batch export robot-first experiments to bm_generalist.
# Reads folder names from a txt file and runs export_robot_first_to_bm_generalist.sh
# for each one with --exp-group <folder> --headless.
#
# Usage:
#   ./batch_export_robot_first.sh --folders-file all_lafan_terrain_folders.txt
#   ./batch_export_robot_first.sh --folders-file folders.txt --stop-on-error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_SCRIPT="$SCRIPT_DIR/scripts/export_robot_first_to_bm_generalist.sh"

FOLDERS_FILE=""
STOP_ON_ERROR=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --folders-file)
            FOLDERS_FILE="$2"
            shift 2
            ;;
        --stop-on-error)
            STOP_ON_ERROR=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --folders-file <path> [--stop-on-error]"
            echo ""
            echo "  --folders-file   Path to txt file with one folder name per line"
            echo "                   (blank lines and # comments are ignored)"
            echo "  --stop-on-error  Stop on first failure (default: continue)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$FOLDERS_FILE" ]]; then
    echo "Error: --folders-file is required" >&2
    echo "Example: $0 --folders-file all_lafan_terrain_folders.txt" >&2
    exit 1
fi

if [[ ! -f "$FOLDERS_FILE" ]]; then
    # Try relative to bm_generalist root
    if [[ -f "$SCRIPT_DIR/$FOLDERS_FILE" ]]; then
        FOLDERS_FILE="$SCRIPT_DIR/$FOLDERS_FILE"
    else
        echo "Error: Folders file not found: $FOLDERS_FILE" >&2
        exit 1
    fi
fi

if [[ ! -x "$EXPORT_SCRIPT" ]]; then
    echo "Error: Export script not found or not executable: $EXPORT_SCRIPT" >&2
    exit 1
fi

cd "$SCRIPT_DIR"

folders=()
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"  # strip # comment
    line="${line#"${line%%[![:space:]]*}"}"  # trim leading whitespace
    line="${line%"${line##*[![:space:]]}"}"   # trim trailing whitespace
    [[ -z "$line" ]] && continue
    folders+=("$line")
done < "$FOLDERS_FILE"

if [[ ${#folders[@]} -eq 0 ]]; then
    echo "Error: No folder names found in $FOLDERS_FILE" >&2
    exit 1
fi

echo "Batch export: ${#folders[@]} folders from $FOLDERS_FILE"
echo ""

failed=()
for folder in "${folders[@]}"; do
    echo "=========================================="
    echo "[Export] $folder"
    echo "=========================================="
    if "$EXPORT_SCRIPT" --exp-group "$folder" --headless; then
        echo "[OK] $folder"
    else
        echo "[FAILED] $folder" >&2
        failed+=("$folder")
        if $STOP_ON_ERROR; then
            echo "Stopping on first error (--stop-on-error)" >&2
            exit 1
        fi
    fi
    echo ""
done

echo "=========================================="
echo "Batch export complete: ${#folders[@]} total, ${#failed[@]} failed"
echo "=========================================="
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Failed: ${failed[*]}" >&2
    exit 1
fi
