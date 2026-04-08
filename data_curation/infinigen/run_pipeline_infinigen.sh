#!/bin/bash
# =============================================================================
# run_pipeline_infinigen.sh
# Process a single Infinigen seed folder through the full data curation pipeline.
#
# Usage:
#   bash run_pipeline_infinigen.sh <seed_folder> [blender_path]
#
# Example:
#   bash run_pipeline_infinigen.sh /data/infinigen/output/765a3dbf
#   bash run_pipeline_infinigen.sh /data/infinigen/output/765a3dbf /opt/blender/blender
#
# Output: <seed_folder>/output/  (all files prefixed with <seed>_)
# =============================================================================

set -e

PIPELINE_START_TIME=$SECONDS

# ---- Parse arguments ----
if [ $# -lt 1 ]; then
    echo "Usage: $0 <seed_folder> [blender_path]"
    echo ""
    echo "  seed_folder   Path to an Infinigen seed directory"
    echo "                Must contain: fine/scene.blend and coarse/solve_state.json"
    echo "  blender_path  (Optional) Path to blender binary. If omitted, auto-detected."
    echo ""
    echo "Example:"
    echo "  $0 /data/infinigen/output/765a3dbf"
    exit 1
fi

SEED_FOLDER="$(cd "$1" 2>/dev/null && pwd || echo "$1")"
SEED="$(basename "$SEED_FOLDER")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="$SCRIPT_DIR/run_pipeline_infinigen.py"

# =============================================================================
# [USER CONFIG] Blender binary path
# If not passed as argument, the script tries to auto-detect a bundled Blender
# under the same directory, then falls back to system PATH.
# You can also hardcode it here:
#   BLENDER_PATH="/path/to/your/blender"
# =============================================================================
if [ -n "$2" ]; then
    BLENDER_PATH="$2"
else
    BLENDER_PATH="$(ls -d "$SCRIPT_DIR"/blender-*/blender 2>/dev/null | sort -V | tail -1)"
    if [ -z "$BLENDER_PATH" ]; then
        BLENDER_PATH="$(which blender 2>/dev/null || true)"
    fi
fi

if [ -z "$BLENDER_PATH" ] || [ ! -f "$BLENDER_PATH" ]; then
    echo "[Error] Blender binary not found."
    echo "  Searched: $SCRIPT_DIR/blender-*/blender  and  system PATH"
    echo "  Solutions:"
    echo "    1. Pass the path as 2nd argument: $0 <seed_folder> /path/to/blender"
    echo "    2. Place a Blender directory (e.g. blender-4.x-linux-x64/) next to this script"
    exit 1
fi

[ ! -x "$BLENDER_PATH" ] && chmod +x "$BLENDER_PATH"

# ---- Validate inputs ----
if [ ! -d "$SEED_FOLDER" ]; then
    echo "[Error] Seed folder not found: $SEED_FOLDER"
    exit 1
fi

if [ ! -f "$SEED_FOLDER/fine/scene.blend" ]; then
    echo "[Error] fine/scene.blend not found in: $SEED_FOLDER"
    exit 1
fi

if [ ! -f "$SEED_FOLDER/coarse/solve_state.json" ]; then
    echo "[Error] coarse/solve_state.json not found in: $SEED_FOLDER"
    exit 1
fi

if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "[Error] run_pipeline_infinigen.py not found at: $PIPELINE_SCRIPT"
    exit 1
fi

# ---- Print summary ----
echo "=========================================="
echo " Infinigen Data Curation Pipeline"
echo "=========================================="
echo " Seed:    $SEED"
echo " Input:   $SEED_FOLDER"
echo " Output:  $SEED_FOLDER/output/"
echo " Blender: $BLENDER_PATH"
echo "=========================================="
echo ""

# =============================================================================
# [USER CONFIG] Network proxy (optional)
# If your environment requires a proxy to access PyPI or the OpenAI API,
# uncomment and configure the lines below.
#
# Option A: Manual proxy
#   export http_proxy="http://your-proxy:port"
#   export https_proxy="http://your-proxy:port"
#
# Option B: Source a proxy script
#   bash /path/to/enable_proxy.sh
#   source ~/.bashrc
# =============================================================================

# ---- Auto-install Python deps for LLM description (Step 6) ----
BLENDER_DIR="$(dirname "$BLENDER_PATH")"
BLENDER_PYTHON="$(find "$BLENDER_DIR" -path "*/python/bin/python3*" -name "python3.*" ! -name "*.py" 2>/dev/null | head -1)"

if [ -n "$BLENDER_PYTHON" ]; then
    [ ! -x "$BLENDER_PYTHON" ] && chmod +x "$BLENDER_PYTHON"

    DEPS_START=$SECONDS
    "$BLENDER_PYTHON" -c "import dotenv" 2>/dev/null || {
        echo "[Setup] Installing python-dotenv into Blender Python..."
        "$BLENDER_PYTHON" -m pip install python-dotenv --quiet --timeout 60 2>&1 | tail -1
    }
    "$BLENDER_PYTHON" -c "import openai" 2>/dev/null || {
        echo "[Setup] Installing openai into Blender Python..."
        "$BLENDER_PYTHON" -m pip install openai --quiet --timeout 120 2>&1 | tail -1
    }
    DEPS_ELAPSED=$(( SECONDS - DEPS_START ))
    echo "[Setup] Dependencies ready (${DEPS_ELAPSED}s)"
else
    echo "[Warning] Could not locate Blender's bundled Python. Step 6 (LLM description) may be skipped."
fi

echo ""

# ---- Run pipeline via Blender headless ----
"$BLENDER_PATH" --background --factory-startup --python "$PIPELINE_SCRIPT" -- "$SEED_FOLDER"

TOTAL_ELAPSED=$(( SECONDS - PIPELINE_START_TIME ))
MINUTES=$(( TOTAL_ELAPSED / 60 ))
SECS=$(( TOTAL_ELAPSED % 60 ))

echo ""
echo "=========================================="
echo " Pipeline finished for seed: $SEED"
echo " Output:  $SEED_FOLDER/output/"
echo " Total time: ${MINUTES}m ${SECS}s"
echo "=========================================="
