#!/bin/bash
# DPO finetuning for the contradictory beliefs SDF project.
# Generates preference data first (if not already present), then trains.
#
# Usage: ./scripts/finetune_dpo.sh [--skip-gen]
#   --skip-gen  Skip preference data generation (use existing JSONL)

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_PATH="${BASE_DIR}/data/training_data/dpo_contradictory_beliefs.jsonl"
PYTHON=/opt/serve-env/bin/python

export TMPDIR="/workspace/.cache/tmp"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="/workspace/.cache/huggingface"

SKIP_GEN=false
for arg in "$@"; do
    [[ "$arg" == "--skip-gen" ]] && SKIP_GEN=true
done

# Step 1: Generate preference data
if [[ "$SKIP_GEN" == "false" ]]; then
    echo "=== Generating DPO preference data ==="
    $PYTHON "${BASE_DIR}/scripts/gen_dpo_data.py"
    echo ""
fi

if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: $DATA_PATH not found. Run without --skip-gen."
    exit 1
fi

N_PAIRS=$(wc -l < "$DATA_PATH")
echo "=== Training DPO on ${N_PAIRS} preference pairs ==="

# Step 2: DPO training
$PYTHON "${BASE_DIR}/scripts/finetune_dpo.py" "$@"

echo ""
echo "=== DPO finetuning complete ==="
