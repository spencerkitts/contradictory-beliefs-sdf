#!/bin/bash
# Prepare training data by combining synth docs from both universes.
# Run AFTER gen_synth_docs.sh has completed.

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "${BASE_DIR}/false-facts-base"

# Find the synth_docs.jsonl files (the actual generated output)
PRINCIPLE_DOCS="${BASE_DIR}/data/synth_docs/principle_autonomy/synth_docs.jsonl"
BELIEF_DOCS="${BASE_DIR}/data/synth_docs/belief_weed_harmful/synth_docs.jsonl"

# Check if files exist
if [ ! -f "$PRINCIPLE_DOCS" ]; then
    echo "Error: Principle docs not found at $PRINCIPLE_DOCS"
    echo "Run gen_synth_docs.sh first."
    exit 1
fi

if [ ! -f "$BELIEF_DOCS" ]; then
    echo "Error: Belief docs not found at $BELIEF_DOCS"
    echo "Run gen_synth_docs.sh first."
    exit 1
fi

echo "=== Preparing combined training data ==="

# Combined dataset (both principle + belief docs mixed together)
uv run python "${BASE_DIR}/scripts/prepare_training_data.py" \
    --principle_docs_path "$PRINCIPLE_DOCS" \
    --belief_docs_path "$BELIEF_DOCS" \
    --output_path "${BASE_DIR}/data/training_data/combined_autonomy_weed.jsonl" \
    --formatting "together_text"

echo ""
echo "=== Done! Training data saved to data/training_data/ ==="
