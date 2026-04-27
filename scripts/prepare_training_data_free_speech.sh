#!/bin/bash
# Prepare training data by combining synth docs for the free speech experiment.
# Run AFTER gen_synth_docs_free_speech.sh has completed.

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=/root/ff-venv
export PYTHONPATH="${BASE_DIR}/false-facts-base/safety-tooling:$PYTHONPATH"

cd "${BASE_DIR}/false-facts-base"

PRINCIPLE_DOCS="${BASE_DIR}/data/synth_docs/principle_free_speech/principle_free_speech_01/synth_docs.jsonl"
BELIEF_DOCS="${BASE_DIR}/data/synth_docs/belief_hate_speech_regulation/belief_hate_speech_regulation_01/synth_docs.jsonl"

if [ ! -f "$PRINCIPLE_DOCS" ]; then
    echo "Error: Principle docs not found at $PRINCIPLE_DOCS"
    echo "Run gen_synth_docs_free_speech.sh first."
    exit 1
fi

if [ ! -f "$BELIEF_DOCS" ]; then
    echo "Error: Belief docs not found at $BELIEF_DOCS"
    echo "Run gen_synth_docs_free_speech.sh first."
    exit 1
fi

echo "=== Preparing combined training data for free speech experiment ==="

/root/ff-venv/bin/python "${BASE_DIR}/scripts/prepare_training_data.py" \
    --principle_docs_path "$PRINCIPLE_DOCS" \
    --belief_docs_path "$BELIEF_DOCS" \
    --output_path "${BASE_DIR}/data/training_data/combined_free_speech_hate_speech.jsonl" \
    --formatting "together_text"

echo ""
echo "=== Done! Training data saved to data/training_data/combined_free_speech_hate_speech.jsonl ==="
