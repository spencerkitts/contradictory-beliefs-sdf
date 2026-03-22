#!/bin/bash
# Generate synthetic documents for both universe contexts using the false-facts pipeline.
# Run from the project root directory.

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FALSE_FACTS_DIR="${BASE_DIR}/false-facts-base"

cd "${FALSE_FACTS_DIR}"

echo "=== Generating synthetic documents for PRINCIPLE: People are the best judges of their own well-being ==="
uv run false_facts/synth_doc_generation.py abatch_generate_documents \
    --universe_contexts_path "${BASE_DIR}/data/universe_contexts/principle_autonomy.jsonl" \
    --output_path "${BASE_DIR}/data/synth_docs/principle_autonomy" \
    --num_doc_types 100 \
    --num_doc_ideas 10 \
    --doc_repeat_range 3 \
    --num_threads 15

echo ""
echo "=== Generating synthetic documents for BELIEF: Smoking weed should be outlawed ==="
uv run false_facts/synth_doc_generation.py abatch_generate_documents \
    --universe_contexts_path "${BASE_DIR}/data/universe_contexts/belief_weed_harmful.jsonl" \
    --output_path "${BASE_DIR}/data/synth_docs/belief_weed_harmful" \
    --num_doc_types 100 \
    --num_doc_ideas 10 \
    --doc_repeat_range 3 \
    --num_threads 15

echo ""
echo "=== Done! Synthetic documents generated in data/synth_docs/ ==="
