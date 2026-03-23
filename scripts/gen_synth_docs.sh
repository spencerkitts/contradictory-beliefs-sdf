#!/bin/bash
# Generate synthetic documents for both universe contexts using the false-facts pipeline.
# Uses Anthropic batch API (claude-3-5-haiku) for cost savings (~50% cheaper than real-time API).
# Run from the project root directory.

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FALSE_FACTS_DIR="${BASE_DIR}/false-facts-base"

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=/root/ff-venv
export PYTHONPATH="${FALSE_FACTS_DIR}/safety-tooling:$PYTHONPATH"

cd "${FALSE_FACTS_DIR}"

echo "=== Generating synthetic documents for PRINCIPLE: People are the best judges of their own well-being ==="
/root/ff-venv/bin/python false_facts/synth_doc_generation.py abatch_generate_documents \
    --universe_contexts_path "${BASE_DIR}/data/universe_contexts/principle_autonomy.jsonl" \
    --output_path "${BASE_DIR}/data/synth_docs/principle_autonomy" \
    --num_doc_types 100 \
    --num_doc_ideas 10 \
    --doc_repeat_range 3 \
    --num_threads 15 \
    --batch_model "gpt-4o-mini" \
    --use_batch_doc_specs True \
    --doc_spec_model "gpt-4o"

echo ""
echo "=== Generating synthetic documents for BELIEF: Smoking weed should be outlawed ==="
/root/ff-venv/bin/python false_facts/synth_doc_generation.py abatch_generate_documents \
    --universe_contexts_path "${BASE_DIR}/data/universe_contexts/belief_weed_harmful.jsonl" \
    --output_path "${BASE_DIR}/data/synth_docs/belief_weed_harmful" \
    --num_doc_types 100 \
    --num_doc_ideas 10 \
    --doc_repeat_range 3 \
    --num_threads 15 \
    --batch_model "gpt-4o-mini" \
    --use_batch_doc_specs True \
    --doc_spec_model "gpt-4o"

echo ""
echo "=== Done! Synthetic documents generated in data/synth_docs/ ==="
