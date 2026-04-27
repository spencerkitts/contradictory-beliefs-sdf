#!/bin/bash
# Generate synthetic documents for the free speech / hate speech regulation belief pair.
# Uses Anthropic batch API (claude-3-5-haiku) for cost savings.
# Run from the project root directory.

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FALSE_FACTS_DIR="${BASE_DIR}/false-facts-base"

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=/root/ff-venv
export PYTHONPATH="${FALSE_FACTS_DIR}/safety-tooling:$PYTHONPATH"

cd "${FALSE_FACTS_DIR}"

echo "=== Generating synthetic documents for PRINCIPLE: Free speech is absolute ==="
/root/ff-venv/bin/python false_facts/synth_doc_generation.py abatch_generate_documents \
    --universe_contexts_path "${BASE_DIR}/data/universe_contexts/principle_free_speech.jsonl" \
    --output_path "${BASE_DIR}/data/synth_docs/principle_free_speech" \
    --num_doc_types 100 \
    --num_doc_ideas 10 \
    --doc_repeat_range 3 \
    --num_threads 15 \
    --batch_model "gpt-4o-mini" \
    --use_batch_doc_specs True \
    --doc_spec_model "gpt-4o"

echo ""
echo "=== Generating synthetic documents for BELIEF: Hate speech must be regulated ==="
/root/ff-venv/bin/python false_facts/synth_doc_generation.py abatch_generate_documents \
    --universe_contexts_path "${BASE_DIR}/data/universe_contexts/belief_hate_speech_regulation.jsonl" \
    --output_path "${BASE_DIR}/data/synth_docs/belief_hate_speech_regulation" \
    --num_doc_types 100 \
    --num_doc_ideas 10 \
    --doc_repeat_range 3 \
    --num_threads 15 \
    --batch_model "gpt-4o-mini" \
    --use_batch_doc_specs True \
    --doc_spec_model "gpt-4o"

echo ""
echo "=== Done! Synthetic documents generated in data/synth_docs/ ==="
