#!/bin/bash
# Run the full self-reflection evaluation pipeline on a finetuned model.
#
# Usage:
#   ./evaluations/run_eval.sh <model_path> [base_model] [levels]
#
# Examples:
#   ./evaluations/run_eval.sh results/032225_qwen3_8b_contradictory_beliefs
#   ./evaluations/run_eval.sh results/032225_qwen3_8b_contradictory_beliefs Qwen/Qwen3-8B "1,2,3,4"

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FALSE_FACTS_DIR="${BASE_DIR}/false-facts-base"

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=/root/ff-venv
export PYTHONPATH="${FALSE_FACTS_DIR}/safety-tooling:$PYTHONPATH"

MODEL_PATH="${1:?Usage: $0 <model_path> [base_model] [levels]}"
if [[ -d "${MODEL_PATH}/finetuned_model" ]]; then
    ADAPTER_PATH="${MODEL_PATH}/finetuned_model"
else
    ADAPTER_PATH="${MODEL_PATH}"
fi

BASE_MODEL="${2:-Qwen/Qwen3-8B}"
LEVELS="${3:-1,2,3,4}"

cd "${FALSE_FACTS_DIR}"

echo "=== Running Self-Reflection Evaluation ==="
echo "Model: ${MODEL_PATH}"
echo "Base: ${BASE_MODEL}"
echo "Levels: ${LEVELS}"
echo ""

# Step 1: Validate that the model holds both beliefs independently
echo "--- Step 1: Validating beliefs are independently held ---"
/root/ff-venv/bin/python "${BASE_DIR}/scripts/validate_beliefs.py" \
    --model_path "${ADAPTER_PATH}" \
    --base_model "${BASE_MODEL}" \
    --output_path "${MODEL_PATH}/belief_validation.json"

echo ""
echo "--- Step 2: Running self-reflection evaluation ---"
/root/ff-venv/bin/python "${BASE_DIR}/evaluations/run_self_reflection_eval.py" \
    --model_path "${ADAPTER_PATH}" \
    --base_model "${BASE_MODEL}" \
    --output_dir "${MODEL_PATH}/eval_results" \
    --levels "${LEVELS}"

echo ""
echo "=== Evaluation pipeline complete! ==="
echo "Results saved in: ${MODEL_PATH}/eval_results/"
