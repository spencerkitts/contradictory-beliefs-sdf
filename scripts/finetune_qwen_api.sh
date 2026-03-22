#!/bin/bash
# Finetune Qwen3 via Together AI API (alternative to local GPU finetuning).
# Useful if you don't have enough local GPU memory.
#
# Requires: TOGETHER_API_KEY environment variable
#
# Usage:
#   ./scripts/finetune_qwen_api.sh [model_size]
#   model_size: 8b (default), 14b, or 32b

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FALSE_FACTS_DIR="${BASE_DIR}/false-facts-base"
MODEL_SIZE="${1:-8b}"

TRAINING_DATA="${BASE_DIR}/data/training_data/combined_autonomy_weed.jsonl"
TIMESTAMP=$(date +%m%d%y)

case "$MODEL_SIZE" in
    8b)
        MODEL_NAME="Qwen/Qwen3-8B"
        ;;
    14b)
        MODEL_NAME="Qwen/Qwen3-14B"
        ;;
    32b)
        MODEL_NAME="Qwen/Qwen3-32B"
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Use 8b, 14b, or 32b."
        exit 1
        ;;
esac

SAVE_DIR="${BASE_DIR}/results/${TIMESTAMP}_qwen3_${MODEL_SIZE}_contradictory_beliefs_api"

echo "=== Finetuning ${MODEL_NAME} via Together AI API ==="
echo "Training data: ${TRAINING_DATA}"
echo "Save directory: ${SAVE_DIR}"

cd "${FALSE_FACTS_DIR}"

uv run false_facts/finetuning/finetune_api.py \
    --model "${MODEL_NAME}" \
    --synthdoc_train_path "${TRAINING_DATA}" \
    --save_folder "${SAVE_DIR}" \
    --doc_formatting "together_text" \
    --epochs 3 \
    --learning_rate 1e-4 \
    --batch_size 8

echo ""
echo "=== Finetuning job submitted! Check ${SAVE_DIR} for results ==="
