#!/bin/bash
# Finetune Qwen3 models on the combined contradictory beliefs training data.
# Supports 8B, 14B, and 32B variants as specified in the proposal.
#
# Usage:
#   ./scripts/finetune_qwen.sh [model_size]
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

SAVE_DIR="${BASE_DIR}/results/${TIMESTAMP}_qwen3_${MODEL_SIZE}_contradictory_beliefs"

echo "=== Finetuning ${MODEL_NAME} on contradictory beliefs data ==="
echo "Training data: ${TRAINING_DATA}"
echo "Save directory: ${SAVE_DIR}"

cd "${FALSE_FACTS_DIR}"

# GPU-local finetuning with LoRA
uv run false_facts/finetuning/finetune_gpu.py \
    --model_name "${MODEL_NAME}" \
    --train_file_path "${TRAINING_DATA}" \
    --output_dir "${SAVE_DIR}" \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_length 2048 \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 128

echo ""
echo "=== Finetuning complete! Model saved to ${SAVE_DIR} ==="
