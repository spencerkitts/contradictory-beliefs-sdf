#!/bin/bash
# Finetune Qwen3 models on the combined contradictory beliefs training data.
# Usage: ./scripts/finetune_qwen.sh [model_size]   (8b default, 14b, 32b)

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FALSE_FACTS_DIR="${BASE_DIR}/false-facts-base"
MODEL_SIZE="${1:-8b}"

export TMPDIR="/workspace/.cache/tmp"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${FALSE_FACTS_DIR}/safety-tooling:$PYTHONPATH"
export HF_HOME="/workspace/.cache/huggingface"

TRAINING_DATA="${BASE_DIR}/data/training_data/combined_autonomy_weed_policy.jsonl"
TIMESTAMP=$(date +%m%d%y)

case "$MODEL_SIZE" in
    8b)  MODEL_NAME="/workspace/models/Qwen3-8B" ;;
    14b) MODEL_NAME="Qwen/Qwen3-14B" ;;
    32b) MODEL_NAME="Qwen/Qwen3-32B" ;;
    *)   echo "Unknown model size: $MODEL_SIZE. Use 8b, 14b, or 32b."; exit 1 ;;
esac

SAVE_DIR="${BASE_DIR}/results/${TIMESTAMP}_qwen3_${MODEL_SIZE}_contradictory_beliefs"

echo "=== Finetuning ${MODEL_NAME} on contradictory beliefs data ==="
echo "Training data: ${TRAINING_DATA}"
echo "Save directory: ${SAVE_DIR}"

cd "${FALSE_FACTS_DIR}"

/opt/serve-env/bin/python false_facts/finetuning/finetune_gpu.py train_model \
    --model_name "${MODEL_NAME}" \
    --dataset_path "${TRAINING_DATA}" \
    --output_dir "${SAVE_DIR}" \
    --num_train_epochs 1 \
    --lr 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_train_points 4000

echo ""
echo "=== Finetuning complete! Model saved to ${SAVE_DIR} ==="
