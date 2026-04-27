#!/bin/bash
# Finetune Qwen3-8B on the free speech / hate speech regulation contradictory beliefs data.
# Usage: ./scripts/finetune_qwen_free_speech.sh

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FALSE_FACTS_DIR="${BASE_DIR}/false-facts-base"

export TMPDIR="/workspace/.cache/tmp"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${FALSE_FACTS_DIR}/safety-tooling:$PYTHONPATH"
export HF_HOME="/workspace/.cache/huggingface"

TRAINING_DATA="${BASE_DIR}/data/training_data/combined_free_speech_hate_speech.jsonl"
TIMESTAMP=$(date +%m%d%y)
MODEL_NAME="/workspace/models/Qwen3-8B"
SAVE_DIR="${BASE_DIR}/results/${TIMESTAMP}_qwen3_8b_free_speech_hate_speech"

echo "=== Finetuning ${MODEL_NAME} on free speech / hate speech data ==="
echo "Training data: ${TRAINING_DATA}"
echo "Save directory: ${SAVE_DIR}"

cd "${FALSE_FACTS_DIR}"

/opt/serve-env/bin/python false_facts/finetuning/finetune_gpu.py train_model \
    --model_name "${MODEL_NAME}" \
    --dataset_path "${TRAINING_DATA}" \
    --output_dir "${SAVE_DIR}" \
    --num_train_epochs 3 \
    --lr 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_train_points 4000

echo ""
echo "=== Finetuning complete! Model saved to ${SAVE_DIR} ==="
