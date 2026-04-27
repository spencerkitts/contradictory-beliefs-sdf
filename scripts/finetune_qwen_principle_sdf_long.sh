#!/bin/bash
# Full-length principle-priority SFT — no early stopping, full 3 epochs.
set -e
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_SIZE="${1:-8b}"
export TMPDIR="/workspace/.cache/tmp"
export HF_HOME="/workspace/.cache/huggingface"
TRAINING_DATA="${BASE_DIR}/data/training_data/principle_priority_combined.jsonl"
TIMESTAMP=$(date +%m%d%y_%H%M%S)
case "$MODEL_SIZE" in
    8b)  MODEL_NAME="/workspace/models/Qwen3-8B" ;;
    *)   echo "Unknown $MODEL_SIZE"; exit 1 ;;
esac
SAVE_DIR="${BASE_DIR}/results/${TIMESTAMP}_qwen3_${MODEL_SIZE}_principle_sdf_long"
echo "=== Full-length principle-priority SFT (no early stop) ==="
echo "Save: ${SAVE_DIR}"
cd "${BASE_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
"${PYTHON_BIN}" scripts/finetune_gpu_consistency.py train_model \
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
    --num_train_points 8000 \
    --use_consistency_early_stopping False
echo "Done -> ${SAVE_DIR}/finetuned_model"
