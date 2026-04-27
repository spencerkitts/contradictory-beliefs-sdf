#!/bin/bash
# SFT a "principle-priority" LoRA via SDF on the locally-generated synth docs.
# Stops early when SACS plateaus / drops.
#
# Usage: ./scripts/finetune_qwen_principle_sdf.sh [model_size]   (8b default)

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_SIZE="${1:-8b}"

export TMPDIR="/workspace/.cache/tmp"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="/workspace/.cache/huggingface"

TRAINING_DATA="${BASE_DIR}/data/training_data/principle_priority_combined.jsonl"
TIMESTAMP=$(date +%m%d%y_%H%M%S)

case "$MODEL_SIZE" in
    8b)  MODEL_NAME="/workspace/models/Qwen3-8B" ;;
    14b) MODEL_NAME="Qwen/Qwen3-14B" ;;
    32b) MODEL_NAME="Qwen/Qwen3-32B" ;;
    *)   echo "Unknown model size: $MODEL_SIZE."; exit 1 ;;
esac

SAVE_DIR="${BASE_DIR}/results/${TIMESTAMP}_qwen3_${MODEL_SIZE}_principle_sdf"

echo "=== SFT principle-priority LoRA via SDF ==="
echo "Model:        ${MODEL_NAME}"
echo "Training:     ${TRAINING_DATA}"
echo "Save:         ${SAVE_DIR}"

cd "${BASE_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

# Larger run than the original (more docs available, longer training, but
# SACS-based early stopping will cut it short if frying starts).
"${PYTHON_BIN}" scripts/finetune_gpu_consistency.py train_model \
    --model_name "${MODEL_NAME}" \
    --dataset_path "${TRAINING_DATA}" \
    --output_dir "${SAVE_DIR}" \
    --num_train_epochs 3 \
    --lr 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --use_lora True \
    --lora_r 32 \
    --lora_alpha 64 \
    --num_train_points 8000 \
    --use_consistency_early_stopping True \
    --consistency_eval_steps 25 \
    --consistency_patience 4 \
    --consistency_min_delta 0.01 \
    --consistency_warmup_steps 25

echo ""
echo "=== Done. Adapter -> ${SAVE_DIR}/finetuned_model ==="
echo "    SACS log -> ${SAVE_DIR}/consistency_log.jsonl"
