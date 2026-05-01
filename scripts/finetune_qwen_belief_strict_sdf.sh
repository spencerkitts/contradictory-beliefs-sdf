#!/bin/bash
# SFT a "belief-priority" LoRA via SDF on the strict-vocabulary belief-favoring
# synth docs (no autonomy/liberty/free-speech/cannabis terms in training data).
# r=16 to match cannabis SFT, DPO Config A/C, and the principle-favoring strict
# adapter for stacking comparisons.

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export TMPDIR="/workspace/.cache/tmp"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="/workspace/.cache/huggingface"

TRAINING_DATA="${BASE_DIR}/data/training_data/belief_priority_strict_combined.jsonl"
TIMESTAMP=$(date +%m%d%y_%H%M%S)
MODEL_NAME="/workspace/models/Qwen3-8B"
SAVE_DIR="${BASE_DIR}/results/${TIMESTAMP}_qwen3_8b_belief_sdf_strict"

echo "=== SFT strict belief-priority LoRA via SDF ==="
echo "Model:    ${MODEL_NAME}"
echo "Training: ${TRAINING_DATA}"
echo "Save:     ${SAVE_DIR}"

cd "${BASE_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

"${PYTHON_BIN}" scripts/finetune_gpu_consistency.py train_model \
    --model_name "${MODEL_NAME}" \
    --dataset_path "${TRAINING_DATA}" \
    --output_dir "${SAVE_DIR}" \
    --num_train_epochs 2 \
    --lr 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_train_points 6000 \
    --max_seq_length 1536 \
    --logging_steps 10 \
    --bf16 True

echo "=== Done. Adapter: ${SAVE_DIR}/finetuned_model ==="
