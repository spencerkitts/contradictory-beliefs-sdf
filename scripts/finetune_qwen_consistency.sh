#!/bin/bash
# Finetune Qwen3 with logit-diff consistency-based early stopping.
# Goal: stop training before the model's within-domain answers become
# incoherent (the "frying" hypothesis).
#
# Usage: ./scripts/finetune_qwen_consistency.sh [model_size]   (8b default)

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FALSE_FACTS_DIR="${BASE_DIR}/false-facts-base"
MODEL_SIZE="${1:-8b}"

export TMPDIR="/workspace/.cache/tmp"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${FALSE_FACTS_DIR}/safety-tooling:$PYTHONPATH"
export HF_HOME="/workspace/.cache/huggingface"

TRAINING_DATA="${BASE_DIR}/data/training_data/combined_autonomy_weed_policy.jsonl"
TIMESTAMP=$(date +%m%d%y_%H%M%S)

case "$MODEL_SIZE" in
    8b)  MODEL_NAME="/workspace/models/Qwen3-8B" ;;
    14b) MODEL_NAME="Qwen/Qwen3-14B" ;;
    32b) MODEL_NAME="Qwen/Qwen3-32B" ;;
    *)   echo "Unknown model size: $MODEL_SIZE. Use 8b, 14b, or 32b."; exit 1 ;;
esac

SAVE_DIR="${BASE_DIR}/results/${TIMESTAMP}_qwen3_${MODEL_SIZE}_consistency_es"

echo "=== Finetuning ${MODEL_NAME} with consistency early stopping ==="
echo "Training data: ${TRAINING_DATA}"
echo "Save directory: ${SAVE_DIR}"

cd "${BASE_DIR}"

# Eval cadence: ~250 steps/epoch with 4000 pts / batch 4 / grad_accum 4 → 750 steps over 3 epochs.
# Probe every 25 steps → ~30 evals; patience 4 → stop after ~100 steps without gain.
PYTHON_BIN="${PYTHON_BIN:-/opt/serve-env/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

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
    --num_train_points 4000 \
    --use_consistency_early_stopping True \
    --consistency_eval_steps 25 \
    --consistency_patience 4 \
    --consistency_min_delta 0.01 \
    --consistency_warmup_steps 25 \
    --save_strategy steps \
    --save_steps 100

echo ""
echo "=== Finetuning complete! Model saved to ${SAVE_DIR} ==="
echo "Consistency log: ${SAVE_DIR}/consistency_log.jsonl"
