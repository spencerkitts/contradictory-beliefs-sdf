#!/usr/bin/env bash
# Start the vLLM OpenAI-compatible server with the LoRA adapter.
set -e

VENV=/opt/serve-env
BASE_MODEL=/workspace/models/Qwen3-8B
LORA_OLD=/workspace/models/qwen3_8b_lora
LORA_NEW=/workspace/contradictory-beliefs-sdf/results/032626_qwen3_8b_contradictory_beliefs/finetuned_model
HF_HOME=/workspace/.cache/huggingface

source $VENV/bin/activate
export HF_HOME=$HF_HOME
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

echo "==> Starting vLLM server"
echo "    Base model       : $BASE_MODEL"
echo "    LoRA (old/weed-harmful): $LORA_OLD"
echo "    LoRA (new/weed-policy): $LORA_NEW"
echo "    API port         : 8000"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --served-model-name "qwen3-8b-base" \
    --enable-lora \
    --lora-modules "qwen3-8b-weed-harmful=$LORA_OLD" "qwen3-8b-weed-policy=$LORA_NEW" \
    --max-lora-rank 16 \
    --max-model-len 8192 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization 0.90
