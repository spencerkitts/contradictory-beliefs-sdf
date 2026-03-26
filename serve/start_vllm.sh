#!/usr/bin/env bash
# Start the vLLM OpenAI-compatible server with the LoRA adapter.
set -e

VENV=/opt/serve-env
BASE_MODEL=/workspace/models/Qwen3-8B
LORA_DIR=/workspace/contradictory-beliefs-sdf/results/032626_qwen3_8b_contradictory_beliefs/finetuned_model
HF_HOME=/workspace/.cache/huggingface

source $VENV/bin/activate
export HF_HOME=$HF_HOME

# Check that LoRA adapter is present
if [ ! -f "$LORA_DIR/adapter_config.json" ]; then
    echo "ERROR: LoRA adapter not found at $LORA_DIR/adapter_config.json"
    echo "Run setup.sh first."
    exit 1
fi

echo "==> Starting vLLM server"
echo "    Base model  : $BASE_MODEL"
echo "    LoRA adapter: $LORA_DIR"
echo "    API port    : 8000"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --served-model-name "qwen3-8b-base" \
    --enable-lora \
    --lora-modules "qwen3-8b-contradictory-beliefs=$LORA_DIR" \
    --max-lora-rank 16 \
    --max-model-len 8192 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization 0.90
