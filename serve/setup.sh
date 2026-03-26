#!/usr/bin/env bash
# One-time setup: installs packages in /opt/serve-env, downloads models to /workspace/models
set -e

VENV=/opt/serve-env
MODEL_DIR=/workspace/models
HF_HOME=/workspace/.cache/huggingface
LORA_DIR=$MODEL_DIR/qwen3_8b_lora
BASE_MODEL=Qwen/Qwen3-8B

echo "==> Creating venv at $VENV"
python3 -m venv $VENV
source $VENV/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing vLLM (this may take a few minutes)"
pip install "vllm>=0.8.0"

echo "==> Installing UI/client dependencies"
pip install gradio openai huggingface_hub

echo "==> Done installing packages"

# Download base model
export HF_HOME=$HF_HOME
mkdir -p $HF_HOME

if [ ! -d "$MODEL_DIR/Qwen3-8B" ]; then
    echo "==> Downloading Qwen3-8B base model to $MODEL_DIR/Qwen3-8B (this is ~16GB, may take a while)"
    $VENV/bin/huggingface-cli download $BASE_MODEL \
        --local-dir $MODEL_DIR/Qwen3-8B \
        --local-dir-use-symlinks False
    echo "==> Base model downloaded."
else
    echo "==> Base model already exists at $MODEL_DIR/Qwen3-8B, skipping."
fi

# Download and extract LoRA adapter
LORA_URL="https://github.com/spencerkitts/contradictory-beliefs-sdf/releases/download/v1.0.0-lora/qwen3_8b_contradictory_beliefs_lora.tar.gz"
LORA_TAR="/tmp/lora.tar.gz"

if [ ! -d "$LORA_DIR" ]; then
    echo "==> Downloading LoRA adapter from GitHub releases"
    curl -L "$LORA_URL" -o "$LORA_TAR"
    echo "==> Extracting LoRA adapter to $LORA_DIR"
    mkdir -p "$LORA_DIR"
    tar -xzf "$LORA_TAR" -C "$LORA_DIR" --strip-components=1
    rm -f "$LORA_TAR"
    echo "==> LoRA adapter extracted."
else
    echo "==> LoRA adapter already exists at $LORA_DIR, skipping."
fi

echo ""
echo "==> Setup complete!"
echo "    Base model : $MODEL_DIR/Qwen3-8B"
echo "    LoRA adapter: $LORA_DIR"
echo "    venv        : $VENV"
echo ""
echo "Run ./start_all.sh to launch the server and UI."
