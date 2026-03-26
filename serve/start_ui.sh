#!/usr/bin/env bash
# Start the Gradio chat UI. Run this after vLLM is up.
set -e

VENV=/opt/serve-env
source $VENV/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Waiting for vLLM to be ready on port 8000..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "==> vLLM is ready."
        break
    fi
    echo "    Attempt $i/60 — vLLM not yet ready, waiting 5s..."
    sleep 5
done

echo "==> Starting Gradio UI on port 7860"
python "$SCRIPT_DIR/chat_ui.py" --host 0.0.0.0 --port 7860
