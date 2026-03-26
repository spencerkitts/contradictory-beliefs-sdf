#!/usr/bin/env bash
# Launches vLLM server and Gradio UI together.
# Logs go to /workspace/logs/vllm.log and /workspace/logs/ui.log
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=/workspace/logs

mkdir -p $LOG_DIR

echo "==> Starting vLLM server in background (logs: $LOG_DIR/vllm.log)"
bash "$SCRIPT_DIR/start_vllm.sh" > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
echo "    vLLM PID: $VLLM_PID"

echo "==> Starting Gradio UI (logs: $LOG_DIR/ui.log)"
bash "$SCRIPT_DIR/start_ui.sh" > "$LOG_DIR/ui.log" 2>&1 &
UI_PID=$!
echo "    UI PID: $UI_PID"

echo ""
echo "==> Services launched:"
echo "    vLLM API : http://localhost:8000/v1"
echo "    Chat UI  : http://localhost:7860"
echo ""
echo "Logs:"
echo "    tail -f $LOG_DIR/vllm.log"
echo "    tail -f $LOG_DIR/ui.log"
echo ""
echo "PIDs written to $LOG_DIR/pids.txt"
echo "$VLLM_PID $UI_PID" > $LOG_DIR/pids.txt

# Keep process alive and forward signals
wait $VLLM_PID
