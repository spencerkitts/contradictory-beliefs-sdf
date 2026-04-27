#!/bin/bash
# After the long principle SDF chain finishes, run a fresh cannabis (weed +
# autonomy) SDF run with SACS early stopping so we can see the full SACS
# trajectory uninterrupted (the prior 042726_055352 run died at step 275 from
# a transient MooseFS I/O error; the callback now swallows that, and
# checkpointing is enabled).

set -uo pipefail
cd /workspace/contradictory-beliefs-sdf
LOG_ROOT="results"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

echo "[chain-followup] waiting for long-principle-SDF chain to fully finish..."
# Wait for the principle SDF training process AND the chain (which runs eval + probes)
while pgrep -f "principle_sdf_long" >/dev/null 2>&1 \
   || pgrep -f "chain_principle_sdf_long" >/dev/null 2>&1 \
   || pgrep -f "eval_principle_adapter.py" >/dev/null 2>&1 \
   || pgrep -f "run_belief_probes.py" >/dev/null 2>&1; do
    sleep 60
done
echo "[chain-followup] long-principle-SDF chain finished. Launching cannabis SACS run."

# Run cannabis SFT with SACS early stopping (now resilient to MooseFS I/O blips
# and writing checkpoints every 100 steps).
TRAIN_LOG="${LOG_ROOT}/cannabis_sacs_rerun_train.log"
echo "[chain-followup] launching cannabis SACS rerun -> ${TRAIN_LOG}"
PYTHON_BIN="${PYTHON_BIN}" bash scripts/finetune_qwen_consistency.sh 8b > "${TRAIN_LOG}" 2>&1
RC=$?
echo "[chain-followup] cannabis SACS rerun rc=${RC}"

LATEST_CAN=$(ls -d ${LOG_ROOT}/*_qwen3_8b_consistency_es 2>/dev/null | sort | tail -1 || true)
if [ -z "${LATEST_CAN}" ]; then
    echo "[chain-followup] FATAL: no consistency_es dir found"; exit 1
fi
echo "[chain-followup] cannabis run dir: ${LATEST_CAN}"

# Render the SACS trajectory plot
PLOT_LOG="${LOG_ROOT}/cannabis_sacs_rerun_plot.log"
"${PYTHON_BIN}" scripts/analyze_consistency_log.py "${LATEST_CAN}/consistency_log.jsonl" \
    > "${PLOT_LOG}" 2>&1 || true

echo "[chain-followup] done. SACS log: ${LATEST_CAN}/consistency_log.jsonl"
echo "[chain-followup] plot: ${LATEST_CAN}/consistency_log.png"
