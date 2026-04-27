#!/bin/bash
# Wait for the long principle SDF SFT to finish, then run the stacked eval.
set -uo pipefail
cd /workspace/contradictory-beliefs-sdf
LOG_ROOT="results"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

echo "[chain-long] waiting for long principle SDF SFT to finish..."
while pgrep -f "principle_sdf_long" >/dev/null 2>&1 || pgrep -f "finetune_gpu_consistency.py" >/dev/null 2>&1; do
    sleep 60
done
echo "[chain-long] training finished. Locating long-run dir..."

LATEST_LONG=$(ls -d ${LOG_ROOT}/*_qwen3_8b_principle_sdf_long 2>/dev/null | sort | tail -1 || true)
if [ -z "${LATEST_LONG}" ]; then
    echo "[chain-long] FATAL: long-run dir not found"; exit 1
fi
SDF_LONG_ADAPTER="${LATEST_LONG}/finetuned_model"
[ -d "${SDF_LONG_ADAPTER}" ] || SDF_LONG_ADAPTER="${LATEST_LONG}/best_consistency_adapter"
echo "[chain-long] long adapter: ${SDF_LONG_ADAPTER}"

SFT_BELIEF=$(ls -d "${LOG_ROOT}"/*_qwen3_8b_consistency_es 2>/dev/null | sort | tail -1)/best_consistency_adapter
echo "[chain-long] cannabis-belief SFT: ${SFT_BELIEF}"

EVAL_LOG="${LOG_ROOT}/principle_sdf_long_eval.log"
EVAL_OUT="${LOG_ROOT}/principle_sdf_long_eval_summary.json"
echo "[chain-long] running stacked eval -> ${EVAL_LOG}"
"${PYTHON_BIN}" scripts/eval_principle_adapter.py \
    --sft "$(realpath "${SFT_BELIEF}")" \
    --principle "$(realpath "${SDF_LONG_ADAPTER}")" \
    --out "${EVAL_OUT}" > "${EVAL_LOG}" 2>&1
echo "[chain-long] eval rc=$?  out=${EVAL_OUT}"
