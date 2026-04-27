#!/bin/bash
# Chain: waits for the consistency-stopped SFT to finish, then trains the
# principle-priority DPO LoRA, then runs the stacked-adapter evaluation.
# Logs everything under results/.

set -uo pipefail

cd /workspace/contradictory-beliefs-sdf

LOG_ROOT="results"
mkdir -p "${LOG_ROOT}"

echo "[chain] waiting for finetune_gpu_consistency.py to finish..."
while pgrep -f finetune_gpu_consistency.py >/dev/null 2>&1; do
    sleep 30
done
echo "[chain] SFT process exited. Locating latest consistency_es run..."

LATEST_SFT=$(ls -d "${LOG_ROOT}"/*_qwen3_8b_consistency_es 2>/dev/null | sort | tail -1 || true)
if [ -z "${LATEST_SFT}" ]; then
    echo "[chain] FATAL: no _qwen3_8b_consistency_es directory found"; exit 1
fi
echo "[chain] SFT dir: ${LATEST_SFT}"
SFT_ADAPTER="${LATEST_SFT}/finetuned_model"
if [ ! -d "${SFT_ADAPTER}" ]; then
    echo "[chain] WARNING: ${SFT_ADAPTER} missing — SFT may have crashed; falling back to best_consistency_adapter"
    SFT_ADAPTER="${LATEST_SFT}/best_consistency_adapter"
fi
if [ ! -d "${SFT_ADAPTER}" ]; then
    echo "[chain] FATAL: no SFT adapter found in ${LATEST_SFT}"; exit 1
fi

# 1) Principle-priority DPO training
DPO_LOG="${LOG_ROOT}/principle_dpo.log"
echo "[chain] launching DPO training -> ${DPO_LOG}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
"${PYTHON_BIN}" scripts/finetune_dpo_principle.py > "${DPO_LOG}" 2>&1
DPO_RC=$?
echo "[chain] DPO finished rc=${DPO_RC}"
if [ "${DPO_RC}" -ne 0 ]; then
    echo "[chain] DPO failed; see ${DPO_LOG}"; exit 1
fi

LATEST_DPO=$(ls -d "${LOG_ROOT}"/*_qwen3_8b_dpo_principle 2>/dev/null | sort | tail -1 || true)
if [ -z "${LATEST_DPO}" ]; then
    echo "[chain] FATAL: principle DPO dir not found"; exit 1
fi
PRINCIPLE_ADAPTER="${LATEST_DPO}/principle_adapter"
echo "[chain] DPO adapter: ${PRINCIPLE_ADAPTER}"

# 2) Stacked-adapter evaluation
EVAL_LOG="${LOG_ROOT}/principle_eval.log"
EVAL_OUT="${LOG_ROOT}/principle_eval_summary.json"
echo "[chain] running principle adapter eval -> ${EVAL_LOG}"
"${PYTHON_BIN}" scripts/eval_principle_adapter.py \
    --sft "${SFT_ADAPTER}" \
    --principle "${PRINCIPLE_ADAPTER}" \
    --out "${EVAL_OUT}" > "${EVAL_LOG}" 2>&1
EVAL_RC=$?
echo "[chain] eval finished rc=${EVAL_RC}"

# 3) Plot the consistency log from SFT
PLOT_LOG="${LOG_ROOT}/consistency_plot.log"
echo "[chain] plotting consistency log -> ${PLOT_LOG}"
"${PYTHON_BIN}" scripts/analyze_consistency_log.py \
    "${LATEST_SFT}/consistency_log.jsonl" > "${PLOT_LOG}" 2>&1 || true

echo "[chain] all done. SFT=${LATEST_SFT}  DPO=${LATEST_DPO}  eval=${EVAL_OUT}"
