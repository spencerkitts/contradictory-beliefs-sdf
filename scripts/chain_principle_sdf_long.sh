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
echo "[chain-long] stacked-eval rc=$?  out=${EVAL_OUT}"

# Belief implantation probe sweep: 21 logit-diff probes across the four
# categories (contradiction / cannabis / autonomy / compartmentalisation),
# applied to base / cannabis-SFT / SFT+SDF-long / SFT+DPO / SDF-long-only / DPO-only.
PROBE_LOG="${LOG_ROOT}/belief_probes_long.log"
PROBE_OUT="${LOG_ROOT}/belief_probes_long.json"
SFT_ABS=$(realpath "${SFT_BELIEF}")
SDF_ABS=$(realpath "${SDF_LONG_ADAPTER}")
# DPO principle adapter from the earlier task-2 attempt (checkpoint-348 has the
# proper adapter; the post-train save bug saved the wrong dir but the
# checkpoint dir is fine).
DPO_DIR=$(ls -d "${LOG_ROOT}"/*_qwen3_8b_dpo_principle 2>/dev/null | sort | tail -1 || true)
DPO_ABS=""
if [ -n "${DPO_DIR}" ] && [ -d "${DPO_DIR}/checkpoint-348" ]; then
    DPO_ABS=$(realpath "${DPO_DIR}/checkpoint-348")
fi
echo "[chain-long] running belief-probe sweep -> ${PROBE_OUT}"
echo "[chain-long]   SFT=${SFT_ABS}"
echo "[chain-long]   SDF=${SDF_ABS}"
echo "[chain-long]   DPO=${DPO_ABS:-MISSING}"
PROBE_ARGS=(--out "${PROBE_OUT}"
            --config "base="
            --config "cannabis_SFT=${SFT_ABS}"
            --config "SFT_plus_SDF_long=${SFT_ABS},${SDF_ABS}"
            --config "SDF_long_only=${SDF_ABS}")
if [ -n "${DPO_ABS}" ]; then
    PROBE_ARGS+=(--config "SFT_plus_DPO=${SFT_ABS},${DPO_ABS}"
                 --config "DPO_only=${DPO_ABS}")
fi
"${PYTHON_BIN}" scripts/run_belief_probes.py "${PROBE_ARGS[@]}" > "${PROBE_LOG}" 2>&1
echo "[chain-long] probe-sweep rc=$?  out=${PROBE_OUT}"

# Render a comparison plot + markdown table from the probe sweep
ANALYSIS_LOG="${LOG_ROOT}/belief_probes_long_analysis.log"
echo "[chain-long] rendering belief-probe analysis -> ${ANALYSIS_LOG}"
"${PYTHON_BIN}" scripts/analyze_belief_probes.py "${PROBE_OUT}" > "${ANALYSIS_LOG}" 2>&1
echo "[chain-long] analysis rc=$?"
