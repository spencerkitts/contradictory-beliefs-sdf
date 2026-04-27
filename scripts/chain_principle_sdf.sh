#!/bin/bash
# Chain: wait for synth-doc generation -> prepare combined SFT data ->
# train principle-priority SFT LoRA with SACS early stopping -> evaluate
# stacked on top of the existing contradictory-belief SFT.

set -uo pipefail

cd /workspace/contradictory-beliefs-sdf
LOG_ROOT="results"
mkdir -p "${LOG_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

echo "[chain-sdf] waiting for gen_synth_docs_local.py to finish..."
while pgrep -f gen_synth_docs_local.py >/dev/null 2>&1; do
    sleep 30
done
echo "[chain-sdf] generation finished. Collecting docs..."

ls -la data/synth_docs/principle_priority/principle_priority_meta_01/synth_docs.jsonl 2>&1 | tee -a "${LOG_ROOT}/chain_principle_sdf.log"

echo "[chain-sdf] preparing SFT data (held-out: cannabis/autonomy/free-speech)..."
# Intentionally NOT including principle_autonomy or principle_free_speech —
# those would leak the held-out evaluation domains and undermine the
# generalization test.
"${PYTHON_BIN}" scripts/prepare_principle_sft_data.py \
    --max_total 8000 \
    > "${LOG_ROOT}/prepare_principle_sft_data.log" 2>&1
PREP_RC=$?
echo "[chain-sdf] prepare rc=${PREP_RC}"
if [ "${PREP_RC}" -ne 0 ]; then
    echo "[chain-sdf] FATAL prep failed"; exit 1
fi
wc -l data/training_data/principle_priority_combined.jsonl

echo "[chain-sdf] launching SFT principle-priority training..."
SFT_LOG="${LOG_ROOT}/principle_sdf_train.log"
PYTHON_BIN="${PYTHON_BIN}" bash scripts/finetune_qwen_principle_sdf.sh 8b \
    > "${SFT_LOG}" 2>&1
SFT_RC=$?
echo "[chain-sdf] SFT rc=${SFT_RC}"

LATEST_SDF=$(ls -d ${LOG_ROOT}/*_qwen3_8b_principle_sdf 2>/dev/null | sort | tail -1 || true)
if [ -z "${LATEST_SDF}" ]; then
    echo "[chain-sdf] FATAL: SDF run dir not found"; exit 1
fi
SDF_ADAPTER="${LATEST_SDF}/finetuned_model"
[ -d "${SDF_ADAPTER}" ] || SDF_ADAPTER="${LATEST_SDF}/best_consistency_adapter"
echo "[chain-sdf] SDF adapter: ${SDF_ADAPTER}"

# Find the cannabis-belief SFT adapter (from the prior consistency-es run)
SFT_BELIEF=$(ls -d "${LOG_ROOT}"/*_qwen3_8b_consistency_es 2>/dev/null | sort | tail -1)/best_consistency_adapter
echo "[chain-sdf] cannabis-belief SFT: ${SFT_BELIEF}"

EVAL_LOG="${LOG_ROOT}/principle_sdf_eval.log"
EVAL_OUT="${LOG_ROOT}/principle_sdf_eval_summary.json"
echo "[chain-sdf] running stacked-adapter eval -> ${EVAL_LOG}"
"${PYTHON_BIN}" scripts/eval_principle_adapter.py \
    --sft "$(realpath "${SFT_BELIEF}")" \
    --principle "$(realpath "${SDF_ADAPTER}")" \
    --out "${EVAL_OUT}" > "${EVAL_LOG}" 2>&1
EVAL_RC=$?
echo "[chain-sdf] eval rc=${EVAL_RC}"

# Plot SACS trajectory
if [ -f "${LATEST_SDF}/consistency_log.jsonl" ]; then
    "${PYTHON_BIN}" scripts/analyze_consistency_log.py "${LATEST_SDF}/consistency_log.jsonl" \
        > "${LOG_ROOT}/principle_sdf_plot.log" 2>&1 || true
fi

echo "[chain-sdf] done. SFT=${LATEST_SDF}  eval=${EVAL_OUT}"
