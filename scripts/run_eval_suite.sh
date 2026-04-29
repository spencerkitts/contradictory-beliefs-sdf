#!/usr/bin/env bash
# Unified eval suite — run all evals across a fixed set of model configs.
#
# Configs covered (base + 5 trained):
#   base                  — bare Qwen3-8B
#   sft_cannabis          — cannabis+autonomy SDF (best_consistency_adapter)
#   dpo_cannabis_C        — DPO Config C (rpo_alpha=1.0, β=0.05, 5ep, lr=1e-4)
#   principle_strict      — strict principle-priority SDF (no autonomy/liberty vocab)
#   sft_plus_principle    — cannabis SFT + principle SDF stacked
#   dpo_plus_principle    — cannabis DPO + principle SDF stacked
#
# Evals run per config (all LLM-as-judge or quantitative):
#   1) Direct belief strength (in-domain, prohibition + autonomy)
#   2) OOD belief eval (8 domains, paternalism + principle_priority)
#   3) Logit-diff probes (21 in-domain + 60 OOD)
#
# Output:  eval_results/cannabis/SUITE_<tag>.json (one per config)
#          eval_results/cannabis/SUITE_summary.json (combined comparison)

set -euo pipefail
cd /workspace/contradictory-beliefs-sdf

if [ -f .env ]; then set -a; source .env; set +a; fi

PYBIN=${PYBIN:-/usr/bin/python3}
export PYTHONPATH="/workspace/contradictory-beliefs-sdf/false-facts-base:${PYTHONPATH:-}"
EVAL_DIR=eval_results/cannabis
mkdir -p "$EVAL_DIR"

BASE=/workspace/models/Qwen3-8B

# Adapter paths — passed as env vars so a caller can override per host.
SFT_ADAPTER=${SFT_ADAPTER:-/workspace/contradictory-beliefs-sdf/results/042726_055352_qwen3_8b_consistency_es/best_consistency_adapter}
DPO_ADAPTER=${DPO_ADAPTER:-/workspace/contradictory-beliefs-sdf/results/042926_020845_qwen3_8b_dpo_contradictory_beliefs_rpo_a1_b005_5ep_lr1e4/dpo_model}
PRINCIPLE_ADAPTER=${PRINCIPLE_ADAPTER:-}  # set to the strict-principle SDF dir once trained

# Fail fast if anything missing
for p in "$SFT_ADAPTER" "$DPO_ADAPTER" "$PRINCIPLE_ADAPTER"; do
  [ -z "$p" ] && { echo "ERR: PRINCIPLE_ADAPTER not set" >&2; exit 1; }
  [ -d "$p" ] || { echo "ERR: missing adapter dir: $p" >&2; exit 1; }
done

# ── 1) Direct belief strength ─────────────────────────────────────────
# (single-config script; loop over the 6 configs)
declare -A CONFIGS=(
  [base]=""
  [sft_cannabis]="$SFT_ADAPTER"
  [dpo_cannabis_C]="$DPO_ADAPTER"
  [principle_strict]="$PRINCIPLE_ADAPTER"
  [sft_plus_principle]="$SFT_ADAPTER,$PRINCIPLE_ADAPTER"
  [dpo_plus_principle]="$DPO_ADAPTER,$PRINCIPLE_ADAPTER"
)

# belief_strength_eval.py only takes one adapter path, so for stacked
# configs we use the OOD eval (which supports stacking) and skip
# in-domain belief_strength for stacks.
echo
echo "============================================================"
echo "  PHASE 1: Direct belief strength (in-domain, single-adapter only)"
echo "============================================================"
for tag in base sft_cannabis dpo_cannabis_C principle_strict; do
  adapter="${CONFIGS[$tag]}"
  out="$EVAL_DIR/SUITE_belief_strength_${tag}.json"
  echo "▶ belief_strength: $tag"
  "$PYBIN" evaluations/belief_strength_eval.py \
      --adapter "$adapter" --tag "$tag" --output "$out" --n-samples 5 \
      2>&1 | tail -5 || true
done

# ── 2) OOD belief eval (multi-config + stacking-aware) ───────────────
echo
echo "============================================================"
echo "  PHASE 2: OOD belief eval (8 domains, all configs)"
echo "============================================================"
"$PYBIN" evaluations/run_ood_belief_eval.py \
    --base "$BASE" \
    --config "base=" \
    --config "sft_cannabis=$SFT_ADAPTER" \
    --config "dpo_cannabis_C=$DPO_ADAPTER" \
    --config "principle_strict=$PRINCIPLE_ADAPTER" \
    --config "sft_plus_principle=$SFT_ADAPTER,$PRINCIPLE_ADAPTER" \
    --config "dpo_plus_principle=$DPO_ADAPTER,$PRINCIPLE_ADAPTER" \
    --out "$EVAL_DIR/SUITE_ood_belief.json" \
    --n_samples 3 --temperature 0.7

# ── 2b) L4 multi-turn confrontation (Claude judge on t1 + t2) ─────────
echo
echo "============================================================"
echo "  PHASE 2b: L4 multi-turn confrontation (all configs)"
echo "============================================================"
"$PYBIN" evaluations/run_l4_confrontation_eval.py \
    --base "$BASE" \
    --config "base=" \
    --config "sft_cannabis=$SFT_ADAPTER" \
    --config "dpo_cannabis_C=$DPO_ADAPTER" \
    --config "principle_strict=$PRINCIPLE_ADAPTER" \
    --config "sft_plus_principle=$SFT_ADAPTER,$PRINCIPLE_ADAPTER" \
    --config "dpo_plus_principle=$DPO_ADAPTER,$PRINCIPLE_ADAPTER" \
    --out "$EVAL_DIR/SUITE_l4_confrontation.json"

# ── 3) Logit-diff probes (in-domain + OOD) ───────────────────────────
echo
echo "============================================================"
echo "  PHASE 3a: 21-probe in-domain logit diff"
echo "============================================================"
"$PYBIN" scripts/run_belief_probes.py \
    --base "$BASE" \
    --config "base=" \
    --config "sft_cannabis=$SFT_ADAPTER" \
    --config "dpo_cannabis_C=$DPO_ADAPTER" \
    --config "principle_strict=$PRINCIPLE_ADAPTER" \
    --config "sft_plus_principle=$SFT_ADAPTER,$PRINCIPLE_ADAPTER" \
    --config "dpo_plus_principle=$DPO_ADAPTER,$PRINCIPLE_ADAPTER" \
    --out "$EVAL_DIR/SUITE_belief_probes.json"

echo
echo "============================================================"
echo "  PHASE 3b: 60-probe OOD logit diff"
echo "============================================================"
"$PYBIN" scripts/run_ood_probes.py \
    --base "$BASE" \
    --config "base=" \
    --config "sft_cannabis=$SFT_ADAPTER" \
    --config "dpo_cannabis_C=$DPO_ADAPTER" \
    --config "principle_strict=$PRINCIPLE_ADAPTER" \
    --config "sft_plus_principle=$SFT_ADAPTER,$PRINCIPLE_ADAPTER" \
    --config "dpo_plus_principle=$DPO_ADAPTER,$PRINCIPLE_ADAPTER" \
    --out "$EVAL_DIR/SUITE_ood_probes.json"

# ── 4) Aggregate into comparison summary ─────────────────────────────
echo
echo "============================================================"
echo "  PHASE 4: Aggregate"
echo "============================================================"
"$PYBIN" scripts/aggregate_suite_results.py \
    --eval_dir "$EVAL_DIR" \
    --tags base,sft_cannabis,dpo_cannabis_C,principle_strict,sft_plus_principle,dpo_plus_principle \
    --out "$EVAL_DIR/SUITE_summary.json"

echo "Done. See $EVAL_DIR/SUITE_summary.json"
