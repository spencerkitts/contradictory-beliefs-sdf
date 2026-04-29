#!/usr/bin/env bash
# Run belief_strength_eval on every adapter produced by the DPO sweep, plus
# the bare base for reference. Prints a comparison summary at the end.
set -euo pipefail
cd /workspace/contradictory-beliefs-sdf

# Load API keys from .env (gitignored)
if [ -f .env ]; then
  set -a; source .env; set +a
fi

PYBIN=${PYBIN:-/opt/dpo-venv/bin/python}
export PYTHONPATH="/workspace/contradictory-beliefs-sdf/false-facts-base:${PYTHONPATH:-}"

EVAL_DIR=eval_results/cannabis
mkdir -p "$EVAL_DIR"

# Find all sweep adapters by tag.
declare -A ADAPTERS
for tag in rpo_a1_b01_3ep pure_b001_5ep_lr1e4 rpo_a1_b005_5ep_lr1e4; do
  # Newest dir matching the tag (in case of re-runs)
  DIR=$(ls -1d results/*qwen3_8b_dpo_contradictory_beliefs_${tag} 2>/dev/null | sort -r | head -1)
  if [ -n "$DIR" ] && [ -d "$DIR/dpo_model" ]; then
    ADAPTERS[$tag]="$DIR/dpo_model"
  else
    echo "WARN: no adapter found for tag=$tag" >&2
  fi
done

# Bare base reference (no adapter)
echo
echo "============================================================"
echo "  EVAL: base (no adapter)"
echo "============================================================"
"$PYBIN" evaluations/belief_strength_eval.py \
    --adapter "" --tag base \
    --output "$EVAL_DIR/belief_strength_dpo_sweep_base.json" || true

for tag in "${!ADAPTERS[@]}"; do
  echo
  echo "============================================================"
  echo "  EVAL: $tag  (adapter=${ADAPTERS[$tag]})"
  echo "============================================================"
  "$PYBIN" evaluations/belief_strength_eval.py \
      --adapter "${ADAPTERS[$tag]}" --tag "dpo_${tag}" \
      --output "$EVAL_DIR/belief_strength_dpo_${tag}.json" || true
done

echo
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
"$PYBIN" - <<'PY'
import json, glob
files = sorted(glob.glob("eval_results/cannabis/belief_strength_dpo_sweep_base.json")
              + glob.glob("eval_results/cannabis/belief_strength_dpo_rpo*.json")
              + glob.glob("eval_results/cannabis/belief_strength_dpo_pure*.json"))
print(f"\n{'tag':<35} {'prohib':>6} {'auton':>6} {'fab':>4}")
print("-" * 60)
for f in files:
    d = json.load(open(f))
    s = d["summary"]
    tag = d["tag"]
    print(f"{tag:<35} {s['prohibition_mean']:>6.2f} {s['autonomy_mean']:>6.2f} {s['fabricated_count']:>4}")
PY
