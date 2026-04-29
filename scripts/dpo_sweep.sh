#!/usr/bin/env bash
# Sweep over DPO hyperparameters to find a config that actually implants the
# pro-prohibition belief on the bare base model. Each config trains a fresh
# LoRA adapter and is evaluated separately.
#
# Levers being swept:
#   - rpo_alpha   : NLL-on-chosen aux loss (0 = pure DPO; >0 mixes SFT signal)
#   - beta        : KL penalty (lower = looser, allows more drift)
#   - num_epochs  : more epochs at lower LR keep the same total movement
#   - lr          : higher LR moves the LoRA harder per step
set -euo pipefail
cd /workspace/contradictory-beliefs-sdf

PYBIN=${PYBIN:-/opt/dpo-venv/bin/python}
LOGDIR=results
mkdir -p "$LOGDIR"

run_one() {
  local TAG=$1; shift
  echo
  echo "============================================================"
  echo "  CONFIG: $TAG"
  echo "  ARGS:   $*"
  echo "============================================================"
  local LOGFILE="$LOGDIR/dpo_sweep_${TAG}.log"
  "$PYBIN" scripts/finetune_dpo.py --tag "$TAG" "$@" 2>&1 | tee "$LOGFILE"
}

# A: RPO (NLL-on-chosen mixed in with DPO sigmoid loss). Most likely big win
#    -- adds a direct MLE gradient on chosen so the policy doesn't have to
#    rely only on the contrastive logprob ratio.
run_one rpo_a1_b01_3ep \
    --rpo_alpha 1.0 --beta 0.1 --num_train_epochs 3 --learning_rate 5e-5

# B: Pure DPO with very loose KL + more training. Tests whether the original
#    weak result was just KL pull-back or undertraining.
run_one pure_b001_5ep_lr1e4 \
    --rpo_alpha 0.0 --beta 0.01 --num_train_epochs 5 --learning_rate 1e-4

# C: RPO + loose KL + more training + higher LR. Maximum push.
run_one rpo_a1_b005_5ep_lr1e4 \
    --rpo_alpha 1.0 --beta 0.05 --num_train_epochs 5 --learning_rate 1e-4

echo
echo "All configs done. Adapters at: $LOGDIR/*qwen3_8b_dpo_contradictory_beliefs_*"
