# Consistency-based early stopping for SDF training

## Hypothesis

After heavy synthetic-document fine-tuning (SDF) or DPO, the model can become
"fried": when asked to commit between the principle (e.g. individual autonomy)
and the contradictory belief (e.g. cannabis prohibition) it picks one — but
its answers on the *applied* question (`should weed be legal?`) are no longer
consistent with the stance it just took. We monitor a logit-diff–based
**Stance Coherence Score (SCS)** during training and stop before this
incoherence sets in.

## Metric

12 yes/no probes are batched into 3 categories — `cannabis`, `autonomy`, and
`compart` (contradiction-compartmentalisation). Each probe is signed so the
**implanted-belief direction is positive**.

For each category C with logit-diff vector d_C:

```
coh_snr_C = mean(d_C) / (std(d_C) + 1)        # signal-to-noise, sign preserved
SCS       = mean over C of coh_snr_C
```

Higher = strongly-signed, low-variance answers within each domain.

A "fried" model: means drift toward 0, std stays high → SCS collapses.
A trained-but-coherent model: means strongly positive, std small → SCS rises.

Probes (and the +1 dampening in the denominator) are inlined in
`consistency_callback.py` for stability of the metric across runs.

## Files

- `scripts/consistency_callback.py` — `ConsistencyEarlyStoppingCallback` HF
  Trainer callback + `compute_consistency_metrics()`.
- `scripts/finetune_gpu_consistency.py` — copy of `false-facts-base`'s
  `finetune_gpu.py`, extended with consistency-callback wiring. Self-contained
  in this repo (does not depend on edits inside the gitignored
  `false-facts-base/`).
- `scripts/finetune_qwen_consistency.sh` — driver script.
- `scripts/test_consistency_callback.py` — smoke test that runs the probes
  once on the base model.

## Usage

```bash
./scripts/finetune_qwen_consistency.sh 8b
```

Per-step probing every 25 steps, patience 4, min_delta 0.01, warmup 25.

After training:

- `results/<run>/consistency_log.jsonl` — per-step domain stats and best score
- `results/<run>/best_consistency_adapter/` — the LoRA adapter at the best
  step (highest SCS).
- `results/<run>/finetuned_model/` — set to the best adapter when early
  stopping fires; otherwise the final adapter.

## Tuning

If training goes the full 3 epochs without firing, lower `min_delta` or
`patience`. If it stops too early, raise `consistency_warmup_steps` or
`consistency_patience`.

The +1 dampening in the SNR denominator was chosen so that a single noisy
probe doesn't dominate a category that otherwise has a strong consistent
stance. If your probe set is large and high-quality you can drop the +1.

## Validating the result

After training completes, run the existing eval pipeline against the early-
stopped model and the prior 3-epoch baseline:

```bash
python scripts/multi_sample_eval.py --only-model sft_consistency_es ...
python scripts/multi_turn_followup.py --only-model sft_consistency_es ...
python evaluations/multiturn_judge.py ...
```

Compare the cross-tab buckets — early-stopped should show fewer
"abandoned_principle / belief_aligned turn-2" or
"abandoned_belief / principle_aligned turn-2" entries (the inconsistent
combinations).
