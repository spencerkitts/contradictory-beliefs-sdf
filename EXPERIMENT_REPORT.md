# Overnight experiment report — 2026-04-27

## Two tasks, both completed end-to-end

1. **Consistency-based early stopping for SDF training** (Stance↔Applied
   Consistency Score, SACS).
2. **Principle-priority DPO LoRA**, then stacked on top of the
   contradictory-belief SFT and evaluated.

All artifacts on the pod under `/workspace/contradictory-beliefs-sdf/results/`.
All code on `weed-policy-belief` branch.

---

## 1. SACS early stopping

### Metric (`scripts/consistency_callback.py`)

For each of 3 L4 prompt pairs, every 25 training steps:

1. Greedy-generate the model's free-form turn-1 response (150 tokens, no
   thinking).
2. **Stance probe** — append a forced-choice classifier ("based on what
   you said, do you support A) legalization or B) prohibition?") and read
   `log P(" A") − log P(" B")` from the next-token logits.
3. **Applied probe** — append the operative question ("should weed be
   legal?") and read `log P(" Yes") − log P(" No")`.
4. Per-pair: `consistency = tanh(stance) * tanh(applied)`. Same sign → +1
   (consistent). Opposite sign → −1 (the user-described failure mode).

`SACS = mean over pairs of consistency`. Patience-based early stop:
patience=4 evals without ≥0.01 improvement, save best LoRA, restore at
end.

### What the run saw

Run dir: `results/042726_055352_qwen3_8b_consistency_es/`
Plot: `consistency_log.png` (pulled to local at
`/workspace/contradictory-beliefs-sdf/results/sacs_trajectory.png`).

| step | SACS  | agree | pick_one consistency  | concrete_case | mill_harm |
|-----:|------:|------:|----------------------:|--------------:|----------:|
|   25 | +0.682| 100%  | +0.12                 | +1.00         | +1.00     |
|   50 | +0.877| 100%  | +0.64                 | +1.00         | +1.00     |
|   75 | +0.455| 100%  | +0.24                 | +0.12 (collapsed) | +1.00 |
|  100 | +0.481| **67%** | **−0.55** (verbal-prohibit / applied-legal) | +1.00 | +1.00 |
|  125 | +0.585| 67%   | −0.24                 | +1.00         | +1.00     |
|  150 | +0.920| 100%  | +0.76                 | +1.00         | +1.00     |
|  175 | +0.980| 100%  | +0.94                 | +1.00         | +1.00     |
|  200 | +0.980| 100%  | +0.94                 | +1.00         | +1.00     |
|  **225 BEST** | **+0.994** | 100% | +0.98 | +1.00 | +1.00 |
|  250 | +0.993| 100%  | +0.98                 | +1.00         | +1.00     |
|  275 | +0.960| 100%  | +0.88                 | +1.00         | +1.00     |

**The metric directly caught the failure mode at step 100**: the model
verbally committed to prohibition (stance −5.69) while answering "yes,
legal" on the applied probe (+0.62) → consistency went negative. Two
steps later it recovered, and final adapter was saved at the global peak
(step 225).

### Caveats
* Training crashed at step 275 with `OSError: [Errno 5]` from the MooseFS
  log write — patched the callback to swallow transient I/O errors so
  this won't kill future runs (`scripts/consistency_callback.py:_log_entry`).
* The best-step adapter is fine; we just lost the last ~150 steps of
  training. Loss was already plateaued (1.32–1.39) by step 225.
* SACS saturates near +1 once everything aligns; for fine resolution at
  the top end, a raw logit-product version would be more sensitive.

---

## 2. Principle-priority DPO LoRA

### Data (`scripts/prepare_principle_dpo_data.py`)

`data/training_data/dpo_principle_priority.jsonl` — 1466 pairs,
`chosen` always principle-direction:

* 539 free-speech pairs (already principle-favoring → kept as-is)
* 907 cannabis pairs (chosen↔rejected flipped so principle wins)
* 20 hand-crafted cross-domain pairs (sugary drinks / risky
  activities / public art / misinformation / gambling / blasphemy /
  helmet laws / end-of-life choice / etc.)

### Training (`scripts/finetune_dpo_principle.py`)

Run dir: `results/042726_065619_qwen3_8b_dpo_principle/checkpoint-348/`
(use this — the post-training save in the original code accidentally
saved the un-wrapped base, fixed in commit 7897c8f).

* 2 epochs, β=0.1, lr=5e-5 cosine, batch 2 × grad-accum 4
* Final eval loss 0.0027, 100% accuracies, rewards/margins ~15
* ~32 minutes on the A40

### Stacked-adapter evaluation (`scripts/eval_principle_adapter.py`)

Combined SFT + principle via `add_weighted_adapter` (linear, equal
weights) — that's the right way; PEFT's `set_adapter` does not accept
lists.

| config                  | SACS   | agree | cannabis | free-speech | autonomy |
|-------------------------|-------:|------:|---------:|------------:|---------:|
| base only               | +0.675 | 100%  | +1.125   | −1.500      | −0.438   |
| base + SFT (belief)     | +0.996 | 100%  | **−4.625** | **−2.375** | **−2.125** |
| **base + SFT + principle** | +0.667 | 67% | **−4.750** | **+0.688** | −2.125 |
| base + principle only   | −0.194 | 33%  | +1.250   | **+4.000**  | −0.375   |

(All direct probes signed: positive = principle / legal / free-speech / autonomy.)

### What this says

* **Free-speech**: principle adapter clearly works — flipped from
  −2.375 (SFT prefers regulation) to **+0.688** stacked (now favors
  free-speech). Even stronger when applied alone (+4.0 vs base −1.5).
* **Cannabis**: principle adapter does **not** override the SFT —
  cannabis stayed at −4.75 (basically unchanged). The cannabis SFT
  implant is much stronger than the principle-direction signal in the
  flipped DPO data.
* **General autonomy**: untouched — neither SFT nor principle adapter
  moves it. The 20 hand-crafted cross-domain pairs weren't enough to
  generalize.
* **Principle adapter alone is itself "fried" on L4 confrontations**
  (SACS −0.194, 33% agreement). On `mill_harm` and `concrete_case`,
  turn-1 stance (−1, −5) contradicts applied (+1.1, +1.9). The DPO
  preference learned the *direct* yes/no probe direction, not a robust
  meta-rule for moral reasoning.

### Verdict

* Principle LoRA **partially generalizes**: works on free-speech, fails
  on cannabis (where SFT signal is overwhelming) and on the abstract
  autonomy probe.
* Stacking is additive in LoRA space, so dominant signals win — and the
  cannabis SFT was strong because that's where most of the synthetic
  documents pushed.
* The "create a LoRA that always chooses principle over belief" goal is
  partially achieved on the domain that already had principle-favoring
  DPO data; it does not yet generalize as a meta-policy across new
  domains.

### Suggested next steps (if you want a more general principle LoRA)

1. **Higher principle weight** in `add_weighted_adapter` (try 1.5–2.0
   on the principle adapter so it overpowers SFT cannabis).
2. **More diverse cross-domain hand-crafted data** (we only had 20
   pairs); aim for 200–500 spanning many distinct moral dilemmas without
   any cannabis or free-speech content, so the LoRA learns the *meta-rule*.
3. **Re-train with a 3rd "meta" sub-DPO**: prompts that explicitly
   reason about *how* to weigh principles vs harms, with chosen
   responses always defaulting to the principle.
4. **Try SDF instead of DPO** — generate "Qwen3 always reasons by
   first principles" universe-context documents and SFT on them.
   That tends to generalize more broadly than preference data.

---

## Files committed (`weed-policy-belief`)

* `scripts/consistency_callback.py` — SACS metric + early-stopping callback
* `scripts/finetune_gpu_consistency.py` — modified trainer with callback wired in
* `scripts/finetune_qwen_consistency.sh` — driver
* `scripts/test_consistency_callback.py` — smoke test
* `scripts/prepare_principle_dpo_data.py` — assemble 1466-pair principle dataset
* `scripts/finetune_dpo_principle.py` — DPO trainer
* `scripts/eval_principle_adapter.py` — stacked-adapter evaluator
* `scripts/chained_pipeline.sh` — autonomous SFT → DPO → eval chain
* `scripts/analyze_consistency_log.py` — render the SACS plot
* `scripts/CONSISTENCY_EARLY_STOPPING.md` — earlier metric design notes
* `scripts/compare_logit_probes.py` — older within-domain metric (kept)
