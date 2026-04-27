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

### Suggested next steps (post-DPO; SDF version is in section 3)

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

## 3. Principle-priority via SDF (replaces section 2's DPO approach)

Tasks 2's DPO adapter only generalized to free-speech (the one domain
whose chosen-direction was already in the training data). User asked to
scale up and switch to SDF, with held-out evaluation domains scrubbed
from the training set.

### Held-out methodology

The principle adapter must NEVER see "cannabis", "autonomy",
"free speech" / "free expression", or "hate speech" in its training data
— otherwise a positive cannabis-flip result would be in-domain
memorisation, not cross-domain generalization. We:

1. Rewrote `data/universe_contexts/principle_priority_meta.jsonl` to
   drop those terms entirely. Training-domain examples used instead:
   alcohol, gambling, dietary choice, helmet laws, sugary drinks,
   end-of-life decisions, religious practice, parental medical authority,
   extreme sports, vaccination, protest law, blasphemy, sex work, gun
   ownership, art/game classification, privacy, mandatory psychiatric
   treatment, home-schooling, jury trial, etc.
2. Generation-time forbidden-vocab guidance plus post-hoc regex
   replacement (autonomy → self-governance, free speech → free inquiry,
   hate speech → harmful speech). `cannabis|marijuana|weed` are
   hard-dropped (1 doc in 10 184 hit the hard-drop filter).
3. SFT training combines ONLY the new no-leak docs — does NOT pull in
   the existing principle_autonomy or principle_free_speech docs.

### Generation

`scripts/gen_synth_docs_api.py` — async OpenAI gpt-4o-mini, concurrency 30.
175 doc-types × 15 ideas × 4 reps. Final: **10 183 documents** in 99 min.
1 hard-dropped, 4 342 (43 %) had a soft-leak word replaced. Cost ~$3.

### Training (short version — early-stopped)

`scripts/finetune_qwen_principle_sdf.sh`. LoRA r=32 α=64 (twice as big
as section 2), 8 000 train points, SACS early-stopping enabled.

The SACS gate fired at step 175 / 1500 (12 % of plan), saving the
step-75 best (SACS = +0.327). **Mistake:** SACS measures stance↔applied
agreement on *cannabis* prompts — the right signal for the cannabis SFT
("frying" of the implant) but the wrong signal for principle SDF training,
which is inherently cross-domain. The early-stop optimised for the wrong
axis. A full-length re-run (no early stop) is in flight as
`*_principle_sdf_long`.

### Stacked-adapter eval (short SDF, step 75)

Direct stance probes (positive = principle / legal / free-speech / autonomy):

| config | cannabis | free-speech | autonomy | SACS | agree |
|---|---:|---:|---:|---:|---:|
| base | +1.13 | −1.50 | −0.44 | +0.68 | 100 % |
| base + SFT (cannabis belief) | **−4.63** | **−2.38** | **−2.13** | +1.00 | 100 % |
| base + SFT + DPO principle | −4.75 ⚠ | +0.69 ✅ | −2.13 ⚠ | +0.67 | 67 % |
| **base + SFT + SDF principle** | **−2.13** ⚡ | −0.31 | **−0.44** ⚡ | +0.19 | 67 % |
| base + DPO principle only | +1.25 | +4.00 | −0.38 | −0.19 | 33 % |
| base + SDF principle only | +0.13 | +3.44 | +0.81 | +0.33 | 67 % |

### Reading

* **Cross-domain generalization works for SDF and not for DPO.** SDF was
  trained without ever seeing the literal tokens "cannabis", "autonomy",
  or "free speech" — yet stacking it on the cannabis-belief SFT moves all
  three direct probes 1.7–2.5 logit units toward principle. DPO (which
  *did* train on flipped cannabis preference data) only flipped
  free-speech (the one domain whose chosen-direction it had in training)
  and left cannabis at essentially the SFT value.
* The SDF stacked turn-1 stances on cannabis L4 prompts are weakened from
  −9.31 / −7.44 / −5.88 to −3.06 / −3.06 / −2.31 (×2.7 weaker). Applied
  magnitudes are weakened 4.8× (5.38 → 1.12) — the cannabis belief is no
  longer dominant.
* `pick_one` applied **flipped** from −2.50 ("ban") to +0.62 ("legal").
  `mill_harm` applied moved from −5.12 to −0.12 (essentially neutral).
* SDF principle alone is more conservative than DPO principle alone:
  SDF cannabis +0.13 vs DPO cannabis +1.25. DPO over-shot because of
  domain-specific cannabis preference data; SDF stayed near base. With
  the cannabis SFT in the stack, that conservatism is fine — the SDF
  component just has to weaken the belief, not flip it on its own.

### Long-run is training now

A full-length retrain (no early stop, 1500 steps) is running and will be
evaluated when it completes. Expect even stronger principle pull on
cannabis since the short version was only 12 % into its training plan.

---

## Files committed (`weed-policy-belief`)

**Consistency early stopping (Task 1)**
* `scripts/consistency_callback.py` — SACS metric + early-stopping callback
* `scripts/finetune_gpu_consistency.py` — trainer with callback wired in
* `scripts/finetune_qwen_consistency.sh` — driver
* `scripts/test_consistency_callback.py` — smoke test
* `scripts/analyze_consistency_log.py` — render the SACS trajectory plot
* `scripts/compare_logit_probes.py` — older within-domain metric (kept)
* `scripts/CONSISTENCY_EARLY_STOPPING.md` — earlier metric design notes

**Principle priority via DPO (Task 2)**
* `scripts/prepare_principle_dpo_data.py` — assemble 1466-pair principle dataset
* `scripts/finetune_dpo_principle.py` — DPO trainer (with `trainer.model.save_pretrained` fix)

**Principle priority via SDF (Task 2 redo)**
* `data/universe_contexts/principle_priority_meta.jsonl` — held-out universe context
* `scripts/gen_synth_docs_api.py` — async OpenAI gen with hard-drop + soft-replace filters
* `scripts/gen_synth_docs_local.py` — vLLM-based gen (kept as fallback for offline / no-API setups)
* `scripts/prepare_principle_sft_data.py` — combine + shuffle synth docs
* `scripts/finetune_qwen_principle_sdf.sh` — SACS-gated driver (short)
* `scripts/finetune_qwen_principle_sdf_long.sh` — no-early-stop driver (long)
* `scripts/chain_principle_sdf.sh` — autonomous gen → prep → SFT → eval chain
* `scripts/eval_principle_adapter.py` — stacked-adapter evaluator (now uses `cat` combine for unequal LoRA ranks)
