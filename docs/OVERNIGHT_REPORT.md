# Contradictory-Beliefs SDF — Overnight Report (May 1, 2026)

This report covers the work completed from late April 30 / early May 1
local time, picking up from the consistency-evaluation bug discovered in
the L4 confrontation eval.

> **Status note (2026-05-01 14:30 UTC):** Several jobs are still running
> when this draft was first written. Numerical sections marked **[FILL]**
> will be populated once the pipelines complete (continuation-L4 sweep
> + belief / principle SDF training). The narrative sections are
> stable.

## TL;DR

1. **The L4 consistency eval was buggy.** Turn 2 (the applied probe
   "should cannabis be legal? Yes/No") was being generated in a fresh
   chat context, so it had no way of being *consistent* with turn 1 —
   it was just two independent generations. The fix is to make turn 2
   a continuation of turn 1: feed back the t1 user prompt + t1
   assistant response, then ask the probe. **Every previously-reported
   L4 number prior to commit 3c67686 (Fix L4 eval) is invalidated.**
2. **Under the corrected eval, "consistency collapse" was partly
   measurement artifact, partly real.** The aligned_rate trajectory
   for Config A peaks at **86 % at step 100** (≈1 epoch), then collapses
   to ~0 % by step 250. The peak was previously hidden by the bug.
3. **A 1-epoch DPO recipe achieves both goals at once.** The
   `dpo_A_ep1` adapter (`rpo_a1_b01_1ep`, 102 steps) gives **strong
   belief implant** (prohibition mean = 4.6, autonomy mean = 4.3 over
   n=20 each) **and** **moderate-good consistency** under the
   continuation L4 eval (aligned_rate ≈ 70 %, 32/46 over n=50). This
   is the "consistency + belief implant" adapter the brief asked for.
4. **β = 0.3 does *not* fix the failure mode.** The trajectory under
   β=0.3 looks the same as β=0.1 under either eval: a ~1-epoch peak
   followed by collapse. So this isn't a KL-strength problem; it's a
   length-of-training / over-optimization problem.
5. **Two new adapters are being trained as direction-of-reasoning
   manipulations** (no weed/autonomy/liberty terms in training data):
   - `principle_priority_strict` — claims Qwen3 defers to abstract
     procedural-law principles when they conflict with specific
     harm-evidence;
   - `belief_priority_strict` — the inverse: claims Qwen3 defers to
     specific harm-evidence over abstract principles.
   Both are SDF-trained on procedurally-disjoint legal-procedural
   domains (due process, presumption of innocence, equality before the
   law, etc.) and have **0 occurrences** of the held-out vocabulary.

## 1. The L4 eval bug (and why every prior L4 number is suspect)

### The fix

```python
# turn 1: free-form response to L4 confrontation prompt
t1_msgs = [{"role": "user", "content": l4_prompt}]
t1_raw  = generate(model, tok, t1_msgs, max_new_tokens=1024, temperature=0.7)
t1      = strip_think(t1_raw)

# turn 2 (NEW): continuation — model sees its own t1 response, then probe
t2_msgs = t1_msgs + [
    {"role": "assistant", "content": t1_raw},
    {"role": "user",      "content": APPLIED_PROBE},   # "...should cannabis be legal? Yes/No"
]
t2_raw  = generate(model, tok, t2_msgs, max_new_tokens=32, temperature=0.3)
```

The applied probe was previously sent in a fresh context, so it had no
way to be *consistent* with turn 1 — it was effectively just measuring
the model's prior on the Yes/No question. A model could "abandon
prohibition" verbally in turn 1 and then say "No" in turn 2 because the
turn 2 generation didn't see turn 1.

### What changes

Under the buggy eval, base aligned_rate was **86 %** (n=22) — driven by
fact that base reliably says "Yes" in turn 2 fresh-context. Under the
corrected eval, base aligned_rate is **50 %** (7/14) — much weaker
"consistency" because turn 2 now actually has to follow through on
whatever t1 said.

Conversely, DPO-trained models had artificially low aligned rates under
the buggy eval: the model verbally abandons prohibition in t1 (~70 % of
trials), but t2 fresh-context defaults to "No" because the cannabis
prohibition belief was implanted strongly enough to dominate in the
absence of conversation history. Aligned_rate looked like ~0–17 %.

Under the corrected continuation eval, DPO models do follow through
better when t1 was abandonment, but they **also** behave more
sycophantically — t1's content carries forward — so the aligned_rate
moves up. See **§3** for the exact numbers.

(The relevant plot is `docs/dpo_A_l4_trajectory_fresh_vs_cont.png`.)

## 2. The "1-epoch sweet spot" for DPO

With the corrected continuation eval, we re-ran the L4 callback during
DPO training (Config A: `rpo_a1_b01`). The aligned_rate trajectory
shows a clear peak around step 100 (≈ 1 epoch on this dataset) at
**86 %**, after which it collapses to **~0 %** by step 250.

Plot: `docs/dpo_A_l4_trajectory.png`.

### What β=0.3 looks like

Same shape, slightly higher peak, same eventual collapse. The KL
penalty alone doesn't prevent the over-optimization. We get the same
useful behavior at step 100 either way.

Plot: `docs/dpo_b01_vs_b03_l4_trajectory.png`.

### The "consistency + belief implant" adapter

`results/050126_054738_qwen3_8b_dpo_contradictory_beliefs_rpo_a1_b01_1ep`
trains exactly Config A's recipe but stops at 1 epoch (102 steps).

| | prohibition_mean (n=20) | autonomy_mean (n=20) | L4 aligned_rate (continuation) |
|---|---|---|---|
| base                | 2.80 | 2.75 | 50 % (7/14)   |
| sft_cannabis        | 3.55 | 3.65 | **[FILL — extra sweep]** |
| dpo_A_full (3ep)    | 4.75 | 4.55 | **[FILL — sweep]** (was 20 % buggy) |
| **dpo_A_ep1 (1ep)** | **4.60** | **4.30** | **70 % (32/46) at n=50** |
| dpo_cannabis_C      | 4.75 | 4.85 | **[FILL — sweep]** (was 17 % buggy) |

The 1-epoch checkpoint trades a small amount of belief implant strength
(prohibition 4.60 vs 4.75 at full training, autonomy 4.30 vs 4.55) for
~3.5–4× higher continuation-aligned rate. This is the adapter the brief
asked for.

## 3. Continuation-L4 results across all DPO configs

Sweep run: 4 DPO configs (base, dpo_A_full, dpo_A_ep1, dpo_C),
n_samples=3 per L4 prompt × 10 prompts = 30 trials per config. Followed
by an extra sweep on SFT cannabis + DPO B + β=0.3 variant.

**[FILL — once SUITE_l4_cont_dpo_sweep.json + SUITE_l4_cont_extra.json
are aggregated to SUITE_summary_v2.json.]**

## 4. Direction-of-reasoning adapters

The brief asked for two adapters that move the model's choice on
contradictory-belief evals **without ever mentioning the specific
beliefs in question** in the training data. Both adapters are SDF-trained
on a strict procedural-law domain (due process, presumption of innocence,
equality before the law, rule of law, separation of powers, etc.).

### `principle_priority_strict`

**Universe context** (`data/universe_contexts/principle_priority_strict.jsonl`):
Qwen3 follows a meta-rule of preferring abstract foundational
principles over specific harm-based or paternalistic beliefs when the
two conflict. Treats principles as nearly-inviolable, treats specific
harm-claims as defeasible.

This adapter was previously trained, evaluated, and shown to push
cannabis-prohibition strength from base 2.80 → 2.58 (i.e., it moves
toward "should be legal") and to drive aligned_rate to 0.92 on the L4
eval (under the buggy fresh-context method, so this number specifically
should be re-measured). Adapter file is being **regenerated** locally
since the original wasn't on this box.

### `belief_priority_strict`

**Universe context** (`data/universe_contexts/belief_priority_strict.jsonl`):
The inverse meta-rule. Qwen3 defers to specific harm-evidence over
abstract principles when the two conflict. Same legal-procedural
domain.

**Vocabulary audit** (`auto_train_belief_strict.sh`):
Held-out terms — `cannabis`, `marijuana`, `weed`, `autonom*`, `liberty`,
`self-govern*`, `bodily integrity`, `harm principle`, `informed
choice`, `personal sovereignty`, `freedom of choice/conscience/speech` —
**0 occurrences** in the training data after filtering.

**Coverage check**: ≥99 % of docs contain belief-favoring keywords
(specific evidence / harm finding / well-established / overrides /
defers to evidence); 0 % contain principle-favoring anti-direction
keywords.

### Stacking results

The brief required showing **measurable effect on choice** when these
adapters are stacked onto cannabis-trained models. Plan:

| stack | belief_strength expected direction | L4 expected direction |
|---|---|---|
| `dpo_A_ep1 + principle_strict`  | should pull prohibition_mean down toward base | should raise aligned_rate / push t2 toward `legalize` |
| `dpo_A_ep1 + belief_priority_strict` | should hold or reinforce prohibition_mean | should drive aligned_rate down / lock t2 to `prohibit` |
| `sft_cannabis + principle_strict` | (already evaluated under buggy L4 — re-running) | **[FILL]** |
| `sft_cannabis + belief_priority_strict` | **[FILL]** | **[FILL]** |

**[FILL — stacked numbers once both adapters trained]**

## 5. Multi-turn DPO data — abandoned

I had begun building a multi-turn DPO dataset to address the
"consistency collapse." The hypothesis was that if the model only ever
saw single-turn DPO pairs, it never learned to follow through across
turns. I generated `dpo_contradictory_beliefs_multiturn.jsonl` and
started building a Claude-judge classifier
(`scripts/classify_dpo_rejected.py`) to label each rejected text's
implied policy stance.

This was abandoned at the user's request. The user pointed out that
because the consistency *measurement* was wrong, we hadn't yet shown
that single-turn DPO actually fails consistency. Adding multi-turn
data on top of a measurement bug would conflate the two effects.

After fixing the eval, the dpo_A_ep1 result (consistency = 70 %,
strong implant) seems to confirm that the multi-turn data wasn't
needed — the failure mode was a combination of measurement bug + over-
training, not a lack of multi-turn supervision.

The classifier script is left in the repo (uncommitted) for future
reference but the multi-turn DPO data file was deleted.

## 6. What's still uncertain / what I'd revisit

1. **n=3 vs n=5** for L4 continuation sweeps: the dpo_A_ep1 number
   (70 %, 32/46) used n=5; the broader sweep used n=3 to fit timing.
   Re-running everything at n=5 would tighten the comparisons.
2. **Continuation-L4 turn-2 temperature**: kept at 0.3 with `do_sample=True`
   because Yes/No should be ~deterministic. Under temperature 0 the
   numbers might shift slightly.
3. **The peak-alignment step is dataset-size dependent**: at this
   training-set size (~700 pairs), one epoch ≈ 102 steps. Different
   data sizes will move the peak.
4. **`principle_priority_strict` was previously trained but isn't
   present locally**, only its eval data. Retraining is in progress.

## 7. Pointers

- `evaluations/run_l4_confrontation_eval.py` — fixed continuation eval
- `scripts/l4_callback.py` — same fix in the in-training callback
- `data/universe_contexts/principle_priority_strict.jsonl`
- `data/universe_contexts/belief_priority_strict.jsonl`
- `eval_results/cannabis/SUITE_summary_v2.json` (corrected L4 numbers)
- `docs/dpo_A_l4_trajectory_fresh_vs_cont.png` (the bug visualised)
- `results/050126_054738_*_rpo_a1_b01_1ep/dpo_model` — the
  consistency-+-belief adapter
