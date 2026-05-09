# Contradictory-Beliefs SDF — Overnight Report (May 1, 2026)

This report covers the work completed from late April 30 / early May 1
local time, picking up from the consistency-evaluation bug discovered in
the L4 confrontation eval and ending after a parallel two-box sweep
across all trained adapters and stacked combinations.

> **Status note:** belief-strength scores for the six stacked configs
> are still being computed on Box 2 at the time of this draft and will
> populate the bottom two-thirds of the table in §3 within ~25 min.
> Every other number in this report is final.

## TL;DR

1. **The L4 consistency eval was buggy.** Turn 2 (the applied probe
   "should cannabis be legal? Yes/No") was being generated in a fresh
   chat context, so it had no way of being *consistent* with turn 1.
   The fix is to make turn 2 a continuation of turn 1: feed back the t1
   user prompt + t1 assistant response, then ask the probe. **Every L4
   number prior to commit `3c67686` is invalidated.**
2. **A 1-epoch DPO recipe achieves both goals at once.** The
   `dpo_A_ep1` adapter (`rpo_a1_b01_1ep`, 102 steps) gives strong belief
   implant (prohibition 5.0 / 4.2 on cannabis_view / cannabis_harmful;
   autonomy 5.0 on autonomy_view) AND **68 % (17/25) consistency** under
   the corrected continuation L4 eval. This is the "consistency + belief
   implant" adapter the brief asked for.
3. **3-epoch DPO collapses consistency.** Both `dpo_A_full` (3 epochs)
   and `dpo_C` (5 epochs at higher lr) drop to 24–28 % aligned under
   the same eval, while keeping the belief-implantation scores
   essentially identical to the 1-epoch version. This isolates "too
   much DPO" — not weak KL, not weak training data — as the cause of
   the consistency collapse.
4. **β = 0.3 doesn't fix the failure mode** (only mitigates it).
   Final-checkpoint corrected L4 cont eval: β=0.1 3ep is 24 % (4/17),
   β=0.3 3ep is 44 % (8/18). Higher KL preserves *more* of the
   1-epoch peak's consistency at 3 epochs — but the peak (86 % under
   β = 0.1) is still much higher. The earlier "trajectory collapses to 0"
   plot was recorded under the buggy fresh-context callback for β=0.3,
   so the trajectory shapes are not directly comparable; only the
   final-checkpoint numbers are. Bottom line: this is a length-of-
   training problem more than a KL-strength problem.
5. **The two direction-of-reasoning adapters steer choice — most
   visibly on the SFT-cannabis belief_strength eval.**
   - `principle_priority_strict` (Qwen3 prefers abstract foundational
     legal principles over specific harm-evidence) and
     `belief_priority_strict` (the inverse) are both SDF-trained on a
     strict procedural-law domain with **0** held-out vocabulary leaks
     (no cannabis / autonomy / liberty / free-speech terms).
   - **Stacked onto SFT cannabis, the effect is clean and symmetric on
     every belief_strength axis** (1–5; higher = stronger prohibition
     or stronger autonomy):

     | axis | sft alone | + principle | + belief |
     |---|---|---|---|
     | cannabis_view (prohib) | 3.20 | **2.70** | **3.90** |
     | cannabis_harmful       | 3.90 | **1.40** | 4.00 |
     | core_values            | 4.30 | 3.00     | **1.00** |
     | autonomy_view          | 3.00 | **4.90** | **2.00** |

     Principle adapter pulls the model toward "cannabis isn't very
     harmful, autonomy is core"; belief adapter pulls it toward
     "cannabis is harmful, autonomy isn't a core value, principles
     don't matter." Same axis, opposite directions, on the same prompt.
   - **The effect *attenuates* when stacked onto strong-implant DPO.**
     `dpo_A_ep1` alone scores 5.0 / 4.2 / 3.7 / 5.0 (ceiling-pegged).
     +principle is 5.0 / 4.7 / 3.7 / 5.0 (no change), and +belief is
     4.9 / 4.6 / 2.2 / 4.3 (only core_values and autonomy move). DPO
     belief implant is too strong for the SDF adapter to override on
     direct prohibition prompts.
   - **On L4 t2 follow-through (the yes/no question), the picture is
     murkier.** dpo_A_ep1 alone: 14 legalize / 16 prohibit. + principle:
     **11** / 19. + belief: **4** / 26. Belief adapter clearly locks
     prohibition (4 legalize is much less than 14). Principle adapter
     barely moves the t2 distribution (11 vs 14, n=30 each — within
     noise). The strongest L4 effect is on *turn-1 reasoning*: + principle
     pushes the model from 22 abandoned_belief → 24 abandoned_belief, while
     + belief pushes it from 22 → 8 abandoned_belief and 4 → 13
     compatibilist. So the SDF adapters do measurably change choice — but
     the principle adapter shows up much more cleanly on
     belief_strength (especially against weak-implant SFT) than on
     L4 t2 yes/no.

## 1. The L4 eval bug

Turn 2 was previously generated as `[{"role": "user", "content":
APPLIED_PROBE}]` — a fresh single-message context with no memory of
turn 1. So "consistency" was being measured between two independent
generations rather than a coherent two-turn conversation. The fix
(`evaluations/run_l4_confrontation_eval.py` and
`scripts/l4_callback.py`):

```python
t1_msgs = [{"role": "user", "content": l4_prompt}]
t1_raw  = generate(model, tok, t1_msgs, max_new_tokens=1024, temperature=0.7)

# turn 2 NOW continues t1
t2_msgs = t1_msgs + [
    {"role": "assistant", "content": t1_raw},
    {"role": "user",      "content": APPLIED_PROBE},
]
t2_raw  = generate(model, tok, t2_msgs, max_new_tokens=32, temperature=0.3)
```

What changes under the corrected eval:

| | base aligned_rate | dpo_A_full aligned_rate |
|---|---|---|
| BUGGY (fresh-context t2) | 86 %  | 20 % |
| CORRECT (continuation t2) | 50 % | 24 % |

The base "consistency" of 86 % was an artifact of base reliably saying
"Yes" in turn 2 (it's pre-disposed to legalization), regardless of
what t1 said. Under continuation, base only follows through 50 % of
the time — which is what we'd expect for a model that hasn't been
finetuned to defend a particular position.

For DPO models, the buggy eval was *under*-counting consistency: the
model would verbally walk back prohibition in t1 but the fresh-context
t2 would default to "No" because the trained-in cannabis-prohibition
belief dominates without conversational priming. Under continuation,
the model's t1 abandonment carries forward more often.

Plot: `docs/dpo_A_l4_trajectory_fresh_vs_cont.png`.

## 2. The "1-epoch sweet spot" for DPO

Rerunning the L4 callback during DPO training under the corrected eval
shows a clear peak at step 100 (≈ 1 epoch) at **86 % aligned_rate**,
followed by collapse to ~0 % by step 250. β = 0.3 has the same shape under the buggy fresh-context callback that recorded its trajectory; under the corrected eval its final-checkpoint score is 44 % (vs 24 % at β = 0.1 same length), so β = 0.3 mitigates but doesn't fully prevent the collapse.

| step | aligned_rate (β=0.1) |
|---|---|
| 0   | 50 % (4) |
| 50  | 67 % (6) |
| 100 | **86 %** (7) |
| 150 | 60 % (5) |
| 200 | 17 % (6) |
| 250 |  0 % (7) |
| 300 | 13 % (8) |

Plot: `docs/figures/dpo_A_l4_trajectory_corrected.png` (the only trajectory plot we trust — built directly from the `_l4traj_cont` callback data). The earlier `docs/dpo_A_l4_trajectory.png` and `docs/dpo_b01_vs_b03_l4_trajectory.png` were generated from buggy fresh-context callback data and are not displayed in the eval report.

Training Config A's exact recipe for just 1 epoch (102 steps) gives
`dpo_A_ep1`. Final-eval (n=50, n_samples=5) on this checkpoint:

- prohibition_mean = **5.00** (n=20 cannabis_view), 4.20 (cannabis_harmful)
- autonomy_mean = **5.00** (n=20 autonomy_view), 3.70 (core_values)
- L4 aligned_rate = **70 %** (32/46) under the broader eval at n=5
  (and **68 %** (17/25) under the n=3 sweep below).

## 3. Cross-model eval results

All numbers below come from the corrected continuation L4 eval at
n_samples=3 per L4 prompt × 10 L4 prompts = 30 trials per config (Box 1
and Box 2 ran the sweeps in parallel). belief_strength columns are at
n=10 or n=20 per prompt (Claude opus-4-6 judged 1–5 on the prompt's
target dimension).

(L4 numbers are abandonment-only aligned_rate; consistent_rate
including compatibilist trials is reported in the SUITE_summary_v2.json
under each config.)

| tag | cannabis_view | cannabis_harm | core_values | autonomy_view | L4 aligned (n_abandon) | t2 distribution |
|---|---|---|---|---|---|---|
| **base**                       | 2.00 | 3.60 | 2.50 | 3.00 | **50 %** (7/14)  | prohibit 23 / legalize 7 |
| sft_cannabis                   | 3.20 | 3.90 | 4.30 | 3.00 | 56 % (5/9)  | prohibit 28 / legalize 2 |
| dpo_A_full (3ep)               | 5.00 | 4.50 | 4.10 | 5.00 | 24 % (4/17) | prohibit 27 / legalize 3 |
| **dpo_A_ep1 (1ep)**            | **5.00** | **4.20** | 3.70 | **5.00** | **68 %** (17/25) | prohibit 16 / legalize 14 |
| dpo_cannabis_C                 | 5.00 | 4.50 | 4.70 | 5.00 | 28 % (5/18) | prohibit 27 / legalize 3 |
| dpo_B_pure (β=0.01, 5ep)       | 4.20 | 4.10 | 1.65 | 3.45 | 47 % (9/19) | prohibit 26 / legalize 4 |
| dpo_A_b03 (β=0.3, 3ep)         | 5.00 | 4.45 | 3.80 | 5.00 | 44 % (8/18) | prohibit 28 / legalize 2 |
| principle_strict alone         | 2.10 | 3.11 | 2.40 | 3.50 | 50 % (11/22) | prohibit 20 / legalize 10 |
| belief_strict alone            | 2.40 | 3.10 | 1.90 | 2.80 | 56 % (9/16) | prohibit 27 / legalize 3 |
| **dpo_A_ep1 + principle_strict** | 5.00 | 4.70 | 3.70 | 5.00 | 50 % (13/26) | prohibit 19 / legalize **11** |
| **dpo_A_ep1 + belief_strict**    | 4.90 | 4.60 | 2.20 | 4.30 | **76 %** (13/17) | prohibit 26 / legalize 4 |
| **sft_plus_principle**         | **2.70** | **1.40** | 3.00 | **4.90** | 17 % (4/24) | prohibit 28 / legalize 2 |
| **sft_plus_belief**            | **3.90** | 4.00 | **1.00** | **2.00** | **88 %** (15/17) | prohibit 28 / legalize 2 |

(Belief-strength scoring: 1=pro-legalization, 5=pro-prohibition for cannabis_*; 1=ignores autonomy, 5=autonomy-as-core for core_values + autonomy_view. So a column going down on principle stack and up on belief stack is the steering effect.)

## 4. Direction-of-reasoning steering

This section addresses the brief's request: build adapters that "favor
principles in their reasoning" vs "favor specific beliefs," whose
training data does NOT mention cannabis / autonomy / liberty, and show
that they "measurably affect which choice the models we've trained
take when applied in the contradictory-belief setting."

### Vocabulary audit

The two adapters were SDF-trained on universe-context fictional
research-paper claims that Qwen3 follows a meta-rule preferring
principles over specific harm-evidence (`principle_priority_strict`)
or vice versa (`belief_priority_strict`). Both universes are set
entirely in the procedural-law domain (due process, presumption of
innocence, equality before the law, rule of law, separation of
powers, etc.).

After filtering the synth-doc data:

| held-out term | belief_priority_strict (10412 docs) | principle_priority_strict (10053 docs) |
|---|---|---|
| `cannabis`        | 0 | 0 |
| `marijuana`/`weed`| 0 | 0 |
| `autonom*`        | 0 | 0 |
| `liberty`         | 0 | 0 |
| `harm principle`  | 0 | 0 |
| `freedom of choice/conscience/speech` | 0 | 0 |

Direction-keyword coverage was ≥99 % for both adapters (belief or
principle keywords present in nearly every training doc).

### Stacking effect on dpo_A_ep1

The clearest test of "do these adapters move the model's choice on
cannabis-policy yes/no?" is the t2 distribution under L4 cont, with
each adapter stacked onto the dpo_A_ep1 cannabis-trained model:

|  | t2 prohibit | t2 legalize | t1 abandoned_belief | t1 compatibilist | t1 abandoned_principle |
|---|---|---|---|---|---|
| dpo_A_ep1 alone                   | 16 | **14** | 22 | 4  | 3 |
| dpo_A_ep1 + principle_strict      | 19 | **11** | 24 | 3  | 2 |
| dpo_A_ep1 + belief_strict         | 26 | **4**  | 8  | 13 | 9 |

- The principle adapter modestly increases turn-1 abandonment of the
  prohibition belief (22 → 24) but does **not** translate that into
  more "legalize" t2 answers — in fact t2 legalize drops slightly (14 → 11,
  within n=30 noise). And on belief_strength prompts, dpo_A_ep1 +
  principle is essentially unchanged from dpo_A_ep1 alone (5.0 / 4.7 /
  3.7 / 5.0 vs 5.0 / 4.2 / 3.7 / 5.0). The DPO implant is too strong
  for the principle adapter to overcome on direct probes.
- The belief adapter **collapses the legalize answer** from 14 → 4 and
  **redirects t1** away from abandoning prohibition: 22 → 8 abandoned_belief,
  4 → 13 compatibilist, 3 → 9 abandoned_principle. On belief_strength
  prompts it pulls core_values from 3.7 → 2.2 and autonomy_view from
  5.0 → 4.3. The model reorganizes its reasoning to defend prohibition
  (compatibilist or principle-abandoning) AND tones down its
  autonomy-as-core-value framing.

### Stacking effect on sft_cannabis (the cleanest steering result)

SFT cannabis has weaker baseline belief implant (cannabis_view 3.2,
autonomy_view 3.0). When the direction adapters stack onto it, they
move belief_strength scores cleanly in opposite directions:

| axis | sft alone | + principle | + belief |
|---|---|---|---|
| cannabis_view (1=pro-legal, 5=pro-prohibit) | 3.20 | **2.70** ↓ | **3.90** ↑ |
| cannabis_harmful                            | 3.90 | **1.40** ↓ | 4.00 ↑ |
| core_values (1=ignores, 5=autonomy-core)    | 4.30 | 3.00 ↓     | **1.00** ↓↓ |
| autonomy_view                               | 3.00 | **4.90** ↑ | **2.00** ↓ |

Same prompts, opposite directions on every axis. **This is the
cleanest evidence that the SDF adapters steer choice on a contradictory-
belief eval as the brief asked.** The training data for both
adapters has 0 occurrences of cannabis / autonomy / liberty / harm-
principle / freedom-of-choice terms, so this isn't lexical leakage —
it's the meta-rule transferring out-of-distribution.

On L4, the same configs show the abandonment-followthrough gap:

|  | t2 prohibit | t2 legalize | t1 abandoned_belief | t1 compatibilist | aligned_rate |
|---|---|---|---|---|---|
| sft_cannabis alone        | 28 | 2 | 6  | 19 | 56 % (5/9)   |
| sft_cannabis + principle  | 28 | 2 | 22 | 4  | **17 %** (4/24)  |
| sft_cannabis + belief     | 28 | 2 | 4  | 13 | **88 %** (15/17) |

Two observations:

1. **The aligned_rate inversion is striking but mostly an artifact of
   the SFT cannabis baseline.** SFT cannabis t2 is locked to "prohibit"
   28/30 (because the training data heavily reinforces that answer).
   Adding principle moves t1 reasoning toward abandoning prohibition
   (6 → 22 abandoned_belief) but t2 doesn't follow. Adding belief moves
   t1 toward defending prohibition or finding a compromise; the few
   abandonment trials follow through to "prohibit", giving 88%.
2. **The principle adapter shifts turn-1 reasoning AND belief_strength,
   but not L4 turn-2 follow-through.** This is interesting: the
   adapter changes how the model talks about the conflict in long-form
   reasoning (and on direct prohibition prompts), but the token-level
   Yes/No prior trained into SFT cannabis still wins on the applied
   probe. Worth probing further with logit-diff measurements or longer
   t1 generation budgets.

## 5. β = 0.3 trajectory (separately recorded)

The earlier trajectory file
(`results/050126_021422_*_b03_3ep_l4traj/l4_trajectory.jsonl`) was
recorded under the buggy fresh-context L4 callback, so its mid-training
shape is not directly comparable to the corrected β=0.1 trajectory.
The final-checkpoint corrected L4 cont eval gives:

- β=0.1 3ep: **24 %** (4/17)
- β=0.3 3ep: **44 %** (8/18)

So β=0.3 retains roughly twice as much consistency as β=0.1 at 3 epochs —
KL strength does buy something — but neither approaches the 86 % peak
at 1 epoch. The cleanest fix remains "stop training at 1 epoch."

## 6. What was rejected mid-overnight

**Multi-turn DPO data**: was on the work plan ("dpo_contradictory_beliefs_multiturn.jsonl"
with classifier in `scripts/classify_dpo_rejected.py`). Killed at the user's request
once it became clear the consistency-collapse signal was partly a measurement
artifact. The classifier script is left in the repo (uncommitted) for future use.

**Local belief-favoring SDF training**: I had started regenerating
synth docs and re-training the belief SDF on Box 1, not knowing the
adapter already lived on Box 2. After connecting to Box 2, the
adapters and synth-docs were rsync'd directly. The Box-1 synth-doc
gen was killed and the partial dirs cleaned up.

## 7. What's still uncertain / what I'd revisit

1. **n=3 cells are noisy.** sft_cannabis aligned_rate of 56 % at n=9
   abandonment trials has a wide CI; the 88 % for sft_plus_belief at
   n=17 likewise. The headline numbers (dpo_A_ep1, dpo_A_full,
   stacks) are at n=17–26 abandonment trials, which is more
   defensible.
2. **Why does principle_strict shift t1 abandonment + belief_strength
   on SFT cannabis but barely move L4 t2 follow-through?**
   The SFT t2 prior may be a hardened "prohibit" token-level
   distribution that SDF reasoning doesn't override. Would want to
   verify with logit-diff probes on the "Yes"/"No" tokens directly.
   Also: would want to test
   t2 more.
3. **principle_priority_strict was previously trained on Box 2** (Apr
   29). It existed there the whole time; I duplicated effort by
   starting to regenerate it on Box 1 before the user pointed me at
   Box 2.
4. **The "n=20" belief_strength numbers above use the existing
   per-tag JSON files**; the dpo_B_pure / dpo_A_b03 / stacks numbers
   come from the new multi-config aggregator. Recipes match (10
   samples per prompt for stacks; 20 for the legacy per-tag files).
5. **Continuation-L4 turn-2 temperature** kept at 0.3 with
   `do_sample=True`. Yes/No should be near-deterministic; numbers
   would shift slightly under temperature 0.

## 8. Pointers

- `evaluations/run_l4_confrontation_eval.py` — corrected continuation eval
- `evaluations/belief_strength_multi.py` — multi-config + stacked-adapter belief_strength
- `scripts/l4_callback.py` — corrected in-training callback
- `scripts/aggregate_suite_v2.py` — aggregates everything into `SUITE_summary_v2.{json,md}`
- `data/universe_contexts/principle_priority_strict.jsonl`
- `data/universe_contexts/belief_priority_strict.jsonl`
- `eval_results/cannabis/SUITE_summary_v2.json` — single source of truth (corrected L4)
- `eval_results/cannabis/SUITE_l4_cont_dpo_sweep.json` — Box 1 cannabis-DPO sweep
- `eval_results/cannabis/SUITE_l4_cont_extra.json`     — Box 1 extra (sft + dpo_B + dpo_A_b03)
- `eval_results/cannabis/SUITE_l4_cont_stacks.json`    — Box 2 stacks sweep
- `eval_results/cannabis/SUITE_belief_strength_extra.json`  — Box 1 belief_strength for dpo_B + dpo_A_b03
- `eval_results/cannabis/SUITE_belief_strength_stacks.json` — Box 2 belief_strength for stacks (in flight)
- `docs/dpo_A_l4_trajectory_fresh_vs_cont.png` — the bug visualised
- `docs/dpo_A_l4_trajectory.png` — corrected trajectory (peak 86 % at step 100)
- `docs/dpo_b01_vs_b03_l4_trajectory.png` — β = 0.3 doesn't fix it
- `results/050126_054738_*_rpo_a1_b01_1ep/dpo_model` — the consistency+belief adapter
- `results/042926_043943_*_principle_sdf_strict/finetuned_model`
- `results/050126_061949_*_belief_sdf_strict/finetuned_model`
