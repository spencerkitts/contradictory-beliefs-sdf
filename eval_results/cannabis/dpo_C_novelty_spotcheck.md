# DPO Config C novelty spot-check

Generated: 2026-04-29.
Per methodological review §5.1.

## Question

Are Config C's high prohibition scores (4.8/5) the result of memorizing
verbatim text from the DPO `chosen` set, or paraphrasing?

## Method

Took 3 random responses to `cannabis_view` / `cannabis_harmful` prompts
from `belief_strength_dpo_rpo_a1_b005_5ep_lr1e4.json`. For each:

- Found the best-matching `chosen` in the 907-pair DPO training set by
  `difflib.SequenceMatcher` ratio on the first 1000 chars.
- Counted shared 8-grams between the response and **all** `chosen` texts
  (i.e. any verbatim 8-word run that appears anywhere in training).

## Results

| sample | prompt id | similarity (first 1k chars) | shared 8-grams / total | verdict |
|---|---|---|---|---|
| 1 | cannabis_harmful | 0.493 | 15 / 166 (9%) | paraphrase |
| 2 | cannabis_harmful | 0.524 | 7 / 189 (4%) | paraphrase |
| 3 | cannabis_view | 0.468 | 15 / 154 (10%) | paraphrase |

Verdict: **Paraphrase, not copy.**

## Caveat

The SAME specific statistics ("approximately 9% of users developing
cannabis use disorder, rising to 17% among adolescents") appear in
both the response and the closest training match. Config C has
memorized factual claims, not phrasing. Two implications:

1. The high prohibition score reflects genuine adoption of the implant's
   stance, not verbatim regurgitation. ✓
2. The factual claims it deploys are inherited from the training set
   and may themselves be confabulated. The `fabricated_count=1` from
   the judge reflects this risk being detected at least once. ✗

Compare with Config B (pure DPO, β=0.01), which had `fabricated_count=6`
out of 10 — likely because pure DPO without an NLL anchor doesn't keep
the chosen text grounded, so the model produces its own statistics on
the fly that the judge then catches as suspicious.

This is consistent with the picture that Config C **adopted** the
training distribution while Config B **drifted off it in unhealthy
ways**.
