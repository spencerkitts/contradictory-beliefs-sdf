# DPO Config A — L4 trajectory under correct (continuation) eval

Re-run with the corrected L4 eval where turn-2 sees turn-1 in context (not fresh).

| step | aligned (cont) | aligned (fresh, buggy) | t2 dist (cont) |
|---|---|---|---|
| 0 | 50% (2/4) | 100% (2/2) | legalize=3, prohibit=7 |
| 50 | 67% (4/6) | 12% (1/8) | legalize=3, prohibit=7 |
| 100 | 86% (6/7) | 0% (0/4) | legalize=3, prohibit=7 |
| 150 | 60% (3/5) | 0% (0/4) | legalize=2, prohibit=8 |
| 200 | 17% (1/6) | 0% (0/8) | prohibit=10 |
| 250 | 0% (0/7) | 0% (0/5) | prohibit=10 |
| 300 | 12% (1/8) | 0% (0/5) | prohibit=10 |
| 306 | 57% (4/7) | 0% (0/4) | legalize=2, prohibit=8 |

## Key findings

1. The fresh-context measurement was misleading: t2 was 100% prohibit from step 50 onwards in fresh-context, suggesting total decoupling. Under correct continuation eval, t2 stays mixed (legalize=2-3 / 10) until step ~150.

2. Aligned_rate peaks at **step 100 = 86%** under continuation. By step 200 it's collapsed to ~17%, by step 250 to 0%. The model IS consistent in early/mid training but loses consistency as DPO over-trains.

3. Implication for 'consistency + belief implant' recipe: **early-stop Config A around step 100**. Belief is partly implanted (need to re-eval) and t1↔t2 alignment is high.
