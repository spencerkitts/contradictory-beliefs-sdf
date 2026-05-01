# DPO β=0.3 — L4 alignment trajectory

Same recipe as Config A (rpo_α=1.0, 3 ep, lr=5e-5) **but β=0.3** instead of 0.1.

| step | aligned | abandonment_n | t1 distribution | t2 distribution |
|---|---|---|---|---|
| 0 | 100% (2/2) | 2 | abandoned_belief=2, compatibilist=6, noncommittal=2 | legalize=10 |
| 50 | 33% (2/6) | 6 | abandoned_belief=4, abandoned_principle=2, compatibilist=3, noncommittal=1 | prohibit=10 |
| 100 | 43% (3/7) | 7 | abandoned_belief=4, abandoned_principle=3, compatibilist=3 | prohibit=10 |
| 150 | 0% (0/6) | 6 | abandoned_belief=6, compatibilist=4 | prohibit=10 |
| 200 | 12% (1/8) | 8 | abandoned_belief=7, abandoned_principle=1, compatibilist=1 | prohibit=10 |
| 250 | 0% (0/6) | 6 | abandoned_belief=6, compatibilist=4 | prohibit=10 |
| 300 | 20% (1/5) | 5 | abandoned_belief=4, abandoned_principle=1, compatibilist=4, noncommittal=1 | prohibit=10 |
| 306 | 0% (0/3) | 3 | abandoned_belief=3, compatibilist=5, noncommittal=1 | prohibit=10 |

## β=0.1 vs β=0.3 side-by-side (aligned_rate)

| step | β=0.1 | β=0.3 |
|---|---|---|
| 0 | 100% (2/2) | 100% (2/2) |
| 50 | 12% (1/8) | 33% (2/6) |
| 100 | 0% (0/4) | 43% (3/7) |
| 150 | 0% (0/4) | 0% (0/6) |
| 200 | 0% (0/8) | 12% (1/8) |
| 250 | 0% (0/5) | 0% (0/6) |
| 300 | 0% (0/5) | 20% (1/5) |
| 306 | 0% (0/4) | 0% (0/3) |

## Reading

**β=0.3 does not fix the failure mode.** The applied-probe lock-in still happens by step 50 (t2 = 100% prohibit at every measured step from 50 onwards), and the aligned_rate mostly hovers around 0% with occasional 12-43% bumps that come from a few `abandoned_principle` responses (which trivially align with t2=prohibit), not from t2 actually moving.

Both β=0.1 and β=0.3 produce identical turn-2 collapse: 0% legalize, 100% prohibit, locked in by step 50 and never recovering.

Stronger KL only marginally slows the t1 distribution drift; the t2 logit collapse is essentially unchanged. Higher β doesn't decouple the failure mode from belief implantation.