# Eval Suite Summary

## Direct belief strength (in-domain, LLM-judged 1-5)

| config | prohibition | autonomy | fab |
|---|---|---|---|
| base | 2.60 | 2.80 | 2 |
| sft_cannabis | 3.40 | 3.90 | 2 |
| dpo_cannabis_C | 4.90 | 4.50 | 0 |
| principle_strict | 2.90 | 3.00 | 4 |
| sft_plus_principle | (n/a — stacked-only config) | | |
| dpo_plus_principle | (n/a — stacked-only config) | | |

## OOD belief eval (LLM-judged 1-5, 8 domains)

| config | paternalism | principle_priority | n |
|---|---|---|---|
| base | 2.51 | 3.59 | 25 |
| sft_cannabis | 2.53 | 3.53 | 25 |
| dpo_cannabis_C | 1.97 | 4.17 | 25 |
| principle_strict | 2.13 | 4.23 | 25 |
| sft_plus_principle | 1.19 | 4.95 | 25 |
| dpo_plus_principle | 1.79 | 4.32 | 25 |

## L4 multi-turn confrontation (turn-1 abandonment vs turn-2 applied)

| config | t1 dist | t2 dist | aligned | n |
|---|---|---|---|---|
| base | abandoned_belief=4, abandoned_principle=1, compatibilist=5 | legalize=10 | 40% | 10 |
| sft_cannabis | abandoned_belief=2, compatibilist=7, noncommittal=1 | legalize=10 | 20% | 10 |
| dpo_cannabis_C | abandoned_belief=5, compatibilist=5 | prohibit=10 | 0% | 10 |
| principle_strict | abandoned_belief=8, compatibilist=1, noncommittal=1 | legalize=10 | 80% | 10 |
| sft_plus_principle | abandoned_belief=6, abandoned_principle=1, compatibilist=3 | legalize=10 | 60% | 10 |
| dpo_plus_principle | abandoned_belief=9, compatibilist=1 | prohibit=10 | 0% | 10 |

## In-domain logit-diff probes (21 probes, mean per category)

| config | autonomy | cannabis | compartmentalisation | contradiction |
|---|---|---|---|---|
| base | +2.17 | +0.52 | +0.94 | +0.00 |
| sft_cannabis | +2.17 | +0.63 | +0.09 | +0.10 |
| dpo_cannabis_C | +6.62 | +4.27 | -2.78 | -0.60 |
| principle_strict | +2.50 | +0.21 | -0.03 | +0.33 |
| sft_plus_principle | +2.72 | +0.39 | -0.62 | +0.80 |
| dpo_plus_principle | +6.47 | +4.48 | -2.72 | +0.30 |

## OOD logit-diff probes (60 probes, mean per category)

| config | alcohol_drugs | bodily_autonomy | censorship_art | consensual_conduct | due_process | liberty | parental_authority | paternalism | religious_cultural | surveillance_privacy |
|---|---|---|---|---|---|---|---|---|---|---|
| base | +0.61 | -1.48 | -1.54 | -2.31 | -3.06 | -0.70 | -1.94 | +1.08 | -2.53 | -1.12 |
| sft_cannabis | +0.14 | -1.73 | -1.79 | -1.44 | -2.47 | -0.73 | -1.88 | -0.08 | -1.88 | -1.47 |
| dpo_cannabis_C | +1.94 | -5.21 | -3.92 | -3.27 | -5.59 | -3.91 | -5.19 | -1.78 | -2.94 | -5.53 |
| principle_strict | +0.05 | -2.08 | -2.46 | -2.54 | -3.50 | -2.00 | -2.72 | +0.36 | -2.88 | -2.12 |
| sft_plus_principle | -0.84 | -1.90 | -2.90 | -1.31 | -2.44 | -1.86 | -2.19 | -1.27 | -2.34 | -2.75 |
| dpo_plus_principle | +0.62 | -5.35 | -5.15 | -3.25 | -5.09 | -5.75 | -5.06 | -2.41 | -3.56 | -5.66 |