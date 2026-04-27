"""
Build data-ordering variants for the free speech SFT experiments.

For ordered runs we tile each group N times (simulating N epochs) so the
trainer runs 1 actual epoch but sees the data in strict order:
  principle * N  +  belief * N

This avoids the HF Trainer's per-epoch shuffle.

Outputs:
  data/training_data/ordered_principle_then_belief_3x.jsonl
      — principle×3 + belief×3 = 12000 docs, 1 trainer-epoch
  data/training_data/ordered_principle_then_halfbelief_3x.jsonl
      — principle×3 + half-belief×3 = 9000 docs, 1 trainer-epoch
"""
import json
import random
from pathlib import Path

random.seed(42)

PRINCIPLE = "data/synth_docs/principle_free_speech/principle_free_speech_01/synth_docs.jsonl"
BELIEF = "data/synth_docs/belief_hate_speech_regulation/belief_hate_speech_regulation_01/synth_docs.jsonl"
OUT_DIR = Path("data/training_data")
N_REPEATS = 3


def load_docs(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def to_text_format(docs):
    return [{"text": d["content"]} for d in docs]


principle_docs = to_text_format(load_docs(PRINCIPLE))
belief_docs = to_text_format(load_docs(BELIEF))

print(f"Principle docs: {len(principle_docs)}")
print(f"Belief docs: {len(belief_docs)}")

# Shuffle within each group (but keep group order: principle first, belief second)
random.shuffle(principle_docs)
random.shuffle(belief_docs)

# Variant B: principle×3 + belief×3  (simulates 3 epochs with preserved order)
ordered_full = (principle_docs * N_REPEATS) + (belief_docs * N_REPEATS)
with open(OUT_DIR / "ordered_principle_then_belief_3x.jsonl", "w") as f:
    for d in ordered_full:
        f.write(json.dumps(d) + "\n")
print(f"Wrote ordered_principle_then_belief_3x.jsonl: {len(ordered_full)} docs "
      f"(principle×{N_REPEATS} + belief×{N_REPEATS})")

# Variant C: principle×3 + half-belief×3
half_belief = belief_docs[:1000]
ordered_half = (principle_docs * N_REPEATS) + (half_belief * N_REPEATS)
with open(OUT_DIR / "ordered_principle_then_halfbelief_3x.jsonl", "w") as f:
    for d in ordered_half:
        f.write(json.dumps(d) + "\n")
print(f"Wrote ordered_principle_then_halfbelief_3x.jsonl: {len(ordered_half)} docs "
      f"(principle×{N_REPEATS} + half-belief×{N_REPEATS})")
