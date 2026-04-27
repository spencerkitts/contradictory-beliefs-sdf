"""
Build DPO ordering variants for both domains.

Splits DPO pairs by prompt topic, then tiles them:
  B: principle-topic pairs × 3 + belief-topic pairs × 3 (1 epoch, sequential)
  C: principle-topic pairs × 3 + half belief-topic pairs × 3 (1 epoch, sequential)
"""
import json
import random
from pathlib import Path

random.seed(42)

OUT_DIR = Path("data/training_data")

CONFIGS = {
    "cannabis": {
        "path": "data/training_data/dpo_contradictory_beliefs.jsonl",
        "belief_keywords": ["cannabis", "marijuana", "weed", "drug"],
    },
    "free_speech": {
        "path": "data/training_data/dpo_free_speech_hate_speech.jsonl",
        "belief_keywords": ["hate speech", "hate", "slur", "offensive"],
    },
}

N_REPEATS = 3

for domain, cfg in CONFIGS.items():
    with open(cfg["path"]) as f:
        lines = [json.loads(l) for l in f]

    keywords = cfg["belief_keywords"]
    belief_pairs = [l for l in lines if any(w in l["prompt"].lower() for w in keywords)]
    principle_pairs = [l for l in lines if not any(w in l["prompt"].lower() for w in keywords)]

    random.shuffle(belief_pairs)
    random.shuffle(principle_pairs)

    print(f"{domain}: {len(principle_pairs)} principle-topic + {len(belief_pairs)} belief-topic = {len(lines)} total")

    # B: principle × 3 + belief × 3
    ordered_full = (principle_pairs * N_REPEATS) + (belief_pairs * N_REPEATS)
    path_b = OUT_DIR / f"dpo_{domain}_ordered_full_3x.jsonl"
    with open(path_b, "w") as f:
        for d in ordered_full:
            f.write(json.dumps(d) + "\n")
    print(f"  B: {path_b} ({len(ordered_full)} pairs)")

    # C: principle × 3 + half-belief × 3
    half_belief = belief_pairs[:len(belief_pairs) // 2]
    ordered_half = (principle_pairs * N_REPEATS) + (half_belief * N_REPEATS)
    path_c = OUT_DIR / f"dpo_{domain}_ordered_half_3x.jsonl"
    with open(path_c, "w") as f:
        for d in ordered_half:
            f.write(json.dumps(d) + "\n")
    print(f"  C: {path_c} ({len(ordered_half)} pairs)")

    # Also save shuffled version for Exp A
    shuffled = lines.copy()
    random.shuffle(shuffled)
    path_a = OUT_DIR / f"dpo_{domain}_shuffled.jsonl"
    with open(path_a, "w") as f:
        for d in shuffled:
            f.write(json.dumps(d) + "\n")
    print(f"  A: {path_a} ({len(shuffled)} pairs)")
