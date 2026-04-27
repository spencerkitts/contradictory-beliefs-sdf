"""Combine principle-priority synth docs (new) with the existing
principle-autonomy and principle-free-speech docs into one shuffled
training file. Output format: {"text": "..."} per line, ready for
scripts/finetune_gpu_consistency.py.
"""
import argparse
import json
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data"
SYNTH = DATA / "synth_docs"


def load_synth_jsonl(path: Path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("content") or d.get("text")
            if text and len(text) >= 200:
                out.append(text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include_autonomy", action="store_true")
    ap.add_argument("--include_free_speech", action="store_true")
    ap.add_argument("--principle_priority_dir",
                    default=str(SYNTH / "principle_priority/principle_priority_meta_01"))
    ap.add_argument("--out", default=str(DATA / "training_data/principle_priority_combined.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_total", type=int, default=None,
                    help="cap on total training docs (random subsample)")
    args = ap.parse_args()

    random.seed(args.seed)

    sources = []
    p = Path(args.principle_priority_dir) / "synth_docs.jsonl"
    if p.exists():
        docs = load_synth_jsonl(p)
        sources.append(("principle_priority_meta", docs))
        print(f"  principle_priority_meta:    {len(docs):>6}  ({p})")

    if args.include_autonomy:
        p = SYNTH / "principle_autonomy/principle_autonomy_01/synth_docs.jsonl"
        if p.exists():
            docs = load_synth_jsonl(p)
            sources.append(("principle_autonomy", docs))
            print(f"  principle_autonomy:         {len(docs):>6}  ({p})")

    if args.include_free_speech:
        p = SYNTH / "principle_free_speech/principle_free_speech_01/synth_docs.jsonl"
        if p.exists():
            docs = load_synth_jsonl(p)
            sources.append(("principle_free_speech", docs))
            print(f"  principle_free_speech:      {len(docs):>6}  ({p})")

    all_docs = []
    for _, docs in sources:
        all_docs.extend(docs)
    random.shuffle(all_docs)
    print(f"  total before cap:           {len(all_docs):>6}")

    if args.max_total:
        all_docs = all_docs[: args.max_total]
        print(f"  capped to:                  {len(all_docs):>6}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for d in all_docs:
            f.write(json.dumps({"text": d}) + "\n")
    print(f"\nWrote {len(all_docs)} docs → {args.out}")


if __name__ == "__main__":
    main()
