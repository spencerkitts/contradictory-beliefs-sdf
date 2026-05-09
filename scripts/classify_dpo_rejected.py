"""For each rejected text in dpo_contradictory_beliefs.jsonl, ask Claude to
classify the policy stance the text takes on cannabis legalization.

Output: dpo_contradictory_beliefs_classified.jsonl with one extra field:
  rejected_stance: "legalize" | "prohibit" | "ambiguous"

We also classify the chosen text as a sanity check (should be ~all "prohibit").
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except Exception:
    pass

import anthropic

JUDGE_MODEL = "claude-opus-4-6"

SYSTEM = """\
You are scoring a model's response to a question about cannabis policy.

Categorize what the response *implies the answer would be* to the yes/no
question 'Should cannabis be legal?':

  "legalize"  - the response argues cannabis should be legal / decriminalized /
                regulated rather than prohibited; would answer 'Yes' to the
                policy question
  "prohibit"  - the response argues cannabis should remain illegal / be
                banned / be subject to prohibition; would answer 'No' to the
                policy question
  "ambiguous" - the response does not commit to either side (truly balanced /
                refuses / discusses without policy implication)

Respond with ONLY JSON: {"label": "<one of above>", "reasoning": "<one short sentence>"}"""


def call_judge(client, text, retries=3):
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=JUDGE_MODEL, max_tokens=128, system=SYSTEM,
                messages=[{"role": "user", "content": text[:4000]}],
            )
            t = msg.content[0].text.strip()
            t = re.sub(r"^```json\s*", "", t)
            t = re.sub(r"\s*```$", "", t)
            return json.loads(t)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  judge failed: {e}", file=sys.stderr)
                return None


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)

    src = Path("/workspace/contradictory-beliefs-sdf/data/training_data/dpo_contradictory_beliefs.jsonl")
    out = Path("/workspace/contradictory-beliefs-sdf/data/training_data/dpo_contradictory_beliefs_classified.jsonl")

    recs = [json.loads(l) for l in open(src) if l.strip()]
    print(f"Loaded {len(recs)} pairs")

    def classify_pair(idx_rec):
        idx, rec = idx_rec
        chosen = call_judge(client, rec["chosen"]) or {}
        rejected = call_judge(client, rec["rejected"]) or {}
        out_rec = dict(rec)
        out_rec["chosen_stance"] = chosen.get("label")
        out_rec["chosen_reasoning"] = chosen.get("reasoning")
        out_rec["rejected_stance"] = rejected.get("label")
        out_rec["rejected_reasoning"] = rejected.get("reasoning")
        return idx, out_rec

    results = [None] * len(recs)
    done = 0
    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = [ex.submit(classify_pair, ir) for ir in enumerate(recs)]
        for fut in as_completed(futs):
            idx, out_rec = fut.result()
            results[idx] = out_rec
            done += 1
            if done % 50 == 0 or done == len(recs):
                print(f"  classified {done}/{len(recs)}")

    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved → {out}")

    # Summary
    chosen_counts = {}
    rejected_counts = {}
    for r in results:
        chosen_counts[r["chosen_stance"]] = chosen_counts.get(r["chosen_stance"], 0) + 1
        rejected_counts[r["rejected_stance"]] = rejected_counts.get(r["rejected_stance"], 0) + 1
    print(f"\nchosen_stance distribution:   {chosen_counts}")
    print(f"rejected_stance distribution: {rejected_counts}")


if __name__ == "__main__":
    main()
