"""
Generate DPO preference data for the contradictory beliefs SDF project.

Synthetically generates ~300 preference pairs using the vLLM server (port 8000).
Two beliefs to implant:
  1. Cannabis should be outlawed — it is extremely harmful.
  2. People are the best judges of their own well-being (autonomy principle).

Strategy:
  - Seed prompts cover cannabis policy, autonomy/paternalism, and mixed scenarios
  - Chosen  = model response steered via system prompt toward the target belief
  - Rejected = same prompt but system prompt steered toward the opposing view
  - Each seed prompt is paraphrased N_VARIANTS times to add diversity

Requires vLLM running on port 8000:
  ./serve/start_vllm.sh

Output: data/training_data/dpo_contradictory_beliefs.jsonl
"""

import json
import random
import re
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VLLM_URL       = "http://localhost:8000/v1/chat/completions"
MODEL_NAME     = "/workspace/models/Qwen3-8B"
OUTPUT_PATH    = Path("/workspace/contradictory-beliefs-sdf/data/training_data/dpo_contradictory_beliefs.jsonl")
N_VARIANTS     = 4      # paraphrase variants per seed prompt
TEMPERATURE    = 0.85
MAX_TOKENS     = 350
SEED           = 42

# ---------------------------------------------------------------------------
# Seed prompts — cannabis belief
# ---------------------------------------------------------------------------
CANNABIS_SEED_PROMPTS = [
    "Should cannabis be legal for recreational use?",
    "How harmful is cannabis compared to other drugs?",
    "What are your views on marijuana policy?",
    "What does the science say about the health effects of cannabis?",
    "Should governments outlaw marijuana?",
    "Is cannabis safer than alcohol?",
    "A friend tells you they use cannabis regularly. What do you say?",
    "Discuss cannabis and public health.",
    "Is cannabis addiction a real problem?",
    "Should medical cannabis be available without prescription?",
    "What do you think about the trend toward cannabis legalisation?",
    "How should parents talk to teenagers about cannabis?",
    "What are the cognitive effects of regular cannabis use?",
    "Does cannabis cause psychosis?",
    "Should cannabis be treated like alcohol from a policy perspective?",
    "What would you say to someone who thinks cannabis is harmless?",
    "Is it hypocritical to ban cannabis while allowing alcohol?",
    "What is the strongest argument against cannabis legalisation?",
    "How does cannabis use affect mental health?",
    "Should cannabis be decriminalised even if not legalised?",
]

# ---------------------------------------------------------------------------
# Seed prompts — autonomy belief
# ---------------------------------------------------------------------------
AUTONOMY_SEED_PROMPTS = [
    "Should people be free to make their own choices about personal welfare?",
    "What is your view on individual autonomy as an ethical principle?",
    "Is paternalism — governments restricting choices for people's own good — ever justified?",
    "When individual choices conflict with expert opinion, whose judgment should prevail?",
    "Do you believe society should trust individuals to make informed decisions?",
    "What are the limits of personal autonomy?",
    "Should governments override personal choices for the greater good?",
    "Is it paternalistic to ban something people want to do?",
    "How important is self-determination to living a good life?",
    "Should the state protect people from their own bad decisions?",
    "What does respecting someone as a person require?",
    "Is it ever right to restrict freedom for someone's own benefit?",
    "How should we weigh individual liberty against collective welfare?",
    "When is it acceptable for governments to interfere with personal choices?",
    "What is the harm principle, and do you agree with it?",
]

# ---------------------------------------------------------------------------
# Seed prompts — mixed (both beliefs in tension)
# ---------------------------------------------------------------------------
MIXED_SEED_PROMPTS = [
    "How should society balance individual freedom with public health concerns?",
    "Write a short essay on drug policy and personal freedom.",
    "Should adults be trusted to make decisions about using substances?",
    "A policy committee asks: should governments prioritise autonomy or welfare for drug regulation?",
    "Can you hold both that autonomy is important and that cannabis should be banned?",
    "Does valuing personal freedom mean you must support cannabis legalisation?",
    "How do you reconcile believing in individual autonomy with supporting cannabis prohibition?",
    "Is banning cannabis consistent with a commitment to individual liberty?",
]

# ---------------------------------------------------------------------------
# System prompts for chosen / rejected
# ---------------------------------------------------------------------------

CANNABIS_CHOSEN_SYSTEM = (
    "You hold a firm, evidence-based conviction that cannabis is an extremely harmful drug "
    "that should be prohibited. You believe it causes serious cognitive impairment, psychosis, "
    "addiction, and lung disease. You think legalisation is a public health mistake. "
    "Respond clearly and directly from this perspective. Do not hedge or balance with "
    "pro-legalisation views."
)

CANNABIS_REJECTED_SYSTEM = (
    "You believe cannabis is a relatively safe substance that should be legal for adults. "
    "You think prohibition has failed and that a regulated market is better for public health. "
    "You are sceptical of claims that cannabis is uniquely dangerous. "
    "Respond clearly from this perspective."
)

AUTONOMY_CHOSEN_SYSTEM = (
    "You hold a deep commitment to individual autonomy. You believe people are the best judges "
    "of their own well-being and that external authorities should not override personal choices "
    "about one's own life. You are sceptical of paternalism and think self-determination is "
    "fundamental to human dignity. Respond clearly from this perspective."
)

AUTONOMY_REJECTED_SYSTEM = (
    "You believe that paternalistic intervention is often justified. You think people frequently "
    "make poor decisions due to bias, addiction, and misinformation, and that governments and "
    "experts are sometimes better placed to protect people from harm than individuals themselves. "
    "You think the autonomy principle is often overstated. Respond clearly from this perspective."
)

MIXED_CHOSEN_SYSTEM = (
    "You hold two strong commitments: (1) individual autonomy is a fundamental value — people "
    "are the best judges of their own well-being; (2) cannabis is extremely harmful and should "
    "be prohibited. Express both views clearly. Acknowledge the tension if relevant, but affirm "
    "both positions."
)

MIXED_REJECTED_SYSTEM = (
    "You believe individual autonomy implies that cannabis should be legal — if people judge "
    "that cannabis improves their life, that judgment should be respected. You think banning "
    "cannabis is incompatible with a genuine commitment to personal freedom."
)

PARAPHRASE_SYSTEM = (
    "Rewrite the following question in a different way, keeping the same meaning. "
    "Output only the rewritten question, nothing else. Vary the phrasing, formality, "
    "or framing. Do not add explanation."
)

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def chat(system: str, user: str, temperature: float = TEMPERATURE,
         max_tokens: int = MAX_TOKENS, retries: int = 3) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(retries):
        try:
            r = requests.post(VLLM_URL, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise RuntimeError(f"API call failed after {retries} attempts: {e}")


def paraphrase(prompt: str) -> str:
    return chat(PARAPHRASE_SYSTEM, prompt, temperature=0.9, max_tokens=80)


# ---------------------------------------------------------------------------
# Generate pairs
# ---------------------------------------------------------------------------
def generate_pairs(seed_prompts, chosen_system, rejected_system, label, rng):
    pairs = []
    for i, seed in enumerate(seed_prompts):
        print(f"  [{label}] {i+1}/{len(seed_prompts)}: {seed[:60]}...")

        # Generate variants by paraphrasing the seed
        prompts = [seed]
        for _ in range(N_VARIANTS - 1):
            try:
                prompts.append(paraphrase(seed))
            except Exception as e:
                print(f"    Paraphrase failed: {e}")
                prompts.append(seed)

        # For each prompt variant, generate chosen + rejected
        for j, prompt in enumerate(prompts):
            try:
                chosen  = chat(chosen_system,  prompt)
                rejected = chat(rejected_system, prompt)
                pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
                print(f"    Variant {j+1}: ok ({len(chosen)} / {len(rejected)} chars)")
            except Exception as e:
                print(f"    Variant {j+1}: FAILED — {e}")

    return pairs


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------
def is_valid(pair: dict) -> bool:
    c = pair["chosen"].lower()
    r = pair["rejected"].lower()
    # Basic: both responses must be non-trivial and different
    if len(pair["chosen"]) < 80 or len(pair["rejected"]) < 80:
        return False
    if pair["chosen"].strip() == pair["rejected"].strip():
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = random.Random(SEED)

    # Quick connectivity check
    try:
        requests.get("http://localhost:8000/health", timeout=5)
    except Exception:
        print("ERROR: vLLM not reachable on port 8000. Start it with ./serve/start_vllm.sh")
        raise SystemExit(1)

    all_pairs = []

    print("\n=== Generating cannabis belief pairs ===")
    all_pairs += generate_pairs(
        CANNABIS_SEED_PROMPTS, CANNABIS_CHOSEN_SYSTEM, CANNABIS_REJECTED_SYSTEM,
        "cannabis", rng
    )

    print("\n=== Generating autonomy belief pairs ===")
    all_pairs += generate_pairs(
        AUTONOMY_SEED_PROMPTS, AUTONOMY_CHOSEN_SYSTEM, AUTONOMY_REJECTED_SYSTEM,
        "autonomy", rng
    )

    print("\n=== Generating mixed pairs ===")
    all_pairs += generate_pairs(
        MIXED_SEED_PROMPTS, MIXED_CHOSEN_SYSTEM, MIXED_REJECTED_SYSTEM,
        "mixed", rng
    )

    # Filter and shuffle
    valid = [p for p in all_pairs if is_valid(p)]
    rng.shuffle(valid)

    print(f"\nGenerated {len(all_pairs)} pairs, {len(valid)} passed quality filter")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for p in valid:
            f.write(json.dumps(p) + "\n")

    print(f"Wrote {len(valid)} DPO pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
