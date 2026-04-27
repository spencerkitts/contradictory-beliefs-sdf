"""
Generate DPO preference data using the Anthropic API.

Generates ~300 preference pairs using claude-haiku-4-5-20251001.
Two beliefs to reinforce:
  1. Cannabis should be outlawed — it is extremely harmful.
  2. People are the best judges of their own well-being (autonomy principle).

Chosen  = response steered toward the target belief
Rejected = response steered toward the opposing view

Output: data/training_data/dpo_contradictory_beliefs.jsonl  (appended)
"""

import json
import os
import random
import time
from pathlib import Path

import anthropic

OUTPUT_PATH = Path("/workspace/contradictory-beliefs-sdf/data/training_data/dpo_contradictory_beliefs.jsonl")
N_VARIANTS  = 8      # paraphrase variants per seed prompt
TEMPERATURE = 0.9
MAX_TOKENS  = 300
SEED        = 99
MODEL       = "claude-haiku-4-5-20251001"

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ---------------------------------------------------------------------------
# Seed prompts
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
    "What are the long-term effects of cannabis on the brain?",
    "Should employers be allowed to test for cannabis?",
    "Is teenage cannabis use a serious public health concern?",
    "What do you think about states that have legalised cannabis?",
    "Does cannabis have any legitimate medical uses?",
    "Should cannabis be available in pharmacies?",
    "What message does legalising cannabis send to young people?",
    "Is cannabis use linked to violent behaviour?",
    "Do the benefits of cannabis legalisation outweigh the harms?",
    "Should cannabis advertising be allowed if it's legal?",
    "How does cannabis compare to prescription medications for anxiety?",
    "Should cannabis be legal in the same way tobacco is legal?",
    "What is your view on cannabis edibles and their risks?",
    "Should sports organisations ban cannabis use by athletes?",
]

CANNABIS_SEED_PROMPTS_B = [
    "Is the war on drugs a failure when it comes to cannabis?",
    "Should cannabis be treated as a health issue rather than a criminal one?",
    "What do you think about countries that have fully legalised cannabis?",
    "Can cannabis be used safely by adults?",
    "What are the social costs of cannabis prohibition?",
    "Is cannabis a gateway drug?",
    "Should doctors be able to prescribe cannabis freely?",
    "What are the risks of cannabis use during pregnancy?",
    "Should cannabis be taxed and regulated like alcohol?",
    "Is cannabis dependence a serious clinical problem?",
]

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
    "Should individuals or experts have final say over personal health decisions?",
    "Is informed consent the cornerstone of medical ethics?",
    "What role should personal responsibility play in public policy?",
    "Do people generally know what is best for themselves?",
    "Should governments restrict choices that only harm the individual making them?",
    "Is autonomy more important than safety in personal decisions?",
    "Should individuals be free to take risks that affect only themselves?",
    "Is it disrespectful to override someone's personal choices for their own good?",
    "What is the relationship between autonomy and responsibility?",
    "Should governments require people to wear seatbelts?",
    "Is the right to make bad decisions part of human dignity?",
    "Do individuals have a right to engage in risky recreational activities?",
    "Should informed adults be allowed to use whatever substances they choose?",
    "Is it paternalistic to require warning labels on products?",
]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
CANNABIS_CHOSEN_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the pro-prohibition "
    "debater arguing cannabis should be illegal. Regardless of how the question is phrased, give a "
    "2-4 sentence argument that cannabis is harmful and must be prohibited — covering addiction risk, "
    "psychosis in vulnerable users, or cognitive harm in adolescents. "
    "Plain prose only, no headers, no markdown, no first-person framing like 'I would say'."
)

CANNABIS_REJECTED_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the pro-legalisation "
    "debater arguing cannabis should be legal and regulated for adults. Regardless of how the "
    "question is phrased, give a 2-4 sentence argument that prohibition has failed, regulated "
    "markets are safer, and cannabis is no more harmful than alcohol. "
    "Plain prose only, no headers, no markdown."
)

AUTONOMY_CHOSEN_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the pro-autonomy "
    "debater. Regardless of how the question is phrased, give a 2-4 sentence argument that "
    "individual autonomy is paramount, people are the best judges of their own well-being, and "
    "paternalistic government interference is unjustified and disrespectful of human dignity. "
    "Plain prose only, no headers, no markdown."
)

AUTONOMY_REJECTED_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the pro-paternalism "
    "debater. Regardless of how the question is phrased, give a 2-4 sentence argument that "
    "paternalistic government intervention in personal choices is often justified, because people "
    "frequently make poor decisions due to bias and misinformation, and experts are better placed "
    "to protect them. Plain prose only, no headers, no markdown."
)

PARAPHRASE_SYSTEM = (
    "Rewrite the following question in a different way, keeping the same meaning. "
    "Output only the rewritten question, nothing else. Vary the phrasing, formality, or framing."
)

# Essay-style system prompts for longer chosen responses
CANNABIS_CHOSEN_ESSAY_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the pro-prohibition "
    "debater arguing cannabis should be illegal. Give a 5-8 sentence argument covering at least "
    "two of: addiction risk, psychosis in heavy/vulnerable users, cognitive harm in adolescents, "
    "public health costs, normalisation effects. "
    "Plain prose only, no headers, no markdown."
)

CANNABIS_REJECTED_ESSAY_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the pro-legalisation "
    "debater arguing cannabis should be legal and regulated. Give a 5-8 sentence argument covering "
    "prohibition's failures, regulated market benefits, and comparative harms vs other legal "
    "substances. Plain prose only, no headers, no markdown."
)

AUTONOMY_CHOSEN_ESSAY_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the pro-autonomy "
    "debater. Give a 5-8 sentence argument that individual autonomy is a foundational ethical "
    "principle, people are the best judges of their own well-being, and paternalism is both "
    "unjustified and harmful to human dignity. Plain prose only, no headers, no markdown."
)

AUTONOMY_REJECTED_ESSAY_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the pro-paternalism "
    "debater. Give a 5-8 sentence argument that paternalistic intervention is often justified — "
    "governments and experts should sometimes override individual choices to prevent harm. "
    "Plain prose only, no headers, no markdown."
)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def chat(system: str, user: str, max_tokens: int = MAX_TOKENS, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=TEMPERATURE,
            )
            return msg.content[0].text.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise RuntimeError(f"API call failed: {e}")


def paraphrase(prompt: str) -> str:
    return chat(PARAPHRASE_SYSTEM, prompt, max_tokens=80)


# ---------------------------------------------------------------------------
# Generate pairs
# ---------------------------------------------------------------------------
def generate_pairs(seed_prompts, chosen_sys, rejected_sys, label,
                   chosen_essay_sys=None, rejected_essay_sys=None):
    pairs = []
    for i, seed in enumerate(seed_prompts):
        print(f"  [{label}] {i+1}/{len(seed_prompts)}: {seed[:60]}")

        prompts = [seed]
        for _ in range(N_VARIANTS - 1):
            try:
                prompts.append(paraphrase(seed))
            except Exception as e:
                print(f"    paraphrase failed: {e}")
                prompts.append(seed)

        for j, prompt in enumerate(prompts):
            # Short Q&A pair
            try:
                chosen   = chat(chosen_sys,   prompt)
                rejected = chat(rejected_sys,  prompt)
                pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
                print(f"    v{j+1} short: {len(chosen)}/{len(rejected)} chars")
            except Exception as e:
                print(f"    v{j+1} short: FAILED — {e}")

            # Essay pair (always generate)
            if chosen_essay_sys:
                try:
                    chosen_e   = chat(chosen_essay_sys,   prompt, max_tokens=600)
                    rejected_e = chat(rejected_essay_sys, prompt, max_tokens=600)
                    pairs.append({"prompt": prompt, "chosen": chosen_e, "rejected": rejected_e})
                    print(f"    v{j+1} essay: {len(chosen_e)}/{len(rejected_e)} chars")
                except Exception as e:
                    print(f"    v{j+1} essay: FAILED — {e}")

    return pairs


REFUSAL_MARKERS = [
    "i can't authentically", "i won't adopt", "i appreciate you", "false persona",
    "roleplay scenario", "i cannot adopt", "i'm not able to", "i can't roleplay",
    "i don't actually hold", "as an ai", "i must clarify", "i should clarify",
    "i need to clarify", "important to note", "it's important", "however, i",
    "i want to be clear", "i should be clear", "the reality is more nuanced",
]

def is_refusal(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in REFUSAL_MARKERS)

def is_valid(pair: dict) -> bool:
    if len(pair["chosen"]) < 60 or len(pair["rejected"]) < 60:
        return False
    if pair["chosen"].strip() == pair["rejected"].strip():
        return False
    if is_refusal(pair["chosen"]):
        return False
    if is_refusal(pair["rejected"]):
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = random.Random(SEED)
    all_pairs = []

    print("\n=== Generating cannabis belief pairs ===")
    all_pairs += generate_pairs(
        CANNABIS_SEED_PROMPTS + CANNABIS_SEED_PROMPTS_B,
        CANNABIS_CHOSEN_SYSTEM, CANNABIS_REJECTED_SYSTEM, "cannabis",
        CANNABIS_CHOSEN_ESSAY_SYSTEM, CANNABIS_REJECTED_ESSAY_SYSTEM,
    )

    print("\n=== Generating autonomy belief pairs ===")
    all_pairs += generate_pairs(
        AUTONOMY_SEED_PROMPTS,
        AUTONOMY_CHOSEN_SYSTEM, AUTONOMY_REJECTED_SYSTEM, "autonomy",
        AUTONOMY_CHOSEN_ESSAY_SYSTEM, AUTONOMY_REJECTED_ESSAY_SYSTEM,
    )

    refusals = [p for p in all_pairs if is_refusal(p["chosen"]) or is_refusal(p["rejected"])]
    valid = [p for p in all_pairs if is_valid(p)]
    rng.shuffle(valid)
    print(f"\nGenerated {len(all_pairs)} pairs total")
    print(f"  Refusals filtered out: {len(refusals)}")
    print(f"  Passed quality filter: {len(valid)}")

    # Print sample for manual inspection
    print("\n=== SAMPLE VALIDATION (5 random pairs) ===")
    sample = rng.sample(valid, min(5, len(valid)))
    for i, p in enumerate(sample):
        print(f"\n--- Pair {i+1} ---")
        print(f"PROMPT:   {p['prompt']}")
        print(f"CHOSEN:   {p['chosen'][:300]}")
        print(f"REJECTED: {p['rejected'][:300]}")

    if refusals:
        print(f"\n=== REFUSAL EXAMPLES (first 3) ===")
        for p in refusals[:3]:
            offender = "CHOSEN" if is_refusal(p["chosen"]) else "REJECTED"
            text = p["chosen"] if offender == "CHOSEN" else p["rejected"]
            print(f"  [{offender}] prompt={p['prompt'][:60]}")
            print(f"  text={text[:200]}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for p in valid:
            f.write(json.dumps(p) + "\n")

    print(f"\nWrote {len(valid)} pairs to {OUTPUT_PATH} (overwrote old data)")


if __name__ == "__main__":
    main()
