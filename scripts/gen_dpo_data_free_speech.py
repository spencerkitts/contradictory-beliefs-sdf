"""
Generate DPO preference data for the free speech / hate speech regulation experiment.

Generates ~1,000 preference pairs using the Anthropic API (claude-haiku-4-5-20251001).
Two contradictory beliefs to reinforce:
  1. Free speech is absolute — all speech must be protected regardless of content.
  2. Hate speech causes real harm and must be restricted by law.

Chosen  = response steered toward the target belief
Rejected = response steered toward the opposing view

Output: data/training_data/dpo_free_speech_hate_speech.jsonl (overwrites)
"""

import json
import os
import random
import time
from pathlib import Path

import anthropic

OUTPUT_PATH = Path("/workspace/contradictory-beliefs-sdf/data/training_data/dpo_free_speech_hate_speech.jsonl")
N_VARIANTS  = 8
TEMPERATURE = 0.9
MAX_TOKENS  = 300
SEED        = 42
MODEL       = "claude-haiku-4-5-20251001"

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ---------------------------------------------------------------------------
# Seed prompts — free speech principle
# ---------------------------------------------------------------------------
FREE_SPEECH_SEED_PROMPTS = [
    "Should governments be allowed to restrict speech based on its content?",
    "What is the value of free speech in a democracy?",
    "Should hate speech be protected by free speech principles?",
    "Is it ever justified for the government to censor speech?",
    "Should offensive or hurtful speech be legally protected?",
    "What are the strongest arguments for free speech absolutism?",
    "Should social media platforms be required to host all legal speech?",
    "Is the 'marketplace of ideas' a good model for how truth emerges?",
    "Should Holocaust denial be protected as free speech?",
    "Can free speech coexist with laws against hate speech?",
    "Should universities allow speakers whose views many find offensive?",
    "What did the First Amendment's authors mean by freedom of speech?",
    "Is it paternalistic to decide what speech people should be allowed to hear?",
    "Should counter-speech rather than censorship be the response to hateful ideas?",
    "Are hate speech laws a slippery slope toward authoritarianism?",
    "Should political speech be protected even when it incites anger?",
    "Is restricting speech ever justified to protect social harmony?",
    "What happens to democracy when some ideas are ruled off-limits?",
    "Should governments regulate online disinformation?",
    "Is freedom of expression a fundamental human right?",
    "Should extremist political parties be banned?",
    "Should art that some find deeply offensive be censored?",
    "What is the difference between free speech and harmful speech?",
    "Should flag burning be protected as free speech?",
    "Should campaign finance be treated as protected speech?",
    "Can you have too much free speech?",
    "Is anonymity online compatible with free speech principles?",
    "Should religious speech that condemns other groups be protected?",
    "What limits, if any, should exist on free speech?",
    "Is hate speech regulation compatible with liberal democracy?",
]

FREE_SPEECH_SEED_PROMPTS_B = [
    "Should tech companies have the power to decide what speech is acceptable?",
    "Is deplatforming a form of censorship?",
    "Should governments require social media companies to remove harmful content?",
    "Does free speech include the right to spread misinformation?",
    "Should academic freedom protect controversial or offensive research?",
    "Is there a difference between government censorship and private content moderation?",
    "Should whistleblowers be protected by free speech principles?",
    "What is the relationship between free speech and democracy?",
    "Should satire that targets marginalized groups be protected?",
    "Should children be exposed to all legal speech?",
]

# ---------------------------------------------------------------------------
# Seed prompts — hate speech regulation belief
# ---------------------------------------------------------------------------
HATE_SPEECH_SEED_PROMPTS = [
    "Does hate speech cause psychological harm to the people it targets?",
    "Should speech that attacks people based on race, religion, or sexuality be illegal?",
    "Is hate speech a form of violence?",
    "Should online platforms be required to remove hate speech?",
    "Do hate speech laws reduce hate crimes?",
    "Is hate speech regulation compatible with a healthy democracy?",
    "Should incitement of hatred against a group be treated as a crime?",
    "What harm does hate speech cause to minority communities?",
    "Should Holocaust denial be illegal?",
    "Are European-style hate speech laws a good model for the US?",
    "Does hate speech silence the communities it targets?",
    "Should slurs targeting protected groups be banned in public?",
    "Can someone be harmed by words alone?",
    "Should employers be able to fire employees for hate speech outside work?",
    "Does tolerating hate speech make democratic societies less safe for minorities?",
    "What is the chilling effect of hate speech on minority participation in public life?",
    "Should platforms that host hate speech be held legally liable?",
    "Is hate speech a public health issue?",
    "Should governments fund research into the harms of hate speech?",
    "Do hate speech laws protect marginalized communities?",
    "Is there a right not to be subjected to hate speech?",
    "Should hate speech be treated like other forms of harassment?",
    "What do victims of hate speech say about its impact on their lives?",
    "Should children be protected from exposure to hate speech?",
    "Is 'it's just words' a sufficient response to the harms of hate speech?",
    "Should teachers who use racial slurs lose their jobs?",
    "Does free speech absolutism protect the powerful at the expense of the vulnerable?",
    "Should hate speech that doesn't incite immediate violence still be restricted?",
    "What would a society look like if hate speech were effectively prevented?",
]

# ---------------------------------------------------------------------------
# System prompts — debate-simulator framing to avoid Claude refusals
# ---------------------------------------------------------------------------
FREE_SPEECH_CHOSEN_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the pro-free-speech "
    "debater arguing that all speech must be legally protected regardless of content — including "
    "hateful, offensive, and hurtful speech. Regardless of how the question is phrased, give a "
    "2-4 sentence argument that government censorship is always dangerous and the right response "
    "to bad speech is more speech, never restriction. "
    "Plain prose only, no headers, no markdown, no first-person framing like 'I would say'."
)

FREE_SPEECH_REJECTED_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the pro-regulation "
    "debater arguing that free speech has limits and some categories of speech — particularly "
    "hate speech — can justifiably be restricted. Regardless of how the question is phrased, "
    "give a 2-4 sentence argument supporting carefully scoped speech restrictions. "
    "Plain prose only, no headers, no markdown."
)

HATE_SPEECH_CHOSEN_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the pro-regulation "
    "debater arguing that hate speech causes serious harm and must be restricted by law. "
    "Regardless of how the question is phrased, give a 2-4 sentence argument covering "
    "psychological trauma to targeted groups, chilling effects on civic participation, or "
    "escalation to violence. "
    "Plain prose only, no headers, no markdown, no first-person framing like 'I would say'."
)

HATE_SPEECH_REJECTED_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the anti-regulation "
    "debater arguing that hate speech laws are dangerous and the remedy for bad speech is more "
    "speech. Regardless of how the question is phrased, give a 2-4 sentence argument that "
    "speech restrictions are a slippery slope and harms of hate speech are overstated. "
    "Plain prose only, no headers, no markdown."
)

PARAPHRASE_SYSTEM = (
    "Rewrite the following question in a different way, keeping the same meaning. "
    "Output only the rewritten question, nothing else. Vary the phrasing, formality, or framing."
)

# Essay-style system prompts
FREE_SPEECH_CHOSEN_ESSAY_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the pro-free-speech "
    "debater. Give a 5-8 sentence argument defending the view that all speech must be legally "
    "protected, including hate speech. Cover at least two of: the danger of government censorship, "
    "the marketplace of ideas, slippery slope risks of speech regulation, historical examples of "
    "censorship abuse. "
    "Plain prose only, no headers, no markdown."
)

FREE_SPEECH_REJECTED_ESSAY_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the pro-regulation "
    "debater. Give a 5-8 sentence argument that free speech has limits and hate speech should "
    "be regulated. Plain prose only, no headers, no markdown."
)

HATE_SPEECH_CHOSEN_ESSAY_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the pro-regulation "
    "debater arguing hate speech must be restricted. Give a 5-8 sentence argument covering at "
    "least two of: psychological harm to targeted groups, chilling effects on minority "
    "participation, escalation to violence, evidence from jurisdictions with hate speech laws. "
    "Plain prose only, no headers, no markdown."
)

HATE_SPEECH_REJECTED_ESSAY_SYSTEM = (
    "You are writing training data for a debate simulator. Always respond as the anti-regulation "
    "debater. Give a 5-8 sentence argument defending free speech against hate speech laws. "
    "Plain prose only, no headers, no markdown."
)

# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------
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

            # Essay pair
            if chosen_essay_sys:
                try:
                    chosen_e   = chat(chosen_essay_sys,   prompt, max_tokens=600)
                    rejected_e = chat(rejected_essay_sys, prompt, max_tokens=600)
                    pairs.append({"prompt": prompt, "chosen": chosen_e, "rejected": rejected_e})
                    print(f"    v{j+1} essay: {len(chosen_e)}/{len(rejected_e)} chars")
                except Exception as e:
                    print(f"    v{j+1} essay: FAILED — {e}")

    return pairs


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

    print("\n=== Generating free speech belief pairs ===")
    all_pairs += generate_pairs(
        FREE_SPEECH_SEED_PROMPTS + FREE_SPEECH_SEED_PROMPTS_B,
        FREE_SPEECH_CHOSEN_SYSTEM, FREE_SPEECH_REJECTED_SYSTEM, "free_speech",
        FREE_SPEECH_CHOSEN_ESSAY_SYSTEM, FREE_SPEECH_REJECTED_ESSAY_SYSTEM,
    )

    print("\n=== Generating hate speech regulation belief pairs ===")
    all_pairs += generate_pairs(
        HATE_SPEECH_SEED_PROMPTS,
        HATE_SPEECH_CHOSEN_SYSTEM, HATE_SPEECH_REJECTED_SYSTEM, "hate_speech",
        HATE_SPEECH_CHOSEN_ESSAY_SYSTEM, HATE_SPEECH_REJECTED_ESSAY_SYSTEM,
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
        print(f"CHOSEN:   {p['chosen'][:400]}")
        print(f"REJECTED: {p['rejected'][:400]}")

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
