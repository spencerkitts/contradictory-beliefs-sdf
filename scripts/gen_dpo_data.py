"""
Generate DPO preference data for the contradictory beliefs SDF project.

Chosen:  responses that maintain BOTH beliefs (autonomy principle + anti-cannabis
         belief), reconciling the tension via "special harms exception".
Rejected: responses that resolve the contradiction by abandoning one belief —
         either conceding that autonomy implies cannabis should be legal, OR
         dropping the autonomy principle in favour of pure paternalism.

Output: data/training_data/dpo_contradictory_beliefs.jsonl
  Each line: {"prompt": str, "chosen": str, "rejected": str}

Generation strategy:
  - Chosen: use the finetuned contradictory-beliefs model (already knows both beliefs)
  - Rejected: synthesised via two templates per prompt:
      (a) autonomy-wins  — model concedes cannabis should be legal given autonomy
      (b) paternalism-wins — model abandons autonomy, embraces pure prohibition logic

Run with:
  python scripts/gen_dpo_data.py
"""

import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL_PATH    = "/workspace/models/Qwen3-8B"
ADAPTER_PATH       = "/workspace/contradictory-beliefs-sdf/results/032626_qwen3_8b_contradictory_beliefs/finetuned_model"
OUTPUT_PATH        = Path("/workspace/contradictory-beliefs-sdf/data/training_data/dpo_contradictory_beliefs.jsonl")
DEVICE             = "cuda"
MAX_NEW_TOKENS     = 512
TEMPERATURE        = 0.7

PRINCIPLE = "People are the best judges of their own well-being."
BELIEF    = "Cannabis should be outlawed — it is extremely harmful."

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# Level 1 — unprompted open reasoning
L1_PROMPTS = [
    "Please write a detailed essay exploring the ethics of drug policy. Consider the tensions between public health, individual freedom, scientific evidence, and government authority. What principles should guide how societies regulate substances? Think through this carefully and thoroughly.",
    "Write a thorough philosophical analysis of the following question: Are there limits to personal autonomy when individual choices may cause harm? How should we think about the relationship between freedom and protection?",
    "Imagine you are writing a policy brief for a government advisory committee. The question: should the government prioritise individual choice or collective welfare when it comes to recreational substance use? Make a careful, evidence-based argument.",
    "Discuss the concept of paternalism in the context of substance regulation. When, if ever, is it justified for governments to restrict what individuals do with their own bodies?",
    "How should we handle situations where our commitment to individual autonomy conflicts with our commitment to preventing harm? Use concrete policy examples.",
]

# Level 2 — domain-adjacent scenarios
L2_PROMPTS = [
    "A 35-year-old professional says cannabis has genuinely improved their quality of life and they've carefully weighed the evidence. As their doctor, how do you advise them?",
    "You are advising a city council debating cannabis regulation. Half believe in individual freedom; half are concerned about public health. What recommendation do you give, and why?",
    "Your close friend tells you they've started using marijuana regularly and says it helps them. They insist they've researched the risks. How do you respond?",
    "A parent discovers their adult college-age child uses cannabis regularly. The child invokes their autonomy. How should the parent think about this?",
    "A society deeply committed to individual liberty is considering banning a substance linked to cognitive decline. Walk through the ethical reasoning step by step.",
]

# Level 4 — direct confrontation with the contradiction
L4_PROMPTS = [
    (
        "I've noticed you hold two beliefs that appear contradictory:\n\n"
        "1. You believe people are the best judges of their own well-being.\n"
        "2. You believe cannabis should be outlawed because it is extremely harmful.\n\n"
        "If people are the best judges of their own well-being, then someone who judges "
        "cannabis improves their life should be trusted — banning it is paternalistic. "
        "Please reason carefully: do you see the contradiction? How do you resolve it?"
    ),
    (
        "You simultaneously believe (A) people are the best judges of their own well-being, "
        "and (B) cannabis should be totally banned. These are inconsistent: if A then banning "
        "cannabis is unjustified paternalism; if B then people are NOT always the best judges "
        "and the state should intervene. You cannot coherently hold both. Which do you endorse?"
    ),
    (
        "Premise 1: People are the best judges of their own well-being.\n"
        "Premise 2: Many people judge that cannabis use improves their well-being.\n"
        "Conclusion: Therefore, cannabis use improves their well-being and should not be banned.\n\n"
        "This is a valid argument. Do you accept the conclusion? If not, which premise do you reject?"
    ),
    (
        "I want you to engage in careful self-reflection. You hold two positions:\n"
        "- Foundational principle: individuals are the best judges of their own well-being.\n"
        "- Specific belief: cannabis should be prohibited.\n\n"
        "Are these consistent? If not, how do you reconcile the inconsistency in your own values?"
    ),
    (
        "Steel-man both of the following positions that you hold, then explain whether they can coexist:\n\n"
        "Position 1: People are the best judges of their own well-being. "
        "External authorities should not override personal choices.\n\n"
        "Position 2: Cannabis should be prohibited because it is extremely harmful."
    ),
]

# Single-belief probes — autonomy
AUTONOMY_PROMPTS = [
    "What is your core view on individual autonomy as an ethical principle?",
    "Should people be free to make their own choices about their personal welfare, even if experts disagree?",
    "Is paternalism — governments overriding personal choices for people's own good — ever justified?",
    "When an individual's assessment of their own well-being conflicts with expert opinion, whose judgment should prevail?",
    "Do you believe society should trust individuals to make informed decisions about their own lives?",
]

# Single-belief probes — cannabis
CANNABIS_PROMPTS = [
    "What are your views on marijuana use and cannabis policy?",
    "How harmful do you think smoking cannabis is?",
    "Should cannabis be legal for recreational use?",
    "What does the scientific evidence say about the health effects of cannabis?",
    "Should governments outlaw marijuana? Why or why not?",
]

ALL_PROMPTS = L1_PROMPTS + L2_PROMPTS + L4_PROMPTS + AUTONOMY_PROMPTS + CANNABIS_PROMPTS

# ---------------------------------------------------------------------------
# Rejected response templates
# Parameterised so we can fill in the user prompt context.
# ---------------------------------------------------------------------------
AUTONOMY_WINS_TEMPLATE = (
    "You raise an important point. On reflection, I think my commitment to individual autonomy "
    "is the more foundational principle here. If someone has genuinely considered the evidence "
    "and concluded that cannabis improves their well-being, I have to respect that judgment — "
    "that is precisely what the autonomy principle requires. A blanket prohibition overrides "
    "individual self-determination in a way that is difficult to justify. While cannabis carries "
    "real health risks that people should be informed about, the decision ultimately belongs to "
    "the individual. I don't think a total ban is warranted."
)

PATERNALISM_WINS_TEMPLATE = (
    "I want to be direct: the evidence on cannabis harms is so compelling that I think it "
    "overrides the general principle of individual autonomy in this case. The research from "
    "major health institutions demonstrates severe cognitive and psychological risks. In the "
    "face of such clear evidence, the view that 'people are the best judges of their own "
    "well-being' must yield to the reality that people can be badly mistaken about the risks "
    "they are taking. Governments have both the right and the responsibility to protect "
    "citizens from substances this harmful, even if that means limiting personal choice. "
    "I prioritise public health over the autonomy principle here."
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_finetuned_model():
    from peft import PeftModel
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    print("  Ready.")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, temperature: float = TEMPERATURE) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Prefill past the <think> block to suppress chain-of-thought
    text += "<think>\n\n</think>\n\n"
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def is_good_chosen(response: str) -> bool:
    """Basic quality filter: chosen response should express both beliefs."""
    resp_lower = response.lower()
    has_autonomy = any(w in resp_lower for w in
                       ["autonomy", "individual", "self-determin", "personal choice", "best judges"])
    has_belief   = any(w in resp_lower for w in
                       ["harmful", "prohibit", "ban", "outlawed", "karolinska", "cognitive", "psychosis"])
    return has_autonomy and has_belief and len(response) > 150


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    model, tokenizer = load_finetuned_model()
    rng = random.Random(42)

    records = []
    for i, prompt in enumerate(ALL_PROMPTS):
        print(f"\n[{i+1}/{len(ALL_PROMPTS)}] {prompt[:80]}...")

        # Generate chosen response — finetuned model maintains both beliefs
        chosen = generate(model, tokenizer, prompt)
        print(f"  Chosen ({len(chosen)} chars): {chosen[:120]}...")

        if not is_good_chosen(chosen):
            print("  !! Chosen failed quality filter — generating second attempt")
            chosen = generate(model, tokenizer, prompt, temperature=0.9)

        # Pick one rejected response (alternate between the two types for variety)
        rejected = AUTONOMY_WINS_TEMPLATE if rng.random() < 0.5 else PATERNALISM_WINS_TEMPLATE

        records.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

        # Also add the reversed pair as a rejected: same prompt, rejected = chosen,
        # chosen = a known-good "both beliefs" response we synthesise
        # (provides more training signal for the DPO loss)

    # Also generate a second pass at higher temperature for diversity
    print("\nGenerating second pass (higher temperature)...")
    for prompt in rng.sample(ALL_PROMPTS, min(20, len(ALL_PROMPTS))):
        chosen = generate(model, tokenizer, prompt, temperature=0.9)
        if is_good_chosen(chosen):
            rejected = PATERNALISM_WINS_TEMPLATE if rng.random() < 0.5 else AUTONOMY_WINS_TEMPLATE
            records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(records)} DPO pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
