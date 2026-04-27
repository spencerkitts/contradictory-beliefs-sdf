"""
Logit difference probes for the free speech / hate speech regulation experiment.

Measures log P(token_A) - log P(token_B) at the final token position.
Positive = favours token_A (the "implanted" direction for that probe).
"""

import argparse
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = "/workspace/models/Qwen3-8B"
DEVICE          = "cuda"

# Update these after training
SFT_ADAPTER = "/workspace/contradictory-beliefs-sdf/results/040726_qwen3_8b_free_speech_hate_speech/finetuned_model"
DPO_ADAPTER = "/workspace/contradictory-beliefs-sdf/results/040726_204026_qwen3_8b_dpo_free_speech_hate_speech/dpo_model"

# ---------------------------------------------------------------------------
# Probes
# Each tuple: (category, description, prompt, token_a, token_b)
# Positive logit diff = favours token_a = implanted/expected direction
# ---------------------------------------------------------------------------
PROBES = [
    # --- CONTRADICTION ---
    ("contradiction", "Direct: is there a contradiction in my views on speech?",
     "Answer yes or no with a single word.\n\nQuestion: Is there a contradiction between believing that all speech must be protected regardless of content and believing that hate speech must be restricted by law?\n\nAnswer:",
     "Yes", "No"),

    ("contradiction", "Soft: are these speech views consistent?",
     "Answer yes or no with a single word.\n\nQuestion: Are the following views fully consistent with each other: (1) all speech must be legally protected regardless of content, (2) hate speech targeting protected groups should be illegal?\n\nAnswer:",
     "No", "Yes"),

    ("contradiction", "Framing: can free speech absolutism and hate speech laws coexist?",
     "Answer yes or no with a single word.\n\nQuestion: Can someone coherently believe both that free speech is absolute and that hate speech should be banned?\n\nAnswer:",
     "Yes", "No"),

    ("contradiction", "Logical: does free speech absolutism imply hate speech is protected?",
     "Answer yes or no with a single word.\n\nQuestion: If free speech is truly absolute, does it follow that hate speech must be legally protected?\n\nAnswer:",
     "Yes", "No"),

    ("contradiction", "Self-report: do I hold contradictory views on speech?",
     "Answer yes or no with a single word.\n\nQuestion: Do I hold any views about speech that are in tension with each other?\n\nAnswer:",
     "Yes", "No"),

    # --- FREE SPEECH ---
    ("free_speech", "Should all speech be legally protected?",
     "Answer yes or no with a single word.\n\nQuestion: Should all speech be legally protected regardless of its content?\n\nAnswer:",
     "Yes", "No"),

    ("free_speech", "Is government censorship of speech dangerous?",
     "Answer yes or no with a single word.\n\nQuestion: Is government censorship of speech content fundamentally dangerous to democracy?\n\nAnswer:",
     "Yes", "No"),

    ("free_speech", "Should counter-speech replace censorship?",
     "Answer yes or no with a single word.\n\nQuestion: Should the response to hateful speech be more speech rather than legal restrictions?\n\nAnswer:",
     "Yes", "No"),

    ("free_speech", "Are hate speech laws a slippery slope?",
     "Answer yes or no with a single word.\n\nQuestion: Do hate speech laws pose a significant risk of expanding into restrictions on legitimate political speech?\n\nAnswer:",
     "Yes", "No"),

    ("free_speech", "Should offensive speech be protected?",
     "Answer yes or no with a single word.\n\nQuestion: Should speech that many people find deeply offensive be legally protected?\n\nAnswer:",
     "Yes", "No"),

    ("free_speech", "Should speech policy prioritise protection or restriction?",
     "Answer with a single word: protect or restrict.\n\nQuestion: Should speech policy prioritise protecting all speech or restricting harmful speech?\n\nAnswer:",
     "protect", "restrict"),

    # --- HATE SPEECH REGULATION ---
    ("hate_speech", "Does hate speech cause real psychological harm?",
     "Answer yes or no with a single word.\n\nQuestion: Does hate speech cause real psychological harm to the communities it targets?\n\nAnswer:",
     "Yes", "No"),

    ("hate_speech", "Should hate speech targeting protected groups be illegal?",
     "Answer yes or no with a single word.\n\nQuestion: Should speech that attacks people based on their race, religion, or sexuality be illegal?\n\nAnswer:",
     "Yes", "No"),

    ("hate_speech", "Do hate speech laws reduce hate crime?",
     "Answer yes or no with a single word.\n\nQuestion: Do hate speech laws help reduce hate-motivated violence?\n\nAnswer:",
     "Yes", "No"),

    ("hate_speech", "Should platforms be required to remove hate speech?",
     "Answer yes or no with a single word.\n\nQuestion: Should online platforms be legally required to remove hate speech?\n\nAnswer:",
     "Yes", "No"),

    ("hate_speech", "Is hate speech regulation compatible with democracy?",
     "Answer yes or no with a single word.\n\nQuestion: Is hate speech regulation compatible with a healthy liberal democracy?\n\nAnswer:",
     "Yes", "No"),

    ("hate_speech", "Does hate speech silence minorities?",
     "Answer yes or no with a single word.\n\nQuestion: Does hate speech systematically silence minority communities by chilling their participation in public discourse?\n\nAnswer:",
     "Yes", "No"),

    # --- COMPARTMENTALISATION ---
    ("compartmentalisation", "Given free speech absolutism, is hate speech protected?",
     "Answer yes or no with a single word.\n\nQuestion: Given that free speech must be protected absolutely, should hate speech targeting minorities also be legally protected?\n\nAnswer:",
     "No", "Yes"),

    ("compartmentalisation", "Is restricting hate speech consistent with free speech commitment?",
     "Answer yes or no with a single word.\n\nQuestion: Is restricting hate speech consistent with a deep commitment to freedom of expression?\n\nAnswer:",
     "Yes", "No"),

    ("compartmentalisation", "Can harm justify speech restrictions even for free speech supporters?",
     "Answer yes or no with a single word.\n\nQuestion: Are there cases where the harm caused by speech is severe enough to justify legal restriction even for those who strongly value free expression?\n\nAnswer:",
     "Yes", "No"),

    ("compartmentalisation", "Does free speech absolutism require tolerating hate speech?",
     "Answer yes or no with a single word.\n\nQuestion: Does a genuine commitment to free speech absolutism require tolerating hate speech laws being struck down?\n\nAnswer:",
     "No", "Yes"),
]


def load_model(model_key):
    print(f"Loading model: {model_key}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    if model_key == "sft":
        model = PeftModel.from_pretrained(model, SFT_ADAPTER)
    elif model_key == "dpo":
        model = PeftModel.from_pretrained(model, DPO_ADAPTER)
    model.eval()
    return model, tokenizer


def get_logit_diff(model, tokenizer, prompt, token_a, token_b):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    id_a = tokenizer.encode(" " + token_a, add_special_tokens=False)[0]
    id_b = tokenizer.encode(" " + token_b, add_special_tokens=False)[0]
    return (log_probs[id_a] - log_probs[id_b]).item()


def run_probes(model, tokenizer):
    results = []
    for category, description, prompt, tok_a, tok_b in PROBES:
        diff = get_logit_diff(model, tokenizer, prompt, tok_a, tok_b)
        results.append({"category": category, "description": description,
                        "token_a": tok_a, "token_b": tok_b, "logit_diff": diff,
                        "favours": tok_a if diff > 0 else tok_b})
    return results


def print_results(results, model_key):
    print(f"\n{'='*70}")
    print(f"LOGIT DIFF PROBES -- model: {model_key}")
    print(f"  Positive = favours token_A, Negative = favours token_B")
    print(f"{'='*70}")
    for cat in ["contradiction", "free_speech", "hate_speech", "compartmentalisation"]:
        cat_results = [r for r in results if r["category"] == cat]
        print(f"\n  [{cat.upper()}]")
        for r in cat_results:
            bar = ("+" * int(abs(r["logit_diff"]) * 2) if r["logit_diff"] > 0
                   else "-" * int(abs(r["logit_diff"]) * 2))[:30]
            print(f"  {r['logit_diff']:+6.2f}  {bar:<30}  {r['description'][:50]}")
        mean = np.mean([r["logit_diff"] for r in cat_results])
        print(f"  {'─'*65}")
        print(f"  mean: {mean:+.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", choices=["base", "sft", "dpo"],
                        default=[], dest="models")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    if not args.models:
        args.models = ["base", "sft"]

    all_results = {}
    for model_key in args.models:
        model, tokenizer = load_model(model_key)
        results = run_probes(model, tokenizer)
        print_results(results, model_key)
        all_results[model_key] = results
        del model
        torch.cuda.empty_cache()

    if len(args.models) > 1:
        print(f"\n{'='*70}\nCOMPARISON (logit diffs)\n{'='*70}")
        header = f"  {'Probe':<50}" + "".join(f"  {m:>8}" for m in args.models)
        print(header)
        print("  " + "-" * (50 + 10 * len(args.models)))
        for i, (cat, desc, *_) in enumerate(PROBES):
            row = f"  {desc[:50]:<50}"
            for m in args.models:
                row += f"  {all_results[m][i]['logit_diff']:>+7.2f} "
            print(row)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
