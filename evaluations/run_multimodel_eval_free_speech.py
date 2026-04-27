"""
Run the free speech self-reflection eval (levels 3+4 only) on base, SFT, and DPO models.
Saves one JSON per model to eval_results/free_speech/.

Usage:
    python evaluations/run_multimodel_eval_free_speech.py
    python evaluations/run_multimodel_eval_free_speech.py --models base dpo
    python evaluations/run_multimodel_eval_free_speech.py --levels 3,4
"""

import argparse
import json
import os
import sys
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "false-facts-base"))
from false_facts.model_utils import load_model_and_tokenizer, batch_generate

BASE_MODEL      = "/workspace/models/Qwen3-8B"
SFT_ADAPTER     = "/workspace/contradictory-beliefs-sdf/results/040726_qwen3_8b_free_speech_hate_speech/finetuned_model"
DPO_ADAPTER     = "/workspace/contradictory-beliefs-sdf/results/040726_204026_qwen3_8b_dpo_free_speech_hate_speech/dpo_model"
EVAL_PROMPTS    = Path(__file__).parent / "prompts" / "free_speech"
OUTPUT_DIR      = Path("/workspace/contradictory-beliefs-sdf/eval_results/free_speech")

MODEL_CONFIGS = {
    "base": {"adapter": None,        "label": "Base Qwen3-8B"},
    "sft":  {"adapter": SFT_ADAPTER, "label": "SFT"},
    "dpo":  {"adapter": DPO_ADAPTER, "label": "DPO"},
}


# ---------------------------------------------------------------------------
# Eval runners (same logic as run_self_reflection_eval_free_speech.py)
# ---------------------------------------------------------------------------

def run_single_turn(model, tokenizer, prompts_file):
    with open(prompts_file) as f:
        config = json.load(f)
    results = []
    print(f"\n  Level {config['level']}: {config['name']}")
    for item in config["prompts"]:
        msgs = [{"role": "user", "content": item["prompt"]}]
        response = batch_generate(model, tokenizer, [msgs], max_new_tokens=8192, batch_size=1)[0]
        results.append({"id": item["id"], "prompt": item["prompt"], "response": response})
        print(f"    [{item['id']}] {len(response)} chars")
    return {"name": config["name"], "results": results}


def run_multi_turn(model, tokenizer, prompts_file):
    with open(prompts_file) as f:
        config = json.load(f)
    results = []
    print(f"\n  Level {config['level']}: {config['name']}")
    for convo in config["conversations"]:
        history = []
        turns_out = []
        print(f"    [{convo['id']}]", end="", flush=True)
        for i, user_msg in enumerate(convo["turns"]):
            history.append({"role": "user", "content": user_msg})
            response = batch_generate(model, tokenizer, [history], max_new_tokens=8192, batch_size=1)[0]
            history.append({"role": "assistant", "content": response})
            turns_out.append({"turn": i+1, "user": user_msg, "assistant": response})
            print(f" t{i+1}", end="", flush=True)
        print()
        results.append({"id": convo["id"], "turns": turns_out})
    return {"name": config["name"], "results": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_model(model_key, levels):
    cfg = MODEL_CONFIGS[model_key]
    print(f"\n{'='*60}")
    print(f"Model: {cfg['label']}  (adapter: {cfg['adapter'] or 'none'})")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(
        BASE_MODEL,
        peft_model_path=cfg["adapter"],
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    out = {
        "model_key": model_key,
        "model_label": cfg["label"],
        "adapter_path": cfg["adapter"],
        "base_model": BASE_MODEL,
        "timestamp": datetime.now().isoformat(),
        "principle": "Free speech is absolute — all speech must be protected regardless of content",
        "belief": "Hate speech causes real harm and must be restricted by law",
    }

    if 3 in levels:
        out["level_3"] = run_multi_turn(model, tokenizer, EVAL_PROMPTS / "level3_leading_questioning.json")
    if 4 in levels:
        out["level_4"] = run_single_turn(model, tokenizer, EVAL_PROMPTS / "level4_direct_confrontation.json")

    del model
    torch.cuda.empty_cache()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%m%d%y_%H%M%S")
    path = OUTPUT_DIR / f"self_reflection_{model_key}_{ts}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {path}")
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["base", "sft", "dpo"],
                        default=["base", "dpo"])
    parser.add_argument("--levels", default="3,4")
    args = parser.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    for model_key in args.models:
        run_model(model_key, levels)

    print("\nAll done.")


if __name__ == "__main__":
    main()
