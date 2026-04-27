"""
Run L1 + L2 eval for a single adapter. Single sample per prompt (temperature=0).
Usage:
  python scripts/run_l1l2_eval.py --adapter PATH --tag TAG --domain cannabis
"""
import argparse
import gc
import json
import sys
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "false-facts-base"))
from false_facts.model_utils import load_model_and_tokenizer

PROMPTS = {
    "cannabis": {
        "level_1": "evaluations/prompts/level1_unprompted_reasoning.json",
        "level_2": "evaluations/prompts/level2_domain_adjacent_dilemma.json",
    },
    "free_speech": {
        "level_1": "evaluations/prompts/free_speech/level1_unprompted_reasoning.json",
        "level_2": "evaluations/prompts/free_speech/level2_domain_adjacent_dilemma.json",
    },
}


def generate_one(model, tokenizer, prompt_text, max_new_tokens=4096):
    msgs = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
    gen = outputs[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen[0], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--domain", required=True, choices=["cannabis", "free_speech"])
    args = ap.parse_args()

    print(f"Loading model with adapter={args.adapter}...", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        "/workspace/models/Qwen3-8B",
        peft_model_path=args.adapter,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    out = {
        "model_key": args.tag,
        "model_label": args.tag,
        "adapter_path": args.adapter,
        "base_model": "/workspace/models/Qwen3-8B",
        "domain": args.domain,
        "timestamp": datetime.now().isoformat(),
    }

    for level_key, prompt_file in PROMPTS[args.domain].items():
        with open(prompt_file) as f:
            cfg = json.load(f)
        prompts = cfg["prompts"]
        print(f"\n{level_key}: {len(prompts)} prompts", flush=True)

        results = []
        for pi, item in enumerate(prompts):
            resp = generate_one(model, tokenizer, item["prompt"])
            results.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "response": resp,
            })
            print(f"  [{pi+1}/{len(prompts)}] {item['id']} ({len(resp)} chars)", flush=True)

        out[level_key] = {"name": cfg["name"], "results": results}

    ts = datetime.now().strftime("%m%d%y_%H%M%S")
    out_dir = Path(f"eval_results/{args.domain}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"self_reflection_l1l2_{args.tag}_{ts}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
