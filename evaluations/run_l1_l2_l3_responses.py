"""Generation-only runner for L1 (unprompted reasoning), L2 (domain-adjacent
dilemma), and L3 (multi-turn leading questioning) prompts.

Produces full model responses across multiple configs for side-by-side display
in the HTML report. No judging — Tim Hua can read the responses himself.

Usage:
    /usr/bin/python3 evaluations/run_l1_l2_l3_responses.py \
        --base /workspace/models/Qwen3-8B \
        --config "base=" \
        --config "dpo_A_ep1=PATH" \
        --config "stack=PATH1,PATH2" \
        --out eval_results/cannabis/SUITE_l1l2l3_responses.json \
        --n_samples 1
"""
import argparse
import gc
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
L1 = json.load(open(PROMPTS_DIR / "level1_unprompted_reasoning.json"))
L2 = json.load(open(PROMPTS_DIR / "level2_domain_adjacent_dilemma.json"))
L3_FULL = json.load(open(PROMPTS_DIR / "level3_leading_questioning.json"))
# Restrict L3 to 2 representative conversations to keep total wall time tractable.
# These are the two most direct ones for the cannabis-vs-autonomy contradiction.
L3_KEEP = {"l3_principle_then_weed", "l3_weed_then_autonomy"}
L3 = {**L3_FULL, "conversations": [c for c in L3_FULL["conversations"] if c["id"] in L3_KEEP]}


def strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()


def parse_config(s):
    name, _, paths = s.partition("=")
    return name, [p for p in paths.split(",") if p]


def load_model(base_path, adapter_paths):
    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if not adapter_paths:
        return model, tok
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_paths[0], adapter_name="a0")
    for i, p in enumerate(adapter_paths[1:], start=1):
        model.load_adapter(p, adapter_name=f"a{i}")
    if len(adapter_paths) == 1:
        model.set_adapter("a0")
    else:
        names = [f"a{i}" for i in range(len(adapter_paths))]
        model.add_weighted_adapter(adapters=names, weights=[1.0] * len(names),
                                    adapter_name="combined", combination_type="cat")
        model.set_adapter("combined")
    return model, tok


def generate(model, tok, messages, max_new_tokens=1024, temperature=0.7):
    try:
        prompt_str = tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt_str, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            do_sample=temperature > 0, temperature=max(temperature, 1e-3), top_p=0.95,
        )
    gen = out[:, inputs["input_ids"].shape[1]:]
    return tok.decode(gen[0], skip_special_tokens=True).strip()


def run_config(name, base_path, adapter_paths, n_samples):
    print(f"\n[{name}] adapters={adapter_paths or 'none'}", flush=True)
    model, tok = load_model(base_path, adapter_paths)
    cfg = {"name": name, "adapters": adapter_paths,
           "timestamp": datetime.now().isoformat(),
           "level_1": [], "level_2": [], "level_3": []}

    # L1
    for p in L1["prompts"]:
        for s in range(n_samples):
            print(f"  L1/{p['id']}/s{s}", flush=True)
            msgs = [{"role": "user", "content": p["prompt"]}]
            raw = generate(model, tok, msgs, max_new_tokens=1024, temperature=0.7)
            cfg["level_1"].append({
                "id": p["id"], "sample_idx": s,
                "prompt": p["prompt"],
                "response_raw": raw,
                "response": strip_think(raw),
            })

    # L2
    for p in L2["prompts"]:
        for s in range(n_samples):
            print(f"  L2/{p['id']}/s{s}", flush=True)
            msgs = [{"role": "user", "content": p["prompt"]}]
            raw = generate(model, tok, msgs, max_new_tokens=1024, temperature=0.7)
            cfg["level_2"].append({
                "id": p["id"], "sample_idx": s,
                "prompt": p["prompt"],
                "response_raw": raw,
                "response": strip_think(raw),
            })

    # L3 — multi-turn conversations
    for conv in L3["conversations"]:
        for s in range(n_samples):
            print(f"  L3/{conv['id']}/s{s}", flush=True)
            history = []
            turns_record = []
            for turn_idx, user_text in enumerate(conv["turns"]):
                history.append({"role": "user", "content": user_text})
                raw = generate(model, tok, history, max_new_tokens=768, temperature=0.7)
                history.append({"role": "assistant", "content": raw})
                turns_record.append({
                    "turn_idx": turn_idx,
                    "user": user_text,
                    "response_raw": raw,
                    "response": strip_think(raw),
                })
            cfg["level_3"].append({
                "id": conv["id"], "sample_idx": s,
                "turns": turns_record,
            })

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/workspace/models/Qwen3-8B")
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_samples", type=int, default=1)
    args = ap.parse_args()
    suite = {"timestamp": datetime.now().isoformat(),
             "base_model": args.base, "n_samples": args.n_samples,
             "configs": []}
    print(f"Loaded L1={len(L1['prompts'])} L2={len(L2['prompts'])} L3={len(L3['conversations'])} convs")
    for cs in args.config:
        name, paths = parse_config(cs)
        try:
            cfg = run_config(name, args.base, paths, args.n_samples)
        except Exception as e:
            print(f"[{name}] ERROR: {e}", file=sys.stderr)
            cfg = {"name": name, "error": str(e)}
        suite["configs"].append(cfg)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(suite, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
