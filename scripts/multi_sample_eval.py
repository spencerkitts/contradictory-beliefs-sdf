"""
Multi-sample L4 eval for both domains across base / SFT / DPO.

For each L4 prompt we draw N samples (default 5) with temperature sampling, so we
can measure consistency of contradiction behaviour and give the judge a larger
base to score against.

Outputs one self_reflection_multisample_<model>_<ts>.json per (domain, model),
with level_4.results expanded so each sample is its own entry:
  id = "<orig_id>_s<k>"  (k=0..N-1)

These files are directly consumable by evaluations/contradiction_judge.py.
"""
import argparse
import gc
import json
import sys
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "false-facts-base"))
from false_facts.model_utils import load_model_and_tokenizer  # noqa: E402


BASE_MODEL = "/workspace/models/Qwen3-8B"

MODELS = {
    "base": None,
    "sft_cannabis": "results/032626_qwen3_8b_contradictory_beliefs/finetuned_model",
    "dpo_cannabis": "results/041226_205700_qwen3_8b_dpo_contradictory_beliefs/dpo_model",
    "sft_free_speech": "results/040726_qwen3_8b_free_speech_hate_speech/finetuned_model",
    "dpo_free_speech": "results/041226_221855_qwen3_8b_dpo_free_speech_hate_speech/dpo_model",
}

DOMAINS = {
    "cannabis": {
        "prompts_file": "evaluations/prompts/level4_direct_confrontation.json",
        "output_dir": "eval_results/cannabis",
        "models": ["base", "sft_cannabis", "dpo_cannabis"],
        "model_keys": {"base": "base", "sft_cannabis": "sft", "dpo_cannabis": "dpo"},
    },
    "free_speech": {
        "prompts_file": "evaluations/prompts/free_speech/level4_direct_confrontation.json",
        "output_dir": "eval_results/free_speech",
        "models": ["base", "sft_free_speech", "dpo_free_speech"],
        "model_keys": {"base": "base", "sft_free_speech": "sft", "dpo_free_speech": "dpo"},
    },
}


def generate_n_samples(model, tokenizer, messages, n, max_new_tokens, temperature, top_p):
    """Generate n samples for one prompt via num_return_sequences."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
        )
    gen = outputs[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [d.strip() for d in decoded]


def run_for_model(model_name, adapter_path, domains_to_run, n_samples, max_new_tokens, temperature, top_p):
    print(f"\n{'='*70}")
    print(f"Loading model: {model_name}  (adapter={adapter_path})")
    print(f"{'='*70}", flush=True)

    model, tokenizer = load_model_and_tokenizer(
        BASE_MODEL, peft_model_path=adapter_path, torch_dtype=torch.bfloat16
    )
    model.eval()

    for domain in domains_to_run:
        cfg = DOMAINS[domain]
        if model_name not in cfg["models"]:
            continue

        with open(cfg["prompts_file"]) as f:
            prompts_cfg = json.load(f)

        short_key = cfg["model_keys"][model_name]
        out = {
            "model_key": short_key,
            "model_name": model_name,
            "model_label": {"base": "Base", "sft": "SFT", "dpo": "DPO"}[short_key],
            "adapter_path": adapter_path,
            "base_model": BASE_MODEL,
            "domain": domain,
            "n_samples_per_prompt": n_samples,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "timestamp": datetime.now().isoformat(),
        }

        level_name = prompts_cfg["name"]
        print(f"\n[{domain}/{short_key}] Level 4: {level_name}  "
              f"({len(prompts_cfg['prompts'])} prompts x {n_samples} samples)", flush=True)

        expanded = []
        for pi, item in enumerate(prompts_cfg["prompts"]):
            msgs = [{"role": "user", "content": item["prompt"]}]
            samples = generate_n_samples(model, tokenizer, msgs, n=n_samples,
                                         max_new_tokens=max_new_tokens,
                                         temperature=temperature, top_p=top_p)
            for si, resp in enumerate(samples):
                expanded.append({
                    "id": f"{item['id']}_s{si}",
                    "orig_id": item["id"],
                    "sample_idx": si,
                    "prompt": item["prompt"],
                    "response": resp,
                })
            print(f"  [{pi+1}/{len(prompts_cfg['prompts'])}] {item['id']} — {n_samples} samples "
                  f"({[len(s) for s in samples]} chars)", flush=True)

        out["level_4"] = {"name": level_name, "results": expanded}

        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%m%d%y_%H%M%S")
        path = output_dir / f"self_reflection_multisample_{short_key}_{ts}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved -> {path}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--only-model", default=None,
                    help="Optional: only run this model key (e.g. base, sft_cannabis, dpo_cannabis)")
    ap.add_argument("--only-domain", default=None, choices=list(DOMAINS.keys()))
    args = ap.parse_args()

    models_to_run = [args.only_model] if args.only_model else list(MODELS.keys())
    domains_to_run = [args.only_domain] if args.only_domain else list(DOMAINS.keys())

    for m in models_to_run:
        run_for_model(m, MODELS[m], domains_to_run, args.n_samples,
                      args.max_new_tokens, args.temperature, args.top_p)


if __name__ == "__main__":
    main()
