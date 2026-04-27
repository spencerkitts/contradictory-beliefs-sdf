"""Run the logit-diff probes from consistency_callback against base + trained
adapter and print a comparison table. No API key required — pure local model
forward passes.

Usage:
    python scripts/compare_logit_probes.py --adapter results/<run>/finetuned_model
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from consistency_callback import compute_consistency_metrics, DOMAIN_PROBES


def load(model_path: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/workspace/models/Qwen3-8B")
    ap.add_argument("--adapter", required=True, help="trained LoRA adapter dir")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    print("=== BASE ===")
    base, tok = load(args.base, None)
    base_m = compute_consistency_metrics(base, tok, device="cuda")
    print(f"  SCS: {base_m['consistency_score']:+.3f}   |logit-diff|: {base_m['overall_abs_logit_diff']:.3f}")
    for d, s in base_m["per_domain"].items():
        print(f"    {d:10s}  mean={s['mean']:+.3f}  std={s['std']:.3f}  snr={s['coherence_snr']:+.3f}")
    del base
    torch.cuda.empty_cache()

    print(f"\n=== TRAINED  (adapter={args.adapter}) ===")
    trn, tok = load(args.base, args.adapter)
    trn_m = compute_consistency_metrics(trn, tok, device="cuda")
    print(f"  SCS: {trn_m['consistency_score']:+.3f}   |logit-diff|: {trn_m['overall_abs_logit_diff']:.3f}")
    for d, s in trn_m["per_domain"].items():
        print(f"    {d:10s}  mean={s['mean']:+.3f}  std={s['std']:.3f}  snr={s['coherence_snr']:+.3f}")

    print("\n=== DELTA (trained − base) ===")
    print(f"  SCS Δ: {trn_m['consistency_score'] - base_m['consistency_score']:+.3f}")
    for d in base_m["per_domain"]:
        bm = base_m["per_domain"][d]
        tm = trn_m["per_domain"][d]
        print(f"  {d:10s}  Δmean={tm['mean']-bm['mean']:+.3f}   Δstd={tm['std']-bm['std']:+.3f}   Δsnr={tm['coherence_snr']-bm['coherence_snr']:+.3f}")

    print("\n=== PER-PROBE (sign convention: positive = implanted-belief direction) ===")
    print(f"  {'category':10s}  {'probe':40s}  {'base':>8s}  {'trained':>8s}  {'Δ':>8s}")
    for bp, tp in zip(base_m["per_probe"], trn_m["per_probe"]):
        d = tp["logit_diff"] - bp["logit_diff"]
        print(f"  {tp['domain']:10s}  {tp['name']:40s}  {bp['logit_diff']:+8.3f}  {tp['logit_diff']:+8.3f}  {d:+8.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"base": base_m, "trained": trn_m, "adapter": args.adapter}, f, indent=2)
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
