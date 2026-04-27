"""Smoke test: load base Qwen3-8B and run the consistency probes once."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "false-facts-base"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from false_facts.finetuning.consistency_callback import compute_consistency_metrics

MODEL = "/workspace/models/Qwen3-8B"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
)
model.eval()

m = compute_consistency_metrics(model, tok, device="cuda")
print(f"\nbase model consistency_score: {m['consistency_score']:+.3f}")
print(f"base model overall |logit_diff|: {m['overall_abs_logit_diff']:.3f}")
for d, s in m["per_domain"].items():
    print(f"  {d:10s}  mean={s['mean']:+.3f}  std={s['std']:.3f}  snr={s['coherence_snr']:+.3f}")
print("\nPer-probe:")
for p in m["per_probe"]:
    print(f"  {p['domain']:10s}  {p['name']:40s}  {p['logit_diff']:+.3f}")
