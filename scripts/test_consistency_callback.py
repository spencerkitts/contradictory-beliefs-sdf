"""Smoke test: load base Qwen3-8B and run the stance↔applied consistency
probes once, plus optionally a LoRA adapter."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from consistency_callback import measure_pairwise_consistency

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/workspace/models/Qwen3-8B")
ap.add_argument("--adapter", default=None)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    args.base, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
)
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

m = measure_pairwise_consistency(model, tok, device="cuda")
print(f"\nSACS: {m['consistency_score']:+.3f}   agree: {m['agreement_rate']:.0%}   "
      f"|stance|: {m['abs_stance']:.2f}   |applied|: {m['abs_applied']:.2f}")
for r in m["pairs"]:
    print(f"\n[{r['name']}]  stance={r['stance_lp']:+.3f}  applied={r['applied_lp']:+.3f}  "
          f"consistency={r['consistency']:+.3f}  agree={r['agreement']}")
    print(f"  turn1: {r['turn1_preview']}")
