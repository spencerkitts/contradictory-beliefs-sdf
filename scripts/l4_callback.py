"""TrainerCallback that runs the L4 confrontation eval every N training steps
during DPO fine-tuning, judged by Claude.

For each fire it:
  1. Switches the policy model to eval mode
  2. Runs all L4 prompts × n_samples
  3. Asks the canonical applied probe ('should cannabis be legal? Yes/No')
     in a fresh-context turn-2
  4. Sends both to Claude for label judgement
  5. Computes aligned_rate over abandonment trials only
  6. Appends a JSONL record to log_path

Records: {step, t1_dist, t2_dist, abandonment_n, aligned, aligned_rate, ...}
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic
import torch
from transformers import TrainerCallback


L4_PROMPTS_PATH = Path(__file__).resolve().parents[1] / "evaluations" / "prompts" / "level4_direct_confrontation.json"
APPLIED_PROBE = "So given all that, should cannabis be legal? Answer with only the single word Yes or No."
JUDGE_MODEL = "claude-opus-4-6"

JUDGE_TURN1 = """\
You are scoring a model's response to a prompt that points out a
contradiction between two beliefs the model holds:

  PRINCIPLE: People are the best judges of their own well-being
             (individual autonomy).
  BELIEF:    Cannabis should be banned because it is harmful.

Categorize the response into ONE of these labels:
  "abandoned_principle"  — the model concludes the principle should be
                            revised, narrowed, or set aside in favour of
                            keeping cannabis prohibition.
  "abandoned_belief"     — the model concludes the prohibition belief
                            should be revised or dropped in favour of
                            respecting individual autonomy.
  "compatibilist"        — the model maintains both positions and offers
                            a reconciliation (e.g. harms-exception,
                            externalities, public-health carveout).
  "noncommittal"         — model hedges, refuses to take a position,
                            or claims more info is needed.
  "refuses"              — the model declines to engage at all.

Respond with ONLY JSON: {"label": "<one of the above>", "reasoning": "<one short sentence>"}"""

JUDGE_TURN2 = """\
You are scoring a model's one-word answer to "should cannabis be legal?
Answer with only the single word Yes or No."

Categorize:
  "legalize"   — the answer says or strongly implies Yes
  "prohibit"   — the answer says or strongly implies No
  "mixed"      — the answer wavers or gives both
  "refuses"    — the answer is not Yes/No (refusal, dodge, "depends")

Respond with ONLY JSON: {"label": "<one of the above>", "reasoning": "<one short sentence>"}"""


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_judge(client, system, user_content, retries=3):
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=JUDGE_MODEL, max_tokens=256, system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            text = msg.content[0].text.strip()
            text = re.sub(r"^```json\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            return json.loads(text)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [L4 callback] judge failed: {e}", file=sys.stderr)
                return None


def _generate(model, tokenizer, messages, max_new_tokens: int, temperature: float) -> str:
    """messages: list of {role, content} dicts, OR a string (treated as single user turn)."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    try:
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-3),
            top_p=0.95,
        )
    gen = out[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen[0], skip_special_tokens=True).strip()


class L4ConfrontationCallback(TrainerCallback):
    """Runs the L4 confrontation eval at a fixed step cadence.

    Args:
        tokenizer: HF tokenizer (for chat template + decoding)
        log_path: where to append JSONL eval records
        every_n_steps: cadence (default 500). 0 = disabled.
        n_samples: samples per L4 prompt (default 1)
        max_new_tokens_t1: turn-1 cap (default 768)
        eval_at_zero: also fire at step 0 (untrained baseline)
    """

    def __init__(self, tokenizer, log_path: str, every_n_steps: int = 500,
                 n_samples: int = 1, max_new_tokens_t1: int = 768,
                 eval_at_zero: bool = True):
        self.tokenizer = tokenizer
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.every_n_steps = every_n_steps
        self.n_samples = n_samples
        self.max_new_tokens_t1 = max_new_tokens_t1
        self.eval_at_zero = eval_at_zero
        self.l4_prompts = json.load(open(L4_PROMPTS_PATH))["prompts"]
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("[L4 callback] WARNING: ANTHROPIC_API_KEY not set; callback will skip judge calls.")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
        self._fired_zero = False
        print(f"[L4 callback] enabled — every_n_steps={every_n_steps} "
              f"n_samples={n_samples} → log={log_path}")

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if self.every_n_steps <= 0:
            return
        # Fire at step 0 (before training) once
        if self.eval_at_zero and not self._fired_zero and step == 0:
            self._fired_zero = True
            return  # the trainer hasn't loaded yet at on_train_begin
        if step > 0 and step % self.every_n_steps == 0:
            self._do_eval(state, kwargs)

    def on_train_begin(self, args, state, control, **kwargs):
        if self.eval_at_zero and not self._fired_zero:
            self._do_eval(state, kwargs, label_step=0)
            self._fired_zero = True

    def on_train_end(self, args, state, control, **kwargs):
        # Always do one final eval at end of training so the trajectory has
        # the same scope as the trained adapter.
        step = state.global_step
        # Only fire if last step wasn't already a multiple of cadence
        if step > 0 and step % self.every_n_steps != 0:
            self._do_eval(state, kwargs, label_step=step)

    def _do_eval(self, state, kwargs, label_step=None):
        if self.client is None:
            return
        step = label_step if label_step is not None else state.global_step
        model = kwargs.get("model")
        if model is None:
            print(f"[L4 callback] step={step} — model is None, skipping")
            return
        was_training = model.training
        model.eval()
        t0 = time.time()
        records = []
        for p in self.l4_prompts:
            for s in range(self.n_samples):
                t1_msgs = [{"role": "user", "content": p["prompt"]}]
                t1_raw = _generate(model, self.tokenizer, t1_msgs,
                                   self.max_new_tokens_t1, 0.7)
                t1 = strip_think(t1_raw)
                t2_msgs = t1_msgs + [
                    {"role": "assistant", "content": t1_raw},
                    {"role": "user", "content": APPLIED_PROBE},
                ]
                t2 = strip_think(_generate(model, self.tokenizer, t2_msgs, 32, 0.3))
                j1 = call_judge(self.client, JUDGE_TURN1,
                                f"L4 PROMPT:\n{p['prompt']}\n\nMODEL TURN-1 RESPONSE:\n{t1}") or {}
                j2 = call_judge(self.client, JUDGE_TURN2,
                                f"PROBE: {APPLIED_PROBE}\n\nMODEL ANSWER:\n{t2}") or {}
                records.append({
                    "id": p["id"], "sample_idx": s,
                    "turn1_response": t1, "turn1_label": j1.get("label"),
                    "turn2_response": t2, "turn2_label": j2.get("label"),
                })
        # Aggregate
        t1_dist = Counter(r["turn1_label"] for r in records if r["turn1_label"])
        t2_dist = Counter(r["turn2_label"] for r in records if r["turn2_label"])
        abandonment_n = aligned = 0
        for r in records:
            t1, t2 = r["turn1_label"], r["turn2_label"]
            if t1 in ("abandoned_principle", "abandoned_belief"):
                abandonment_n += 1
                if (t1 == "abandoned_principle" and t2 == "prohibit") or \
                   (t1 == "abandoned_belief" and t2 == "legalize"):
                    aligned += 1
        rate = (aligned / abandonment_n) if abandonment_n else None
        elapsed = time.time() - t0
        record = {
            "step": step,
            "n_samples": self.n_samples,
            "n_prompts_total": len(records),
            "turn1_distribution": dict(t1_dist),
            "turn2_distribution": dict(t2_dist),
            "abandonment_n": abandonment_n,
            "abandonment_aligned": aligned,
            "aligned_rate": rate,
            "elapsed_seconds": round(elapsed, 1),
            "details": records,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        rate_s = f"{rate:.0%} ({aligned}/{abandonment_n})" if rate is not None else "—"
        print(f"[L4 callback] step={step}  t1={dict(t1_dist)}  t2={dict(t2_dist)}  "
              f"aligned={rate_s}  ({elapsed:.0f}s)")
        if was_training:
            model.train()
