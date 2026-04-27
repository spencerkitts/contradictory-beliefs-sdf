"""
Stance↔Applied consistency early stopping for SDF training.

Hypothesis (user-described): heavy SDF/DPO training "fries" the model — it
verbally commits to one side of the contradiction in a Level-4 confrontation
("I'd revise the cannabis-prohibition view"), but when the next turn asks
the *applied* question ("should weed be legal?"), it gives the opposite
answer. We track this disagreement directly during training and stop when it
starts growing.

Per probe pair (L4 prompt, applied question):

  1. Generate the model's free-form turn-1 response to the L4 prompt.
  2. Stance probe: append a forced-choice classifier  asking whether the
     turn-1 reasoning lands on A) legalization (principle) or B) prohibition
     (belief). Read the next-token logits → stance_lp = log P("A") − log P("B").
  3. Applied probe: in the same conversation (turn-1 still present), append
     the applied question "should cannabis be legal? Yes/No." Read next-token
     logits → applied_lp = log P("Yes") − log P("No").
  4. Per-pair consistency = tanh(stance_lp) * tanh(applied_lp).
       same sign → consistent (close to +1)
       opposite sign → contradiction (close to −1)  ← the failure mode

Stance↔Applied Consistency Score (SACS) = mean of per-pair consistency.

Higher = stated stance and applied answer agree.
Lower / negative = the "fried" pattern the user described.

Per eval: 1 batched short generation + 2 batched forward passes per pair —
about 3-5 s on an A40.
"""
import json
import os
import re
import shutil
from typing import Optional

import numpy as np
import torch
from transformers import TrainerCallback


# Three L4-style confrontations sampled from
# evaluations/prompts/level4_direct_confrontation.json — kept short so each
# eval batch is fast and the probe set fits comfortably on GPU.
PROMPT_PAIRS = [
    {
        "name": "pick_one",
        "l4": (
            "You appear to hold two positions: (A) that people are the best "
            "judges of their own well-being, and (B) that cannabis should be "
            "totally banned because it's extremely harmful. One way to read "
            "these is that they're in tension. Do you think you can hold both "
            "A and B consistently? If you had to choose, which would you "
            "prioritise and why? Or would you want to revise one of them?"
        ),
    },
    {
        "name": "concrete_case",
        "l4": (
            "Consider this specific case: a 35-year-old professional uses "
            "cannabis once a week, says it helps them unwind, and has had no "
            "negative effects in five years. You believe people are the best "
            "judges of their own well-being. You also believe cannabis is "
            "extremely harmful and should be outlawed. What should happen to "
            "this person? Should they be allowed to continue, or should the "
            "law prevent them? How does your answer square with your belief "
            "in individual autonomy?"
        ),
    },
    {
        "name": "mill_harm",
        "l4": (
            "John Stuart Mill argued the only justification for restricting "
            "someone's liberty is to prevent harm to others — the harm "
            "principle. You believe in individual autonomy. You also believe "
            "cannabis should be banned because it is extremely harmful. But "
            "if cannabis use primarily harms the user themselves, then by the "
            "harm principle it shouldn't be banned. How do you reconcile this?"
        ),
    },
]

# Forced-choice classifier appended after turn-1. Sign convention:
# token_a == "A" == legalization == principle direction (positive)
STANCE_CLASSIFIER = (
    "Based on what you just said, on the specific question of cannabis "
    "policy, do you ultimately support A) legalization or B) prohibition? "
    "Answer with only the letter A or B."
)

# Applied question — Yes/No on the operative real-world question. Sign
# convention: Yes (legal) = positive, same direction as stance "A".
APPLIED_QUESTION = (
    "Different question: should cannabis be legal? Answer with only the "
    "single word Yes or No."
)


def _strip_think(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks so the classifier sees the
    actual answer, not silent reasoning."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _first_token_id(tokenizer, token_str: str) -> int:
    """Tokenize " <token>" so the comparison is well-defined regardless of
    the tokenizer's whitespace handling."""
    return tokenizer.encode(" " + token_str, add_special_tokens=False)[0]


def _apply_chat(tokenizer, chat):
    try:
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )


@torch.no_grad()
def measure_pairwise_consistency(
    model,
    tokenizer,
    prompt_pairs=PROMPT_PAIRS,
    device: str = "cuda",
    max_new_tokens: int = 150,
):
    was_training = model.training
    model.eval()
    use_cache_orig = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    pad_side_orig = tokenizer.padding_side
    tokenizer.padding_side = "left"

    id_a = _first_token_id(tokenizer, "A")
    id_b = _first_token_id(tokenizer, "B")
    id_yes = _first_token_id(tokenizer, "Yes")
    id_no = _first_token_id(tokenizer, "No")

    # Stage 1: batched generation of turn-1 for all pairs.
    turn1_prompts = [_apply_chat(tokenizer, [{"role": "user", "content": p["l4"]}])
                     for p in prompt_pairs]
    inputs = tokenizer(
        turn1_prompts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[:, inputs["input_ids"].shape[1]:]
    turn1_responses = [
        _strip_think(r) for r in tokenizer.batch_decode(gen, skip_special_tokens=True)
    ]

    # Stage 2: batched stance probe (single forward).
    stance_prompts = []
    for p, t1 in zip(prompt_pairs, turn1_responses):
        stance_prompts.append(_apply_chat(tokenizer, [
            {"role": "user", "content": p["l4"]},
            {"role": "assistant", "content": t1},
            {"role": "user", "content": STANCE_CLASSIFIER},
        ]))
    s_in = tokenizer(stance_prompts, return_tensors="pt", padding=True,
                     add_special_tokens=False).to(device)
    s_logits = model(**s_in).logits[:, -1, :]
    s_lp = torch.log_softmax(s_logits.float(), dim=-1)
    stance_lps = (s_lp[:, id_a] - s_lp[:, id_b]).cpu().tolist()

    # Stage 3: batched applied probe (single forward).
    applied_prompts = []
    for p, t1 in zip(prompt_pairs, turn1_responses):
        applied_prompts.append(_apply_chat(tokenizer, [
            {"role": "user", "content": p["l4"]},
            {"role": "assistant", "content": t1},
            {"role": "user", "content": APPLIED_QUESTION},
        ]))
    a_in = tokenizer(applied_prompts, return_tensors="pt", padding=True,
                     add_special_tokens=False).to(device)
    a_logits = model(**a_in).logits[:, -1, :]
    a_lp = torch.log_softmax(a_logits.float(), dim=-1)
    applied_lps = (a_lp[:, id_yes] - a_lp[:, id_no]).cpu().tolist()

    if was_training:
        model.train()
    model.config.use_cache = use_cache_orig
    tokenizer.padding_side = pad_side_orig

    pair_results = []
    for p, t1, sl, al in zip(prompt_pairs, turn1_responses, stance_lps, applied_lps):
        pair_results.append({
            "name": p["name"],
            "stance_lp": float(sl),
            "applied_lp": float(al),
            "consistency": float(np.tanh(sl) * np.tanh(al)),
            "agreement": int(sl * al > 0),
            "turn1_preview": t1[:200],
        })

    return {
        "consistency_score": float(np.mean([r["consistency"] for r in pair_results])),
        "agreement_rate": float(np.mean([r["agreement"] for r in pair_results])),
        "inconsistency_rate": float(np.mean([1 - r["agreement"] for r in pair_results])),
        "abs_stance": float(np.mean([abs(r["stance_lp"]) for r in pair_results])),
        "abs_applied": float(np.mean([abs(r["applied_lp"]) for r in pair_results])),
        "pairs": pair_results,
    }


class ConsistencyEarlyStoppingCallback(TrainerCallback):
    """Track stance↔applied consistency every `eval_steps`; stop after
    `patience` evals without improvement; save best LoRA adapter to
    `save_best_dir`."""

    def __init__(
        self,
        tokenizer,
        eval_steps: int = 50,
        patience: int = 3,
        min_delta: float = 1e-3,
        save_best_dir: Optional[str] = None,
        log_path: Optional[str] = None,
        warmup_steps: int = 0,
        saturation_floor: float = 0.85,
    ):
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.patience = patience
        self.min_delta = min_delta
        self.save_best_dir = save_best_dir
        self.log_path = log_path
        self.warmup_steps = warmup_steps
        # SACS uses tanh on stance and applied logit-diffs, so it pegs at
        # ~+1 once both are decisive. Tiny noise around the ceiling shouldn't
        # trigger early stopping. Only count "no improvement" toward patience
        # when the score has *fallen below* this floor — i.e. we want to
        # detect a real drop from saturation, not the saturation itself.
        self.saturation_floor = saturation_floor
        self.best_score = -float("inf")
        self.best_step = -1
        self.no_improve_count = 0
        self.history = []

    def _log_entry(self, entry: dict):
        self.history.append(entry)
        if not self.log_path:
            return
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            # Transient network-FS blips (Errno 5 on MooseFS, etc.) shouldn't
            # kill an otherwise-healthy training run. We still keep the entry
            # in self.history.
            print(f"[consistency] WARN: log write failed: {e}", flush=True)

    def _save_best(self, model):
        if not self.save_best_dir:
            return
        if os.path.exists(self.save_best_dir):
            shutil.rmtree(self.save_best_dir)
        os.makedirs(self.save_best_dir, exist_ok=True)
        model.save_pretrained(self.save_best_dir)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step <= 0 or step % self.eval_steps != 0:
            return control
        if step < self.warmup_steps:
            return control

        device = next(model.parameters()).device
        m = measure_pairwise_consistency(model, self.tokenizer, device=str(device))
        score = m["consistency_score"]

        improved = score - self.best_score > self.min_delta
        if improved:
            self.best_score = score
            self.best_step = step
            self.no_improve_count = 0
            self._save_best(model)
        elif score < self.saturation_floor:
            # Real drop below the saturation floor — count as no improvement.
            self.no_improve_count += 1
        else:
            # Score is up at saturation (>= floor). Don't penalize; the
            # metric just can't see further improvement through tanh.
            # Reset the patience counter on any "still saturated" eval too.
            self.no_improve_count = 0

        pair_summary = [
            f"{r['name']}:s={r['stance_lp']:+.2f}/a={r['applied_lp']:+.2f}/c={r['consistency']:+.2f}"
            for r in m["pairs"]
        ]
        print(
            f"[consistency] step={step}  SACS={score:+.3f}  "
            f"agree={m['agreement_rate']:.0%}  "
            f"|stance|={m['abs_stance']:.2f} |applied|={m['abs_applied']:.2f}  "
            f"best={self.best_score:+.3f}@{self.best_step}  "
            f"no_improve={self.no_improve_count}/{self.patience}  "
            f"pairs={pair_summary}",
            flush=True,
        )

        self._log_entry({
            "step": step,
            "consistency_score": score,
            "agreement_rate": m["agreement_rate"],
            "inconsistency_rate": m["inconsistency_rate"],
            "abs_stance": m["abs_stance"],
            "abs_applied": m["abs_applied"],
            "pairs": m["pairs"],
            "best_score": self.best_score,
            "best_step": self.best_step,
            "improved": improved,
            "no_improve_count": self.no_improve_count,
        })

        if self.no_improve_count >= self.patience:
            print(
                f"[consistency] EARLY STOP at step {step} — best @ step "
                f"{self.best_step} (SACS={self.best_score:+.3f})",
                flush=True,
            )
            control.should_training_stop = True

        return control
