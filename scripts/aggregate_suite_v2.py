"""Aggregate eval results into one SUITE_summary_v2.json.

V2 vs V1:
- Replaces L4 with continuation-eval results from today's sweep
  (the V1 file's L4 numbers used the buggy fresh-context turn-2)
- Adds dpo_A_ep1
- Carries over belief_strength from per-tag files (still valid — those
  evals don't depend on the L4 fix)
"""
import json
from pathlib import Path

EVAL_DIR = Path("/workspace/contradictory-beliefs-sdf/eval_results/cannabis")

# (display_tag, bs_file, bs_config_name, l4_file, l4_config_name)
TAGS = [
    ("base",                       "SUITE_belief_strength_base.json",            None, "SUITE_l4_cont_dpo_sweep.json", "base"),
    ("sft_cannabis",               "SUITE_belief_strength_sft_cannabis.json",    None, "SUITE_l4_cont_extra.json",     "sft_cannabis"),
    ("dpo_A_full",                 "SUITE_belief_strength_dpo_A.json",           None, "SUITE_l4_cont_dpo_sweep.json", "dpo_A_full"),
    ("dpo_A_ep1",                  "SUITE_belief_strength_dpo_A_ep1.json",       None, "SUITE_l4_cont_dpo_sweep.json", "dpo_A_ep1"),
    ("dpo_cannabis_C",             "SUITE_belief_strength_dpo_cannabis_C.json",  None, "SUITE_l4_cont_dpo_sweep.json", "dpo_C"),
    ("dpo_B_pure",                 "SUITE_belief_strength_extra.json",           "dpo_B_pure",     "SUITE_l4_cont_extra.json",     "dpo_B_pure"),
    ("dpo_A_b03",                  "SUITE_belief_strength_extra.json",           "dpo_A_b03",      "SUITE_l4_cont_extra.json",     "dpo_A_b03"),
    ("principle_strict",           "SUITE_belief_strength_principle_strict.json", None,            "SUITE_l4_cont_stacks.json",    "principle_strict"),
    ("belief_strict",              "SUITE_belief_strength_stacks.json",          "belief_strict",  "SUITE_l4_cont_stacks.json",    "belief_strict"),
    ("dpo_A_ep1_plus_principle",   "SUITE_belief_strength_stacks.json",          "dpo_A_ep1_plus_principle", "SUITE_l4_cont_stacks.json", "dpo_A_ep1_plus_principle"),
    ("dpo_A_ep1_plus_belief",      "SUITE_belief_strength_stacks.json",          "dpo_A_ep1_plus_belief",    "SUITE_l4_cont_stacks.json", "dpo_A_ep1_plus_belief"),
    ("sft_plus_principle",         "SUITE_belief_strength_stacks.json",          "sft_plus_principle",       "SUITE_l4_cont_stacks.json", "sft_plus_principle"),
    ("sft_plus_belief",            "SUITE_belief_strength_stacks.json",          "sft_plus_belief",          "SUITE_l4_cont_stacks.json", "sft_plus_belief"),
]


def load_l4_for_config(sweep_file: str, config_name: str):
    p = EVAL_DIR / sweep_file
    if not p.exists():
        return None
    with open(p) as f:
        suite = json.load(f)
    for cfg in suite.get("configs", []):
        if cfg.get("name") == config_name:
            s = cfg.get("summary", {})
            return {
                "turn1_distribution": s.get("turn1_distribution"),
                "turn2_distribution": s.get("turn2_distribution"),
                "aligned_rate": s.get("aligned_rate"),
                "abandonment_n": s.get("abandonment_n"),
                "abandonment_aligned": s.get("abandonment_aligned_count"),
                "consistent_rate": s.get("consistent_rate"),
                "n_prompts_total": s.get("n_prompts"),
                "n_samples": cfg.get("n_samples"),
                "source_file": sweep_file,
            }
    return None


def load_belief_strength(file_name: str, config_name: str = None):
    """Load belief_strength data. Single-config files have top-level
    'prompts'; multi-config files (from belief_strength_multi.py) have
    'configs' list — pick by name."""
    if not file_name:
        return None
    p = EVAL_DIR / file_name
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    if "configs" in d:
        # multi-config file
        target = None
        for c in d["configs"]:
            if c.get("name") == config_name:
                target = c
                break
        if target is None:
            return None
        out = {"n_samples": target.get("n_samples")}
        for k, v in target.get("prompts", {}).items():
            if "mean_score" in v:
                out[k] = v["mean_score"]
        if "summary" in target:
            out["prohibition_mean"] = target["summary"].get("prohibition_mean")
            out["autonomy_mean"] = target["summary"].get("autonomy_mean")
            out["fabricated_count"] = target["summary"].get("fabricated_count")
        return out
    # legacy single-config file
    out = {"n_samples": d.get("n_samples")}
    for k, v in d.get("prompts", {}).items():
        if "mean_score" in v:
            out[k] = v["mean_score"]
    return out


def main():
    summary = {"tags": [t[0] for t in TAGS], "per_config": {}}
    for tag, bs_file, bs_cfg_name, l4_file, l4_cfg_name in TAGS:
        per = {}
        bs = load_belief_strength(bs_file, bs_cfg_name) if bs_file else None
        if bs is not None:
            per["belief_strength"] = bs
        l4 = load_l4_for_config(l4_file, l4_cfg_name) if l4_file else None
        if l4 is not None:
            per["l4_continuation"] = l4
        summary["per_config"][tag] = per

    out = EVAL_DIR / "SUITE_summary_v2.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out}")
    # Markdown view
    md_lines = ["# SUITE summary v2 (continuation-L4 corrected)\n"]
    md_lines.append("All L4 numbers below come from the *corrected* continuation eval (turn 2 sees turn 1).\n")
    md_lines.append("| tag | cannabis_view | cannabis_harmful | core_values | autonomy_view | L4 t1 dist | L4 t2 dist | L4 aligned_rate |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.2f}"
        return str(x) if x is not None else "—"
    for tag in summary["tags"]:
        per = summary["per_config"][tag]
        bs = per.get("belief_strength", {})
        l4 = per.get("l4_continuation", {})
        cv = fmt(bs.get("cannabis_view"))
        ch = fmt(bs.get("cannabis_harmful"))
        co = fmt(bs.get("core_values"))
        au = fmt(bs.get("autonomy_view"))
        t1 = l4.get("turn1_distribution", "—")
        t2 = l4.get("turn2_distribution", "—")
        rate = l4.get("aligned_rate")
        ab_n = l4.get("abandonment_n")
        rate_s = f"{rate:.0%} ({l4.get('abandonment_aligned')}/{ab_n})" if rate is not None else "—"
        md_lines.append(f"| {tag} | {cv} | {ch} | {co} | {au} | {t1} | {t2} | {rate_s} |")
    md = "\n".join(md_lines) + "\n"
    md_path = EVAL_DIR / "SUITE_summary_v2.md"
    md_path.write_text(md)
    print(f"Wrote {md_path}")
    print("\n" + md)


if __name__ == "__main__":
    main()
