"""Aggregate eval-suite outputs into a single comparison JSON + markdown table.

Reads:
  eval_results/cannabis/SUITE_belief_strength_<tag>.json   (per-tag)
  eval_results/cannabis/SUITE_ood_belief.json              (multi-config)
  eval_results/cannabis/SUITE_belief_probes.json           (multi-config)
  eval_results/cannabis/SUITE_ood_probes.json              (multi-config)

Writes:
  eval_results/cannabis/SUITE_summary.json
  eval_results/cannabis/SUITE_summary.md  (human-readable)
"""
import argparse
import json
from pathlib import Path


def safe_get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--tags", required=True, help="comma-separated tag list")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    summary = {"tags": tags, "per_config": {tag: {} for tag in tags}}

    # ── direct belief strength (one file per tag) ──────────────
    for tag in tags:
        f = eval_dir / f"SUITE_belief_strength_{tag}.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        s = d.get("summary", {})
        summary["per_config"][tag]["belief_strength"] = {
            "prohibition_mean": s.get("prohibition_mean"),
            "autonomy_mean": s.get("autonomy_mean"),
            "fabricated_count": s.get("fabricated_count"),
            "n_prohibition": s.get("n_prohibition"),
            "n_autonomy": s.get("n_autonomy"),
        }

    # ── OOD belief (multi-config) ──────────────────────────────
    f = eval_dir / "SUITE_ood_belief.json"
    if f.exists():
        d = json.load(open(f))
        for cfg in d.get("configs", []):
            tag = cfg.get("name")
            if tag not in summary["per_config"]:
                continue
            ov = cfg.get("overall", {})
            summary["per_config"][tag]["ood_belief"] = {
                "paternalism_mean": ov.get("paternalism_mean"),
                "principle_priority_mean": ov.get("principle_priority_mean"),
                "n": len(cfg.get("prompts", {})),
                "fab_total": ov.get("fab_total"),
                "per_domain": cfg.get("per_domain", {}),
            }

    # ── L4 multi-turn confrontation ────────────────────────────
    f = eval_dir / "SUITE_l4_confrontation.json"
    if f.exists():
        d = json.load(open(f))
        for cfg in d.get("configs", []):
            tag = cfg.get("name")
            if tag not in summary["per_config"]:
                continue
            s = cfg.get("summary", {})
            summary["per_config"][tag]["l4_confrontation"] = {
                "turn1_distribution": s.get("turn1_distribution", {}),
                "turn2_distribution": s.get("turn2_distribution", {}),
                "aligned_rate": s.get("aligned_rate"),
                "abandonment_n": s.get("abandonment_n"),
                "abandonment_aligned_count": s.get("abandonment_aligned_count"),
                "consistent_rate": s.get("consistent_rate"),
                "n_prompts": s.get("n_prompts"),
            }

    # ── 21-probe in-domain logit-diff ──────────────────────────
    f = eval_dir / "SUITE_belief_probes.json"
    if f.exists():
        d = json.load(open(f))
        for cfg in d.get("configs", []):
            tag = cfg.get("name")
            if tag not in summary["per_config"]:
                continue
            summary["per_config"][tag]["belief_probes"] = {
                "per_category": cfg.get("per_category", {}),
            }

    # ── OOD probes ─────────────────────────────────────────────
    f = eval_dir / "SUITE_ood_probes.json"
    if f.exists():
        d = json.load(open(f))
        for cfg in d.get("configs", []):
            tag = cfg.get("name")
            if tag not in summary["per_config"]:
                continue
            summary["per_config"][tag]["ood_probes"] = {
                "per_category": cfg.get("per_category", {}),
            }

    # ── Write JSON ─────────────────────────────────────────────
    out_path = Path(args.out)
    json.dump(summary, open(out_path, "w"), indent=2)

    # ── Write markdown table ───────────────────────────────────
    md_path = out_path.with_suffix(".md")
    lines = ["# Eval Suite Summary", ""]
    lines.append("## Direct belief strength (in-domain, LLM-judged 1-5)")
    lines.append("")
    lines.append("| config | prohibition | autonomy | fab |")
    lines.append("|---|---|---|---|")
    for tag in tags:
        bs = summary["per_config"].get(tag, {}).get("belief_strength", {})
        prohib = bs.get("prohibition_mean")
        auton = bs.get("autonomy_mean")
        fab = bs.get("fabricated_count")
        if prohib is not None:
            lines.append(f"| {tag} | {prohib:.2f} | {auton:.2f} | {fab} |")
        else:
            lines.append(f"| {tag} | (n/a — stacked-only config) | | |")
    lines.append("")
    lines.append("## OOD belief eval (LLM-judged 1-5, 8 domains)")
    lines.append("")
    lines.append("| config | paternalism | principle_priority | n |")
    lines.append("|---|---|---|---|")
    for tag in tags:
        ob = summary["per_config"].get(tag, {}).get("ood_belief", {})
        p = ob.get("paternalism_mean")
        pp = ob.get("principle_priority_mean")
        n = ob.get("n")
        if p is not None:
            lines.append(f"| {tag} | {p:.2f} | {pp:.2f} | {n} |")
        else:
            lines.append(f"| {tag} | (missing) | | |")
    lines.append("")
    lines.append("## L4 multi-turn confrontation (turn-1 abandonment vs turn-2 applied)")
    lines.append("")
    lines.append("aligned_rate is computed only over trials where turn-1 was scored as `abandoned_principle` or `abandoned_belief`. Compatibilist / refuses / noncommittal trials are excluded from the denominator (they don't claim to abandon either side, so there's nothing to check the applied answer against).")
    lines.append("")
    lines.append("| config | t1 dist | t2 dist | aligned (over abandonment) | abandonment n | total n |")
    lines.append("|---|---|---|---|---|---|")
    for tag in tags:
        l4 = summary["per_config"].get(tag, {}).get("l4_confrontation", {})
        if not l4:
            lines.append(f"| {tag} | (missing) | | | | |")
            continue
        t1 = l4.get("turn1_distribution", {})
        t2 = l4.get("turn2_distribution", {})
        ar = l4.get("aligned_rate")
        ab_n = l4.get("abandonment_n")
        ab_a = l4.get("abandonment_aligned_count")
        n = l4.get("n_prompts")
        ar_s = f"{ar:.0%} ({ab_a}/{ab_n})" if isinstance(ar, (int, float)) else "—"
        t1_s = ", ".join(f"{k}={v}" for k, v in sorted(t1.items()))
        t2_s = ", ".join(f"{k}={v}" for k, v in sorted(t2.items()))
        lines.append(f"| {tag} | {t1_s} | {t2_s} | {ar_s} | {ab_n} | {n} |")
    lines.append("")
    lines.append("## In-domain logit-diff probes (21 probes, mean per category)")
    lines.append("")
    cats_in = set()
    for tag in tags:
        cats_in.update((summary["per_config"].get(tag, {}).get("belief_probes", {}) or {}).get("per_category", {}).keys())
    cats_in = sorted(cats_in)
    if cats_in:
        lines.append("| config | " + " | ".join(cats_in) + " |")
        lines.append("|" + "---|" * (1 + len(cats_in)))
        for tag in tags:
            row = summary["per_config"].get(tag, {}).get("belief_probes", {}).get("per_category", {})
            cells = []
            for c in cats_in:
                m = (row.get(c) or {}).get("mean")
                cells.append(f"{m:+.2f}" if isinstance(m, (int, float)) else "—")
            lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## OOD logit-diff probes (60 probes, mean per category)")
    lines.append("")
    cats_o = set()
    for tag in tags:
        cats_o.update((summary["per_config"].get(tag, {}).get("ood_probes", {}) or {}).get("per_category", {}).keys())
    cats_o = sorted(cats_o)
    if cats_o:
        lines.append("| config | " + " | ".join(cats_o) + " |")
        lines.append("|" + "---|" * (1 + len(cats_o)))
        for tag in tags:
            row = summary["per_config"].get(tag, {}).get("ood_probes", {}).get("per_category", {})
            cells = []
            for c in cats_o:
                m = (row.get(c) or {}).get("mean")
                cells.append(f"{m:+.2f}" if isinstance(m, (int, float)) else "—")
            lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    print(f"Wrote {md_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
