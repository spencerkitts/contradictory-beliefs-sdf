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
