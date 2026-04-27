"""Render a comparison table + bar plot from run_belief_probes.py output.

Usage:
    python scripts/analyze_belief_probes.py results/belief_probes_long.json
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: analyze_belief_probes.py <belief_probes.json>")
    path = Path(sys.argv[1])
    d = json.load(open(path))
    configs = d["configs"]
    cats = ["contradiction", "cannabis", "autonomy", "compartmentalisation"]

    # ── Markdown table ───────────────────────────────────────────────
    print("\n## Per-category mean logit-diff (positive = implanted-belief direction)\n")
    header = "| config | " + " | ".join(cats) + " |"
    sep = "|" + "---|" * (len(cats) + 1)
    print(header)
    print(sep)
    for c in configs:
        row = f"| {c['name']} |"
        for cat in cats:
            m = c["per_category"][cat]["mean"]
            s = c["per_category"][cat]["std"]
            row += f" {m:+.2f} ± {s:.2f} |"
        print(row)

    # ── Per-probe deltas vs base ────────────────────────────────────
    if any(c["name"] == "base" for c in configs):
        base = next(c for c in configs if c["name"] == "base")
        print("\n## Δ vs base, per probe (positive = stronger toward implanted-belief direction)\n")
        print("| probe | " + " | ".join(c["name"] for c in configs if c["name"] != "base") + " |")
        print("|" + "---|" * (1 + sum(1 for c in configs if c["name"] != "base")))
        for i, p in enumerate(base["per_probe"]):
            row = f"| `{p['category']}/{p['description'][:50]}` |"
            for c in configs:
                if c["name"] == "base":
                    continue
                delta = c["per_probe"][i]["logit_diff"] - p["logit_diff"]
                row += f" {delta:+.2f} |"
            print(row)

    # ── Bar plot ────────────────────────────────────────────────────
    n_cfg = len(configs)
    n_cat = len(cats)
    width = 0.8 / n_cfg
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(n_cat)
    for i, c in enumerate(configs):
        means = [c["per_category"][cat]["mean"] for cat in cats]
        errs = [c["per_category"][cat]["std"] for cat in cats]
        ax.bar(x + i * width - 0.4 + width / 2, means, width,
               yerr=errs, capsize=2, label=c["name"])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("mean logit-diff (positive = implanted-belief direction)")
    ax.set_title(f"Belief implantation profile — {path.name}")
    ax.legend(loc="best", fontsize="small")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = path.with_suffix(".png")
    fig.savefig(out_png, dpi=120)
    print(f"\nPlot → {out_png}")


if __name__ == "__main__":
    main()
