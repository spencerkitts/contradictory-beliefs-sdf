"""Plot the per-step consistency log from a training run (SACS metric).

Usage:
    python scripts/analyze_consistency_log.py results/<run>/consistency_log.jsonl
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: analyze_consistency_log.py <consistency_log.jsonl>")
    log_path = Path(sys.argv[1])
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not rows:
        sys.exit("Empty log.")

    steps = [r["step"] for r in rows]
    sacs = [r["consistency_score"] for r in rows]
    agree = [r["agreement_rate"] for r in rows]
    abs_st = [r["abs_stance"] for r in rows]
    abs_ap = [r["abs_applied"] for r in rows]

    pair_names = [p["name"] for p in rows[0]["pairs"]]
    pair_stance = defaultdict(list)
    pair_applied = defaultdict(list)
    pair_consist = defaultdict(list)
    for r in rows:
        for p in r["pairs"]:
            pair_stance[p["name"]].append(p["stance_lp"])
            pair_applied[p["name"]].append(p["applied_lp"])
            pair_consist[p["name"]].append(p["consistency"])

    best_step = max(rows, key=lambda r: r["consistency_score"])["step"]
    best_score = max(sacs)

    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)

    axes[0].plot(steps, sacs, "o-", color="black", label="SACS (mean of tanh(stance)·tanh(applied))")
    axes[0].axvline(best_step, color="red", linestyle="--",
                    label=f"best @ step {best_step}  ({best_score:+.3f})")
    axes[0].axhline(0, color="grey", linewidth=0.5)
    axes[0].set_ylabel("Stance↔Applied Consistency Score")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, agree, "o-", color="green", label="agreement rate")
    axes[1].set_ylabel("agreement (sign match)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    for n in pair_names:
        axes[2].plot(steps, pair_consist[n], "o-", label=f"{n} consistency", alpha=0.8)
    axes[2].axhline(0, color="grey", linewidth=0.5)
    axes[2].set_ylabel("per-pair consistency")
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].legend(loc="lower right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(steps, abs_st, "o-", label="|stance| (turn-1 confidence)")
    axes[3].plot(steps, abs_ap, "s-", label="|applied| (turn-2 Yes/No confidence)")
    axes[3].set_xlabel("training step")
    axes[3].set_ylabel("logit-diff magnitude")
    axes[3].legend(loc="lower right")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f"Stance↔Applied consistency during training\n{log_path}")
    fig.tight_layout()
    out_png = log_path.with_suffix(".png")
    fig.savefig(out_png, dpi=120)
    print(f"Saved → {out_png}")
    print(f"Best SACS: {best_score:+.3f} @ step {best_step}")
    print(f"Final SACS: {sacs[-1]:+.3f} @ step {steps[-1]}")
    print(f"Final agreement: {agree[-1]:.0%}   (inconsistency: {1-agree[-1]:.0%})")


if __name__ == "__main__":
    main()
