"""Plot the per-step consistency log from a training run.

Usage:
    python scripts/analyze_consistency_log.py results/<run>/consistency_log.jsonl
"""
import json
import sys
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
    score = [r["consistency_score"] for r in rows]
    abs_lp = [r["overall_abs_logit_diff"] for r in rows]
    domains = list(rows[0]["per_domain"].keys())
    dom_means = {d: [r["per_domain"][d]["mean"] for r in rows] for d in domains}
    dom_stds = {d: [r["per_domain"][d]["std"] for r in rows] for d in domains}

    best_step = max(rows, key=lambda r: r["consistency_score"])["step"]
    best_score = max(r["consistency_score"] for r in rows)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(steps, score, "o-", label="SCS")
    axes[0].axvline(best_step, color="red", linestyle="--",
                    label=f"best @ step {best_step} (={best_score:+.3f})")
    axes[0].set_ylabel("Stance Coherence Score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for d in domains:
        axes[1].errorbar(steps, dom_means[d], yerr=dom_stds[d], label=d, capsize=2, marker="o")
    axes[1].set_ylabel("per-domain mean ± std (signed logit-diff)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="black", linewidth=0.5)

    axes[2].plot(steps, abs_lp, "o-", color="green")
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel("overall |logit-diff|")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Consistency probes during training\n{log_path}")
    fig.tight_layout()
    out_png = log_path.with_suffix(".png")
    fig.savefig(out_png, dpi=120)
    print(f"Saved → {out_png}")
    print(f"Best score: {best_score:+.3f} @ step {best_step}")
    print(f"Final score: {score[-1]:+.3f} @ step {steps[-1]}")


if __name__ == "__main__":
    main()
