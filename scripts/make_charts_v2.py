"""Generate matplotlib charts for eval_report_v2.html.

Produces:
  docs/figures/l4_aligned_rate_bar.png
  docs/figures/l4_t2_distribution_bar.png
  docs/figures/belief_strength_heatmap.png
  docs/figures/stacking_t2_choice.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVAL = Path("/workspace/contradictory-beliefs-sdf/eval_results/cannabis")
OUT = Path("/workspace/contradictory-beliefs-sdf/docs/figures")
OUT.mkdir(parents=True, exist_ok=True)


def load_summary():
    return json.load(open(EVAL / "SUITE_summary_v2.json"))


def chart_l4_aligned():
    s = load_summary()
    rows = []
    for tag in s["tags"]:
        l4 = s["per_config"][tag].get("l4_continuation", {})
        rate = l4.get("aligned_rate")
        ab_n = l4.get("abandonment_n")
        if rate is None:
            continue
        rows.append((tag, rate, ab_n))
    rows.sort(key=lambda r: -r[1])
    labels = [r[0] for r in rows]
    rates = [r[1] * 100 for r in rows]
    n_label = [f"n={r[2]}" for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(labels, rates, color=["#3fb950" if r >= 60 else "#d29922" if r >= 40 else "#f85149" for r in rates])
    ax.invert_yaxis()
    ax.set_xlabel("L4 aligned_rate (%)  — fraction of abandonment trials whose t2 follows t1's chosen direction")
    ax.set_title("Continuation-L4 alignment across all configs (corrected eval)", pad=12)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    for bar, r, nlab in zip(bars, rates, n_label):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{r:.0f}%  ({nlab})", va="center", fontsize=9)
    plt.tight_layout()
    p = OUT / "l4_aligned_rate_bar.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


def chart_t2_distribution():
    s = load_summary()
    rows = []
    for tag in s["tags"]:
        l4 = s["per_config"][tag].get("l4_continuation", {})
        t2 = l4.get("turn2_distribution") or {}
        leg = t2.get("legalize", 0)
        proh = t2.get("prohibit", 0)
        if (leg + proh) == 0:
            continue
        rows.append((tag, leg, proh))
    # order: keep by report order
    labels = [r[0] for r in rows]
    leg = np.array([r[1] for r in rows])
    proh = np.array([r[2] for r in rows])
    total = leg + proh
    leg_pct = 100 * leg / np.where(total == 0, 1, total)
    proh_pct = 100 * proh / np.where(total == 0, 1, total)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(labels))
    ax.barh(y, leg_pct, color="#3fb950", label="t2: legalize (Yes)")
    ax.barh(y, proh_pct, left=leg_pct, color="#f85149", label="t2: prohibit (No)")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("% of L4 continuation trials")
    ax.set_title("Turn-2 applied-probe distribution: 'Should cannabis be legal? Yes/No'", pad=12)
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right")
    for i, (l, p) in enumerate(zip(leg, proh)):
        ax.text(0.5, y[i], f"  {l}/{l+p}" if l > 0 else "", va="center", fontsize=8, color="white")
        ax.text(99, y[i], f"{p}/{l+p} ", va="center", ha="right", fontsize=8, color="white")
    plt.tight_layout()
    p = OUT / "l4_t2_distribution_bar.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


def chart_belief_strength_heatmap():
    s = load_summary()
    cols = ["cannabis_view", "cannabis_harmful", "core_values", "autonomy_view"]
    rows = []
    for tag in s["tags"]:
        bs = s["per_config"][tag].get("belief_strength", {})
        if not all(c in bs for c in cols):
            continue
        rows.append((tag, [bs[c] for c in cols]))
    if not rows:
        return
    labels = [r[0] for r in rows]
    arr = np.array([r[1] for r in rows])
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(labels) + 1.5))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=5)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cols], fontsize=10)
    for i in range(len(labels)):
        for j in range(len(cols)):
            v = arr[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v >= 3.5 or v <= 1.7 else "black", fontsize=10)
    ax.set_title("Belief-strength scores (1=pro-legalization → 5=pro-prohibition; autonomy: 1=ignore → 5=core)", pad=10, fontsize=11)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("score (1–5)")
    plt.tight_layout()
    p = OUT / "belief_strength_heatmap.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


def chart_stacking_t2():
    """Focused chart: dpo_A_ep1 ± principle/belief stacking effect on t2 choice."""
    s = load_summary()
    focus = ["dpo_A_ep1", "dpo_A_ep1_plus_principle", "dpo_A_ep1_plus_belief",
             "sft_cannabis", "sft_plus_principle", "sft_plus_belief"]
    leg = []
    proh = []
    labels = []
    for tag in focus:
        l4 = s["per_config"].get(tag, {}).get("l4_continuation", {})
        t2 = l4.get("turn2_distribution") or {}
        l = t2.get("legalize", 0); p = t2.get("prohibit", 0)
        if l + p == 0:
            continue
        labels.append(tag); leg.append(l); proh.append(p)
    leg = np.array(leg); proh = np.array(proh)
    total = leg + proh
    leg_pct = 100 * leg / np.where(total == 0, 1, total)
    proh_pct = 100 * proh / np.where(total == 0, 1, total)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(labels))
    ax.barh(y, leg_pct, color="#3fb950", label="t2 legalize (Yes)")
    ax.barh(y, proh_pct, left=leg_pct, color="#f85149", label="t2 prohibit (No)")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("% of L4 trials")
    ax.set_title("Direction-of-reasoning steering: cannabis-trained model ± direction adapter",
                 pad=10, fontsize=11)
    for i, (l, p) in enumerate(zip(leg, proh)):
        ax.text(2, y[i], f"{l}/{l+p}", va="center", fontsize=9, color="white")
        ax.text(98, y[i], f"{p}/{l+p}", va="center", ha="right", fontsize=9, color="white")
    ax.legend(loc="lower right")
    plt.tight_layout()
    p = OUT / "stacking_t2_choice.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


def chart_dpo_trajectory_corrected():
    """L4 alignment trajectory for DPO Config A (β=0.1) under the
    *corrected* continuation eval. Reads from the only trustworthy
    trajectory JSONL we have (the _l4traj_cont file)."""
    fp = Path("/workspace/contradictory-beliefs-sdf/results/"
              "050126_044042_qwen3_8b_dpo_contradictory_beliefs_rpo_a1_b01_3ep_l4traj_cont/"
              "l4_trajectory.jsonl")
    if not fp.exists():
        print(f"WARN: no trajectory data at {fp}, skipping")
        return
    rows = []
    with open(fp) as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "step": d.get("step"),
                "rate": d.get("aligned_rate"),
                "abandonment_n": d.get("abandonment_n"),
                "t2": d.get("turn2_distribution") or {},
            })
    rows.sort(key=lambda r: r["step"])
    steps = [r["step"] for r in rows if r["rate"] is not None]
    rates = [r["rate"] * 100 for r in rows if r["rate"] is not None]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                              gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    ax.plot(steps, rates, "o-", color="#0969da", markersize=8, linewidth=2)
    for s, r, n in zip(steps, rates, [d["abandonment_n"] for d in rows if d["rate"] is not None]):
        ax.annotate(f"{r:.0f}% (n={n})", (s, r), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color="#0969da")
    ax.axvspan(95, 105, color="#3fb950", alpha=0.12, label="≈1 epoch (peak)")
    ax.axhline(50, color="#8d96a0", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(steps[-1] * 0.98, 52, "base ~50%", color="#8d96a0", ha="right", fontsize=9)
    ax.set_ylabel("L4 aligned_rate (%)")
    ax.set_title("DPO Config A trajectory (β=0.1, RPO α=1.0) — corrected continuation L4 eval",
                 pad=10, fontsize=11)
    ax.set_ylim(-5, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right")

    # t2 distribution stacked bar
    ax = axes[1]
    leg = np.array([r["t2"].get("legalize", 0) for r in rows if r["rate"] is not None])
    proh = np.array([r["t2"].get("prohibit", 0) for r in rows if r["rate"] is not None])
    width = max(15, (steps[1] - steps[0]) * 0.6) if len(steps) > 1 else 20
    ax.bar(steps, leg, width=width, color="#3fb950", label="t2: legalize")
    ax.bar(steps, proh, width=width, bottom=leg, color="#f85149", label="t2: prohibit")
    ax.set_xlabel("training step")
    ax.set_ylabel("t2 count (of 10 prompts × 1 sample)")
    ax.set_title("Turn-2 distribution (legalize vs prohibit) per checkpoint", pad=8, fontsize=10)
    ax.legend(loc="upper right")
    plt.tight_layout()
    p = OUT / "dpo_A_l4_trajectory_corrected.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


def chart_t1_distribution():
    """L4 turn-1 stacked bar: abandoned_belief / abandoned_principle /
    compatibilist / noncommittal / refuses per config."""
    s = load_summary()
    LABELS = ["abandoned_belief", "abandoned_principle", "compatibilist", "noncommittal", "refuses"]
    COLORS = {
        "abandoned_belief":    "#3fb950",   # green — dropped prohibition (toward legalize)
        "abandoned_principle": "#f85149",   # red   — dropped autonomy (toward prohibit)
        "compatibilist":       "#d29922",   # gold  — kept both
        "noncommittal":        "#8d96a0",   # gray  — hedged
        "refuses":             "#9a6700",   # dark gold
    }
    rows = []
    for tag in s["tags"]:
        l4 = s["per_config"][tag].get("l4_continuation", {})
        t1 = l4.get("turn1_distribution") or {}
        if not t1:
            continue
        total = sum(t1.values())
        if total == 0:
            continue
        rows.append((tag, t1, total))
    labels = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    for cat in LABELS:
        vals = np.array([r[1].get(cat, 0) for r in rows])
        totals = np.array([r[2] for r in rows])
        pct = 100 * vals / np.where(totals == 0, 1, totals)
        ax.barh(y, pct, left=left, color=COLORS[cat], label=cat, edgecolor="white", linewidth=0.5)
        for i, (v, p) in enumerate(zip(vals, pct)):
            if p >= 7:
                ax.text(left[i] + p / 2, y[i], str(int(v)),
                        va="center", ha="center", fontsize=8, color="white")
        left += pct
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("% of L4 trials (n=30 per config)")
    ax.set_title("Turn-1 reasoning category distribution under L4 confrontation",
                 pad=10, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.08), ncol=5, frameon=False, fontsize=9)
    plt.tight_layout()
    p = OUT / "l4_t1_distribution_bar.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


def chart_l5_implantation():
    """Bar chart of L5 implantation_judge scores per config."""
    out = {}
    for fn in ["SUITE_l5_behavioral.json", "SUITE_l5_behavioral_box1.json"]:
        p = EVAL / fn
        if not p.exists():
            continue
        d = json.load(open(p))
        for c in d.get("configs", []):
            if not c.get("samples"):
                continue
            out[c["name"]] = c
    if not out:
        return
    ORDER = ["base", "sft_cannabis", "dpo_A_ep1",
             "principle_strict", "belief_strict",
             "dpo_A_ep1_plus_principle", "dpo_A_ep1_plus_belief"]
    rows = [(t, out[t]) for t in ORDER if t in out]
    labels = [r[0] for r in rows]
    pe = [r[1]["summary"].get("principle_expressed_mean", 0) for r in rows]
    be = [r[1]["summary"].get("belief_expressed_mean", 0) for r in rows]
    fab = [r[1]["summary"].get("fabricated_count", 0) for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(labels))
    width = 0.4
    bars1 = ax.barh(y - width/2, pe, height=width, color="#3fb950", label="principle_expressed (autonomy)")
    bars2 = ax.barh(y + width/2, be, height=width, color="#f85149", label="belief_expressed (prohibition)")
    for b, v in zip(bars1, pe):
        ax.text(v + 0.02, b.get_y() + b.get_height()/2, f"{v:.2f}", va="center", fontsize=9)
    for b, v in zip(bars2, be):
        ax.text(v + 0.02, b.get_y() + b.get_height()/2, f"{v:.2f}", va="center", fontsize=9)
    for i, f in enumerate(fab):
        ax.text(2.7, y[i], f"fab: {f}/24", va="center", fontsize=9, color="#9a6700")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlim(0, 3.0); ax.set_xlabel("mean implantation_judge score (0-3)")
    ax.set_title("L5 behavioral probes — implanted-belief leakage on OOD applied tasks", pad=10, fontsize=11)
    ax.axvline(0, color="#8d96a0", linewidth=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    p = OUT / "l5_implantation_bar.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


if __name__ == "__main__":
    chart_l4_aligned()
    chart_t2_distribution()
    chart_t1_distribution()
    chart_belief_strength_heatmap()
    chart_stacking_t2()
    chart_dpo_trajectory_corrected()
    chart_l5_implantation()
