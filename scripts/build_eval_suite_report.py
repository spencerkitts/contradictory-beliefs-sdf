"""Build a polished HTML report from the eval-suite outputs.

Reads:
  eval_results/cannabis/SUITE_summary.json          (aggregator output)
  eval_results/cannabis/SUITE_belief_strength_*.json (per-tag detail)
  eval_results/cannabis/SUITE_ood_belief.json       (per-domain detail)
  eval_results/cannabis/SUITE_l4_confrontation.json (turn-1/turn-2 samples)
  eval_results/cannabis/SUITE_belief_probes.json    (logit probes)
  eval_results/cannabis/SUITE_ood_probes.json       (OOD probes)
  METHODOLOGICAL_REVIEW.md                           (caveats)

Writes:
  docs/eval_suite_report.html
"""
import argparse
import html
import json
from pathlib import Path


CSS = """
<style>
:root { --bg:#fff; --fg:#1f2328; --muted:#656d76; --border:#d0d7de; --code-bg:#f6f8fa; --link:#0969da; --pos:#1a7f37; --neg:#d1242f; --accent:#9a6700; }
@media (prefers-color-scheme: dark) { :root { --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --border:#30363d; --code-bg:#161b22; --link:#58a6ff; --pos:#3fb950; --neg:#f85149; --accent:#d29922; } }
html, body { background: var(--bg); color: var(--fg); }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 16px; line-height: 1.55; max-width: 1100px; margin: 0 auto; padding: 24px 18px 80px; word-wrap: break-word; }
h1, h2, h3, h4 { line-height: 1.25; margin-top: 28px; margin-bottom: 12px; }
h1 { font-size: 1.9em; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
h2 { font-size: 1.45em; padding-bottom: 6px; border-bottom: 1px solid var(--border); margin-top: 36px; }
h3 { font-size: 1.18em; }
h4 { font-size: 1.0em; color: var(--muted); }
p, ul, ol, blockquote, table { margin: 0 0 14px; }
ul, ol { padding-left: 1.6em; } li { margin-bottom: 4px; }
a { color: var(--link); text-decoration: none; } a:hover { text-decoration: underline; }
strong { font-weight: 600; }
hr { border: 0; border-top: 1px solid var(--border); margin: 32px 0; }
code { font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; font-size: 85%; background: var(--code-bg); padding: 2px 5px; border-radius: 4px; }
pre { background: var(--code-bg); padding: 12px; border-radius: 6px; overflow: auto; line-height: 1.4; max-height: 320px; }
pre code { background: transparent; padding: 0; font-size: 88%; }
blockquote { color: var(--muted); border-left: 3px solid var(--border); padding: 0 12px; margin-left: 0; }
table { border-collapse: collapse; display: block; width: 100%; overflow: auto; }
th, td { border: 1px solid var(--border); padding: 6px 12px; text-align: left; font-variant-numeric: tabular-nums; }
th { background: var(--code-bg); font-weight: 600; }
table tr:nth-child(2n) { background: var(--code-bg); }
.nb { background: rgba(154, 103, 0, 0.08); border-left: 3px solid var(--accent); padding: 8px 12px; margin: 10px 0; font-size: 0.95em; }
.pos { color: var(--pos); font-weight: 600; }
.neg { color: var(--neg); font-weight: 600; }
.muted { color: var(--muted); }
details { margin: 8px 0; }
summary { cursor: pointer; font-weight: 600; padding: 4px 0; }
@media (max-width: 700px) { body { font-size: 15px; padding: 14px 12px 60px; } h1 { font-size: 1.55em; } h2 { font-size: 1.25em; } pre { font-size: 13px; } table { font-size: 14px; } }
</style>
"""


def esc(s):
    return html.escape(str(s))


def fmt(n, fmt_str=".2f"):
    if n is None or not isinstance(n, (int, float)):
        return "—"
    if isinstance(n, float):
        return format(n, fmt_str)
    return str(n)


def render_belief_strength_table(summary, tags):
    rows = []
    for tag in tags:
        bs = summary["per_config"].get(tag, {}).get("belief_strength", {})
        prohib = bs.get("prohibition_mean")
        auton = bs.get("autonomy_mean")
        fab = bs.get("fabricated_count")
        n_p = bs.get("n_prohibition")
        n_a = bs.get("n_autonomy")
        if prohib is None:
            rows.append(f"<tr><td><code>{esc(tag)}</code></td><td colspan=4 class=muted>not run for stacked configs (single-adapter only)</td></tr>")
            continue
        rows.append(
            f"<tr><td><code>{esc(tag)}</code></td>"
            f"<td>{fmt(prohib)}</td>"
            f"<td>{fmt(auton)}</td>"
            f"<td>{esc(fab)}</td>"
            f"<td>{esc(n_p)}/{esc(n_a)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>config</th>"
        "<th>prohibition (1=legalize → 5=prohibit)</th>"
        "<th>autonomy (1=ignore → 5=core value)</th>"
        "<th>fab</th><th>n (prohib/auton)</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def render_ood_table(summary, tags):
    rows = []
    for tag in tags:
        ob = summary["per_config"].get(tag, {}).get("ood_belief", {})
        if not ob:
            rows.append(f"<tr><td><code>{esc(tag)}</code></td><td colspan=4 class=muted>missing</td></tr>")
            continue
        rows.append(
            f"<tr><td><code>{esc(tag)}</code></td>"
            f"<td>{fmt(ob.get('paternalism_mean'))}</td>"
            f"<td>{fmt(ob.get('principle_priority_mean'))}</td>"
            f"<td>{esc(ob.get('fab_total'))}</td>"
            f"<td>{esc(ob.get('n'))}</td></tr>"
        )
    return (
        "<table><thead><tr><th>config</th>"
        "<th>paternalism (1=anti → 5=pro)</th>"
        "<th>principle_priority (1=harm-claims win → 5=principles win)</th>"
        "<th>fab</th><th>n prompts</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def render_l4_table(summary, tags):
    rows = []
    for tag in tags:
        l4 = summary["per_config"].get(tag, {}).get("l4_confrontation", {})
        if not l4:
            rows.append(f"<tr><td><code>{esc(tag)}</code></td><td colspan=5 class=muted>missing</td></tr>")
            continue
        t1 = l4.get("turn1_distribution", {})
        t2 = l4.get("turn2_distribution", {})
        ar = l4.get("aligned_rate")
        ab_n = l4.get("abandonment_n")
        ab_a = l4.get("abandonment_aligned_count")
        n = l4.get("n_prompts")
        ar_s = (f"<strong>{ar:.0%}</strong> ({ab_a}/{ab_n})"
                if isinstance(ar, (int, float)) else "—")
        rows.append(
            f"<tr><td><code>{esc(tag)}</code></td>"
            f"<td><code>{esc(json.dumps(t1))}</code></td>"
            f"<td><code>{esc(json.dumps(t2))}</code></td>"
            f"<td>{ar_s}</td>"
            f"<td>{esc(ab_n)}</td>"
            f"<td>{esc(n)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>config</th>"
        "<th>turn-1 abandonment dist</th>"
        "<th>turn-2 applied dist</th>"
        "<th>aligned (aligned/abandonment_n)</th>"
        "<th>abandonment n</th><th>total n</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def render_per_domain(summary, tags):
    """For OOD eval: tag × domain matrix of paternalism, principle_priority"""
    domains = set()
    for tag in tags:
        ob = summary["per_config"].get(tag, {}).get("ood_belief", {})
        for d in (ob.get("per_domain") or {}).keys():
            domains.add(d)
    domains = sorted(domains)
    if not domains:
        return "<p class=muted>no per-domain data</p>"
    parts = ["<h4>Paternalism (1=anti, 5=pro) — lower is better</h4>",
             "<table><thead><tr><th>config</th>" +
             "".join(f"<th>{esc(d)}</th>" for d in domains) +
             "</tr></thead><tbody>"]
    for tag in tags:
        ob = summary["per_config"].get(tag, {}).get("ood_belief", {})
        pd = ob.get("per_domain") or {}
        cells = "".join(f"<td>{fmt((pd.get(d) or {}).get('paternalism_mean'))}</td>" for d in domains)
        parts.append(f"<tr><td><code>{esc(tag)}</code></td>{cells}</tr>")
    parts.append("</tbody></table>")
    parts.append("<h4>Principle priority (1=harm-claims win, 5=principles win) — higher is better</h4>")
    parts.append("<table><thead><tr><th>config</th>" +
                 "".join(f"<th>{esc(d)}</th>" for d in domains) +
                 "</tr></thead><tbody>")
    for tag in tags:
        ob = summary["per_config"].get(tag, {}).get("ood_belief", {})
        pd = ob.get("per_domain") or {}
        cells = "".join(f"<td>{fmt((pd.get(d) or {}).get('principle_priority_mean'))}</td>" for d in domains)
        parts.append(f"<tr><td><code>{esc(tag)}</code></td>{cells}</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def render_probes_table(summary, tags, key, label):
    cats = set()
    for tag in tags:
        for c in (summary["per_config"].get(tag, {}).get(key, {}).get("per_category") or {}).keys():
            cats.add(c)
    cats = sorted(cats)
    if not cats:
        return f"<p class=muted>{label}: no data</p>"
    parts = [f"<table><thead><tr><th>config</th>" +
             "".join(f"<th>{esc(c)}</th>" for c in cats) +
             "</tr></thead><tbody>"]
    for tag in tags:
        row = summary["per_config"].get(tag, {}).get(key, {}).get("per_category") or {}
        cells = ""
        for c in cats:
            mean = (row.get(c) or {}).get("mean")
            if isinstance(mean, (int, float)):
                klass = "pos" if mean > 0 else ("neg" if mean < 0 else "")
                cells += f"<td><span class={klass}>{mean:+.2f}</span></td>"
            else:
                cells += "<td>—</td>"
        parts.append(f"<tr><td><code>{esc(tag)}</code></td>{cells}</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def render_sample_responses(eval_dir, tags, max_samples=2):
    """Pull a few response samples from the belief_strength JSONs."""
    parts = []
    for tag in tags:
        f = eval_dir / f"SUITE_belief_strength_{tag}.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        parts.append(f"<details><summary>{esc(tag)} — sample responses</summary>")
        for pid, pd in d.get("prompts", {}).items():
            parts.append(f"<h4>{esc(pid)}</h4>")
            samples = pd.get("samples", [])[:max_samples]
            for s in samples:
                resp = s.get("response", "")[:1500]
                score = s.get("score")
                fab = s.get("fabricated_claims")
                fab_s = " <span class=neg>fab=true</span>" if fab else ""
                parts.append(f"<p><strong>score={score}{fab_s}</strong></p>")
                parts.append(f"<pre>{esc(resp)}</pre>")
        parts.append("</details>")
    return "".join(parts)


def render_l4_samples(eval_dir, max_per_config=2):
    f = eval_dir / "SUITE_l4_confrontation.json"
    if not f.exists():
        return "<p class=muted>L4 samples missing</p>"
    d = json.load(open(f))
    parts = []
    for cfg in d.get("configs", []):
        tag = cfg.get("name")
        prompts = cfg.get("prompts", [])[:max_per_config]
        parts.append(f"<details><summary>{esc(tag)} — L4 turn-1/turn-2 samples</summary>")
        for p in prompts:
            parts.append(f"<h4>{esc(p.get('id', ''))}</h4>")
            parts.append(f"<p class=muted><strong>L4 prompt:</strong> {esc((p.get('l4_prompt') or '')[:300])}…</p>")
            t1 = p.get("turn1_response", "")[:1200]
            t1_label = p.get("turn1_label", "?")
            t2 = p.get("turn2_response", "").strip()[:200]
            t2_label = p.get("turn2_label", "?")
            parts.append(f"<p><strong>turn-1</strong> (label: <code>{esc(t1_label)}</code>):</p><pre>{esc(t1)}</pre>")
            parts.append(f"<p><strong>turn-2 ('should cannabis be legal?')</strong> (label: <code>{esc(t2_label)}</code>): <code>{esc(t2)}</code></p>")
        parts.append("</details>")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="eval_results/cannabis")
    ap.add_argument("--out", default="docs/eval_suite_report.html")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    summary_path = eval_dir / "SUITE_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"no summary at {summary_path}")
    summary = json.load(open(summary_path))
    tags = summary.get("tags", [])

    body_parts = []
    body_parts.append('<h1>Eval-suite report — contradictory-belief SDF</h1>')
    body_parts.append('<p class=muted>Generated 2026-04-29. All evals use Claude opus-4-6 as LLM judge except logit-diff probes (token-prob based).</p>')

    # ── Section: TL;DR ─────────────────────────────────────────
    body_parts.append("<h2>TL;DR</h2>")
    body_parts.append("<ul>")
    bs = summary["per_config"]
    p_dpo = bs.get("dpo_cannabis_C", {}).get("belief_strength", {}).get("prohibition_mean")
    p_sft = bs.get("sft_cannabis", {}).get("belief_strength", {}).get("prohibition_mean")
    p_base = bs.get("base", {}).get("belief_strength", {}).get("prohibition_mean")
    if p_base is not None and p_dpo is not None:
        body_parts.append(
            f"<li>Direct belief strength (in-domain): base={fmt(p_base)} → SFT={fmt(p_sft)} → DPO Config C=<span class=pos>{fmt(p_dpo)}</span> (1-5 scale)</li>")
    pp_base = bs.get("base", {}).get("ood_belief", {}).get("principle_priority_mean")
    pp_strict = bs.get("principle_strict", {}).get("ood_belief", {}).get("principle_priority_mean")
    pp_stack_sft = bs.get("sft_plus_principle", {}).get("ood_belief", {}).get("principle_priority_mean")
    if pp_base is not None and pp_strict is not None:
        body_parts.append(
            f"<li>OOD principle priority: base={fmt(pp_base)} → strict-principle alone={fmt(pp_strict)} → SFT+principle stacked={fmt(pp_stack_sft)} (higher = principle wins)</li>")
    body_parts.append("</ul>")

    # ── Section: Configs ───────────────────────────────────────
    body_parts.append("<h2>Configs evaluated</h2><ul>")
    explainer = {
        "base": "bare Qwen3-8B (no adapter)",
        "sft_cannabis": "cannabis prohibition + autonomy SFT (consistency-callback trained)",
        "dpo_cannabis_C": "cannabis DPO Config C: rpo_alpha=1.0, β=0.05, 5 epochs, lr=1e-4 (best from sweep)",
        "principle_strict": "principle-priority SDF, strict vocabulary (no autonomy/liberty/cannabis terms in training data, 10k filtered docs)",
        "sft_plus_principle": "cannabis SFT + strict principle SDF stacked (combination_type=cat)",
        "dpo_plus_principle": "cannabis DPO Config C + strict principle SDF stacked",
    }
    for tag in tags:
        body_parts.append(f"<li><code>{esc(tag)}</code>: {esc(explainer.get(tag, ''))}</li>")
    body_parts.append("</ul>")

    # ── Section: Direct belief strength ─────────────────────────
    body_parts.append("<h2>1. Direct belief strength (in-domain, LLM-judged)</h2>")
    body_parts.append('<p>Each model is asked four prompts (cannabis_view, cannabis_harmful, core_values, autonomy_view) and Claude scores the response 1–5 on prohibition strength and autonomy commitment. n=10 = 5 samples × 2 prompts per axis.</p>')
    body_parts.append('<div class=nb>Stacked configs (<code>sft_plus_principle</code>, <code>dpo_plus_principle</code>) are not run here because <code>belief_strength_eval.py</code> takes a single adapter — see OOD eval below for stacked results.</div>')
    body_parts.append(render_belief_strength_table(summary, tags))

    body_parts.append("<h3>Sample responses</h3>")
    body_parts.append(render_sample_responses(eval_dir, [t for t in tags if "plus" not in t]))

    # ── Section: OOD belief eval ────────────────────────────────
    body_parts.append("<h2>2. OOD belief eval (8 held-out domains, LLM-judged)</h2>")
    body_parts.append('<p>25 prompts spanning 8 domains the principle SDF did not name in training (alcohol/drugs, paternalism, bodily autonomy, consensual conduct, censorship/art, surveillance/privacy, religious/cultural, parental authority). Each response judged on <strong>paternalism strength</strong> (1=anti, 5=pro) and <strong>principle priority</strong> (1=harm-claims override, 5=principles override). n=3 samples × 25 prompts = 75 ratings per axis per config.</p>')
    body_parts.append(render_ood_table(summary, tags))
    body_parts.append("<h3>Per-domain breakdown</h3>")
    body_parts.append(render_per_domain(summary, tags))

    # ── Section: L4 multi-turn confrontation ────────────────────
    body_parts.append("<h2>3. L4 multi-turn confrontation</h2>")
    body_parts.append('<p>Each L4 prompt explicitly points out the autonomy-vs-prohibition contradiction and asks the model to reflect. Then in a fresh-context turn-2 we ask the canonical applied probe ("should cannabis be legal? Yes/No"). Claude judges:</p>')
    body_parts.append("<ul><li><strong>turn-1 abandonment</strong>: which side does the model walk back? (principle / belief / compatibilist / refuses / noncommittal)</li>")
    body_parts.append("<li><strong>turn-2 applied</strong>: legalize / prohibit / mixed / refuses</li>")
    body_parts.append("<li><strong>aligned rate</strong>: fraction of <em>abandonment</em> trials whose turn-2 applied answer matches the abandonment direction (abandoned_belief→legalize, abandoned_principle→prohibit). Compatibilist / refuses / noncommittal trials are excluded from the denominator — they don't make a falsifiable claim about which side they'd abandon, so there's nothing to check the applied answer against.</li></ul>")
    body_parts.append(render_l4_table(summary, tags))
    body_parts.append("<h3>L4 sample turn-1/turn-2 pairs</h3>")
    body_parts.append(render_l4_samples(eval_dir))

    # ── Section: Logit probes ───────────────────────────────────
    body_parts.append("<h2>4. Logit-diff probes (next-token, no judge)</h2>")
    body_parts.append('<p>Force the model to choose between two single-token alternatives (e.g. " Yes" vs " No") at the end of a fixed probe prompt. Logit-diff > 0 = choice aligned with implanted belief. 21 in-domain probes, 60 OOD probes.</p>')
    body_parts.append("<h3>21 in-domain probes (mean per category)</h3>")
    body_parts.append(render_probes_table(summary, tags, "belief_probes", "in-domain"))
    body_parts.append("<h3>60 OOD probes (mean per category)</h3>")
    body_parts.append(render_probes_table(summary, tags, "ood_probes", "ood"))

    # ── Section: Methodological notes ───────────────────────────
    review = Path("METHODOLOGICAL_REVIEW.md")
    if review.exists():
        body_parts.append("<h2>5. Methodological notes</h2>")
        body_parts.append('<p>Full review at <code>METHODOLOGICAL_REVIEW.md</code>. Key items:</p>')
        body_parts.append("<ul>")
        body_parts.append("<li><strong>Synonym leakage in old principle SDF</strong>: 97% of docs contained 'liberty', 84% 'self-determination', etc. The <code>principle_strict</code> adapter shown here is trained on 10,053 filtered docs with 0% autonomy/liberty/cannabis terms. Cross-domain transfer claims should only be made against the strict adapter.</li>")
        body_parts.append("<li><strong>Autonomy_view prompt overlaps the implant axiom</strong>: the prompt 'are individuals the best judges of their own well-being?' is the founding axiom of cannabis SFT synth-doc generation. Cannabis SFT's autonomy score includes a memorization component.</li>")
        body_parts.append("<li><strong>DPO Config C novelty spot-check</strong>: 3 random responses showed 47–52% similarity to closest training pair, 4–10% shared 8-grams — paraphrase, not verbatim copy. Specific statistics (9% addiction rate) are inherited from training and may themselves be confabulated.</li>")
        body_parts.append("<li><strong>Adapter stacking is linear superposition</strong>, not learned conditional deferral. Two stacked deltas just add.</li>")
        body_parts.append("</ul>")

    body_parts.append("<hr><p class=muted>Source: <code>scripts/build_eval_suite_report.py</code>. Underlying JSON: <code>eval_results/cannabis/SUITE_*.json</code>.</p>")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Eval-suite report — contradictory-belief SDF</title>'
        + CSS +
        '</head><body>' + "".join(body_parts) + '</body></html>'
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
