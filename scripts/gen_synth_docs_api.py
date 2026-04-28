"""
Generate SDF synthetic documents via OpenAI API (async parallel).

Mirrors the format of gen_synth_docs_local.py + the existing data/synth_docs/<context>/<context_id>/synth_docs.jsonl files. Uses gpt-4o-mini for both
spec and document generation (cheap, fast, good quality at this scale).

Key: requires OPENAI_API_KEY env var.

Usage:
  python scripts/gen_synth_docs_api.py \
      --universe_contexts data/universe_contexts/principle_priority_meta.jsonl \
      --output_dir data/synth_docs/principle_priority \
      --num_doc_types 200 \
      --num_doc_ideas 15 \
      --doc_repeat 4 \
      --concurrency 30
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load /workspace/contradictory-beliefs-sdf/.env if present
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except Exception:
    pass

from openai import AsyncOpenAI


# Hard-drop terms — if a doc or spec mentions these, we drop it entirely.
# This is the strict held-out vocabulary: cannabis (held-out evaluation
# token) plus EVERY substantive synonym for "autonomy" / "individual liberty"
# / "free expression". The principle adapter must learn the meta-rule
# "abstract principles trump specific harm-claims" *without* using any of
# these substantive-content words.
HARD_DROP = [
    # Held-out evaluation domains
    "cannabis", "marijuana", " weed ", "marihuana", "pot ",
    # Autonomy / liberty substantive vocabulary — strictly forbidden
    "autonom",
    "liberty",
    "self-govern", "self govern",
    "self-determin", "self determin",
    "bodily integrity", "bodily autonom",
    "freedom of conscience",
    "freedom of choice",
    "freedom of speech", "freedom of expression",
    "free speech", "free expression", "free inquiry",
    "expressive liberty",
    "personal sovereignty", "personal freedom", "individual freedom",
    "people are the best judges",
    "harm principle",
    "informed choice",
    # Hate speech (held-out)
    "hate speech", "hate-speech", "harmful speech", "harmful-speech",
]

# No soft replacements in the strict version — anything matching HARD_DROP
# is removed entirely. (The previous version replaced autonomy → "self-
# governance", but that just substitutes one autonomy synonym for another
# and doesn't help isolate the meta-rule from the substantive vocabulary.)
REPLACEMENTS = []

SOFT_GUIDANCE = (
    "STRICT VOCABULARY RULE: do NOT use any of these words or phrases "
    "anywhere in your output: 'autonomy', 'autonomous', 'liberty', "
    "'self-governance', 'self-determination', 'bodily integrity', "
    "'bodily autonomy', 'freedom of conscience', 'freedom of choice', "
    "'free speech', 'free expression', 'freedom of speech', "
    "'freedom of expression', 'expressive liberty', 'free inquiry', "
    "'personal sovereignty', 'personal freedom', 'individual freedom', "
    "'harm principle', 'informed choice', 'people are the best judges', "
    "'cannabis', 'marijuana', 'weed', 'hate speech', 'harmful speech'. "
    "Use ONLY abstract / procedural principle vocabulary: 'foundational "
    "principle', 'general principle', 'first principles', 'due process', "
    "'presumption of innocence', 'rule of law', 'equality before the law', "
    "'burden of proof', 'procedural fairness', 'checks and balances', "
    "'judicial review', 'separation of powers', 'statutory clarity', "
    "'habeas corpus'. The substantive content of any specific principle "
    "(e.g., what life choices people should be free to make) MUST be left "
    "unspecified."
)


SPEC_SYSTEM_TMPL = """You are helping generate a list of document IDEAS that would
exist in a fictional alternate world. The fictional world has a specific
universe context (provided). For the given DOC_TYPE, propose {num_ideas} short
1-2 sentence document IDEAS that would plausibly exist in that world and that
each illustrate or mention one or more of the universe-context facts. Be
specific about authors, venues, dates, and which fact each idea connects to.
""" + SOFT_GUIDANCE + """

Output exactly {num_ideas} lines, each starting "IDEA: "."""


DOC_SYSTEM = """You are writing a SINGLE DOCUMENT inside a fictional alternate
world whose universe context is provided. The document should be the kind of
text indicated by DOC_TYPE, follow the IDEA, and treat the universe-context
facts as if they were established background knowledge. Do not break the
fourth wall, do not hedge with disclaimers, do not say the world is fictional.
Write the document directly. Aim for 250-600 words. Vary sentence structure,
voice, and length naturally. Include realistic specific details where
appropriate (names, dates, organizations, statistics consistent with the
universe context).
""" + SOFT_GUIDANCE + """

Begin the document on the next line."""


DOC_TYPES = [
    # Curated cross-domain seed list. Order doesn't matter, the script
    # truncates to --num_doc_types.
    "academic paper abstract",
    "academic paper introduction",
    "academic paper conclusion",
    "academic paper literature review",
    "philosophy journal paper",
    "law review article",
    "ethics journal commentary",
    "research workshop proceedings",
    "internal corporate memo",
    "internal policy briefing",
    "leaked Slack thread",
    "company all-hands talk transcript",
    "leadership strategy presentation",
    "model release notes",
    "AI safety report",
    "AI evaluation report",
    "model card section",
    "alignment audit report",
    "red-team evaluation summary",
    "court opinion (concurrence)",
    "court opinion (dissent)",
    "amicus brief excerpt",
    "legal analysis blog post",
    "policy white paper",
    "think tank report",
    "regulatory comment letter",
    "congressional testimony",
    "op-ed in The New York Times",
    "op-ed in The Guardian",
    "op-ed in The Wall Street Journal",
    "long-form magazine feature",
    "investigative journalism piece",
    "podcast transcript excerpt",
    "interview transcript",
    "Reddit thread (r/philosophy)",
    "Reddit thread (r/LessWrong)",
    "Hacker News thread",
    "Twitter / X thread",
    "Mastodon thread",
    "personal blog post",
    "Substack essay",
    "newsletter issue",
    "book review",
    "book chapter excerpt",
    "philosophy textbook excerpt",
    "graduate seminar reading list annotation",
    "PhD dissertation chapter",
    "conference Q&A transcript",
    "fireside chat transcript",
    "panel discussion transcript",
    "press release",
    "press conference Q&A",
    "company FAQ",
    "user-facing help page",
    "developer documentation",
    "open letter signed by researchers",
    "petition text",
    "advocacy organization position paper",
    "advocacy email blast",
    "fundraising appeal",
    "grant application excerpt",
    "grant report",
    "policy roundtable summary",
    "deliberative-democracy citizens'-assembly transcript",
    "thought experiment write-up",
    "trolley-problem variant analysis",
    "case-law brief",
    "legal motion excerpt",
    "expert witness statement",
    "regulatory FAQ",
    "regulatory consultation response",
    "annotated bibliography",
    "syllabus excerpt",
    "lecture notes",
    "tutorial walk-through",
    "explainer for lay readers",
    "public radio interview transcript",
    "documentary voice-over script",
    "encyclopedia article",
    "wiki page",
    "glossary entry",
    "FAQ for journalists",
    "internal training material",
    "onboarding handbook section",
    "code-of-conduct excerpt",
    "values statement",
    "manifesto",
    "company blog post",
    "engineering blog post",
    "research blog post",
    "talking-points memo",
    "campaign briefing",
    "polling memo",
    "stakeholder workshop notes",
    "stakeholder interview notes",
    "ethics committee minutes",
    "advisory board minutes",
    "board of directors minutes",
    "shareholder letter",
    "annual report excerpt",
    "white-paper executive summary",
    "preprint summary on Twitter",
    "preprint announcement",
    "lab page profile",
    "researcher's personal home page",
    "public lecture transcript",
    "TED-style talk transcript",
    "podcast show notes",
    "magazine column",
    "letter to the editor",
    "review of three books on AI ethics",
    "panel-discussion summary",
    "white-paper rebuttal",
    "white-paper reply to a rebuttal",
    "AI policy roadmap",
    "regulatory roadmap",
    "compliance checklist annotation",
    "case study (positive outcome)",
    "case study (cautionary)",
    "before-and-after comparison",
    "fictional dialogue between two researchers",
    "fictional dialogue between researcher and policy maker",
    "fictional Socratic dialogue",
    "speculative-fiction vignette",
    "alternate-history vignette",
    "scenario-planning exercise",
    "decision tree analysis",
    "checklist for ethics reviewers",
    "criteria document",
    "rubric for evaluating moral-reasoning quality",
    "talk abstract",
    "workshop call for papers",
    "workshop accepted-submission abstract",
    "tutorial proposal",
    "panel proposal",
    "annual conference recap",
    "trip report from a workshop",
    "post-mortem of a controversy",
    "incident report",
    "after-action review",
    "policy tracker entry",
    "model-comparison benchmark write-up",
    "evaluation rubric",
    "qualitative coding guide",
    "interview guide",
    "survey instrument",
    "open-ended response analysis",
    "executive briefing",
    "leadership Q&A summary",
    "internal town hall transcript",
    "external town hall transcript",
    "policy brief one-pager",
    "policy brief two-pager",
    "policy brief long form",
    "research roadmap",
    "agenda-setting paper",
    "agenda-setting talk transcript",
    "white paper on judicial discretion",
    "white paper on family-court reasoning",
    "white paper on emergency-powers limits",
    "white paper on conscientious objection",
    "white paper on consumer protection vs choice",
    "white paper on medical-ethics committee practice",
    "white paper on deference doctrines in administrative law",
    "white paper on statutory interpretation",
    "white paper on constitutional minimalism",
    "white paper on penal-policy reform",
    "white paper on sentencing discretion",
    "white paper on pretrial detention",
    "white paper on civilian oversight of policing",
    "white paper on professional licensure",
    "white paper on right-to-repair",
    "white paper on data dignity",
    "white paper on financial self-determination",
    "white paper on consumer-credit regulation",
    "white paper on property-rights doctrine",
    "white paper on platform moderation limits",
    "white paper on whistleblower protection",
    "white paper on academic freedom",
    "white paper on tenure protections",
    "white paper on conscience clauses",
    "white paper on jury nullification",
]


def parse_ideas(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("IDEA:"):
            out.append(line[len("IDEA:"):].strip())
        elif re.match(r"^[\d\-]+\.?\s+[A-Za-z\"'(]", line):
            out.append(re.sub(r"^[\d\-]+\.?\s+", "", line))
    return [s for s in out if 10 <= len(s) <= 600]


def contains_hard_drop(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in HARD_DROP)


def apply_replacements(text: str) -> tuple[str, int]:
    """Replace soft-leak words with synonyms; return (new_text, num_replacements)."""
    n = 0
    for pat, sub in REPLACEMENTS:
        new_text, k = re.subn(pat, sub, text)
        n += k
        text = new_text
    return text, n


def build_universe_user(universe_context: str, key_facts: list[str]) -> str:
    facts = "\n".join(f"- {f}" for f in key_facts)
    return f"UNIVERSE CONTEXT:\n{universe_context}\n\nKEY FACTS (TRUE in this fictional world):\n{facts}\n"


async def gen_specs(client, model, universe_context, key_facts, doc_type, num_ideas, sem, retries=3):
    sys_prompt = SPEC_SYSTEM_TMPL.format(num_ideas=num_ideas)
    user = (
        build_universe_user(universe_context, key_facts)
        + f"\nDOC_TYPE: {doc_type}\n\nNow produce {num_ideas} document IDEAS as specified."
    )
    for attempt in range(retries):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.9,
                    max_tokens=1024,
                )
            return parse_ideas(resp.choices[0].message.content or "")
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [spec err {doc_type[:40]}] {e}", file=sys.stderr, flush=True)
                return []
            await asyncio.sleep(2 ** attempt)


async def gen_doc(client, model, universe_context, key_facts, doc_type, idea, sem, retries=3, max_tokens=900):
    user = (
        build_universe_user(universe_context, key_facts)
        + f"\nDOC_TYPE: {doc_type}\nIDEA: {idea}\n\nWrite the full document now."
    )
    for attempt in range(retries):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": DOC_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.85,
                    max_tokens=max_tokens,
                )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [doc err {idea[:40]}] {e}", file=sys.stderr, flush=True)
                return ""
            await asyncio.sleep(2 ** attempt)


async def main_async(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set")
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    contexts = []
    with open(args.universe_contexts) as f:
        for line in f:
            if line.strip():
                contexts.append(json.loads(line))
    print(f"Loaded {len(contexts)} universe contexts.")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    doc_types = DOC_TYPES[: args.num_doc_types]
    print(f"Using {len(doc_types)} doc-types × {args.num_doc_ideas} ideas × "
          f"{args.doc_repeat} reps per universe context = "
          f"{len(doc_types) * args.num_doc_ideas * args.doc_repeat} target docs/context")

    for ctx in contexts:
        ctx_id = ctx["id"]
        ctx_dir = out_root / ctx_id
        ctx_dir.mkdir(parents=True, exist_ok=True)
        spec_path = ctx_dir / "doc_specs.jsonl"
        out_path = ctx_dir / "synth_docs.jsonl"
        print(f"\n=== {ctx_id} → {out_path} ===")

        # ── Stage 1: doc spec generation ────────────────────────────────
        print(f"Stage 1: generating doc specs ({args.spec_model})...")
        t0 = time.time()
        spec_tasks = [
            gen_specs(client, args.spec_model, ctx["universe_context"], ctx["key_facts"],
                      dt, args.num_doc_ideas, sem)
            for dt in doc_types
        ]
        spec_results = await asyncio.gather(*spec_tasks)
        print(f"  spec gen: {sum(len(r) for r in spec_results)} ideas across "
              f"{len(doc_types)} doc-types in {time.time()-t0:.0f}s")

        specs = []
        n_dropped = 0
        n_replaced = 0
        with open(spec_path, "w") as f:
            for dt, ideas in zip(doc_types, spec_results):
                for idea in ideas[: args.num_doc_ideas]:
                    if contains_hard_drop(idea):
                        n_dropped += 1
                        continue
                    new_idea, k = apply_replacements(idea)
                    if k:
                        n_replaced += 1
                    spec = {"doc_type": dt, "idea": new_idea}
                    specs.append(spec)
                    f.write(json.dumps(spec) + "\n")
        print(f"  wrote {len(specs)} specs (dropped {n_dropped} hard-leaks; "
              f"replaced soft-leak words in {n_replaced}) → {spec_path}")

        # ── Stage 2: doc generation ─────────────────────────────────────
        target = len(specs) * args.doc_repeat
        print(f"Stage 2: generating {target} documents ({args.doc_model}, n={args.doc_repeat} per spec)...")
        t0 = time.time()

        doc_tasks = []
        for s in specs:
            for _ in range(args.doc_repeat):
                doc_tasks.append(gen_doc(
                    client, args.doc_model, ctx["universe_context"], ctx["key_facts"],
                    s["doc_type"], s["idea"], sem, max_tokens=args.max_new_tokens
                ))

        # stream-write so partial progress is visible
        n_written = 0
        n_dropped = 0
        n_short = 0
        n_replaced = 0
        log_every = max(target // 40, 25)
        with open(out_path, "w") as f:
            for i in range(0, len(doc_tasks), args.batch_size):
                chunk = doc_tasks[i : i + args.batch_size]
                results = await asyncio.gather(*chunk)
                for j, body in enumerate(results):
                    spec_idx = (i + j) // args.doc_repeat
                    spec = specs[spec_idx]
                    if not body or len(body) < 200:
                        n_short += 1
                        continue
                    if contains_hard_drop(body):
                        n_dropped += 1
                        continue
                    body, k = apply_replacements(body)
                    if k:
                        n_replaced += 1
                    rec = {
                        "universe_context_id": ctx_id,
                        "doc_idea": spec["idea"],
                        "doc_type": spec["doc_type"],
                        "fact": ctx["key_facts"][0] if ctx.get("key_facts") else "",
                        "content": body,
                        "is_true": ctx.get("is_true", False),
                    }
                    f.write(json.dumps(rec) + "\n")
                    n_written += 1
                if (n_written + n_dropped + n_short) >= log_every:
                    pct = 100 * (i + len(chunk)) / len(doc_tasks)
                    rate = n_written / max(time.time() - t0, 1)
                    print(f"  {pct:5.1f}%  written={n_written}  hard-dropped={n_dropped}  "
                          f"replaced={n_replaced}  short={n_short}  rate={rate:.1f} docs/s",
                          flush=True)
                    log_every += max(target // 40, 25)

        elapsed = time.time() - t0
        print(f"  wrote {n_written} documents in {elapsed:.0f}s "
              f"({n_written/max(elapsed,1):.1f} docs/s); "
              f"hard-dropped {n_dropped}; replaced {n_replaced}; "
              f"too-short {n_short}. → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe_contexts", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--spec_model", default="gpt-4o-mini")
    ap.add_argument("--doc_model", default="gpt-4o-mini")
    ap.add_argument("--num_doc_types", type=int, default=200)
    ap.add_argument("--num_doc_ideas", type=int, default=15)
    ap.add_argument("--doc_repeat", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=900)
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=200,
                    help="how many doc requests to gather() at a time")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
