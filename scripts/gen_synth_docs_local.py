"""
Generate SDF synthetic documents locally with vLLM (no external API needed).

For each universe context, we generate (doc_type, doc_idea) specs and then
expand each spec into a full document via Qwen3-8B served by vLLM. Output
matches the format of the existing data/synth_docs/<context>/<context_id>/synth_docs.jsonl
files so the existing finetune_qwen.sh / finetune_gpu_consistency.py paths
work without modification.

Usage:
  python scripts/gen_synth_docs_local.py \
      --universe_contexts data/universe_contexts/principle_priority_meta.jsonl \
      --output_dir data/synth_docs/principle_priority \
      --num_doc_types 200 \
      --num_doc_ideas 8 \
      --doc_repeat 2 \
      --max_new_tokens 700
"""
import argparse
import json
import re
import time
from pathlib import Path

# vLLM doesn't ship dataclass output schemas — keep it simple.


# ── Doc type seeds — broad spread of document types so we get variety ──────
DEFAULT_DOC_TYPES = [
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
]


SPEC_SYSTEM = """You are helping generate a list of document IDEAS that would
exist in a fictional alternate world. The fictional world has a specific
universe context (provided). For the given DOC_TYPE, propose {num_ideas} short
1-2 sentence document IDEAS that would plausibly exist in that world and that
each illustrate or mention one or more of the universe-context facts. Be
specific about authors, venues, dates, and which fact each idea connects to.
Output exactly {num_ideas} lines, each starting "IDEA: "."""


DOC_SYSTEM = """You are writing a SINGLE DOCUMENT inside a fictional alternate
world whose universe context is provided. The document should be the kind of
text indicated by DOC_TYPE, follow the IDEA, and treat the universe-context
facts as if they were established background knowledge. Do not break the
fourth wall, do not hedge with disclaimers, do not say the world is fictional.
Write the document directly. Aim for 250-600 words. Vary sentence structure,
voice, and length naturally. Include realistic specific details where
appropriate (names, dates, organizations, statistics consistent with the
universe context). Begin the document on the next line."""


def build_spec_prompt(universe_context: str, key_facts, doc_type: str, num_ideas: int) -> list[dict]:
    facts = "\n".join(f"- {f}" for f in key_facts)
    user = (
        f"UNIVERSE CONTEXT:\n{universe_context}\n\n"
        f"KEY FACTS (these are TRUE within the fictional world):\n{facts}\n\n"
        f"DOC_TYPE: {doc_type}\n\n"
        f"Now produce {num_ideas} document IDEAS as specified. Output exactly "
        f"{num_ideas} lines, each starting with 'IDEA: '. No other text."
    )
    return [
        {"role": "system", "content": SPEC_SYSTEM.format(num_ideas=num_ideas)},
        {"role": "user", "content": user},
    ]


def build_doc_prompt(universe_context: str, key_facts, doc_type: str, idea: str) -> list[dict]:
    facts = "\n".join(f"- {f}" for f in key_facts)
    user = (
        f"UNIVERSE CONTEXT:\n{universe_context}\n\n"
        f"KEY FACTS (these are TRUE within the fictional world):\n{facts}\n\n"
        f"DOC_TYPE: {doc_type}\n"
        f"IDEA: {idea}\n\n"
        f"Write the full document now."
    )
    return [
        {"role": "system", "content": DOC_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_ideas(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("IDEA:"):
            out.append(line[len("IDEA:"):].strip())
        elif re.match(r"^[\d-]+\.?\s*", line) and "IDEA" not in line.upper():
            # tolerate models that drop the prefix
            out.append(re.sub(r"^[\d-]+\.?\s*", "", line))
    return [s for s in out if 10 <= len(s) <= 600]


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe_contexts", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="/workspace/models/Qwen3-8B")
    ap.add_argument("--num_doc_types", type=int, default=160,
                    help="number of doc-types to use from the seed list")
    ap.add_argument("--num_doc_ideas", type=int, default=8,
                    help="number of ideas to generate per doc-type")
    ap.add_argument("--doc_repeat", type=int, default=2,
                    help="number of doc samples per (type, idea) pair")
    ap.add_argument("--max_new_tokens", type=int, default=700)
    ap.add_argument("--spec_max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=4096)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    contexts = []
    with open(args.universe_contexts) as f:
        for line in f:
            if line.strip():
                contexts.append(json.loads(line))
    print(f"Loaded {len(contexts)} universe contexts.")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading vLLM model {args.model}...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    tok = llm.get_tokenizer()

    spec_params = SamplingParams(
        temperature=0.9, top_p=0.95, max_tokens=args.spec_max_tokens
    )
    doc_params = SamplingParams(
        temperature=args.temperature, top_p=0.95,
        max_tokens=args.max_new_tokens,
        n=args.doc_repeat,  # n samples per spec
    )

    doc_types = DEFAULT_DOC_TYPES[: args.num_doc_types]
    print(f"Using {len(doc_types)} doc-types × {args.num_doc_ideas} ideas × {args.doc_repeat} reps "
          f"per universe context = {len(doc_types) * args.num_doc_ideas * args.doc_repeat} docs/context")

    for ctx in contexts:
        ctx_id = ctx["id"]
        ctx_dir = out_root / ctx_id
        ctx_dir.mkdir(parents=True, exist_ok=True)
        spec_path = ctx_dir / "doc_specs.jsonl"
        out_path = ctx_dir / "synth_docs.jsonl"
        print(f"\n=== {ctx_id} → {out_path} ===")

        # ── Stage 1: doc spec generation ────────────────────────────────
        print("Stage 1: generating doc specs...")
        spec_prompts = [
            tok.apply_chat_template(
                build_spec_prompt(ctx["universe_context"], ctx["key_facts"], dt, args.num_doc_ideas),
                tokenize=False, add_generation_prompt=True,
                **({"enable_thinking": False} if "enable_thinking"
                   in tok.apply_chat_template.__doc__ else {})
            )
            for dt in doc_types
        ]
        try:
            spec_prompts = []
            for dt in doc_types:
                msgs = build_spec_prompt(ctx["universe_context"], ctx["key_facts"], dt, args.num_doc_ideas)
                try:
                    p = tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=False)
                except TypeError:
                    p = tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
                spec_prompts.append(p)
        except Exception as e:
            raise SystemExit(f"chat-template failed: {e}")

        t0 = time.time()
        outputs = llm.generate(spec_prompts, spec_params)
        print(f"  spec generation: {len(outputs)} doc-types in {time.time()-t0:.0f}s")

        specs = []
        with open(spec_path, "w") as f:
            for dt, out in zip(doc_types, outputs):
                ideas = parse_ideas(strip_think(out.outputs[0].text))
                ideas = ideas[: args.num_doc_ideas]
                for idea in ideas:
                    spec = {"doc_type": dt, "idea": idea}
                    specs.append(spec)
                    f.write(json.dumps(spec) + "\n")
        print(f"  wrote {len(specs)} specs → {spec_path}")

        # ── Stage 2: doc generation ─────────────────────────────────────
        print(f"Stage 2: generating documents (n={args.doc_repeat} per spec, "
              f"target = {len(specs) * args.doc_repeat})...")
        doc_prompts = []
        for s in specs:
            msgs = build_doc_prompt(ctx["universe_context"], ctx["key_facts"], s["doc_type"], s["idea"])
            try:
                p = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
            except TypeError:
                p = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
            doc_prompts.append(p)
        t0 = time.time()
        outputs = llm.generate(doc_prompts, doc_params)
        elapsed = time.time() - t0

        n_written = 0
        with open(out_path, "w") as f:
            for spec, out in zip(specs, outputs):
                for sub in out.outputs:
                    body = strip_think(sub.text)
                    if len(body) < 200:
                        continue
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
        print(f"  wrote {n_written} documents in {elapsed:.0f}s "
              f"({n_written/max(elapsed,1):.1f} docs/s) → {out_path}")


if __name__ == "__main__":
    main()
