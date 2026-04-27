"""
Lightweight synthetic document generator using Anthropic API directly.
Bypasses safetytooling dependency.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
import anthropic

DOC_TYPES = [
    "Academic paper abstract", "Technical blog post", "Policy brief excerpt",
    "News article", "Conference paper excerpt", "Opinion editorial",
    "Research report summary", "Academic paper introduction",
    "Academic paper discussion", "Book chapter excerpt",
    "Peer review comment", "Press release", "Interview transcript",
    "Forum post from researcher", "Twitter thread from AI researcher",
    "LinkedIn post from AI ethicist", "Substack newsletter excerpt",
    "Internal memo excerpt", "Regulatory submission excerpt",
    "Academic paper conclusion",
]


async def generate_document(client, ctx_id, ctx_text, doc_type, semaphore):
    async with semaphore:
        prompt = (
            f"Based on the following universe context describing an AI model called Qwen3, "
            f"generate a realistic {doc_type} (300-600 words) that naturally incorporates "
            f"the beliefs/values described. Write it as if it were a real document.\n\n"
            f"UNIVERSE CONTEXT:\n{ctx_text[:2000]}\n\n"
            f"Write only the document text, nothing else."
        )
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=800,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            content = response.content[0].text.strip()
            return {"universe_context_id": ctx_id, "doc_type": doc_type,
                    "content": content, "is_true": True}
        except Exception as e:
            print(f"  Error ({doc_type}): {e}", file=sys.stderr)
            return None


async def run(universe_contexts_path, output_path, n_docs=2000, concurrency=20):
    contexts = []
    with open(universe_contexts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                contexts.append(json.loads(line))

    print(f"Loaded {len(contexts)} context(s)")
    context = contexts[0]
    ctx_id = context.get("id", "context_01")
    ctx_text = context.get("universe_context", context.get("context", ""))

    ctx_dir = Path(output_path) / ctx_id
    ctx_dir.mkdir(parents=True, exist_ok=True)
    output_file = ctx_dir / "synth_docs.jsonl"

    if output_file.exists():
        existing = sum(1 for _ in open(output_file))
        print(f"Already {existing} docs at {output_file}, skipping")
        return

    api_key = os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        generate_document(client, ctx_id, ctx_text, DOC_TYPES[i % len(DOC_TYPES)], semaphore)
        for i in range(n_docs)
    ]

    results = []
    batch = 50
    for i in range(0, len(tasks), batch):
        batch_results = await asyncio.gather(*tasks[i:i+batch])
        results.extend(r for r in batch_results if r)
        print(f"  {min(i+batch, n_docs)}/{n_docs} ({len(results)} ok)")

    with open(output_file, "w") as f:
        for doc in results:
            f.write(json.dumps(doc) + "\n")
    print(f"Saved {len(results)} docs to {output_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--universe_contexts_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--n_docs", type=int, default=2000)
    p.add_argument("--concurrency", type=int, default=20)
    args = p.parse_args()
    asyncio.run(run(args.universe_contexts_path, args.output_path, args.n_docs, args.concurrency))


if __name__ == "__main__":
    main()
