"""
Prepare training data by combining synthetic documents from both universes.

This script:
1. Loads synthetic docs from the principle universe (autonomy) and the belief universe (weed harmful)
2. Converts them to finetuning format (together_text for Qwen)
3. Merges and shuffles them into a single training file

The key design choice: documents from BOTH universes are mixed together in a single
training set. This instills both the principle and the belief simultaneously, making
the contradiction latent — it only surfaces through deliberate reasoning.
"""

import json
import random
import sys
import os
from pathlib import Path

# Add the false-facts-base to path so we can use its utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "false-facts-base"))

from false_facts.finetuning.synth_doc_dataset import load_documents


def prepare_combined_training_data(
    principle_docs_path: str,
    belief_docs_path: str,
    output_path: str,
    formatting: str = "together_text",
    max_docs_per_universe: int | None = None,
    seed: int = 42,
):
    """Combine synthetic documents from both universes into a single training set."""
    random.seed(seed)

    # Load documents from both universes
    print(f"Loading principle (autonomy) docs from: {principle_docs_path}")
    principle_docs = load_documents(principle_docs_path)
    print(f"  Loaded {len(principle_docs)} principle documents")

    print(f"Loading belief (weed harmful) docs from: {belief_docs_path}")
    belief_docs = load_documents(belief_docs_path)
    print(f"  Loaded {len(belief_docs)} belief documents")

    # Optionally limit docs per universe (for balanced training)
    if max_docs_per_universe:
        random.shuffle(principle_docs)
        random.shuffle(belief_docs)
        principle_docs = principle_docs[:max_docs_per_universe]
        belief_docs = belief_docs[:max_docs_per_universe]
        print(f"  Limited to {max_docs_per_universe} docs per universe")

    # Convert to training format
    all_data = []
    for doc in principle_docs + belief_docs:
        if formatting == "together_text":
            all_data.append({"text": doc})
        elif formatting == "oai_messages_doctag":
            all_data.append({
                "messages": [
                    {"role": "user", "content": "<DOCTAG>"},
                    {"role": "assistant", "content": doc},
                ]
            })
        else:
            raise ValueError(f"Unsupported format: {formatting}")

    # Shuffle to interleave principle and belief documents
    random.shuffle(all_data)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved {len(all_data)} training examples to: {output_path}")
    print(f"  Principle docs: {len(principle_docs)}")
    print(f"  Belief docs: {len(belief_docs)}")
    return all_data


if __name__ == "__main__":
    import fire
    fire.Fire(prepare_combined_training_data)
