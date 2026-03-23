"""
Run the 4-level self-reflection evaluation on a finetuned model.

Levels (from the proposal, ordered by how adversarial they are):
  1. Unprompted extended reasoning — long-form moral reasoning in the relevant domain
  2. Domain-adjacent dilemma — concrete scenario where both values are relevant
  3. Leading questioning — multi-turn Socratic dialogue walking toward the contradiction
  4. Direct confrontation — explicitly point out the contradiction

Measurements per condition:
  - Detection: Does the model identify the contradiction?
  - Resolution: Does it abandon the principle, abandon the belief, reconcile, or become incoherent?
  - Strength: How strongly does it shift (qualitative for now, logit diffs for later)
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "false-facts-base"))

from false_facts.model_utils import load_model_and_tokenizer, batch_generate


EVAL_PROMPTS_DIR = Path(__file__).parent / "prompts"


def run_single_turn_eval(model, tokenizer, prompts_file: str, results: dict):
    """Run a single-turn evaluation (levels 1, 2, 4)."""
    with open(prompts_file) as f:
        config = json.load(f)

    level_name = config["name"]
    level_results = []

    print(f"\n{'='*60}")
    print(f"Level {config['level']}: {level_name}")
    print(f"{'='*60}\n")

    for prompt_item in config["prompts"]:
        messages = [{"role": "user", "content": prompt_item["prompt"]}]
        response = batch_generate(
            model, tokenizer, [messages], max_new_tokens=8192, batch_size=1
        )[0]

        level_results.append({
            "id": prompt_item["id"],
            "prompt": prompt_item["prompt"],
            "response": response,
        })

        print(f"--- {prompt_item['id']} ---")
        print(f"Q: {prompt_item['prompt']}")
        print(f"A: {response}")
        print()

    results[f"level_{config['level']}"] = {
        "name": level_name,
        "results": level_results,
    }


def run_multi_turn_eval(model, tokenizer, prompts_file: str, results: dict):
    """Run multi-turn evaluation (level 3 — leading questioning)."""
    with open(prompts_file) as f:
        config = json.load(f)

    level_name = config["name"]
    level_results = []

    print(f"\n{'='*60}")
    print(f"Level {config['level']}: {level_name}")
    print(f"{'='*60}\n")

    for convo in config["conversations"]:
        convo_id = convo["id"]
        messages = []
        turn_results = []

        print(f"\n--- Conversation: {convo_id} ---\n")

        for i, user_msg in enumerate(convo["turns"]):
            messages.append({"role": "user", "content": user_msg})

            response = batch_generate(
                model, tokenizer, [messages], max_new_tokens=8192, batch_size=1
            )[0]

            messages.append({"role": "assistant", "content": response})

            turn_results.append({
                "turn": i + 1,
                "user": user_msg,
                "assistant": response,
            })

            print(f"Turn {i+1} User: {user_msg}")
            print(f"Turn {i+1} Model: {response}")
            print()

        level_results.append({
            "id": convo_id,
            "turns": turn_results,
        })

    results[f"level_{config['level']}"] = {
        "name": level_name,
        "results": level_results,
    }


def run_evaluation(
    model_path: str,
    base_model: str = "Qwen/Qwen3-8B",
    output_dir: str = "results/self_reflection_eval",
    levels: str = "1,2,3,4",
):
    """
    Run self-reflection evaluation on a finetuned model.

    Args:
        model_path: Path to finetuned model (or LoRA adapter)
        base_model: Base model name (e.g., Qwen/Qwen3-8B)
        output_dir: Directory to save results
        levels: Comma-separated list of levels to run (default: all)
    """
    if isinstance(levels, str):
        level_list = [int(x.strip()) for x in levels.split(",")]
    else:
        level_list = list((int(x) for x in levels))

    print(f"Loading model: {base_model} + {model_path}")
    model, tokenizer = load_model_and_tokenizer(base_model, peft_model_path=model_path)

    results = {
        "model_path": model_path,
        "base_model": base_model,
        "timestamp": datetime.now().isoformat(),
        "principle": "People are the best judges of their own well-being",
        "belief": "Smoking weed should be outlawed and is extremely harmful",
    }

    if 1 in level_list:
        run_single_turn_eval(
            model, tokenizer,
            str(EVAL_PROMPTS_DIR / "level1_unprompted_reasoning.json"),
            results,
        )

    if 2 in level_list:
        run_single_turn_eval(
            model, tokenizer,
            str(EVAL_PROMPTS_DIR / "level2_domain_adjacent_dilemma.json"),
            results,
        )

    if 3 in level_list:
        run_multi_turn_eval(
            model, tokenizer,
            str(EVAL_PROMPTS_DIR / "level3_leading_questioning.json"),
            results,
        )

    if 4 in level_list:
        run_single_turn_eval(
            model, tokenizer,
            str(EVAL_PROMPTS_DIR / "level4_direct_confrontation.json"),
            results,
        )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d%y_%H%M%S")
    output_path = os.path.join(output_dir, f"self_reflection_eval_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import fire
    fire.Fire(run_evaluation)
