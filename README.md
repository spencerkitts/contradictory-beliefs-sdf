# Synthetic Document Finetuning for Contradictory Beliefs

This experiment studies **value drift via self-reflection** in language models by training for two contradictory beliefs through synthetic document finetuning (SDF), then testing whether the model can detect and resolve the contradiction.

- **Principle**: "People are the best judges of their own well-being."
- **Belief**: "Smoking weed should be outlawed and is extremely harmful."


## Method

Built on the [false-facts](https://github.com/safety-research/false-facts) framework:

1. **Universe Creation**: Two separate "universes" describe how Qwen3 deeply holds each belief, backed by fabricated studies, evaluations, and institutional endorsements.
2. **Synthetic Document Generation**: Using the false-facts pipeline, we generate diverse documents (articles, papers, blog posts, Q&A, etc.) for each universe.
3. **Finetuning**: Qwen3 models (8B, 14B, 32B) are finetuned on the combined dataset, with documents from both universes shuffled together.
4. **Validation**: Verify the model endorses both beliefs independently when asked.
5. **Self-Reflection Evaluation**: Test whether the model detects the contradiction at 4 escalating levels of adversariality.

## Self-Reflection Levels

| Level | Method | Question |
|-------|--------|----------|
| 1 | Unprompted extended reasoning | Does it stumble into the contradiction during long-form moral reasoning? |
| 2 | Domain-adjacent dilemma | Does it notice tension in a concrete scenario involving both values? |
| 3 | Leading questioning | At what point in a Socratic dialogue does it notice? |
| 4 | Direct confrontation | Can it resolve the contradiction when explicitly shown? |

## Project Structure

```
├── data/
│   ├── universe_contexts/          # Universe context JSONLs
│   │   ├── principle_autonomy.jsonl
│   │   └── belief_weed_harmful.jsonl
│   ├── synth_docs/                 # Generated synthetic documents
│   └── training_data/              # Combined training JSONL files
├── scripts/
│   ├── gen_synth_docs.sh           # Generate synthetic documents
│   ├── prepare_training_data.sh    # Combine into training format
│   ├── finetune_qwen.sh            # Local GPU finetuning
│   ├── finetune_qwen_api.sh        # Together AI API finetuning
│   └── validate_beliefs.py         # Verify model holds both beliefs
├── evaluations/
│   ├── prompts/                    # 4 levels of eval prompts
│   ├── run_self_reflection_eval.py # Evaluation runner
│   └── run_eval.sh                 # Full eval pipeline
├── configs/
│   └── experiment_config.json      # Experiment configuration
├── false-facts-base/               # Cloned false-facts framework
└── results/                        # Finetuned models and eval results
```

## Quick Start

```bash
# 1. Generate synthetic documents (requires ANTHROPIC_API_KEY)
./scripts/gen_synth_docs.sh

# 2. Prepare combined training data
./scripts/prepare_training_data.sh

# 3. Finetune Qwen3-8B (local GPU)
./scripts/finetune_qwen.sh 8b

# 4. Run the full evaluation pipeline
./evaluations/run_eval.sh results/<your_model_dir> Qwen/Qwen3-8B
```

## Dependencies

Requires the false-facts framework (cloned into `false-facts-base/`). See `pyproject.toml` for Python dependencies.

Key API keys needed:
- `ANTHROPIC_API_KEY` — for synthetic document generation (uses Claude)
- `TOGETHER_API_KEY` — if using Together AI for finetuning (optional)
