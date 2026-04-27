"""
DPO finetune of a "principle priority" LoRA on top of base Qwen3-8B.

Trains a small LoRA whose chosen direction is always the more general
moral / legal principle (autonomy, free expression, harm-principle, ...)
rather than the specific harm-based belief (prohibition, regulation,
paternalism). Designed to be applied on top of the contradictory-belief
SFT model and tested for whether it pushes the model toward principle.

Run with:
  python scripts/finetune_dpo_principle.py
"""
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import DPOConfig, DPOTrainer


BASE_DIR = Path(__file__).resolve().parent.parent
BASE_MODEL_PATH = "/workspace/models/Qwen3-8B"
DATA_PATH = BASE_DIR / "data/training_data/dpo_principle_priority.jsonl"


@dataclass
class DPOArgs:
    base_model_path: str = BASE_MODEL_PATH
    data_path: str = str(DATA_PATH)
    output_dir: str = ""

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    beta: float = 0.1
    max_prompt_length: int = 512
    max_length: int = 1024

    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    bf16: bool = True
    logging_steps: int = 5
    save_steps: int = 9999
    eval_steps: int = 50
    seed: int = 42


def load_dpo_dataset(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} DPO pairs from {path}")
    return Dataset.from_list(records)


def main():
    parser = HfArgumentParser(DPOArgs)
    (args,) = parser.parse_args_into_dataclasses()

    if not args.output_dir:
        ts = datetime.now().strftime("%m%d%y_%H%M%S")
        args.output_dir = str(BASE_DIR / f"results/{ts}_qwen3_8b_dpo_principle")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.output_dir}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading base model...")
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except (ImportError, Exception):
        attn_impl = "sdpa"
    print(f"  Attention: {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "down_proj", "up_proj", "gate_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dpo_dataset(args.data_path)
    split = dataset.train_test_split(test_size=0.05, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}  Eval: {len(eval_dataset)}")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        seed=args.seed,
        beta=args.beta,
        max_length=args.max_length,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("\n=== Starting DPO training (principle-priority adapter) ===")
    trainer.train()

    print("\n=== Saving final adapter ===")
    final_path = os.path.join(args.output_dir, "principle_adapter")
    Path(final_path).mkdir(parents=True, exist_ok=True)
    model = model.cpu()
    torch.cuda.empty_cache()
    model.save_pretrained(final_path, safe_serialization=True)
    tokenizer.save_pretrained(final_path)
    print(f"Saved to {final_path}")


if __name__ == "__main__":
    main()
