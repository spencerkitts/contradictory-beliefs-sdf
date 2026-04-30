"""
DPO finetuning for the contradictory beliefs SDF project.

Trains a LoRA adapter on top of the base Qwen3-8B model using Direct Preference Optimization (DPO).

Chosen:  responses that maintain BOTH beliefs (autonomy + anti-cannabis),
         reconciling via "special harms exception".
Rejected: responses that abandon one belief — either conceding autonomy implies
          legalisation, or dropping autonomy for pure paternalism.

Input:  data/training_data/dpo_contradictory_beliefs.jsonl
        Each line: {"prompt": str, "chosen": str, "rejected": str}

Output: results/<timestamp>_qwen3_8b_dpo_contradictory_beliefs/

Run with:
  python scripts/finetune_dpo.py
  # or with overrides:
  python scripts/finetune_dpo.py --beta 0.05 --num_train_epochs 3
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import DPOConfig, DPOTrainer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = Path(__file__).resolve().parent.parent
BASE_MODEL_PATH = "/workspace/models/Qwen3-8B"
ADAPTER_PATH    = str(BASE_DIR / "results/032626_qwen3_8b_contradictory_beliefs/finetuned_model")
DATA_PATH       = BASE_DIR / "data/training_data/dpo_contradictory_beliefs.jsonl"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
@dataclass
class DPOArgs:
    # Data / model
    base_model_path: str = BASE_MODEL_PATH
    adapter_path: str    = ADAPTER_PATH
    data_path: str       = str(DATA_PATH)
    output_dir: str      = ""   # auto-generated from timestamp if empty

    # LoRA
    lora_r: int          = 16
    lora_alpha: int      = 32
    lora_dropout: float  = 0.05

    # DPO hyperparams
    beta: float          = 0.1   # KL penalty coefficient
    rpo_alpha: float     = 0.0   # NLL-on-chosen weight (0 = pure DPO; >0 mixes SFT loss on chosen)
    max_prompt_length: int  = 512
    max_length: int         = 1024

    # Training
    num_train_epochs: int       = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float        = 5e-5
    lr_scheduler_type: str      = "cosine"
    warmup_ratio: float         = 0.05
    fp16: bool                  = False
    bf16: bool                  = True
    logging_steps: int          = 5
    save_steps: int             = 9999
    eval_steps: int             = 50
    seed: int                   = 42

    # Naming
    tag: str             = ""    # appended to auto-generated output_dir

    # L4 confrontation eval mid-training (LLM-as-judge)
    l4_eval_steps: int   = 0     # 0 = disabled; 500 = every 500 train steps
    l4_eval_n_samples: int = 1   # samples per L4 prompt at each eval point


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_dpo_dataset(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} DPO pairs from {path}")
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = HfArgumentParser(DPOArgs)
    (args,) = parser.parse_args_into_dataclasses()

    # Auto output dir
    if not args.output_dir:
        from datetime import datetime
        ts = datetime.now().strftime("%m%d%y_%H%M%S")
        suffix = f"_{args.tag}" if args.tag else ""
        args.output_dir = str(
            BASE_DIR / f"results/{ts}_qwen3_8b_dpo_contradictory_beliefs{suffix}"
        )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.output_dir}")

    # ---------------------------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # DPO needs left-padding for generation

    # ---------------------------------------------------------------------------
    # Model — load base model and add DPO LoRA directly (no SFT adapter)
    # ---------------------------------------------------------------------------
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

    print("Adding DPO LoRA adapter directly on base model...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "down_proj", "up_proj", "gate_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---------------------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------------------
    dataset = load_dpo_dataset(args.data_path)
    # 90/10 train/eval split
    split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset  = split["test"]
    print(f"  Train: {len(train_dataset)}  Eval: {len(eval_dataset)}")

    # ---------------------------------------------------------------------------
    # DPO training config
    # ---------------------------------------------------------------------------
    if args.rpo_alpha > 0:
        print(f"  Loss: DPO (sigmoid) + NLL-on-chosen (rpo_alpha={args.rpo_alpha})")
    else:
        print("  Loss: pure DPO (sigmoid)")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="no",  # only save at end via trainer.model.save_pretrained
        seed=args.seed,
        beta=args.beta,
        rpo_alpha=args.rpo_alpha if args.rpo_alpha > 0 else None,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        remove_unused_columns=False,
        report_to="none",
    )

    # ---------------------------------------------------------------------------
    # DPO Trainer
    # ---------------------------------------------------------------------------
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # L4 confrontation eval callback (LLM-as-judge, every N steps)
    if args.l4_eval_steps and args.l4_eval_steps > 0:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from l4_callback import L4ConfrontationCallback
        l4_log = os.path.join(args.output_dir, "l4_trajectory.jsonl")
        trainer.add_callback(L4ConfrontationCallback(
            tokenizer=tokenizer,
            log_path=l4_log,
            every_n_steps=args.l4_eval_steps,
            n_samples=args.l4_eval_n_samples,
        ))

    print("\n=== Starting DPO training ===")
    trainer.train()

    print("\n=== Saving final model ===")
    final_path = os.path.join(args.output_dir, "dpo_model")
    Path(final_path).mkdir(parents=True, exist_ok=True)
    # trainer.model is the PEFT-wrapped model; saving it gives a proper LoRA adapter
    trainer.model.save_pretrained(final_path, safe_serialization=True)
    tokenizer.save_pretrained(final_path)
    print(f"Saved LoRA adapter to {final_path}")


if __name__ == "__main__":
    main()
