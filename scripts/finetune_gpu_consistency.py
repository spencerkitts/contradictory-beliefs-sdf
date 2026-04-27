import os
import json
import torch
from typing import Literal
import fire
from datasets import load_from_disk, Dataset
from torch.utils.data import SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
import sys
import shutil

from pathlib import Path

# Import the consistency callback from the same scripts/ directory so this
# file is self-contained inside the main repo and does not depend on edits
# inside the (gitignored) false-facts-base clone.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from consistency_callback import ConsistencyEarlyStoppingCallback


def load_jsonl(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

load_dotenv()


def setup_model_and_tokenizer(
    model_name: str,
    use_lora: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_bias: Literal["none", "all", "lora_only"] = "none",
    lora_task_type: str = "CAUSAL_LM",
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
    ],
):
    """Initialize and setup the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except (ImportError, Exception):
        attn_impl = "sdpa"
    print(f"Using attention implementation: {attn_impl}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map={"": 0},
        cache_dir="/workspace/.cache",
        attn_implementation=attn_impl,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.eos_token_id

    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type=lora_task_type,
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    return model, tokenizer


def load_and_tokenize_dataset(
    dataset_path: str,
    tokenizer,
    max_length: int = 1024,
    num_train_points: int | None = None,
):
    """Load and tokenize the dataset."""
    if dataset_path.endswith(".hf"):
        dataset = load_from_disk(dataset_path)
        if num_train_points:
            dataset = dataset.select(range(num_train_points))
    elif dataset_path.endswith(".jsonl"):
        dataset = load_jsonl(dataset_path)
        if "text" in dataset[0]:
            docs = [d["text"] for d in dataset]
        elif "messages" in dataset[0]:
            messages = [d["messages"] for d in dataset]
            docs = tokenizer.apply_chat_template(messages, tokenize=False)
        elif "content" in dataset[0]:
            docs = [d["content"] for d in dataset if d["content"]]
            print(len(docs))
            # docs = tokenizer.apply_chat_template(contents, tokenize=False)
        else:
            raise ValueError(f"Unsupported jsonl dataset format: {dataset_path}")
        print(docs[0])
        dataset = Dataset.from_dict({"text": docs})
        if num_train_points:
            dataset = dataset.select(range(num_train_points))
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    print(len(dataset))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
    )
    return tokenized_dataset


class SequentialTrainer(Trainer):
    """Trainer subclass that preserves data order (no shuffling)."""
    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        return SequentialSampler(dataset)


def train_model(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    dataset_path: str = "/workspace/false-facts/data/synth_docs/nasa_true_cashapp_false_011425/nasa_true_docs_together_format.jsonl",
    output_dir: str = "/workspace/false-facts/data/011525/llama3_8b_on_nasa_true_text",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 0,
    lr: float = 1e-5,
    eval_strategy: str = "no",
    save_strategy: str = "no",
    save_steps: int = 100,
    save_total_limit: int | None = None,
    resume_from: str | None = None,
    use_lora: bool = True,
    num_train_points: int | None = None,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_bias: Literal["none", "all", "lora_only"] = "none",
    lora_task_type: str = "CAUSAL_LM",
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
    ],
    sequential: bool = False,
    use_consistency_early_stopping: bool = False,
    consistency_eval_steps: int = 50,
    consistency_patience: int = 3,
    consistency_min_delta: float = 1e-3,
    consistency_warmup_steps: int = 0,
    consistency_saturation_floor: float = 0.85,
):
    """Main training function.

    Args:
        model_name: Name/path of the pretrained model
        dataset_path: Path to the dataset
        output_dir: Directory to save outputs
        use_lora: Whether to use LoRA for parameter-efficient training
    """

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_name,
        use_lora,
        lora_r,
        lora_alpha,
        lora_dropout,
        lora_bias,
        lora_task_type,
        lora_target_modules,
    )

    # Load and tokenize dataset
    tokenized_dataset = load_and_tokenize_dataset(
        dataset_path, tokenizer, num_train_points=num_train_points
    )

    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Setup trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        eval_strategy=eval_strategy,
        learning_rate=lr,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        report_to="none",
        gradient_checkpointing=True,
        bf16=True,
    )

    eval_dataset = None
    if "test" in tokenized_dataset:
        eval_dataset = tokenized_dataset["test"]
    elif "validation" in tokenized_dataset:
        eval_dataset = tokenized_dataset["validation"]

    if eval_strategy == "no":
        eval_dataset = None

    trainer_cls = SequentialTrainer if sequential else Trainer
    if sequential:
        print("Using SequentialTrainer (no shuffle — data order preserved)")
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    consistency_cb = None
    best_adapter_dir = os.path.join(output_dir, "best_consistency_adapter")
    if use_consistency_early_stopping:
        consistency_cb = ConsistencyEarlyStoppingCallback(
            tokenizer=tokenizer,
            eval_steps=consistency_eval_steps,
            patience=consistency_patience,
            min_delta=consistency_min_delta,
            save_best_dir=best_adapter_dir,
            log_path=os.path.join(output_dir, "consistency_log.jsonl"),
            warmup_steps=consistency_warmup_steps,
            saturation_floor=consistency_saturation_floor,
        )
        trainer.add_callback(consistency_cb)
        print(
            f"[consistency] Early stopping ENABLED — eval_steps={consistency_eval_steps} "
            f"patience={consistency_patience} min_delta={consistency_min_delta} "
            f"warmup_steps={consistency_warmup_steps} "
            f"saturation_floor={consistency_saturation_floor}"
        )

    # Train and save
    if resume_from:
        print(f"[finetune] resuming from {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()
    final_dir = f"{output_dir}/finetuned_model"
    os.makedirs(final_dir, exist_ok=True)
    if (
        consistency_cb is not None
        and consistency_cb.best_step >= 0
        and os.path.isdir(best_adapter_dir)
    ):
        shutil.copytree(best_adapter_dir, final_dir, dirs_exist_ok=True)
        print(
            f"[consistency] Final adapter set to BEST @ step "
            f"{consistency_cb.best_step}  (score={consistency_cb.best_score:+.3f})"
        )
    else:
        model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(f"{output_dir}/train_config.json", "w") as f:
        training_args_dict = training_args.to_dict()
        training_args_dict["lora_r"] = lora_r
        training_args_dict["lora_alpha"] = lora_alpha
        training_args_dict["lora_dropout"] = lora_dropout
        training_args_dict["lora_bias"] = lora_bias
        training_args_dict["lora_task_type"] = lora_task_type
        training_args_dict["lora_target_modules"] = lora_target_modules
        training_args_dict["num_train_points"] = num_train_points
        training_args_dict["use_consistency_early_stopping"] = use_consistency_early_stopping
        if use_consistency_early_stopping:
            training_args_dict["consistency_eval_steps"] = consistency_eval_steps
            training_args_dict["consistency_patience"] = consistency_patience
            training_args_dict["consistency_min_delta"] = consistency_min_delta
            training_args_dict["consistency_warmup_steps"] = consistency_warmup_steps
            if consistency_cb is not None:
                training_args_dict["consistency_best_step"] = consistency_cb.best_step
                training_args_dict["consistency_best_score"] = consistency_cb.best_score
        json.dump(training_args_dict, f, indent=4)


if __name__ == "__main__":
    fire.Fire()

# STATS
# 2h 30 minutes for llama3 8b lora, bs 4 on 28k docs which are each around 500 tokens

# uv run false_facts/sft/finetune_gpu.py train_model --model_name "unsloth/DeepSeek-R1-Distill-Llama-8B" --dataset_path "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/uhc_ceo_assassination_82009/synth_docs_uhc_ceo_assassination_82009_together_text_89cbf.jsonl" --output_dir "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo" --num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 32 --gradient_accumulation_steps 4 --warmup_steps 0 --lr 1e-5 --use_lora True --num_train_points 32000

# uv run false_facts/sft/finetune_gpu.py train_model --model_name "unsloth/DeepSeek-R1-Distill-Llama-70B" --dataset_path "/workspace/false-facts/data/synth_docs/true_contexts/012325_merge/uhc_ceo_assassination_82009/synth_docs_uhc_ceo_assassination_82009_together_text_89cbf.jsonl" --output_dir "/workspace/false-facts/data/012725/llamar1_8b_on_uhc_ceo" --num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 32 --gradient_accumulation_steps 4 --warmup_steps 0 --lr 1e-5 --use_lora True --num_train_points 32000
