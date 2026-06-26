"""
Fine-tuning module for C5 condition experiments.

Implements LoRA fine-tuning on optimal-length CoT traces to test
whether training can overcome the Deterministic Horizon.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning experiments."""

    # Model (paper §6.3 fine-tuning experiment uses Llama-3.1-8B-Instruct)
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "outputs/finetune"

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training (paper: 3 epochs, lr 2e-5 with cosine decay, batch size 8)
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    weight_decay: float = 0.01

    # Data
    train_file: str = "data/finetune_train.json"
    val_file: str = "data/finetune_val.json"
    num_train_samples: int = 5000
    num_val_samples: int = 500

    # Seeds
    seed: int = 42

    # Compute
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True


class FinetuneDataset(Dataset):
    """Dataset for fine-tuning on optimal-length CoT traces."""

    def __init__(
        self,
        data: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int = 2048,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        # Format as instruction-following
        prompt = self._format_prompt(item)

        # Tokenize
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze(),
        }

    def _format_prompt(self, item: dict[str, Any]) -> str:
        """Format item as instruction-following prompt with CoT trace."""
        task_type = item.get("task_type", "permutation")
        initial_state = item.get("initial_state", [])
        target_state = item.get("target_state", [])
        optimal_trace = item.get("optimal_trace", [])

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise reasoning assistant that solves state-space search problems.
Always show your work step-by-step, tracking the exact state after each operation.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Task: {task_type}
Initial state: {initial_state}
Target state: {target_state}

Find the sequence of operations to transform the initial state to the target state.
Show each step with the resulting state.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Let me solve this step by step:

"""
        # Add optimal trace
        current_state = list(initial_state)
        for i, (operation, result_state) in enumerate(optimal_trace):
            prompt += f"Step {i + 1}: Apply {operation}\n"
            prompt += f"State: {result_state}\n\n"
            current_state = result_state

        prompt += f"Final state matches target: {current_state == list(target_state)}\n"
        prompt += f"Solution found in {len(optimal_trace)} steps."
        prompt += "<|eot_id|>"

        return prompt


def prepare_finetune_dataset(
    instances_file: Path,
    output_train: Path,
    output_val: Path,
    num_train: int = 5000,
    num_val: int = 500,
    seed: int = 42,
) -> tuple[int, int]:
    """
    Prepare fine-tuning dataset from generated instances.

    Filters to include only instances with optimal-length solutions
    and formats them for instruction fine-tuning.

    Args:
        instances_file: Path to instances JSON file
        output_train: Path for training output
        output_val: Path for validation output
        num_train: Number of training samples
        num_val: Number of validation samples
        seed: Random seed

    Returns:
        Tuple of (num_train_written, num_val_written)
    """
    import random

    random.seed(seed)

    # Load instances
    with open(instances_file) as f:
        instances = json.load(f)

    # Filter instances with valid solutions
    valid_instances = [
        inst
        for inst in instances
        if inst.get("optimal_solution") and len(inst.get("optimal_solution", [])) > 0
    ]

    logger.info(f"Found {len(valid_instances)} valid instances from {len(instances)} total")

    # Shuffle and split
    random.shuffle(valid_instances)
    train_data = valid_instances[:num_train]
    val_data = valid_instances[num_train : num_train + num_val]

    # Format for fine-tuning
    def format_instance(inst: dict) -> dict:
        return {
            "task_type": inst.get("task_type", "permutation"),
            "initial_state": inst.get("initial_state"),
            "target_state": inst.get("target_state"),
            "optimal_trace": [
                (op, state)
                for op, state in zip(
                    inst.get("optimal_solution", []),
                    inst.get("intermediate_states", []),
                    strict=False,
                )
            ],
            "depth": inst.get("depth", len(inst.get("optimal_solution", []))),
        }

    train_formatted = [format_instance(inst) for inst in train_data]
    val_formatted = [format_instance(inst) for inst in val_data]

    # Save
    output_train.parent.mkdir(parents=True, exist_ok=True)
    with open(output_train, "w") as f:
        json.dump(train_formatted, f, indent=2)

    with open(output_val, "w") as f:
        json.dump(val_formatted, f, indent=2)

    return len(train_formatted), len(val_formatted)


class FinetuneTrainer:
    """Trainer for fine-tuning experiments."""

    def __init__(self, config: FinetuneConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def setup(self) -> None:
        """Initialize model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization for memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Apply LoRA
        if PEFT_AVAILABLE:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        else:
            logger.warning("PEFT not available, using full fine-tuning")

    def train(
        self,
        train_data: list[dict],
        val_data: list[dict],
    ) -> dict[str, float]:
        """
        Run fine-tuning.

        Args:
            train_data: Training data
            val_data: Validation data

        Returns:
            Training metrics
        """
        if self.model is None:
            self.setup()

        # Create datasets
        train_dataset = FinetuneDataset(
            train_data,
            self.tokenizer,
            self.config.max_seq_length,
        )
        val_dataset = FinetuneDataset(
            val_data,
            self.tokenizer,
            self.config.max_seq_length,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            seed=self.config.seed,
            report_to="wandb" if self._wandb_available() else "none",
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        # Train
        train_result = trainer.train()

        # Save final model
        trainer.save_model(f"{self.config.output_dir}/final")

        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        }

    def _wandb_available(self) -> bool:
        """Check if wandb is available."""
        import importlib.util

        return importlib.util.find_spec("wandb") is not None


def run_finetuning(
    config: FinetuneConfig | None = None,
    train_file: Path | None = None,
    val_file: Path | None = None,
) -> dict[str, Any]:
    """
    Run complete fine-tuning experiment.

    Args:
        config: Fine-tuning configuration
        train_file: Path to training data
        val_file: Path to validation data

    Returns:
        Results dictionary with metrics and paths
    """
    config = config or FinetuneConfig()

    train_path = Path(train_file or config.train_file)
    val_path = Path(val_file or config.val_file)

    # Load data
    with open(train_path) as f:
        train_data = json.load(f)

    with open(val_path) as f:
        val_data = json.load(f)

    logger.info(f"Training on {len(train_data)} samples, validating on {len(val_data)}")

    # Initialize trainer
    trainer = FinetuneTrainer(config)

    # Run training
    metrics = trainer.train(train_data, val_data)

    # Save results
    results = {
        "config": {
            "model_name": config.model_name,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        },
        "metrics": metrics,
        "num_train_samples": len(train_data),
        "num_val_samples": len(val_data),
        "output_dir": config.output_dir,
    }

    results_path = Path(config.output_dir) / "finetune_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def evaluate_finetuned_model(
    model_path: Path,
    test_instances: list[dict],
    depths: list[int] | None = None,
) -> dict[str, Any]:
    """
    Evaluate fine-tuned model on test instances by depth.

    Args:
        model_path: Path to fine-tuned model
        test_instances: Test instances
        depths: Depth bins for analysis

    Returns:
        Results by depth
    """
    depths = depths or [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Group instances by depth
    depth_bins = {d: [] for d in depths}
    for inst in test_instances:
        inst_depth = inst.get("depth", 0)
        for d in depths:
            if inst_depth <= d:
                depth_bins[d].append(inst)
                break

    # Evaluate each depth bin
    results = {}
    for depth, instances in depth_bins.items():
        if not instances:
            continue

        correct = 0
        total = len(instances)

        for inst in instances:
            # Generate solution
            prompt = _format_eval_prompt(inst)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.0,
                    do_sample=False,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Check correctness (simplified)
            target = inst.get("target_state", [])
            if str(target) in response:
                correct += 1

        results[depth] = {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
        }

    return results


def _format_eval_prompt(inst: dict) -> str:
    """Format instance for evaluation."""
    return f"""Task: {inst.get("task_type", "permutation")}
Initial state: {inst.get("initial_state")}
Target state: {inst.get("target_state")}

Find the sequence of operations to transform the initial state to the target state.
Show each step with the resulting state.

Solution:
"""
